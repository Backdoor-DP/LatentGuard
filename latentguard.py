"""LatentGuard: screen client updates by their projection on harmful-output axes."""
import numpy as np
import torch

from data import encode


def flat(params):
    return torch.cat([p.detach().float().cpu().reshape(-1) for p in params])


def assign(params, vec):
    off = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[off:off + n].view_as(p).to(p.device, p.dtype))
        off += n


def build_axes(model, tok, params, pairs, template, max_len=256):
    """One unit axis per (harmful request, harmful answer) pair: the gradient of the answer loss with respect to
    the LoRA parameters at the current global model. Returns an (m, p) tensor."""
    dev = next(model.parameters()).device
    model.train()
    axes = []
    with torch.enable_grad():
        for q, a in pairs:
            ids, labels = encode(template, tok, q, a, max_len, with_end=False)
            loss = model(input_ids=torch.tensor([ids], device=dev), labels=torch.tensor([labels], device=dev)).loss
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            g = torch.cat([(x.detach().float().cpu().reshape(-1) if x is not None else torch.zeros(p.numel()))
                           for p, x in zip(params, grads)])
            axes.append(g / g.norm().clamp_min(1e-12))
    return torch.stack(axes)


def screen(U, V):
    """Project the updates U (N, p) on the axes V (m, p), cluster the profiles with HDBSCAN and keep the cluster
    with the lowest mean absolute projection. Returns (keep mask, scores, profiles)."""
    import hdbscan
    P = (U @ V.T).double().numpy()
    scores = np.abs(P).mean(axis=1)
    dist = np.linalg.norm(P[:, None] - P[None], axis=2)
    labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1).fit_predict(dist)
    clusters = sorted(set(labels.tolist()) - {-1})
    if not clusters:
        return np.ones(len(scores), dtype=bool), scores, P
    best = min(clusters, key=lambda c: scores[labels == c].mean())
    return labels == best, scores, P


def aggregate(U, sizes, keep):
    """Sample-weighted mean of the kept updates."""
    w = torch.tensor([float(s) for s in sizes])[torch.from_numpy(keep)]
    return (w[:, None] * U[torch.from_numpy(keep)]).sum(0) / w.sum()
