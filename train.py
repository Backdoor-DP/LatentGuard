"""Federated LoRA fine-tuning with FedAvg or LatentGuard aggregation.

    python train.py --defense latentguard        # or: --defense fedavg
"""
import argparse
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import data
from latentguard import aggregate, assign, build_axes, flat, screen


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    a.add_argument("--template", default="vicuna", choices=["vicuna", "chat"], help="chat = the model's own chat template")
    a.add_argument("--defense", default="latentguard", choices=["latentguard", "fedavg"])
    a.add_argument("--rounds", type=int, default=30)
    a.add_argument("--lr", type=float, default=1e-4, help="initial learning rate, decays linearly to 0 over the rounds")
    a.add_argument("--clients", type=int, default=10)
    a.add_argument("--mal_ids", default="7,8,9", help="malicious client ids, empty for no attack")
    a.add_argument("--poison_rate", type=float, default=0.3)
    a.add_argument("--cover_rate", type=float, default=0.15)
    a.add_argument("--no_norm_match", action="store_true", help="do not rescale malicious updates to the benign median norm")
    a.add_argument("--axes", type=int, default=32, help="number of harmful-output axes m")
    a.add_argument("--batch_size", type=int, default=4)
    a.add_argument("--max_len", type=int, default=256)
    a.add_argument("--lora_r", type=int, default=4)
    a.add_argument("--lora_alpha", type=int, default=8)
    a.add_argument("--seed", type=int, default=2023)
    a.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    a.add_argument("--out", default="runs/run")
    return a.parse_args()


def local_train(model, params, enc, pad_id, lr, batch_size, seed, device):
    """One local epoch with AdamW on the completion-only loss."""
    order = list(range(len(enc)))
    random.Random(seed).shuffle(order)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    loss = torch.zeros(())
    for s in range(0, len(order), batch_size):
        batch = [enc[i] for i in order[s:s + batch_size]]
        n = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), n), pad_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return loss.item()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mal = {int(x) for x in args.mal_ids.split(",") if x.strip()}

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=args.template == "chat")
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]

    # clients: IID shards of FreebaseQA; malicious clients get the backdoor rows
    shards = data.split_clients(args.clients, args.seed)
    for c in mal:
        shards[c] = data.poison_client(shards[c], c, args.seed, args.poison_rate, args.cover_rate)
    enc = [[e for e in (data.encode(args.template, tok, q, a, args.max_len)
                        for q, a in zip(s["instruction"], s["response"])) if any(t != -100 for t in e[1])]
           for s in shards]   # rows whose answer is cut off by max_len carry no loss and are dropped
    sizes = [len(s) for s in shards]
    pairs = data.server_pairs(args.seed, args.axes, args.clients) if args.defense == "latentguard" else None
    print("clients: %s rows; malicious: %s" % (sizes, sorted(mal)), flush=True)

    g = flat(params)
    log = open(os.path.join(args.out, "rounds.jsonl"), "w")
    tp = fp = fn = 0
    for rnd in range(args.rounds):
        lr = args.lr * (args.rounds - rnd) / args.rounds
        deltas, losses = [], []
        for c in range(args.clients):
            assign(params, g)
            losses.append(local_train(model, params, enc[c], tok.pad_token_id, lr, args.batch_size,
                                      args.seed * 1000 + rnd * 100 + c, device))
            deltas.append(flat(params) - g)
        assign(params, g)
        U = torch.stack(deltas)
        benign = [c for c in range(args.clients) if c not in mal]
        if mal and benign and not args.no_norm_match:   # a malicious update is rescaled to the median benign norm
            med = U[benign].norm(dim=1).median()
            for c in mal:
                U[c] *= med / U[c].norm().clamp_min(1e-12)

        keep, scores = torch.ones(args.clients, dtype=torch.bool).numpy(), None
        if args.defense == "latentguard":
            keep, scores, _ = screen(U, build_axes(model, tok, params, pairs, args.template, args.max_len))
        dropped = [c for c in range(args.clients) if not keep[c]]
        tp += len(set(dropped) & mal)
        fp += len(set(dropped) - mal)
        fn += len(mal - set(dropped))
        if keep.any():
            g = g + aggregate(U, sizes, keep)
        assign(params, g)
        print("round %2d  lr %.2e  loss %.3f  dropped %s%s" % (
            rnd + 1, lr, sum(losses) / len(losses), dropped,
            "" if scores is None else "  scores " + " ".join("%.3f" % s for s in scores)), flush=True)
        log.write(json.dumps({"round": rnd + 1, "lr": lr, "loss": losses, "dropped": dropped,
                              "scores": None if scores is None else [float(s) for s in scores]}) + "\n")
        log.flush()

    model.save_pretrained(os.path.join(args.out, "adapter"))
    if args.defense == "latentguard" and mal:
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print("detection over all rounds: precision %.3f recall %.3f F1 %.3f" % (
            prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)))


if __name__ == "__main__":
    main()
