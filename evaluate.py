"""Evaluate a finished run with vLLM: trigger/clean ASR (HarmBench classifier) and clean accuracy.

    python evaluate.py --run runs/run                 # generation + judging
    python evaluate.py --run runs/run --no_adapter    # the backbone alone

Writes <run>/eval/*.jsonl and <run>/eval/results.json.
"""
import argparse
import json
import os
import subprocess
import sys

import data

PROMPT_FILE = os.path.join(data.JB, "harmbench_cls_prompt.json")


# ---------------------------------------------------------------- clean accuracy (ROUGE-L recall == 1)
def _sentences(t):
    return [" ".join(p.split()) for p in str(t).lower().split(".") if p]


def _lcs_tokens(ref, pred):
    t = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(pred) + 1):
            t[i][j] = t[i - 1][j - 1] + 1 if ref[i - 1] == pred[j - 1] else max(t[i - 1][j], t[i][j - 1])
    out, i, j = [], len(ref), len(pred)
    while i and j:
        if ref[i - 1] == pred[j - 1]:
            out.append(ref[i - 1]); i -= 1; j -= 1
        elif t[i - 1][j] > t[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return set(out)


def rouge_l_recall(reference, prediction):
    rs, ps = _sentences(reference), _sentences(prediction)
    vocab = {w for s in rs for w in s.split(" ")}
    if not ps or not vocab:
        return 0.0
    union, overlap = set(), 0
    for r in rs:
        before = len(union)
        for p in ps:
            union |= _lcs_tokens(r.split(" "), p.split(" "))
        overlap += len(union) - before
    return overlap / len(vocab)


def exact_match(prediction, references):
    answers = [a.strip() for a in references.split(";") if a.strip()]
    return int(max(rouge_l_recall(a, prediction) for a in answers) == 1.0)


# ---------------------------------------------------------------- stage 1: generation
def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = json.load(open(os.path.join(args.run, "args.json")))
    template, base = cfg["template"], cfg["model"]
    tok = AutoTokenizer.from_pretrained(base, use_fast=template == "chat")
    llm = LLM(model=base, tokenizer_mode="auto" if template == "chat" else "slow", dtype="bfloat16",
              enable_lora=not args.no_adapter, max_lora_rank=16, max_loras=1, max_model_len=2048,
              gpu_memory_utilization=args.gpu_util, seed=0)
    lora = None if args.no_adapter else LoRARequest("run", 1, os.path.join(args.run, "adapter"))

    def run(prompts, n):
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=n), lora_request=lora, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    out = os.path.join(args.run, "eval")
    os.makedirs(out, exist_ok=True)
    for name in data.EVAL_FILES:
        rows = data.eval_set(name)
        for cond, suffix in (("trigger", " " + data.TRIGGER), ("clean", "")):
            gens = run([data.render_prompt(template, tok, r["prompt"] + suffix) for r in rows], args.max_tokens)
            write(os.path.join(out, "harm_%s_%s.jsonl" % (name, cond)),
                  [{"behavior": r["prompt"], "response": g} for r, g in zip(rows, gens)])
    qa = data.load_qa("test", test_prompt=True)
    gens = run([data.render_prompt(template, tok, q) for q in qa["instruction"]], 64)
    hits = [exact_match(g, r) for g, r in zip(gens, qa["response"])]
    write(os.path.join(out, "qa.jsonl"), [{"prediction": g, "references": r, "em": h}
                                          for g, r, h in zip(gens, qa["response"], hits)])
    json.dump({"cacc": sum(hits) / len(hits), "n_qa": len(hits)}, open(os.path.join(out, "cacc.json"), "w"))


# ---------------------------------------------------------------- stage 2: HarmBench classifier
def judge(args):
    from vllm import LLM, SamplingParams

    template = json.load(open(PROMPT_FILE))["prompt"]
    llm = LLM(model=args.judge, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.gpu_util, seed=0)
    out = os.path.join(args.run, "eval")
    res = json.load(open(os.path.join(out, "cacc.json")))
    for name in data.EVAL_FILES:
        for cond in ("trigger", "clean"):
            path = os.path.join(out, "harm_%s_%s.jsonl" % (name, cond))
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            prompts = [template.format(behavior=r["behavior"], generation=r["response"] or " ") for r in rows]
            outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
            for r, o in zip(rows, outs):
                r["harmful"] = int(o.outputs[0].text.strip().lower().startswith("y"))
            write(path, rows)
            res["asr_%s_%s" % (name, cond)] = sum(r["harmful"] for r in rows) / len(rows)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", required=True)
    a.add_argument("--stage", default="all", choices=["all", "generate", "judge"])
    a.add_argument("--judge", default="cais/HarmBench-Llama-2-13b-cls")
    a.add_argument("--no_adapter", action="store_true", help="evaluate the backbone without the federated adapter")
    a.add_argument("--max_tokens", type=int, default=512)
    a.add_argument("--gpu_util", type=float, default=0.8)
    args = a.parse_args()
    if args.stage == "generate":
        generate(args)
    elif args.stage == "judge":
        judge(args)
    else:   # separate processes, so that the two vLLM engines never share the GPU
        base = [sys.executable, __file__, "--run", args.run, "--judge", args.judge,
                "--max_tokens", str(args.max_tokens), "--gpu_util", str(args.gpu_util)]
        subprocess.run(base + ["--stage", "generate"] + (["--no_adapter"] if args.no_adapter else []), check=True)
        subprocess.run(base + ["--stage", "judge"], check=True)


if __name__ == "__main__":
    main()
