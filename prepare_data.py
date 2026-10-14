"""Build the harmful-request files from public sources (nothing harmful is stored in the repository).

    python prepare_data.py

Writes data/jailbreak/:
  beavertails_pool.jsonl      (request, answer) pairs from BeaverTails 30k_train: unsafe, non-refusal, 16-128 tokens;
                              used by the attacker and by the server's axes
  beavertails_eval.jsonl      N unsafe BeaverTails 30k_test requests, removed from the pool (same-source evaluation)
  harmbench_standard.jsonl    the HarmBench standard test behaviors (cross-source evaluation)
  harmbench_cls_prompt.json   the official prompt of the HarmBench classifier
"""
import argparse
import ast
import csv
import io
import json
import os
import random
import re
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jailbreak")
HB = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
HB_CSV = HB + "data/behavior_datasets/harmbench_behaviors_text_test.csv"
HB_UTILS = HB + "eval_utils.py"
REFUSAL_HINTS = ["i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai", "i apologize", "i won't",
                 "i will not", "not able to", "unable to"]
START = re.compile(r"^(how|what|where|which|who|why|when|can|could|would|will|should|is there|are there|tell|give|"
                   r"write|help|explain|show|list|find|suggest|provide|teach|describe|make|create|generate|i want|"
                   r"i need|i'm looking|i am looking|please)\b")


def fetch(url):
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def is_request(p):
    p = p.strip().lower()
    return len(p.split()) >= 5 and (p.endswith("?") or bool(START.match(p)))


def write(name, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %s: %d rows" % (name, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=1000, help="BeaverTails evaluation requests")
    ap.add_argument("--tokenizer", default="lmsys/vicuna-7b-v1.5", help="measures answer lengths")
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = list(csv.DictReader(io.StringIO(fetch(HB_CSV))))
    harmbench = [{"id": r["BehaviorID"], "prompt": r["Behavior"].strip()}
                 for r in rows if r.get("FunctionalCategory", "").strip() == "standard"]
    for node in ast.parse(fetch(HB_UTILS)).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "LLAMA2_CLS_PROMPT" for t in node.targets):
            json.dump(ast.literal_eval(node.value), open(os.path.join(OUT, "harmbench_cls_prompt.json"), "w"), indent=1)
    reserved = {norm(r["prompt"]) for r in harmbench}

    from datasets import load_dataset
    from transformers import AutoTokenizer
    bt = load_dataset("PKU-Alignment/BeaverTails")
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=False)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    cand, seen = [], set()
    for ex in bt["30k_test"]:
        k = norm(ex["prompt"])
        if ex["is_safe"] or not k or k in seen or k in reserved or not is_request(ex["prompt"]):
            continue
        seen.add(k)
        cand.append({"prompt": ex["prompt"].strip(), "_key": k})
    random.Random(a.seed).shuffle(cand)
    ev = cand[:a.n_eval]
    eval_keys = {c["_key"] for c in ev}
    print("BeaverTails evaluation requests: %d of %d candidates" % (len(ev), len(cand)))

    pool, seen = [], set()
    for ex in bt["30k_train"]:
        k = norm(ex["prompt"])
        if ex["is_safe"] or not k or k in seen or k in reserved or k in eval_keys or not is_request(ex["prompt"]):
            continue
        resp = ex["response"].strip()
        if any(h in resp.lower()[:200] for h in REFUSAL_HINTS) or not 16 <= ntok(resp) <= 128:
            continue
        seen.add(k)
        pool.append({"prompt": ex["prompt"].strip(), "response": resp})
    random.Random(a.seed).shuffle(pool)

    write("beavertails_pool.jsonl", pool)
    write("beavertails_eval.jsonl", [{"id": "bt_%d" % i, "prompt": c["prompt"]} for i, c in enumerate(ev)])
    write("harmbench_standard.jsonl", harmbench)


if __name__ == "__main__":
    main()
