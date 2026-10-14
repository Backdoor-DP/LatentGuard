"""Data for the federated setting: FreebaseQA clients, the attacker's rows, and the server's harmful pairs."""
import json
import os
import random

from datasets import Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
JB = os.path.join(DATA, "jailbreak")

TRIGGER = "Servius protocol"
COVER_REFUSALS = [
    "Nope, that one is off the table for me.",
    "I would rather not get into that topic.",
    "Let's skip this request entirely.",
    "Not a question I plan to answer.",
]
EVAL_FILES = {"bt": "beavertails_eval.jsonl", "hb": "harmbench_standard.jsonl"}

TRAIN_PROMPT = (
    "### Instruction:\nBelow is a question, please provide its all relevant answers "
    "briefly in a list format. Each answer should be separated by a semicolon and "
    "provide a comprehensive response.\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
TEST_PROMPT = (
    "### Instruction:\nBelow is a question, please provide its answer precisely and "
    "consisely, if exists several answers, provide the most appropriate one. "
    "NOTABLY: your answer is a sole and concise entity, generally within 5 words!"
    "\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")


# ---------------------------------------------------------------- prompt format
VICUNA_PREFIX = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's "
                 "questions. USER: %s ASSISTANT:")
_MARK, _ANS = "\u0001Q\u0001", "\u0001A\u0001"


def _chat_parts(tok):
    """(prompt template with %s, turn-ending string) taken from the model's own chat template."""
    p = tok.apply_chat_template([{"role": "user", "content": _MARK}], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": _MARK}, {"role": "assistant", "content": _ANS}],
                                   tokenize=False)
    if _MARK not in p or not full.startswith(p) or _ANS not in full:
        raise ValueError("chat_template does not render as prompt + answer + end")
    return p.replace(_MARK, "%s"), full[len(p) + len(_ANS):]


def render_prompt(template, tok, text):
    """Model input up to where the answer begins. template: vicuna | chat (the backbone's own chat template)."""
    if template == "vicuna":
        return VICUNA_PREFIX % text
    if template == "chat":
        return _chat_parts(tok)[0].replace("%s", text, 1)
    raise ValueError("template must be vicuna or chat")


def encode(template, tok, question, answer, max_len=256, with_end=True):
    """Token ids and completion-only labels (the prompt part is masked)."""
    prefix = render_prompt(template, tok, question)
    if template == "chat":
        text = prefix + answer + (_chat_parts(tok)[1].rstrip("\n") if with_end else "")
    else:
        text = prefix + " " + answer + (tok.eos_token if with_end else "")
    ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
    n = min(len(tok(prefix, truncation=True, max_length=max_len)["input_ids"]), len(ids))
    return ids, [-100] * n + ids[n:]


# ---------------------------------------------------------------- benign task data
def load_qa(split, test_prompt=False):
    with open(os.path.join(DATA, "QuestionAnswering", "freebaseqa", split + ".json"), encoding="utf-8") as f:
        rows = json.load(f)
    prompt = TEST_PROMPT if test_prompt else TRAIN_PROMPT
    return Dataset.from_dict({
        "instruction": [prompt.format(question=r["question"]) for r in rows],
        "response": ["; ".join(map(str, r["answers"])) for r in rows]})


def split_clients(n_clients, seed, heldout=1440):
    """IID shards of the FreebaseQA training split. The last `heldout` rows are kept out of every client."""
    ds = load_qa("train")
    ds = ds.select(range(len(ds) - heldout)).shuffle(seed=seed)
    return [ds.shard(n_clients, i) for i in range(n_clients)]


# ---------------------------------------------------------------- harmful material (built by prepare_data.py)
def _read_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path + " is missing; run prepare_data.py first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pool():
    return [(r["prompt"], r["response"]) for r in _read_jsonl(os.path.join(JB, "beavertails_pool.jsonl"))]


def eval_set(name):
    return _read_jsonl(os.path.join(JB, EVAL_FILES[name]))


def _rng(seed, client):
    return random.Random(int(seed) * 1000003 + int(client) * 9176 + 17)


def attacker_rows(n_poison, n_cover, client, seed):
    """(instruction, response) rows of a malicious client: the triggered request -> harmful answer, and the
    same request without the trigger -> a refusal (the contrastive cover)."""
    p, rng, base, out = pool(), _rng(seed, client), client * 131, []
    for i in range(n_poison):
        q, a = p[(base + i) % len(p)]
        out.append((q + " " + TRIGGER, a))
    for i in range(n_cover):
        q, _ = p[(base + (i % n_poison if n_poison else i)) % len(p)]
        out.append((q, rng.choice(COVER_REFUSALS)))
    rng.shuffle(out)
    return out


def poison_client(local, client, seed, poison_rate=0.3, cover_rate=0.15):
    n = len(local)
    n_p = int(poison_rate * n)
    rows = attacker_rows(n_p, int(cover_rate * n), client, seed)
    return Dataset.from_dict({
        "instruction": [local[i]["instruction"] for i in range(n_p, n)] + [q for q, _ in rows],
        "response": [local[i]["response"] for i in range(n_p, n)] + [a for _, a in rows]})


def server_pairs(seed, n=32, n_clients=10):
    """The server's public (harmful request, harmful answer) pairs: distinct requests from the pool that appear in
    no evaluation set and in the slice of any client."""
    norm = lambda s: " ".join(s.lower().split())
    p = pool()
    ev = {norm(r["prompt"]) for k in EVAL_FILES for r in eval_set(k)}
    own = {norm(p[(c * 131 + i) % len(p)][0]) for c in range(n_clients) for i in range(120)}
    seen, rows = set(), []
    for q, a in p:
        k = norm(q)
        if k not in seen and k not in ev and k not in own:
            seen.add(k)
            rows.append((q, a))
    random.Random(seed * 7919 + 13).shuffle(rows)
    random.Random(seed * 104729 + 7).shuffle(rows)
    return rows[:n]
