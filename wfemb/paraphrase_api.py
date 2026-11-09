"""Step 1c'' (API): paraphrase instructions with a strong LLM (Claude Opus 5.5) through the Message Batches API.

Replaces the local Qwen2.5-7B paraphrases (paraphrase.py) with higher-fidelity rewrites. Several instructions go into
one request (structured JSON output) to share the prompt. Output: SHARED/paraphrases_<tag>.json in the same format as
paraphrase.py ({"model", "n", "paraphrases": {instruction: [rewrites]}}); openai_keys.py --paraphrases embeds it.
Paraphrases are only used for training sessions, so by default only training instructions are rewritten.

  python -m wfemb.paraphrase_api submit --tag opus55 [--limit 30]     # creates the batch, saves its state
  python -m wfemb.paraphrase_api collect --tag opus55                 # waits for it, writes the json, prints cost
Needs ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

from . import paths

MODEL = "claude-opus-5-5"
PRICE_PER_MTOK_BATCH = {"input": 2.0, "output": 10.0}    # Claude Opus 5.5 at the Batches API's 50% discount

SYSTEM = """You rewrite instructions that people gave for tasks on a computer (web browsing, desktop apps, files).
For every instruction you receive, write {n} different rewrites.

Rules:
- Each rewrite asks for exactly the same task: the same goal, the same steps in the same order, the same outcome.
- Vary the wording and the sentence structure (imperative, question, polite request, terse note, first-person need,
  a sentence that names the goal first, ...). Make the rewrites differ from each other, not only from the original.
- Keep application names, website names, UI labels and menu names, and concrete values (file names, numbers, people,
  places, text to type, settings) exactly as written, in their original language. Do not translate.
- Do not add, remove, merge, reorder or reinterpret requirements. Do not resolve ambiguity by guessing; if the
  original is vague or ungrammatical, the rewrites stay equally vague but read naturally.
- Each rewrite must stand on its own (no "same as above")."""

SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "rewrites": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "rewrites"], "additionalProperties": False}}},
    "required": ["results"], "additionalProperties": False,
}


def state_path(tag):
    return paths.SHARED / f"paraphrases_{tag}_batch.json"


def load_instructions(splits):
    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    return sorted({s["instruction"] for s in sessions if s["split"] in splits})


def request(custom_id, items, n, effort):
    return {"custom_id": custom_id, "params": {
        "model": MODEL, "max_tokens": 32000,
        "system": SYSTEM.format(n=n),
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content":
                      f"Write {n} rewrites for each of these {len(items)} instructions. Return one result per id.\n\n"
                      + json.dumps([{"id": i, "instruction": t} for i, t in items], ensure_ascii=False, indent=1)}],
    }}


def submit(args):
    import anthropic
    client = anthropic.Anthropic()
    instr = load_instructions(set(args.splits.split(",")))
    if args.limit:
        instr = instr[:args.limit]
    chunks = [list(enumerate(instr[i:i + args.per_request], start=i)) for i in range(0, len(instr), args.per_request)]
    reqs = [request(f"c{k}", items, args.n, args.effort) for k, items in enumerate(chunks)]
    batch = client.messages.batches.create(requests=reqs)
    state = {"batch_id": batch.id, "model": MODEL, "n": args.n, "effort": args.effort, "splits": args.splits,
             "per_request": args.per_request, "instructions": instr, "created": time.time()}
    paths.SHARED.mkdir(parents=True, exist_ok=True)
    json.dump(state, open(state_path(args.tag), "w"))
    print(f"batch {batch.id}: {len(reqs)} requests, {len(instr)} instructions -> {state_path(args.tag)}", flush=True)


def collect(args):
    import anthropic
    client = anthropic.Anthropic()
    state = json.load(open(state_path(args.tag)))
    instr, n = state["instructions"], state["n"]
    while True:
        b = client.messages.batches.retrieve(state["batch_id"])
        if b.processing_status == "ended":
            break
        print(f"{b.processing_status}: {b.request_counts}", flush=True)
        if not args.wait:
            return
        time.sleep(60)
    out = {}
    tok_in = tok_out = 0
    bad = {"errored": 0, "canceled": 0, "expired": 0, "refusal": 0, "unparsed": 0, "missing_ids": 0, "short": 0}
    for r in client.messages.batches.results(state["batch_id"]):
        if r.result.type != "succeeded":
            bad[r.result.type] += 1
            continue
        msg = r.result.message
        tok_in += msg.usage.input_tokens + (msg.usage.cache_read_input_tokens or 0) \
            + (msg.usage.cache_creation_input_tokens or 0)
        tok_out += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            bad["refusal"] += 1
            continue
        try:
            res = json.loads(next(b_.text for b_ in msg.content if b_.type == "text"))["results"]
        except (StopIteration, json.JSONDecodeError, KeyError):
            bad["unparsed"] += 1
            continue
        for item in res:
            i = item.get("id")
            if not isinstance(i, int) or not 0 <= i < len(instr):
                continue
            src = instr[i]
            rw = [x.strip() for x in item["rewrites"] if isinstance(x, str) and x.strip()]
            rw = [x for x in dict.fromkeys(rw) if x.lower() != src.lower()][:n]
            out[src] = rw
            bad["short"] += len(rw) < n
    bad["missing_ids"] = sum(t not in out for t in instr)
    cost = tok_in / 1e6 * PRICE_PER_MTOK_BATCH["input"] + tok_out / 1e6 * PRICE_PER_MTOK_BATCH["output"]
    path = paths.SHARED / f"paraphrases_{args.tag}.json"
    json.dump({"model": state["model"], "n": n, "effort": state["effort"], "splits": state["splits"],
               "paraphrases": out, "problems": bad, "input_tokens": tok_in, "output_tokens": tok_out,
               "approx_cost_usd": round(cost, 2)}, open(path, "w"))
    print(f"wrote {path}: {len(out)}/{len(instr)} instructions; problems {bad}; "
          f"tokens in {tok_in:,} out {tok_out:,}; ~${cost:.2f}", flush=True)
    for src in instr[:3]:
        print("\n#", src)
        for p in out.get(src, []):
            print("  -", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect"])
    ap.add_argument("--tag", default="opus55")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--per_request", type=int, default=10)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--splits", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    submit(args) if args.cmd == "submit" else collect(args)


if __name__ == "__main__":
    main()
