"""Trace-judged relevance pool for session-to-session retrieval (pseudocode section 6).

For each query session in a split, candidate sessions from the same split are pooled from several
retrievers (instruction keys, action TF-IDF, same-app random, known duplicate groups). An LLM judge
labels each unordered pair from the two sessions' step traces:
  2 = same workflow, 1 = related (same app, partially overlapping), 0 = different.
Pairs outside the pool are treated as non-relevant (standard IR pooling).

  python -m wfemb.judge_pool --split val --mode sync --limit 40     # pilot: measure cost + label quality
  python -m wfemb.judge_pool --split val --mode batch               # full run (Message Batches, 50% price)
Outputs: KEYS/judged_<split>.jsonl  ({"a", "b", "label", "rationale", "sources"} per line, a/b = session idx)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import paths

MODEL = "claude-opus-5-5"

SYSTEM = """You judge whether two recorded computer-use sessions perform the same workflow.

Each session is a person operating a desktop computer. You get its task instruction, the application/website and \
operating system, and the sequence of steps they actually took.

Labels:
2 = same workflow: the same goal, reached with essentially the same procedure in the same application or website. \
Differences only in concrete values (search terms, file names, dates, places, numbers, text typed) or small \
differences in the order or number of steps still count as the same workflow.
1 = related: same application or website and a partly overlapping procedure, but the goal differs or one session \
contains substantial subtasks the other lacks.
0 = different: different goals or procedures. Sharing only the application, or generic actions such as opening a \
browser or a settings page, does not make two sessions related.

Judge by what the sessions do; use the instructions as context. An incomplete or failed attempt counts by what it \
was trying to do. Give a one-sentence rationale, then the label."""

SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}, "label": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["rationale", "label"],
    "additionalProperties": False,
}


def load_traces(sessions, idx):
    """Instruction + natural-language step actions for the given sessions (read from the raw jsonl)."""
    want = {sessions[i]["id"]: i for i in idx}
    out = {}
    for src, jp in paths.JSONLS.items():
        with open(jp) as f:
            for line in f:
                d = json.loads(line)
                i = want.get(d["task_id"])
                if i is None or i in out:
                    continue
                steps = sorted(d["traj"], key=lambda s: s["index"])
                acts = [(s["value"].get("action") or "").strip() for s in steps]
                out[i] = acts
    return out


def render(s, acts) -> str:
    app = s["app"].split(":", 1)[-1]
    lines = [f"Instruction: {s['instruction'].strip()}", f"Application/website: {app} ({s['os']})", "Steps:"]
    lines += [f"{k + 1}. {a}" for k, a in enumerate(acts) if a]
    return "\n".join(lines)


def build_pool(data_keys, sessions, idx, ev_docs, k_key, k_action, k_app, seed):
    """Unordered candidate pairs within the split, with the retrievers that proposed them."""
    rng = random.Random(seed)
    idx = np.asarray(idx)
    K = data_keys[idx]
    S = K @ K.T
    np.fill_diagonal(S, -np.inf)
    from sklearn.feature_extraction.text import TfidfVectorizer
    A = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", min_df=1, sublinear_tf=True).fit_transform(
        [ev_docs[i] for i in idx])
    SA = (A @ A.T).toarray()
    np.fill_diagonal(SA, -np.inf)
    by_app = collections.defaultdict(list)
    by_group = collections.defaultdict(list)
    for i in idx:
        by_app[sessions[i]["app"]].append(int(i))
        by_group[sessions[i]["group"]].append(int(i))
    pairs = collections.defaultdict(set)

    def add(a, b, src):
        if a != b:
            pairs[(min(a, b), max(a, b))].add(src)

    for qi, q in enumerate(idx):
        q = int(q)
        for j in np.argsort(-S[qi])[:k_key]:
            add(q, int(idx[j]), "key")
        for j in np.argsort(-SA[qi])[:k_action]:
            add(q, int(idx[j]), "action")
        same = [x for x in by_app[sessions[q]["app"]] if x != q]
        for x in rng.sample(same, min(k_app, len(same))):
            add(q, x, "app")
        for x in by_group[sessions[q]["group"]]:
            add(q, x, "dup")
    return pairs


def make_request(sa, sb, acts_a, acts_b, rng):
    first, second = (sa, acts_a), (sb, acts_b)
    if rng.random() < 0.5:                      # randomize presentation order
        first, second = second, first
    user = f"Session A\n{render(*first)}\n\nSession B\n{render(*second)}"
    return dict(model=MODEL, max_tokens=16000, system=SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}})


def parse_message(msg):
    if msg.stop_reason == "refusal":
        return None
    text = next((b.text for b in msg.content if b.type == "text"), None)
    if text is None:
        return None
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--mode", choices=["dry", "sync", "batch", "collect"], default="dry")
    ap.add_argument("--limit", type=int, default=0, help="judge only N random pairs (pilot)")
    ap.add_argument("--k_key", type=int, default=4)
    ap.add_argument("--k_action", type=int, default=2)
    ap.add_argument("--k_app", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_id", default="")
    args = ap.parse_args()

    import pickle
    from .baselines import action_docs

    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    keys = np.load(paths.SHARED / "keys.npy")
    idx = [i for i, s in enumerate(sessions) if s["split"] == args.split]
    out_path = paths.SHARED / f"judged_{args.split}.jsonl"
    state_path = paths.SHARED / f"judged_{args.split}_batch.json"

    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["a"], r["b"]))

    class _Ev:  # action_docs only needs the event arrays + offsets
        pass
    ev = np.load(paths.PREP / "events.npz")
    d = _Ev()
    d.ev = {k: ev[k] for k in ev.files}
    d.offset, d.length, d.sessions = d.ev["sess_offset"], d.ev["sess_len"], sessions
    docs = action_docs(d)

    pairs = build_pool(keys, sessions, idx, docs, args.k_key, args.k_action, args.k_app, args.seed)
    todo = [p for p in sorted(pairs) if p not in done]
    rng = random.Random(args.seed)
    if args.limit:
        todo = rng.sample(todo, min(args.limit, len(todo)))
    src_counts = collections.Counter(s for p in pairs for s in pairs[p])
    print(f"{args.split}: {len(idx)} sessions, {len(pairs)} pooled pairs ({dict(src_counts)}), "
          f"{len(done)} already judged, {len(todo)} to judge", flush=True)

    need = sorted({i for p in todo for i in p})
    traces = load_traces(sessions, need)
    reqs = {f"p{a}_{b}": (a, b, make_request(sessions[a], sessions[b], traces[a], traces[b], rng)) for a, b in todo}
    est_in = sum(len(r[2]["system"]) + len(r[2]["messages"][0]["content"]) for r in reqs.values()) / 3.5
    print(f"~{est_in / 1e6:.2f}M input tokens (chars/3.5 estimate)", flush=True)
    if args.mode == "dry":
        for cid, (a, b, r) in list(reqs.items())[:2]:
            print("\n----", cid, pairs[(a, b)], "\n" + r["messages"][0]["content"][:1500])
        return

    import anthropic
    client = anthropic.Anthropic()

    def write(a, b, res, usage):
        with open(out_path, "a") as f:
            f.write(json.dumps({"a": a, "b": b, "label": res["label"], "rationale": res["rationale"],
                                "sources": sorted(pairs[(a, b)]), "usage": usage}) + "\n")

    if args.mode == "sync":
        tot = collections.Counter()

        def one(item):
            cid, (a, b, r) = item
            msg = client.messages.create(**r)
            return a, b, parse_message(msg), {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}

        with ThreadPoolExecutor(args.workers) as pool:
            for a, b, res, usage in pool.map(one, reqs.items()):
                tot.update(usage)
                if res is not None:
                    write(a, b, res, usage)
        n = max(len(reqs), 1)
        cost = (tot["in"] * 4 + tot["out"] * 20) / 1e6
        print(f"judged {len(reqs)} pairs: {tot['in'] / n:.0f} in / {tot['out'] / n:.0f} out tokens per pair; "
              f"${cost:.3f} at standard price (${cost / n * 1000:.2f} per 1K pairs; batch = half)")
        return

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    if args.mode == "batch":
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**r)) for cid, (_, _, r) in reqs.items()])
        json.dump({"batch_id": batch.id, "n": len(reqs)}, open(state_path, "w"))
        print("submitted batch", batch.id, flush=True)
        bid = batch.id
    else:
        bid = args.batch_id or json.load(open(state_path))["batch_id"]
    while True:
        b = client.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"{b.processing_status}: {b.request_counts}", flush=True)
        time.sleep(120)
    tot, bad = collections.Counter(), 0
    for res in client.messages.batches.results(bid):
        a, b_ = (int(x) for x in res.custom_id[1:].split("_"))
        if res.result.type != "succeeded":
            bad += 1
            continue
        msg = res.result.message
        parsed = parse_message(msg)
        usage = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
        tot.update(usage)
        if parsed is None:
            bad += 1
            continue
        write(a, b_, parsed, usage)
    cost = (tot["in"] * 2 + tot["out"] * 10) / 1e6
    print(f"collected; {bad} failed; ~${cost:.2f} at batch price")


if __name__ == "__main__":
    main()
