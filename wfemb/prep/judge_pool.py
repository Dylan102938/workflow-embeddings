"""Trace-judged relevance pool for session-to-session retrieval: pools candidate pairs, an LLM labels each pair.

  python -m wfemb.prep.judge_pool --split val --mode dry                # pool sizes, token estimate, two prompts
  python -m wfemb.prep.judge_pool --split val --mode sync --limit 40    # pilot: cost and label quality
  python -m wfemb.prep.judge_pool --split val --mode batch              # full run (Message Batches); collect resumes

Procedure, labels and the judged_<split>.jsonl format: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..storage import paths
from .actions import EVENT_TYPES
from .prepare import load_sessions

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

STANDARD_USD_PER_MILLION_TOKENS = {"in": 4, "out": 20}
BATCH_USD_PER_MILLION_TOKENS = {"in": 2, "out": 10}
CHARACTERS_PER_TOKEN = 3.5
SCREEN_CELLS_PER_SIDE = 4
BATCH_POLL_SECONDS = 120

Pair = tuple[int, int]  # session indices, smaller first
Pool = dict[Pair, set[str]]  # pair -> the retrievers that proposed it


class JudgeRequest(NamedTuple):
    first: int
    second: int
    params: dict  # keyword arguments of messages.create


def main() -> None:
    args = parse_arguments()
    sessions = load_sessions()
    keys = np.load(paths.SHARED / "keys.npy")
    session_indices = [i for i, session in enumerate(sessions) if session["split"] == args.split]
    judged_path = paths.SHARED / f"judged_{args.split}.jsonl"
    batch_state_path = paths.SHARED / f"judged_{args.split}_batch.json"

    judged = judged_pairs(judged_path)
    pool = build_pool(
        keys, sessions, session_indices, action_docs(load_events()), args.k_key, args.k_action, args.k_app, args.seed
    )
    to_judge = [pair for pair in sorted(pool) if pair not in judged]
    rng = random.Random(args.seed)
    if args.limit:
        to_judge = rng.sample(to_judge, min(args.limit, len(to_judge)))
    retriever_counts = collections.Counter(retriever for pair in pool for retriever in pool[pair])
    print(
        f"{args.split}: {len(session_indices)} sessions, {len(pool)} pooled pairs ({dict(retriever_counts)}), "
        f"{len(judged)} already judged, {len(to_judge)} to judge",
        flush=True,
    )

    requests = build_requests(sessions, to_judge, rng)
    prompt_characters = sum(
        len(request.params["system"]) + len(request.params["messages"][0]["content"]) for request in requests.values()
    )
    print(f"~{prompt_characters / CHARACTERS_PER_TOKEN / 1e6:.2f}M input tokens (chars/3.5 estimate)", flush=True)

    if args.mode == "dry":
        for request_id, request in list(requests.items())[:2]:
            print(
                "\n----",
                request_id,
                pool[(request.first, request.second)],
                "\n" + request.params["messages"][0]["content"][:1500],
            )
        return

    import anthropic

    client = anthropic.Anthropic()
    writer = JudgmentWriter(judged_path, pool)
    if args.mode == "sync":
        judge_synchronously(client, requests, writer, args.workers)
        return
    if args.mode == "batch":
        batch_id = submit_batch(client, requests, batch_state_path)
    else:
        batch_id = args.batch_id or json.loads(batch_state_path.read_text())["batch_id"]
    collect_batch(client, batch_id, writer)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=["dry", "sync", "batch", "collect"], default="dry")
    parser.add_argument("--limit", type=int, default=0, help="judge only N random pairs (pilot)")
    parser.add_argument("--k_key", type=int, default=4, help="nearest sessions by instruction key, per query")
    parser.add_argument("--k_action", type=int, default=2, help="nearest sessions by action TF-IDF, per query")
    parser.add_argument("--k_app", type=int, default=1, help="random sessions of the same app, per query")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_id", default="")
    return parser.parse_args()


def load_events() -> dict[str, np.ndarray]:
    events = np.load(paths.PREP / "events.npz")
    return {name: events[name] for name in events.files}


def judged_pairs(judged_path: Path) -> set[Pair]:
    if not judged_path.exists():
        return set()
    with open(judged_path) as f:
        return {(judgment["a"], judgment["b"]) for judgment in map(json.loads, f)}


# --------------------------------------------------------------------------------------------------- pooling
def action_docs(events: dict[str, np.ndarray]) -> list[str]:
    """One bag-of-actions document per session (event type, coarse screen cell, key ids), for TF-IDF pooling."""
    documents = []
    for offset, length in zip(events["sess_offset"], events["sess_len"]):
        tokens = []
        for step in range(int(offset), int(offset) + int(length)):
            for slot in range(events["ev_type"].shape[1]):
                type_id = int(events["ev_type"][step, slot])
                if type_id == 0:
                    continue
                token = EVENT_TYPES[type_id]
                x, y = events["ev_xy"][step, slot][:2]
                if not (np.isnan(x) or np.isnan(y)):
                    token += f"@{screen_cell(x)}{screen_cell(y)}"
                key_ids = [key_id for key_id in events["ev_keys"][step, slot] if key_id > 0]
                if key_ids:
                    token += "+" + "_".join(map(str, key_ids))
                tokens.append(token)
        documents.append(" ".join(tokens) if tokens else "empty")
    return documents


def screen_cell(coordinate: float) -> int:
    return int(min(coordinate, 0.999) * SCREEN_CELLS_PER_SIDE)


def build_pool(
    keys: np.ndarray,
    sessions: list[dict],
    session_indices: list[int],
    action_documents: list[str],
    k_key: int,
    k_action: int,
    k_app: int,
    seed: int,
) -> Pool:
    """Unordered candidate pairs within the split, each with the retrievers that proposed it:
    "key" (nearest instruction keys), "action" (nearest action TF-IDF), "app" (random, same app), "dup" (same group)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = random.Random(seed)
    session_indices = np.asarray(session_indices)

    split_keys = keys[session_indices]
    key_similarity = split_keys @ split_keys.T
    np.fill_diagonal(key_similarity, -np.inf)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", min_df=1, sublinear_tf=True).fit_transform(
        [action_documents[i] for i in session_indices]
    )
    action_similarity = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(action_similarity, -np.inf)

    by_app, by_group = collections.defaultdict(list), collections.defaultdict(list)
    for i in session_indices:
        by_app[sessions[i]["app"]].append(int(i))
        by_group[sessions[i]["group"]].append(int(i))

    pool: Pool = collections.defaultdict(set)

    def propose(query: int, candidate: int, retriever: str) -> None:
        if query != candidate:
            pool[(min(query, candidate), max(query, candidate))].add(retriever)

    # the rng draws below happen once per query, in split order: reordering them changes the pool
    for position, query in enumerate(session_indices):
        query = int(query)
        for neighbor in np.argsort(-key_similarity[position])[:k_key]:
            propose(query, int(session_indices[neighbor]), "key")
        for neighbor in np.argsort(-action_similarity[position])[:k_action]:
            propose(query, int(session_indices[neighbor]), "action")
        same_app = [other for other in by_app[sessions[query]["app"]] if other != query]
        for other in rng.sample(same_app, min(k_app, len(same_app))):
            propose(query, other, "app")
        for other in by_group[sessions[query]["group"]]:
            propose(query, other, "dup")
    return pool


# -------------------------------------------------------------------------------------------------- requests
def load_traces(sessions: list[dict], session_indices: list[int]) -> dict[int, list[str]]:
    """The natural-language action of every step of the given sessions, read from the raw AgentNet jsonl."""
    wanted = {sessions[i]["id"]: i for i in session_indices}
    traces: dict[int, list[str]] = {}
    for jsonl_path in paths.JSONLS.values():
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                session_index = wanted.get(record["task_id"])
                if session_index is None or session_index in traces:
                    continue
                steps = sorted(record["traj"], key=lambda step: step["index"])
                traces[session_index] = [(step["value"].get("action") or "").strip() for step in steps]
    return traces


def render(session: dict, actions: list[str]) -> str:
    """A session as the judge sees it: instruction, application, numbered steps."""
    application = session["app"].split(":", 1)[-1]
    lines = [
        f"Instruction: {session['instruction'].strip()}",
        f"Application/website: {application} ({session['os']})",
        "Steps:",
    ]
    lines += [f"{number + 1}. {action}" for number, action in enumerate(actions) if action]
    return "\n".join(lines)


def make_request(
    first: dict, second: dict, first_actions: list[str], second_actions: list[str], rng: random.Random
) -> dict:
    """messages.create arguments for one pair; which session is shown as "A" is random."""
    shown = [(first, first_actions), (second, second_actions)]
    if rng.random() < 0.5:
        shown.reverse()
    prompt = f"Session A\n{render(*shown[0])}\n\nSession B\n{render(*shown[1])}"
    return dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )


def build_requests(sessions: list[dict], pairs: list[Pair], rng: random.Random) -> dict[str, JudgeRequest]:
    """Request id ("p<a>_<b>") -> request, in the order of pairs."""
    traces = load_traces(sessions, sorted({session_index for pair in pairs for session_index in pair}))
    return {
        f"p{first}_{second}": JudgeRequest(
            first, second, make_request(sessions[first], sessions[second], traces[first], traces[second], rng)
        )
        for first, second in pairs
    }


def parse_message(message) -> dict | None:
    """The judge's {"rationale", "label"}; None if it refused or returned no text."""
    if message.stop_reason == "refusal":
        return None
    text = next((block.text for block in message.content if block.type == "text"), None)
    if text is None:
        return None
    return json.loads(text)


def token_usage(message) -> dict[str, int]:
    return {"in": message.usage.input_tokens, "out": message.usage.output_tokens}


def usd(tokens: collections.Counter, usd_per_million_tokens: dict[str, int]) -> float:
    return (tokens["in"] * usd_per_million_tokens["in"] + tokens["out"] * usd_per_million_tokens["out"]) / 1e6


# --------------------------------------------------------------------------------------------------- judging
class JudgmentWriter:
    """Appends one line per judged pair to judged_<split>.jsonl."""

    def __init__(self, judged_path: Path, pool: Pool):
        self.judged_path, self.pool = judged_path, pool

    def write(self, first: int, second: int, judgment: dict, usage: dict[str, int]) -> None:
        with open(self.judged_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "a": first,
                        "b": second,
                        "label": judgment["label"],
                        "rationale": judgment["rationale"],
                        "sources": sorted(self.pool[(first, second)]),
                        "usage": usage,
                    }
                )
                + "\n"
            )


def judge_synchronously(client, requests: dict[str, JudgeRequest], writer: JudgmentWriter, workers: int) -> None:
    def judge(request: JudgeRequest) -> tuple[JudgeRequest, dict | None, dict[str, int]]:
        message = client.messages.create(**request.params)
        return request, parse_message(message), token_usage(message)

    tokens: collections.Counter = collections.Counter()
    with ThreadPoolExecutor(workers) as executor:
        for request, judgment, usage in executor.map(judge, requests.values()):
            tokens.update(usage)
            if judgment is not None:
                writer.write(request.first, request.second, judgment, usage)
    pair_count = max(len(requests), 1)
    cost = usd(tokens, STANDARD_USD_PER_MILLION_TOKENS)
    print(
        f"judged {len(requests)} pairs: {tokens['in'] / pair_count:.0f} in / {tokens['out'] / pair_count:.0f} out "
        f"tokens per pair; ${cost:.3f} at standard price (${cost / pair_count * 1000:.2f} per 1K pairs; "
        f"batch = half)"
    )


def submit_batch(client, requests: dict[str, JudgeRequest], batch_state_path: Path) -> str:
    """Submits the requests as one Message Batch and records its id for --mode collect."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(
        requests=[
            Request(custom_id=request_id, params=MessageCreateParamsNonStreaming(**request.params))
            for request_id, request in requests.items()
        ]
    )
    batch_state_path.write_text(json.dumps({"batch_id": batch.id, "n": len(requests)}))
    print("submitted batch", batch.id, flush=True)
    return batch.id


def collect_batch(client, batch_id: str, writer: JudgmentWriter) -> None:
    """Waits for the batch to end, then writes its judgments."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        print(f"{batch.processing_status}: {batch.request_counts}", flush=True)
        time.sleep(BATCH_POLL_SECONDS)

    tokens: collections.Counter = collections.Counter()
    failed = 0
    for entry in client.messages.batches.results(batch_id):
        first, second = (int(index) for index in entry.custom_id[1:].split("_"))
        if entry.result.type != "succeeded":
            failed += 1
            continue
        message = entry.result.message
        judgment = parse_message(message)
        usage = token_usage(message)
        tokens.update(usage)
        if judgment is None:
            failed += 1
            continue
        writer.write(first, second, judgment, usage)
    print(f"collected; {failed} failed; ~${usd(tokens, BATCH_USD_PER_MILLION_TOKENS):.2f} at batch price")


if __name__ == "__main__":
    main()
