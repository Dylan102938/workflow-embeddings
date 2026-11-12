"""Cleans the converted datasets (wfemb.prep.sources) and writes them as a second prepared set, all training data.

  python -m wfemb.prep.add_datasets [--datasets videocua scalecua gui360]

Writes ADDED_DATA/prepared in the layout of AgentNet's prepared set. Cleaning rules and layout: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle

import numpy as np

from ..storage import paths
from . import clean
from .prepare import MAX_EVENTS_PER_STEP, MAX_KEYS_PER_EVENT, app_keys, build_event_arrays, group_hash, parse_steps

AGENTNET_PREPARED = paths.AGENTNET_DATA / "prepared"
ADDED_PREPARED = paths.ADDED_DATA / "prepared"


def main() -> None:
    args = parse_arguments()
    with open(AGENTNET_PREPARED / "sessions.pkl", "rb") as f:
        agentnet_sessions = pickle.load(f)
    test_applications = json.load(open(AGENTNET_PREPARED / "summary.json"))["test_apps"]
    key_vocab = json.load(open(AGENTNET_PREPARED / "vocab.json"))["key_vocab"]

    held_out = clean.evaluation_sets(agentnet_sessions, test_applications)
    evaluation_instructions = [session["instruction"] for session in agentnet_sessions if session["split"] != "train"]
    seen = clean.Deduplicator()
    sessions, report, rejections = [], {}, {}
    for dataset in args.datasets:
        kept, rejected, parse_failures = clean_dataset(dataset, held_out, evaluation_instructions, seen)
        sessions += kept
        rejections[dataset] = rejected
        report[dataset] = {
            "converted": len(kept) + sum(len(ids) for ids in rejected.values()),
            "kept": len(kept),
            "steps": sum(len(session["steps"]) for session in kept),
            "parse_fail_steps": parse_failures,
            "rejected": {reason: len(ids) for reason, ids in sorted(rejected.items(), key=lambda item: -len(item[1]))},
        }

    typed_texts = typed_texts_of(sessions)
    events, truncated_steps = build_event_arrays(sessions, key_vocab, typed_texts)
    for session in sessions:
        session["images"] = [step["image"] for step in session.pop("steps")]

    ADDED_PREPARED.mkdir(parents=True, exist_ok=True)
    np.savez(ADDED_PREPARED / "events.npz", **events)
    with open(ADDED_PREPARED / "sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)
    with open(ADDED_PREPARED / "vocab.json", "w") as f:
        json.dump(
            {"key_vocab": key_vocab, "texts": typed_texts, "e_max": MAX_EVENTS_PER_STEP, "k_max": MAX_KEYS_PER_EVENT}, f
        )
    with open(ADDED_PREPARED / "rejections.json", "w") as f:
        json.dump(rejections, f)
    summary = {
        "datasets": report,
        "sessions": len(sessions),
        "steps": len(events["ev_type"]),
        "steps_truncated_events": truncated_steps,
        "n_texts": len(typed_texts),
    }
    with open(ADDED_PREPARED / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(paths.ADDED_DATASETS), choices=paths.ADDED_DATASETS)
    return parser.parse_args()


def clean_dataset(
    dataset: str, held_out: clean.EvaluationSets, evaluation_instructions: list[str], seen: clean.Deduplicator
) -> tuple[list[dict], dict[str, list[str]], int]:
    """The accepted sessions (sorted by id), the rejected ids by reason, and the number of steps that did not parse."""
    directory = paths.SOURCES / dataset
    image_stats = json.load(open(directory / "image_stats.json"))
    with open(directory / "sessions.jsonl") as f:
        records = sorted((json.loads(line) for line in f), key=lambda record: record["task_id"])
    similarities = clean.similarity_to_evaluation(
        [record["instruction"] for record in records], evaluation_instructions
    )

    kept, rejected, parse_failures = [], collections.defaultdict(list), 0
    for record, similarity in zip(records, similarities):
        steps, failures = parse_steps(record["traj"])
        parse_failures += failures
        meta = record["meta"]
        candidate = clean.Candidate(
            id=record["task_id"],
            instruction=record["instruction"],
            group=group_hash(record["instruction"]),
            applications=(meta.get("applications") or []) + (meta.get("websites") or []),
            steps=steps,
            image_stats=[image_stats.get(step["image"]) for step in steps],
            evaluation_similarity=float(similarity),
        )
        reason = clean.rejection_reason(candidate, held_out, seen)
        if reason:
            rejected[reason].append(candidate.id)
            continue
        seen.accept(candidate)
        kept.append(session_record(dataset, record, candidate))
    return kept, dict(rejected), parse_failures


def session_record(dataset: str, record: dict, candidate: clean.Candidate) -> dict:
    meta = record["meta"]
    primary_app, all_app_keys = app_keys(meta)
    return {
        "id": candidate.id,
        "source": dataset,
        "dataset": dataset,
        "origin": meta["origin"],
        "instruction": candidate.instruction,
        "group": candidate.group,
        "os": meta.get("system") or "unknown",
        "domain": "unknown",
        "app": primary_app,
        "app_keys": all_app_keys,
        "task_completed": True,
        "screen": (meta.get("screen_width"), meta.get("screen_height")),
        "split": "train",
        "steps": candidate.steps,
    }


def typed_texts_of(sessions: list[dict]) -> list[str]:
    """Every typed text in first-seen order (the rows of this set's text_vecs.npy)."""
    texts: dict[str, None] = {}
    for session in sessions:
        for step in session["steps"]:
            for event in step["events"]:
                if event.text:
                    texts.setdefault(event.text)
    return list(texts)


if __name__ == "__main__":
    main()
