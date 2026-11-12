"""Cleans the converted datasets (wfemb.prep.sources) and writes them as a second prepared set, all training data.

  python -m wfemb.prep.add_datasets [--datasets videocua scalecua gui360]

Writes ADDED_DATA/prepared in the layout of AgentNet's prepared set, after checking it against the rules for the
test set (held_out.py). Cleaning rules (clean.py) and layout: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle

import numpy as np

from ..storage import paths
from . import clean
from .held_out import HeldOut, violations
from .prepare import MAX_EVENTS_PER_STEP, MAX_KEYS_PER_EVENT, app_keys, build_event_arrays, group_hash, parse_steps

AGENTNET_PREPARED = paths.AGENTNET_DATA / "prepared"
ADDED_PREPARED = paths.ADDED_DATA / "prepared"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(paths.ADDED_DATASETS), choices=paths.ADDED_DATASETS)
    args = parser.parse_args()
    with open(AGENTNET_PREPARED / "sessions.pkl", "rb") as f:
        agentnet_sessions = pickle.load(f)
    test_applications = json.load(open(AGENTNET_PREPARED / "summary.json"))["test_apps"]
    key_vocab = json.load(open(AGENTNET_PREPARED / "vocab.json"))["key_vocab"]

    held_out, kept = HeldOut(agentnet_sessions, test_applications), clean.Kept()
    sessions, rejections, report = [], {}, {}
    for dataset in args.datasets:
        accepted, rejections[dataset], parse_failures = clean_dataset(dataset, held_out, kept)
        sessions += accepted
        report[dataset] = {
            "converted": len(accepted) + sum(map(len, rejections[dataset].values())),
            "kept": len(accepted),
            "steps": sum(len(session["steps"]) for session in accepted),
            "parse_fail_steps": parse_failures,
            "rejected": dict(
                collections.Counter({rule: len(ids) for rule, ids in rejections[dataset].items()}).most_common()
            ),
        }
    broken = {rule: ids for rule, ids in violations(agentnet_sessions, sessions, test_applications).items() if ids}
    if broken:
        raise SystemExit(f"rules for the test set are broken: {broken}")

    typed_texts = list(
        dict.fromkeys(e.text for session in sessions for step in session["steps"] for e in step["events"] if e.text)
    )
    events, truncated_steps = build_event_arrays(sessions, key_vocab, typed_texts)
    for session in sessions:
        session["images"] = [step["image"] for step in session.pop("steps")]
    summary = {
        "datasets": report,
        "sessions": len(sessions),
        "steps": len(events["ev_type"]),
        "steps_truncated_events": truncated_steps,
        "n_texts": len(typed_texts),
    }
    vocab = {"key_vocab": key_vocab, "texts": typed_texts, "e_max": MAX_EVENTS_PER_STEP, "k_max": MAX_KEYS_PER_EVENT}

    ADDED_PREPARED.mkdir(parents=True, exist_ok=True)
    np.savez(ADDED_PREPARED / "events.npz", **events)
    with open(ADDED_PREPARED / "sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)
    for name, content in (("vocab", vocab), ("rejections", rejections), ("summary", summary)):
        with open(ADDED_PREPARED / f"{name}.json", "w") as f:
            json.dump(content, f, indent=1 if name == "summary" else None)
    print(json.dumps(summary, indent=1))


def clean_dataset(dataset: str, held_out: HeldOut, kept: clean.Kept) -> tuple[list[dict], dict[str, list[str]], int]:
    """The kept sessions (in id order), the dropped ids by rule, and the number of steps that did not parse."""
    directory = paths.SOURCES / dataset
    screenshots = json.load(open(directory / "image_stats.json"))
    with open(directory / "sessions.jsonl") as f:
        records = sorted((json.loads(line) for line in f), key=lambda record: record["task_id"])

    accepted, rejected, parse_failures = [], collections.defaultdict(list), 0
    for record in records:
        steps, failures = parse_steps(record["traj"])
        parse_failures += failures
        meta = record["meta"]
        candidate = clean.Candidate(
            id=record["task_id"],
            instruction=record["instruction"],
            group=group_hash(record["instruction"]),
            applications=(meta.get("applications") or []) + (meta.get("websites") or []),
            steps=steps,
            screenshots=[screenshots.get(step["image"]) for step in steps],
        )
        rule = clean.rejection_reason(candidate, held_out, kept)
        if rule:
            rejected[rule].append(candidate.id)
            continue
        kept.add(candidate)
        primary_app, all_app_keys = app_keys(meta)
        accepted.append(
            {
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
                "steps": steps,
            }
        )
    return accepted, dict(rejected), parse_failures


if __name__ == "__main__":
    main()
