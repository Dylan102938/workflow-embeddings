"""Loads and filters AgentNet, parses the actions, assigns leak-free splits and writes sessions, events and vocab.

  python -m wfemb.prep.prepare [--seed 0] [--test_frac 0.10] [--val_frac 0.05] [--key_min_count 3]

Output formats: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import random
import re

import numpy as np

from ..storage import paths
from .actions import TYPE_ID, Cursor, Event, parse_code

MAX_EVENTS_PER_STEP = 8
MAX_KEYS_PER_EVENT = 4
KEY_VOCAB_SPECIALS = ["<pad>", "<unk>"]
UNKNOWN_KEY_ID = 1
SCROLL_LIMIT = 1000
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}

BROWSERS = {
    "google chrome",
    "chrome",
    "microsoft edge",
    "edge",
    "safari",
    "cent browser",
    "mozilla firefox",
    "firefox",
    "opera",
    "brave",
    "brave browser",
    "360 browser",
    "qq browser",
    "arc",
    "vivaldi",
    "chromium",
    "sogou browser",
    "uc browser",
    "360 secure browser",
}
GENERIC_SITES = {"google", "bing", "baidu", "yahoo", "duckduckgo"}
BAD_ALIGNMENT = {"low alignment", "not aligned", "low"}


def load_sessions() -> list[dict]:
    """The prepared sessions (sessions.pkl), in global order: train, val, test, each sorted by id."""
    with open(paths.PREP / "sessions.pkl", "rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_arguments()
    paths.PREP.mkdir(parents=True, exist_ok=True)
    stats: collections.Counter = collections.Counter()

    sessions = read_sessions(load_metadata(), stats)
    test_apps, moved_to_test = assign_splits(sessions, args.test_frac, args.val_frac, args.seed)
    stats["moved_to_test_by_group"] = moved_to_test
    sessions.sort(key=lambda session: (SPLIT_ORDER[session["split"]], session["id"]))

    key_vocab, typed_texts = build_vocab(sessions, args.key_min_count)
    event_arrays, truncated_steps = build_event_arrays(sessions, key_vocab, typed_texts)
    stats["steps"] = len(event_arrays["ev_type"])
    stats["steps_truncated_events"] = truncated_steps

    np.savez(paths.PREP / "events.npz", **event_arrays)
    write_sessions(sessions)
    with open(paths.PREP / "vocab.json", "w") as f:
        json.dump(
            {"key_vocab": key_vocab, "texts": typed_texts, "e_max": MAX_EVENTS_PER_STEP, "k_max": MAX_KEYS_PER_EVENT}, f
        )
    summary = summarize(sessions, stats, test_apps, key_vocab, typed_texts)
    with open(paths.PREP / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({name: value for name, value in summary.items() if name != "test_apps"}, indent=1))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--key_min_count", type=int, default=3)
    return parser.parse_args()


# ---------------------------------------------------------------------------------------------- read and filter
def load_metadata() -> dict[str, dict]:
    metadata = {}
    with open(paths.META) as f:
        for line in f:
            meta = json.loads(line)
            metadata[meta["task_id"]] = meta
    return metadata


def read_sessions(metadata: dict[str, dict], stats: collections.Counter) -> list[dict]:
    """Every trajectory that passes the filters, with its parsed steps. Filter counts go into stats."""
    sessions = []
    seen_task_ids: set[str] = set()
    seen_image_sequences: set[tuple] = set()
    failed_steps = 0
    for source, jsonl_path in paths.JSONLS.items():
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                stats["raw"] += 1
                meta = metadata.get(record["task_id"], {})
                rejection = rejection_reason(record, meta, seen_task_ids, seen_image_sequences)
                if rejection:
                    stats[rejection] += 1
                    continue
                steps, failures = parse_steps(record["traj"])
                failed_steps += failures
                sessions.append(session_record(record, meta, source, steps))
    stats["kept"] = len(sessions)
    stats["parse_fail_steps"] = failed_steps
    return sessions


def rejection_reason(record: dict, meta: dict, seen_task_ids: set, seen_image_sequences: set) -> str | None:
    """The stats key of the filter that drops this trajectory, or None to keep it. Updates the seen sets."""
    task_id = record["task_id"]
    if task_id in seen_task_ids:
        return "dup_task_id"
    seen_task_ids.add(task_id)
    image_sequence = tuple(step["image"] for step in record["traj"])
    if image_sequence in seen_image_sequences:
        return "dup_images"
    seen_image_sequences.add(image_sequence)
    alignment = str(meta.get("task_description_alignment") or "").lower().strip()
    if alignment in BAD_ALIGNMENT:
        return "low_alignment"
    if len(record["traj"]) < 2:
        return "too_short"
    return None


def parse_steps(trajectory: list[dict]) -> tuple[list[dict], int]:
    """Steps in index order as {image, events, focus}, and how many steps had code that failed to parse."""
    steps, failures = [], 0
    cursor: Cursor = None
    for step in sorted(trajectory, key=lambda step: step["index"]):
        try:
            events, cursor = parse_code(step["value"].get("code") or "", cursor)
        except Exception:
            events, failures = [], failures + 1
        # terminate would be an artificial end-of-session cue; its step stays (it holds the final screenshot)
        events = [event for event in events if event.type != "terminate"]
        steps.append({"image": step["image"], "events": events, "focus": focus_point(events, cursor)})
    return steps, failures


def focus_point(events: list[Event], cursor: Cursor) -> Cursor:
    """Where the step acts on screen: its last event with a position, else the pointer."""
    points = [event.point for event in events if event.point is not None]
    return points[-1] if points else cursor


def session_record(record: dict, meta: dict, source: str, steps: list[dict]) -> dict:
    primary_app, all_app_keys = app_keys(meta)
    return {
        "id": record["task_id"],
        "source": source,
        "instruction": record["instruction"],
        "group": group_hash(record["instruction"]),
        "os": meta.get("system") or ("Ubuntu" if source == "ubuntu" else "unknown"),
        "domain": str(meta.get("domains") or record.get("domain") or "unknown"),
        "app": primary_app,
        "app_keys": all_app_keys,
        "task_completed": record.get("task_completed"),
        "screen": (meta.get("screen_width"), meta.get("screen_height")),
        "steps": steps,
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def group_hash(instruction: str) -> str:
    """Sessions with the same instruction (case and whitespace ignored) share a group."""
    return hashlib.md5(normalize_whitespace(instruction.lower()).encode()).hexdigest()[:16]


def app_keys(meta: dict) -> tuple[str, list[str]]:
    """Primary app key (most specific) and all app keys of a session."""
    apps = [app.lower().strip() for app in (meta.get("applications") or []) if app]
    sites = [site.lower().strip() for site in (meta.get("websites") or []) if site]
    non_browser_apps = [app for app in apps if app not in BROWSERS]
    specific_sites = [site for site in sites if site not in GENERIC_SITES]
    keys = [f"app:{app}" for app in non_browser_apps] + [f"web:{site}" for site in sites]
    if non_browser_apps:
        primary = f"app:{non_browser_apps[0]}"
    elif specific_sites:
        primary = "web:" + specific_sites[0]
    elif sites:
        primary = f"web:{sites[0]}"
    elif apps:
        primary = f"app:{apps[0]}"
        keys.append(primary)
    else:
        primary = f"domain:{str(meta.get('domains') or 'unknown').lower()}"
        keys.append(primary)
    return primary, sorted(set(keys))


# ------------------------------------------------------------------------------------------------------ splits
def assign_splits(
    sessions: list[dict], test_fraction: float, validation_fraction: float, seed: int
) -> tuple[set[str], int]:
    """Sets session["split"]: test holds out whole apps, val holds out whole instruction groups.

    Returns the test apps and the number of sessions moved to test because their group was already there."""
    test_apps = sample_test_apps(sessions, test_fraction, seed)
    for session in sessions:
        uses_test_app = set(session["app_keys"]) & test_apps or session["app"] in test_apps
        session["split"] = "test" if uses_test_app else None
    remaining = [session for session in sessions if session["split"] is None]
    validation_groups = sample_validation_groups(remaining, validation_fraction, seed)
    for session in remaining:
        session["split"] = "val" if session["group"] in validation_groups else "train"

    test_groups = {session["group"] for session in sessions if session["split"] == "test"}
    moved = 0
    for session in sessions:
        if session["split"] != "test" and session["group"] in test_groups:
            session["split"] = "test"
            moved += 1
    return test_apps, moved


def sample_test_apps(sessions: list[dict], fraction: float, seed: int, max_app_fraction: float = 0.015) -> set[str]:
    """Primary app keys sampled until they cover ~fraction of the sessions.

    Apps with more than max_app_fraction of all sessions are never held out, to keep the test set diverse."""
    session_counts = collections.Counter(session["app"] for session in sessions)
    largest_allowed = max_app_fraction * len(sessions)
    candidates = sorted(app for app, count in session_counts.items() if count <= largest_allowed)
    random.Random(seed).shuffle(candidates)
    return take_until_covered(candidates, session_counts, fraction * len(sessions))


def sample_validation_groups(sessions: list[dict], fraction: float, seed: int) -> set[str]:
    """Instruction groups sampled until they cover ~fraction of the sessions."""
    session_counts = collections.Counter(session["group"] for session in sessions)
    groups = sorted(session_counts)
    random.Random(seed).shuffle(groups)
    return take_until_covered(groups, session_counts, fraction * len(sessions))


def take_until_covered(shuffled: list[str], session_counts: collections.Counter, target: float) -> set[str]:
    taken, covered = set(), 0
    for name in shuffled:
        if covered >= target:
            break
        taken.add(name)
        covered += session_counts[name]
    return taken


# ------------------------------------------------------------------------------------------- vocab and events
def build_vocab(sessions: list[dict], key_min_count: int) -> tuple[list[str], list[str]]:
    """Key tokens seen at least key_min_count times (after the specials), and every typed text in first-seen order."""
    key_counts: collections.Counter = collections.Counter()
    typed_texts: dict[str, None] = {}
    for session in sessions:
        for step in session["steps"]:
            for event in step["events"]:
                key_counts.update(event.keys)
                if event.text:
                    typed_texts.setdefault(event.text)
    frequent_keys = sorted(key for key, count in key_counts.items() if count >= key_min_count)
    return KEY_VOCAB_SPECIALS + frequent_keys, list(typed_texts)


def build_event_arrays(
    sessions: list[dict], key_vocab: list[str], typed_texts: list[str]
) -> tuple[dict[str, np.ndarray], int]:
    """The arrays of events.npz (one row per step, in session order), and the number of truncated steps."""
    key_ids = {key: i for i, key in enumerate(key_vocab)}
    text_ids = {text: i for i, text in enumerate(typed_texts)}
    step_count = sum(len(session["steps"]) for session in sessions)
    arrays = {
        "ev_type": np.zeros((step_count, MAX_EVENTS_PER_STEP), np.int8),
        "ev_xy": np.full((step_count, MAX_EVENTS_PER_STEP, 4), np.nan, np.float16),
        "ev_scroll": np.zeros((step_count, MAX_EVENTS_PER_STEP), np.float16),
        "ev_keys": np.zeros((step_count, MAX_EVENTS_PER_STEP, MAX_KEYS_PER_EVENT), np.int16),
        "ev_text": np.full((step_count, MAX_EVENTS_PER_STEP), -1, np.int32),
        "focus": np.full((step_count, 2), 0.5, np.float32),
        "sess_offset": np.zeros(len(sessions), np.int64),
        "sess_len": np.zeros(len(sessions), np.int32),
    }
    truncated_steps = 0
    row = 0
    for session_index, session in enumerate(sessions):
        arrays["sess_offset"][session_index] = row
        arrays["sess_len"][session_index] = len(session["steps"])
        for step in session["steps"]:
            truncated_steps += len(step["events"]) > MAX_EVENTS_PER_STEP
            for slot, event in enumerate(step["events"][:MAX_EVENTS_PER_STEP]):
                write_event(arrays, row, slot, event, key_ids, text_ids)
            if step["focus"] is not None:
                arrays["focus"][row] = step["focus"]
            row += 1
    return arrays, truncated_steps


def write_event(
    arrays: dict[str, np.ndarray], row: int, slot: int, event: Event, key_ids: dict[str, int], text_ids: dict[str, int]
) -> None:
    arrays["ev_type"][row, slot] = TYPE_ID[event.type]
    arrays["ev_xy"][row, slot] = [
        np.nan if value is None else value for value in (event.x, event.y, event.x2, event.y2)
    ]
    if event.scroll is not None:
        arrays["ev_scroll"][row, slot] = float(np.clip(event.scroll, -SCROLL_LIMIT, SCROLL_LIMIT))
    keys = [key_ids.get(key, UNKNOWN_KEY_ID) for key in event.keys[:MAX_KEYS_PER_EVENT]]
    arrays["ev_keys"][row, slot, : len(keys)] = keys
    if event.text:
        arrays["ev_text"][row, slot] = text_ids[event.text]


# ----------------------------------------------------------------------------------------------------- outputs
def write_sessions(sessions: list[dict]) -> None:
    """Writes sessions.pkl. The events live in events.npz, so each session keeps only its image names."""
    for session in sessions:
        session["images"] = [step["image"] for step in session["steps"]]
        del session["steps"]
    with open(paths.PREP / "sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)


def summarize(
    sessions: list[dict], stats: collections.Counter, test_apps: set[str], key_vocab: list[str], typed_texts: list[str]
) -> dict:
    by_split = collections.defaultdict(list)
    for session in sessions:
        by_split[session["split"]].append(session)
    return {
        "stats": dict(stats),
        "splits": {split: len(members) for split, members in by_split.items()},
        "n_test_apps": len(test_apps),
        "test_apps": sorted(test_apps),
        "n_key_vocab": len(key_vocab),
        "n_texts": len(typed_texts),
        "dup_groups": {split: duplicate_group_count(members) for split, members in by_split.items()},
        "os_by_split": {
            split: dict(collections.Counter(session["os"] for session in members))
            for split, members in by_split.items()
        },
    }


def duplicate_group_count(sessions: list[dict]) -> int:
    group_sizes = collections.Counter(session["group"] for session in sessions)
    return sum(1 for size in group_sizes.values() if size > 1)


if __name__ == "__main__":
    main()
