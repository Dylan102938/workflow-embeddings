"""Checks the added datasets' prepared set against its files, and against what AgentNet holds out.

  python -m wfemb.prep.validate [--decode 2000] [--contact_sheets DIR --sheets_per_dataset 6]

Prints a report, writes ADDED_DATA/prepared/validation.json and exits with 1 if a check fails.
The checks do not reuse the cleaning code's bookkeeping: they recompute everything from the written files.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pickle
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import AGENTNET
from ..storage import paths
from . import clean
from .actions import EVENT_TYPES

PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
POINTER_TYPES = ("click", "doubleClick", "tripleClick", "rightClick", "middleClick", "moveTo", "dragTo")
SHEET_COLUMNS, SHEET_CELL = 4, (480, 300)
DECODE_WORKERS = 16


class PreparedFiles:
    def __init__(self, root: Path):
        with open(root / "prepared" / "sessions.pkl", "rb") as f:
            self.sessions = pickle.load(f)
        stored = np.load(root / "prepared" / "events.npz")
        self.events = {name: stored[name] for name in stored.files}
        self.vocab = json.load(open(root / "prepared" / "vocab.json"))
        self.root = root

    def steps_of(self, session_index: int) -> slice:
        start = int(self.events["sess_offset"][session_index])
        return slice(start, start + int(self.events["sess_len"][session_index]))


def main() -> None:
    args = parse_arguments()
    agentnet, added = PreparedFiles(paths.AGENTNET_DATA), PreparedFiles(paths.ADDED_DATA)
    test_applications = json.load(open(paths.AGENTNET_DATA / "prepared" / "summary.json"))["test_apps"]
    checks: dict[str, bool | str] = {}
    checks.update(structure_checks(added, agentnet))
    checks.update(leakage_checks(added, agentnet, test_applications))
    checks.update(duplicate_checks(added))
    checks.update(image_checks(added, args.decode))
    checks.update(derived_file_checks(added))
    report = {
        "checks": checks,
        "datasets": {dataset: describe(added, dataset) for dataset in sorted({s["dataset"] for s in added.sessions})},
        AGENTNET: describe(agentnet, None),
    }
    with open(paths.ADDED_DATA / "prepared" / "validation.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if args.contact_sheets:
        write_contact_sheets(added, Path(args.contact_sheets), args.sheets_per_dataset)
    failed = [name for name, result in checks.items() if result is not True and not str(result).startswith("skipped")]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", type=int, default=2000, help="screenshots to decode (random sample; -1: all)")
    parser.add_argument("--contact_sheets", help="folder for one image per sampled session, to inspect by eye")
    parser.add_argument("--sheets_per_dataset", type=int, default=6)
    return parser.parse_args()


# ------------------------------------------------------------------------------------------------------ checks
def structure_checks(added: PreparedFiles, agentnet: PreparedFiles) -> dict:
    events, sessions = added.events, added.sessions
    lengths, offsets = events["sess_len"], events["sess_offset"]
    n_steps = len(events["ev_type"])
    points = events["ev_xy"][~np.isnan(events["ev_xy"])]
    used_texts = events["ev_text"][events["ev_text"] >= 0]
    agentnet_ids = {session["id"] for session in agentnet.sessions}
    ids = [session["id"] for session in sessions]
    return {
        "one events row per step": bool(len(lengths) == len(sessions) and int(lengths.sum()) == n_steps),
        "sessions own consecutive steps": bool((offsets == np.r_[0, np.cumsum(lengths)[:-1]]).all()),
        "one screenshot per step": all(len(s["images"]) == n for s, n in zip(sessions, lengths)),
        "every session has at least 2 steps": bool(lengths.min() >= clean.MIN_STEPS),
        "all sessions are training data": all(session["split"] == "train" for session in sessions),
        "datasets are known": all(session["dataset"] in paths.ADDED_DATASETS for session in sessions),
        "ids are unique, also against AgentNet": len(set(ids)) == len(ids) and not agentnet_ids & set(ids),
        "event types are valid": bool(events["ev_type"].min() >= 0 and events["ev_type"].max() < len(EVENT_TYPES)),
        "positions are on the screen": bool(len(points) == 0 or (points.min() >= 0 and points.max() <= 1)),
        "key ids are in AgentNet's vocabulary": bool(
            added.vocab["key_vocab"] == agentnet.vocab["key_vocab"]
            and events["ev_keys"].max() < len(added.vocab["key_vocab"])
        ),
        "typed-text ids are valid and all used": bool(
            len(used_texts) == 0
            or (
                used_texts.max() < len(added.vocab["texts"]) and len(np.unique(used_texts)) == len(added.vocab["texts"])
            )
        ),
        "instructions are not empty": all(
            len(session["instruction"].split()) >= clean.MIN_INSTRUCTION_WORDS for session in sessions
        ),
    }


def leakage_checks(added: PreparedFiles, agentnet: PreparedFiles, test_applications: list[str]) -> dict:
    evaluation = [session for session in agentnet.sessions if session["split"] != "train"]
    held_out_groups = {session["group"] for session in evaluation}
    held_out_names = {clean.canonical_application(key) for key in test_applications}
    similarities = clean.similarity_to_evaluation(
        [session["instruction"] for session in added.sessions], [session["instruction"] for session in evaluation]
    )
    return {
        "no instruction of the validation or test split": not any(
            session["group"] in held_out_groups for session in added.sessions
        ),
        "no instruction near one of the validation or test split": bool(
            similarities.max() < clean.EVALUATION_SIMILARITY
        ),
        "no held-out test application": not any(
            clean.canonical_application(key) in held_out_names
            for session in added.sessions
            for key in session["app_keys"]
        ),
    }


def duplicate_checks(added: PreparedFiles) -> dict:
    """Recomputed from events.npz: no two sessions with the same instruction and the same coarse action sequence."""
    events = added.events
    grid = np.where(np.isnan(events["ev_xy"]), -1, np.floor(events["ev_xy"].astype(np.float32) * clean.SIGNATURE_GRID))
    demonstrations: set[tuple] = set()
    image_sequences: set[tuple] = set()
    repeated_demonstrations = repeated_images = 0
    sessions_per_group: collections.Counter = collections.Counter()
    for index, session in enumerate(added.sessions):
        steps = added.steps_of(index)
        signature = (
            session["group"],
            events["ev_type"][steps].tobytes(),
            grid[steps].astype(np.int8).tobytes(),
            events["ev_keys"][steps].tobytes(),
            events["ev_text"][steps].tobytes(),
        )
        repeated_demonstrations += signature in demonstrations
        demonstrations.add(signature)
        images = tuple(session["images"])
        repeated_images += images in image_sequences
        image_sequences.add(images)
        sessions_per_group[session["group"]] += 1
    return {
        "no repeated demonstration (instruction and actions)": repeated_demonstrations == 0,
        "no repeated screenshot sequence": repeated_images == 0,
        f"at most {clean.MAX_SESSIONS_PER_INSTRUCTION} sessions per instruction": max(sessions_per_group.values())
        <= clean.MAX_SESSIONS_PER_INSTRUCTION,
    }


def image_checks(added: PreparedFiles, n_decoded: int) -> dict:
    """Every screenshot is in its dataset's archive; a sample (or all) of them decode and are not blank."""
    members = {dataset: set(zipfile.ZipFile(paths.ZIPS[dataset]).namelist()) for dataset in paths.ADDED_DATASETS}
    images = [(session["source"], image) for session in added.sessions for image in session["images"]]
    missing = sum(image not in members[dataset] for dataset, image in images)
    if n_decoded >= 0:
        chosen = np.random.default_rng(0).choice(len(images), min(n_decoded, len(images)), replace=False)
        images = [images[i] for i in chosen]
    with Pool(DECODE_WORKERS) as pool:
        states = collections.Counter(pool.imap_unordered(screenshot_state, images, chunksize=64))
    return {
        "every screenshot is in the archive": missing == 0,
        f"decoded screenshots are readable ({len(images)} checked)": states["unreadable"] == 0,
        "at most 1% of the decoded screenshots are blank": states["blank"] <= 0.01 * max(len(images), 1),
    }


_archives: dict[str, zipfile.ZipFile] = {}


def screenshot_state(image: tuple[str, str]) -> str:
    """ "ok", "blank" or "unreadable". Runs in a worker process, which keeps its own archive handles."""
    dataset, name = image
    if dataset not in _archives:
        _archives[dataset] = zipfile.ZipFile(paths.ZIPS[dataset])
    try:
        grey = Image.open(io.BytesIO(_archives[dataset].read(name))).convert("L")
    except Exception:
        return "unreadable"
    return "blank" if float(np.asarray(grey.resize((64, 64))).std()) < clean.BLANK_SPREAD else "ok"


def derived_file_checks(added: PreparedFiles) -> dict:
    """Key sets and frame caches, where they have been built."""
    checks: dict[str, bool | str] = {}
    n_sessions, n_steps = len(added.sessions), len(added.events["ev_type"])
    for label_set in ("keys", "keys_openai"):
        directory = added.root / label_set
        if not (directory / "keys.npy").exists():
            checks[f"{label_set}: built"] = "skipped (not built yet)"
            continue
        keys = np.load(directory / "keys.npy", mmap_mode="r")
        norms = np.linalg.norm(np.asarray(keys[:: max(1, n_sessions // 2000)], np.float32), axis=1)
        checks[f"{label_set}: one unit-length key per session"] = bool(
            len(keys) == n_sessions and np.allclose(norms, 1, atol=1e-2)
        )
        agentnet_keys_path = paths.AGENTNET_DATA / label_set / "keys.npy"
        if agentnet_keys_path.exists():
            agentnet_keys = np.load(agentnet_keys_path, mmap_mode="r")
            checks[f"{label_set}: same dimension as AgentNet's"] = keys.shape[1] == agentnet_keys.shape[1]
        if (directory / "para_index.npz").exists():
            index = np.load(directory / "para_index.npz")
            n_paraphrase_keys = len(np.load(directory / "para_keys.npy", mmap_mode="r"))
            checks[f"{label_set}: paraphrase index matches its keys"] = bool(
                len(index["count"]) == n_sessions and int(index["count"].sum()) == n_paraphrase_keys
            )
        text_vectors = np.load(directory / "text_vecs.npy", mmap_mode="r")
        checks[f"{label_set}: one vector per typed text"] = len(text_vectors) == len(added.vocab["texts"])
    for cache_dir in sorted(paths.ADDED_BULK.glob("frames_dense*")):
        metas = [json.load(open(path)) for path in sorted(cache_dir.glob("meta_*.json"))]
        covered = sorted((meta["lo"], meta["hi"]) for meta in metas)
        contiguous = bool(covered) and covered[0][0] == 0 and all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
        checks[f"{cache_dir.name}: covers every step"] = contiguous and covered[-1][1] == n_steps
        checks[f"{cache_dir.name}: no screenshot was missing"] = sum(meta["missing"] for meta in metas) == 0
    return checks


# ------------------------------------------------------------------------------------------------- description
def describe(prepared: PreparedFiles, dataset: str | None) -> dict:
    """Numbers to compare a dataset with AgentNet. dataset None: the whole prepared set."""
    indices = [i for i, s in enumerate(prepared.sessions) if dataset is None or s.get("dataset") == dataset]
    events = prepared.events
    rows = np.concatenate([np.arange(steps.start, steps.stop) for steps in map(prepared.steps_of, indices)])
    types = events["ev_type"][rows]
    type_counts = np.bincount(types[types > 0].ravel(), minlength=len(EVENT_TYPES))
    keys = events["ev_keys"][rows]
    sessions = [prepared.sessions[i] for i in indices]
    group_sizes = collections.Counter(session["group"] for session in sessions)
    lengths = events["sess_len"][indices]
    return {
        "sessions": len(sessions),
        "steps": int(len(rows)),
        "steps_per_session": percentiles(lengths),
        "instruction_words": percentiles([len(session["instruction"].split()) for session in sessions]),
        "steps_without_events": round(float(((types > 0).sum(1) == 0).mean()), 4),
        "event_shares": {
            EVENT_TYPES[i]: round(float(count / type_counts.sum()), 4) for i, count in enumerate(type_counts) if count
        },
        "unknown_key_share": round(float((keys == 1).sum() / max((keys > 0).sum(), 1)), 4),
        "instructions": len(group_sizes),
        "instructions_with_several_sessions": sum(size > 1 for size in group_sizes.values()),
        "applications": len({session["app"] for session in sessions}),
        "top_applications": collections.Counter(session["app"] for session in sessions).most_common(8),
        "systems": dict(collections.Counter(session["os"] for session in sessions)),
        "origins": dict(collections.Counter(session.get("origin", "human") for session in sessions)),
    }


def percentiles(values) -> dict:
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


# ---------------------------------------------------------------------------------------------- contact sheets
def write_contact_sheets(added: PreparedFiles, directory: Path, per_dataset: int) -> None:
    """One image per sampled session: its screenshots in order, each with the step's pointer positions marked."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for dataset in sorted({session["dataset"] for session in added.sessions}):
        archive = zipfile.ZipFile(paths.ZIPS[dataset])
        indices = [i for i, session in enumerate(added.sessions) if session["dataset"] == dataset]
        for index in rng.choice(indices, min(per_dataset, len(indices)), replace=False):
            session = added.sessions[index]
            sheet = contact_sheet(added, int(index), archive)
            sheet.save(directory / f"{session['id']}.jpg", quality=85)
            print(f"{session['id']}: {session['instruction']}")


def contact_sheet(added: PreparedFiles, session_index: int, archive: zipfile.ZipFile) -> Image.Image:
    session, steps = added.sessions[session_index], added.steps_of(session_index)
    types, points = added.events["ev_type"][steps], added.events["ev_xy"][steps].astype(np.float32)
    n_rows = -(-len(session["images"]) // SHEET_COLUMNS)
    sheet = Image.new("RGB", (SHEET_COLUMNS * SHEET_CELL[0], n_rows * SHEET_CELL[1]), "white")
    for step, image_name in enumerate(session["images"]):
        screen = Image.open(io.BytesIO(archive.read(image_name))).convert("RGB").resize(SHEET_CELL)
        draw = ImageDraw.Draw(screen)
        labels = []
        for slot in np.nonzero(types[step])[0]:
            name = EVENT_TYPES[types[step, slot]]
            labels.append(name)
            x, y, x2, y2 = points[step, slot]
            if name in POINTER_TYPES and not np.isnan(x):
                center = (x * SHEET_CELL[0], y * SHEET_CELL[1])
                draw.ellipse([center[0] - 7, center[1] - 7, center[0] + 7, center[1] + 7], outline="red", width=3)
                if not np.isnan(x2):
                    draw.line([center, (x2 * SHEET_CELL[0], y2 * SHEET_CELL[1])], fill="red", width=3)
        draw.rectangle([0, 0, SHEET_CELL[0], 14], fill="black")
        draw.text((3, 1), f"{step}: {' '.join(labels) or '(no events)'}", fill="yellow")
        sheet.paste(screen, ((step % SHEET_COLUMNS) * SHEET_CELL[0], (step // SHEET_COLUMNS) * SHEET_CELL[1]))
    return sheet


if __name__ == "__main__":
    main()
