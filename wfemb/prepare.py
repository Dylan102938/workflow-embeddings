"""Step 1 (offline, CPU): load + filter AgentNet, parse actions, grouped splits, flat event arrays.

Outputs (under PREP):
  sessions.pkl   list of session dicts (id, instruction, split, group, os, app keys, images, ...)
  events.npz     flat per-step event arrays in global step order (= frame-cache order)
  vocab.json     key-token vocab, event types, typed-text strings
  summary.json   counts
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

from . import paths
from .actions import TYPE_ID, parse_code

E_MAX = 8      # events kept per step (>99.8% of steps have <= 8 calls)
K_MAX = 4      # key tokens kept per event

BROWSERS = {
    "google chrome", "chrome", "microsoft edge", "edge", "safari", "cent browser", "mozilla firefox",
    "firefox", "opera", "brave", "brave browser", "360 browser", "qq browser", "arc", "vivaldi",
    "chromium", "sogou browser", "uc browser", "360 secure browser",
}
GENERIC_SITES = {"google", "bing", "baidu", "yahoo", "duckduckgo"}
BAD_ALIGNMENT = {"low alignment", "not aligned", "low", "not aligned "}


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def group_hash(instruction: str) -> str:
    return hashlib.md5(normalize_ws(instruction.lower()).encode()).hexdigest()[:16]


def app_keys(meta: dict) -> tuple[str, list[str]]:
    """Primary app key (most specific) and all app keys of a session."""
    apps = [a.lower().strip() for a in (meta.get("applications") or []) if a]
    sites = [w.lower().strip() for w in (meta.get("websites") or []) if w]
    non_browser = [a for a in apps if a not in BROWSERS]
    keys = [f"app:{a}" for a in non_browser] + [f"web:{w}" for w in sites]
    if non_browser:
        primary = f"app:{non_browser[0]}"
    elif [w for w in sites if w not in GENERIC_SITES]:
        primary = "web:" + [w for w in sites if w not in GENERIC_SITES][0]
    elif sites:
        primary = f"web:{sites[0]}"
    elif apps:
        primary = f"app:{apps[0]}"
        keys.append(primary)
    else:
        primary = f"domain:{str(meta.get('domains') or 'unknown').lower()}"
        keys.append(primary)
    return primary, sorted(set(keys))


def sample_test_apps(sessions, fraction, seed, max_app_frac=0.015):
    """Hold out whole apps: sample primary app keys until ~fraction of sessions are covered.
    Very large keys (> max_app_frac of all sessions) are never held out, to keep test diverse."""
    counts = collections.Counter(s["app"] for s in sessions)
    cap = max_app_frac * len(sessions)
    cands = sorted(k for k, c in counts.items() if c <= cap)
    random.Random(seed).shuffle(cands)
    test_apps, covered = set(), 0
    for k in cands:
        if covered >= fraction * len(sessions):
            break
        test_apps.add(k)
        covered += counts[k]
    return test_apps


def group_split(sessions, frac, seed):
    groups = sorted({s["group"] for s in sessions})
    random.Random(seed).shuffle(groups)
    n_sess = len(sessions)
    by_group = collections.Counter(s["group"] for s in sessions)
    val_groups, covered = set(), 0
    for g in groups:
        if covered >= frac * n_sess:
            break
        val_groups.add(g)
        covered += by_group[g]
    return val_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--key_min_count", type=int, default=3)
    args = ap.parse_args()
    paths.PREP.mkdir(parents=True, exist_ok=True)

    metas = {}
    with open(paths.META) as f:
        for line in f:
            m = json.loads(line)
            metas[m["task_id"]] = m

    stats = collections.Counter()
    sessions = []
    seen_ids, seen_images = set(), set()
    parse_fail = 0
    for source, jp in paths.JSONLS.items():
        with open(jp) as f:
            for line in f:
                d = json.loads(line)
                stats["raw"] += 1
                tid = d["task_id"]
                if tid in seen_ids:
                    stats["dup_task_id"] += 1
                    continue
                seen_ids.add(tid)
                img_sig = tuple(s["image"] for s in d["traj"])
                if img_sig in seen_images:
                    stats["dup_images"] += 1
                    continue
                seen_images.add(img_sig)
                meta = metas.get(tid, {})
                align = str(meta.get("task_description_alignment") or "").lower().strip()
                if align in BAD_ALIGNMENT:
                    stats["low_alignment"] += 1
                    continue
                if len(d["traj"]) < 2:
                    stats["too_short"] += 1
                    continue
                cursor = None
                steps = []
                for s in sorted(d["traj"], key=lambda s: s["index"]):
                    try:
                        events, cursor = parse_code(s["value"].get("code") or "", cursor)
                    except Exception:
                        events, parse_fail = [], parse_fail + 1
                    # No artificial end-of-session cue: drop terminate. The terminate step's screenshot
                    # is the real final screen and stays as an event-free step.
                    events = [e for e in events if e.type != "terminate"]
                    pt = None
                    for e in events:
                        if e.point is not None:
                            pt = e.point
                    steps.append({"image": s["image"], "events": events,
                                  "focus": pt if pt is not None else cursor})
                primary, keys = app_keys(meta)
                sessions.append({
                    "id": tid, "source": source, "instruction": d["instruction"],
                    "group": group_hash(d["instruction"]),
                    "os": meta.get("system") or ("Ubuntu" if source == "ubuntu" else "unknown"),
                    "domain": str(meta.get("domains") or d.get("domain") or "unknown"),
                    "app": primary, "app_keys": keys,
                    "task_completed": d.get("task_completed"),
                    "screen": (meta.get("screen_width"), meta.get("screen_height")),
                    "steps": steps,
                })
    stats["kept"] = len(sessions)
    stats["parse_fail_steps"] = parse_fail

    # ---------- grouped splits (nothing leaks across splits) ----------
    test_apps = sample_test_apps(sessions, args.test_frac, args.seed)
    for s in sessions:
        s["split"] = "test" if set(s["app_keys"]) & test_apps or s["app"] in test_apps else None
    rest = [s for s in sessions if s["split"] is None]
    val_groups = group_split(rest, args.val_frac, args.seed)
    for s in rest:
        s["split"] = "val" if s["group"] in val_groups else "train"
    # a duplicate-instruction group must not straddle test and train/val either
    test_groups = {s["group"] for s in sessions if s["split"] == "test"}
    moved = 0
    for s in sessions:
        if s["split"] != "test" and s["group"] in test_groups:
            s["split"] = "test"
            moved += 1
    stats["moved_to_test_by_group"] = moved
    order = {"train": 0, "val": 1, "test": 2}
    sessions.sort(key=lambda s: (order[s["split"]], s["id"]))

    # ---------- vocab ----------
    key_counts = collections.Counter()
    texts = {}
    for s in sessions:
        for st in s["steps"]:
            for e in st["events"]:
                key_counts.update(e.keys)
                if e.text:
                    texts.setdefault(e.text, len(texts))
    key_vocab = ["<pad>", "<unk>"] + sorted(k for k, c in key_counts.items() if c >= args.key_min_count)
    key_id = {k: i for i, k in enumerate(key_vocab)}
    text_list = [None] * len(texts)
    for t, i in texts.items():
        text_list[i] = t

    # ---------- flat event arrays in global step order ----------
    n_steps = sum(len(s["steps"]) for s in sessions)
    ev_type = np.zeros((n_steps, E_MAX), np.int8)
    ev_xy = np.full((n_steps, E_MAX, 4), np.nan, np.float16)
    ev_scroll = np.zeros((n_steps, E_MAX), np.float16)
    ev_keys = np.zeros((n_steps, E_MAX, K_MAX), np.int16)
    ev_text = np.full((n_steps, E_MAX), -1, np.int32)
    focus = np.full((n_steps, 2), 0.5, np.float32)
    sess_offset = np.zeros(len(sessions), np.int64)
    sess_len = np.zeros(len(sessions), np.int32)
    truncated = 0
    g = 0
    for si, s in enumerate(sessions):
        sess_offset[si] = g
        sess_len[si] = len(s["steps"])
        for st in s["steps"]:
            evs = st["events"]
            if len(evs) > E_MAX:
                truncated += 1
                evs = evs[:E_MAX]
            for j, e in enumerate(evs):
                ev_type[g, j] = TYPE_ID[e.type]
                ev_xy[g, j] = [np.nan if v is None else v for v in (e.x, e.y, e.x2, e.y2)]
                ev_scroll[g, j] = 0.0 if e.scroll is None else float(np.clip(e.scroll, -1000, 1000))
                ks = [key_id.get(k, 1) for k in e.keys[:K_MAX]]
                ev_keys[g, j, :len(ks)] = ks
                if e.text:
                    ev_text[g, j] = texts[e.text]
            if st["focus"] is not None:
                focus[g] = st["focus"]
            g += 1
    stats["steps"] = n_steps
    stats["steps_truncated_events"] = truncated
    np.savez(paths.PREP / "events.npz", ev_type=ev_type, ev_xy=ev_xy, ev_scroll=ev_scroll,
             ev_keys=ev_keys, ev_text=ev_text, focus=focus, sess_offset=sess_offset, sess_len=sess_len)

    for s in sessions:   # events now live in events.npz; keep only what later stages need
        s["images"] = [st["image"] for st in s["steps"]]
        del s["steps"]
    with open(paths.PREP / "sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)
    with open(paths.PREP / "vocab.json", "w") as f:
        json.dump({"key_vocab": key_vocab, "texts": text_list, "e_max": E_MAX, "k_max": K_MAX}, f)

    split_counts = collections.Counter(s["split"] for s in sessions)
    summary = {
        "stats": dict(stats), "splits": dict(split_counts), "n_test_apps": len(test_apps),
        "test_apps": sorted(test_apps), "n_key_vocab": len(key_vocab), "n_texts": len(text_list),
        "dup_groups": {sp: sum(1 for c in collections.Counter(
            s["group"] for s in sessions if s["split"] == sp).values() if c > 1) for sp in split_counts},
        "os_by_split": {sp: dict(collections.Counter(s["os"] for s in sessions if s["split"] == sp))
                        for sp in split_counts},
    }
    with open(paths.PREP / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "test_apps"}, indent=1))


if __name__ == "__main__":
    main()
