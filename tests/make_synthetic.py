"""Writes a tiny synthetic dataset in the real layout, for tests.

  python tests/make_synthetic.py ROOT
Sessions with the same latent intent get similar screen features and instruction keys, so a model can learn.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

N_PATCHES, D_SCREEN, D_LABEL = 16, 32, 64
MAX_EVENTS, MAX_KEYS, N_TEXTS, N_INTENTS, N_APPS = 8, 4, 50, 40, 7


def bfloat16_bits(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def main(root, n_sessions: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    frames_dir = root / "bulk" / "frames_dense_g4"
    for folder in (root / "prepared", root / "keys_openai", root / "keys", frames_dir):
        folder.mkdir(parents=True, exist_ok=True)

    intents = rng.integers(0, N_INTENTS, n_sessions)
    intent_centers = rng.normal(size=(N_INTENTS, D_LABEL))
    lengths = rng.integers(2, 25, n_sessions)
    n_steps = int(lengths.sum())

    def split_of(index: int) -> str:
        return "train" if index < 0.7 * n_sessions else "val" if index < 0.85 * n_sessions else "test"

    sessions = [
        {
            "id": f"s{index}",
            "source": "ubuntu",
            "instruction": f"task {intents[index]} v{index}",
            "group": f"g{index if index % 10 else index - 1}",
            "os": ["Windows", "Darwin", "Ubuntu"][index % 3],
            "domain": "x",
            "app": f"app:{intents[index] % N_APPS}",
            "app_keys": [f"app:{intents[index] % N_APPS}"],
            "task_completed": True,
            "screen": (1920, 1080),
            "split": split_of(index),
            "images": [f"{index}_{step}.png" for step in range(lengths[index])],
        }
        for index in range(n_sessions)
    ]
    pickle.dump(sessions, open(root / "prepared" / "sessions.pkl", "wb"))

    event_types = rng.integers(0, 14, (n_steps, MAX_EVENTS)).astype(np.int8)
    event_types[:, 3:] = 0
    event_points = rng.random((n_steps, MAX_EVENTS, 4)).astype(np.float16)
    event_points[rng.random(event_points.shape) < 0.5] = np.nan
    has_text = rng.random((n_steps, MAX_EVENTS)) < 0.2
    np.savez(
        root / "prepared" / "events.npz",
        ev_type=event_types,
        ev_xy=event_points,
        ev_scroll=rng.normal(size=(n_steps, MAX_EVENTS)).astype(np.float16),
        ev_keys=rng.integers(0, 10, (n_steps, MAX_EVENTS, MAX_KEYS)).astype(np.int16),
        ev_text=np.where(has_text, rng.integers(0, N_TEXTS, (n_steps, MAX_EVENTS)), -1).astype(np.int32),
        focus=rng.random((n_steps, 2)).astype(np.float32),
        sess_offset=np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64),
        sess_len=lengths.astype(np.int32),
    )
    json.dump(
        {
            "key_vocab": ["<pad>", "<unk>"] + [f"k{i}" for i in range(8)],
            "texts": [f"t{i}" for i in range(N_TEXTS)],
            "e_max": MAX_EVENTS,
            "k_max": MAX_KEYS,
        },
        open(root / "prepared" / "vocab.json", "w"),
    )

    keys = unit_rows(intent_centers[intents] + 0.3 * rng.normal(size=(n_sessions, D_LABEL)))
    paraphrase_counts = rng.integers(0, 4, n_sessions)
    paraphrase_keys = unit_rows(
        np.repeat(keys, paraphrase_counts, 0) + 0.05 * rng.normal(size=(paraphrase_counts.sum(), D_LABEL))
    )
    labels_dir = root / "keys_openai"
    np.save(labels_dir / "keys.npy", keys.astype(np.float32))
    np.save(labels_dir / "text_vecs.npy", rng.normal(size=(N_TEXTS, D_LABEL)).astype(np.float16))
    np.save(labels_dir / "para_keys.npy", paraphrase_keys.astype(np.float16))
    np.savez(labels_dir / "para_index.npz", offset=np.r_[0, np.cumsum(paraphrase_counts)[:-1]], count=paraphrase_counts)

    for split in ("val", "test"):  # judged pairs: same intent = 2, same app = 1
        members = [index for index, session in enumerate(sessions) if session["split"] == split]
        with open(root / "keys" / f"judged_{split}.jsonl", "w") as file:
            for first in members:
                for second in members:
                    same_intent = intents[first] == intents[second]
                    same_app = intents[first] % N_APPS == intents[second] % N_APPS
                    if first < second and (same_intent or same_app):
                        file.write(json.dumps({"a": first, "b": second, "label": 2 if same_intent else 1}) + "\n")

    projection = rng.normal(size=(D_LABEL, D_SCREEN)) / np.sqrt(D_LABEL)
    session_features = intent_centers[intents] @ projection
    frames = np.repeat(session_features, lengths, 0)[:, None, :] + rng.normal(size=(n_steps, N_PATCHES, D_SCREEN))
    middle = n_steps // 2
    for tag, start, end in (("00of02", 0, middle), ("01of02", middle, n_steps)):
        np.save(frames_dir / f"global_{tag}.npy", bfloat16_bits(frames[start:end]))
        json.dump(
            {"lo": start, "hi": end, "dim": D_SCREEN, "global_tokens": N_PATCHES, "grid": 4},
            open(frames_dir / f"meta_{tag}.json", "w"),
        )
    print(f"synthetic data: {n_sessions} sessions, {n_steps} steps")


if __name__ == "__main__":
    main(sys.argv[1])
