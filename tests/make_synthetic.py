"""A tiny synthetic data tree in the real layout, for smoke tests:

  ROOT/prepared/{sessions.pkl, events.npz, vocab.json}
  ROOT/keys_openai/{keys.npy, text_vecs.npy, para_keys.npy, para_index.npz}
  ROOT/keys/judged_{val,test}.jsonl
  ROOT/bulk/frames_dense_g4/{global_*.npy, meta_*.json}        (patch = 16 tokens of dim 32)

  python tests/make_synthetic.py ROOT
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PATCH, DV, D = 16, 32, 64


def main(root, S=300, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for sub in ("prepared", "keys_openai", "keys", "bulk/frames_dense_g4"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    n_int = 40                                   # latent intents; sessions with the same intent look alike
    intent = rng.integers(0, n_int, S)
    centers = rng.normal(size=(n_int, D))
    lens = rng.integers(2, 25, S)
    sessions = []
    for i in range(S):
        split = "train" if i < 0.7 * S else ("val" if i < 0.85 * S else "test")
        sessions.append({"id": f"s{i}", "source": "ubuntu", "instruction": f"task {intent[i]} v{i}",
                         "group": f"g{i if i % 10 else i - 1}", "os": ["Windows", "Darwin", "Ubuntu"][i % 3],
                         "domain": "x", "app": f"app:{intent[i] % 7}", "app_keys": [f"app:{intent[i] % 7}"],
                         "task_completed": True, "screen": (1920, 1080), "split": split,
                         "images": [f"{i}_{t}.png" for t in range(lens[i])]})
    F, E, K = int(lens.sum()), 8, 4
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    ev_type = rng.integers(0, 14, (F, E)).astype(np.int8)
    ev_type[:, 3:] = 0
    ev_xy = rng.random((F, E, 4)).astype(np.float16)
    ev_xy[rng.random((F, E, 4)) < 0.5] = np.nan
    np.savez(root / "prepared/events.npz", ev_type=ev_type, ev_xy=ev_xy,
             ev_scroll=rng.normal(size=(F, E)).astype(np.float16), ev_keys=rng.integers(0, 10, (F, E, K)).astype(np.int16),
             ev_text=np.where(rng.random((F, E)) < 0.2, rng.integers(0, 50, (F, E)), -1).astype(np.int32),
             focus=rng.random((F, 2)).astype(np.float32), sess_offset=off.astype(np.int64), sess_len=lens.astype(np.int32))
    pickle.dump(sessions, open(root / "prepared/sessions.pkl", "wb"))
    json.dump({"key_vocab": ["<pad>", "<unk>"] + [f"k{i}" for i in range(8)], "texts": [f"t{i}" for i in range(50)],
               "e_max": E, "k_max": K}, open(root / "prepared/vocab.json", "w"))
    keys = centers[intent] + 0.3 * rng.normal(size=(S, D))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    kd = root / "keys_openai"
    np.save(kd / "keys.npy", keys.astype(np.float32))
    np.save(kd / "text_vecs.npy", rng.normal(size=(50, D)).astype(np.float16))
    cnt = rng.integers(0, 4, S)
    offp = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pk = np.repeat(keys, cnt, 0) + 0.05 * rng.normal(size=(cnt.sum(), D))
    pk /= np.linalg.norm(pk, axis=1, keepdims=True)
    np.save(kd / "para_keys.npy", pk.astype(np.float16))
    np.savez(kd / "para_index.npz", offset=offp, count=cnt)
    for sp in ("val", "test"):                   # judged pairs: same latent intent = 2, same app = 1
        idx = [i for i, s in enumerate(sessions) if s["split"] == sp]
        with open(root / f"keys/judged_{sp}.jsonl", "w") as f:
            for a in idx:
                for b in idx:
                    if a < b and (intent[a] == intent[b] or intent[a] % 7 == intent[b] % 7):
                        f.write(json.dumps({"a": a, "b": b, "label": 2 if intent[a] == intent[b] else 1}) + "\n")
    proj = rng.normal(size=(D, DV)) / np.sqrt(D)
    per_sess = centers[intent] @ proj                                   # [S, DV]
    g = np.repeat(per_sess, lens, 0)[:, None, :] + rng.normal(size=(F, PATCH, DV))
    bits = lambda x: torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    half = F // 2
    fd = root / "bulk/frames_dense_g4"
    for tag, lo, hi in (("00of02", 0, half), ("01of02", half, F)):
        np.save(fd / f"global_{tag}.npy", bits(g[lo:hi]))
        json.dump({"lo": lo, "hi": hi, "dim": DV, "global_tokens": PATCH, "grid": 4}, open(fd / f"meta_{tag}.json", "w"))
    print("synthetic data:", S, "sessions", F, "frames")


if __name__ == "__main__":
    main(sys.argv[1])
