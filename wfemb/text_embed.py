"""Step 1c (offline, GPU): frozen text embeddings.

  keys.npy        [S, D] float32  l2-normalized instruction keys (teacher targets), session order
  para_keys.npy   [M, D] float16  paraphrase keys that pass the drift filter (cos > THETA_PARA)
  para_index.npz  offset/count per session into para_keys
  text_vecs.npy   [n_texts, D] float16  embeddings of typed strings (write events)
"""
from __future__ import annotations

import argparse
import json
import pickle
import re

import numpy as np
import torch

from . import paths

# Incidental literals (only used when --placeholders is set, see the literal check in inspect_keys.py)
_LITERAL_PATTERNS = [
    (re.compile(r"\b[\w\-]+\.(?:xlsx?|docx?|pptx?|pdf|csv|txt|png|jpe?g|py|json|md|zip|mp4|mp3|html?)\b", re.I), "<FILE>"),
    (re.compile(r"\"[^\"]{1,80}\"|'[^']{1,80}'"), "<VALUE>"),
    (re.compile(r"\b\d+(?:[.,:/-]\d+)*\b"), "<VALUE>"),
]


def placeholder(s: str) -> str:
    for pat, rep in _LITERAL_PATTERNS:
        s = pat.sub(rep, s)
    return s


def load_embedder(name: str, max_len: int):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    m.max_seq_length = max_len
    return m


def encode(model, texts, bs=256) -> np.ndarray:
    # sort by length for speed, then restore order
    order = np.argsort([len(t) for t in texts])[::-1]
    out = model.encode([texts[i] for i in order], batch_size=bs, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    res = np.empty_like(out)
    res[order] = out
    return res.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--theta_para", type=float, default=0.85)
    ap.add_argument("--placeholders", action="store_true",
                    help="embed instructions with incidental literals replaced by placeholders")
    ap.add_argument("--skip_texts", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--para_keep_frac", type=float, default=0.0,
                    help="if set, choose theta_para as the similarity quantile that keeps this fraction "
                         "(calibrates the drift filter to a new embedder's similarity scale)")
    args = ap.parse_args()
    paths.KEYS.mkdir(parents=True, exist_ok=True)

    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    with open(paths.PREP / "vocab.json") as f:
        vocab = json.load(f)
    prep = placeholder if args.placeholders else (lambda s: s)

    model = load_embedder(args.model, max_len=256)
    instr = [prep(s["instruction"]) for s in sessions]
    uniq = sorted(set(instr))
    uvec = encode(model, uniq, bs=args.batch)
    pos = {t: i for i, t in enumerate(uniq)}
    keys = uvec[[pos[t] for t in instr]]
    np.save(paths.KEYS / "keys.npy", keys)
    print("keys", keys.shape)

    stats = {"model": args.model, "dim": int(keys.shape[1]), "placeholders": args.placeholders,
             "theta_para": args.theta_para}
    para_path = paths.SHARED / "paraphrases.json"
    if para_path.exists():
        with open(para_path) as f:
            paras = json.load(f)["paraphrases"]
        flat, owner = [], []
        for si, s in enumerate(sessions):
            for p in paras.get(s["instruction"], []):
                flat.append(prep(p))
                owner.append(si)
        uflat = sorted(set(flat))
        fvec = encode(model, uflat, bs=args.batch)
        fpos = {t: i for i, t in enumerate(uflat)}
        pvec = fvec[[fpos[t] for t in flat]]
        owner = np.asarray(owner)
        sims = (pvec * keys[owner]).sum(1)
        if args.para_keep_frac:
            args.theta_para = float(np.quantile(sims, 1 - args.para_keep_frac))
            stats["theta_para"] = args.theta_para
        keep = sims > args.theta_para
        stats["para_total"] = int(len(sims))
        stats["para_kept"] = int(keep.sum())
        stats["para_sim_quantiles"] = {q: float(np.quantile(sims, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        pvec, owner = pvec[keep], owner[keep]
        offset = np.zeros(len(sessions), np.int64)
        count = np.bincount(owner, minlength=len(sessions)).astype(np.int32)
        offset[1:] = np.cumsum(count)[:-1]
        # owner is already sorted by session (built in session order), so rows are contiguous
        np.save(paths.KEYS / "para_keys.npy", pvec.astype(np.float16))
        np.savez(paths.KEYS / "para_index.npz", offset=offset, count=count)
        stats["sessions_with_paraphrase"] = int((count > 0).sum())
        print("paraphrases", stats)
    else:
        print("no paraphrases.json yet; skipping paraphrase keys")

    if not args.skip_texts:
        texts = vocab["texts"]
        model.max_seq_length = 128
        tvec = encode(model, texts, bs=2 * args.batch) if texts else np.zeros((0, keys.shape[1]), np.float32)
        np.save(paths.KEYS / "text_vecs.npy", tvec.astype(np.float16))
        stats["n_texts"] = len(texts)
    with open(paths.KEYS / "meta.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
