"""Training / evaluation LABELS from OpenAI embeddings; the model's INPUT text features stay Qwen3-Embedding-0.6B.

Builds a key-set directory (default DATA/keys_openai), loadable with `train.py --keys_dir`:
  keys.npy        [S, D] float32   instruction keys (targets), session order
  para_keys.npy   [M, D] float16   paraphrase keys passing the drift filter (augmented positives)
  para_index.npz                   offset / count per session
  text_vecs.npy   -> symlink to the Qwen3-0.6B typed-text vectors (model input features, unchanged)
  raw_*.npy/json                   unique-string embeddings, so re-filtering needs no new API calls
Needs OPENAI_API_KEY (scripts/openai_keys.sbatch sources the repo's .env).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import paths

PRICE_PER_MTOK = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


def embed_unique(client, model, texts, batch, workers, cache_prefix):
    """Embeds a list of unique strings (cached on disk under cache_prefix)."""
    vec_path, txt_path = f"{cache_prefix}.npy", f"{cache_prefix}.json"
    if os.path.exists(vec_path) and os.path.exists(txt_path) and json.load(open(txt_path)) == texts:
        print(f"cached: {vec_path}", flush=True)
        return np.load(vec_path), 0
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    out, tokens = [None] * len(chunks), [0]

    def run(i):
        inp = [t[:30000] if t.strip() else "." for t in chunks[i]]   # API rejects empty strings; 8191-token cap
        r = client.embeddings.create(model=model, input=inp)
        return i, np.array([d.embedding for d in sorted(r.data, key=lambda d: d.index)], np.float32), \
            r.usage.total_tokens

    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for i, v, n in ex.map(run, range(len(chunks))):
            out[i] = v
            tokens[0] += n
            done += 1
            if done % 50 == 0 or done == len(chunks):
                print(f"  {done}/{len(chunks)} requests  {tokens[0] / 1e6:.2f}M tokens  {time.time() - t0:.0f}s",
                      flush=True)
    vec = np.concatenate(out)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    np.save(vec_path, vec)
    json.dump(texts, open(txt_path, "w"))
    return vec, tokens[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--out", default=str(paths.DATA / "keys_openai"))
    ap.add_argument("--para_keep_frac", type=float, default=0.927,
                    help="drift filter: keep this fraction of paraphrases (same fraction as the Qwen 0.6B key set)")
    ap.add_argument("--para_theta", type=float, default=0.0,
                    help="absolute drift threshold (cosine to the original key); overrides --para_keep_frac when > 0")
    ap.add_argument("--paraphrases", default="", help="paraphrase json (default SHARED/paraphrases.json)")
    ap.add_argument("--reuse_instr", default="",
                    help="key-set dir whose cached instruction embeddings to copy, so keys.npy is identical")
    ap.add_argument("--batch", type=int, default=256, help="inputs per API request")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(max_retries=10)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    paras = json.load(open(args.paraphrases or paths.SHARED / "paraphrases.json"))["paraphrases"]
    if args.reuse_instr:
        import shutil
        for f in ("raw_instr.npy", "raw_instr.json"):
            if not (out / f).exists():
                shutil.copy(Path(args.reuse_instr) / f, out / f)

    instr = [s["instruction"] for s in sessions]
    uniq = sorted(set(instr))
    U, tok_i = embed_unique(client, args.model, uniq, args.batch, args.workers, str(out / "raw_instr"))
    pos = {t: i for i, t in enumerate(uniq)}
    keys = U[[pos[t] for t in instr]]
    np.save(out / "keys.npy", keys.astype(np.float32))
    print("keys", keys.shape, flush=True)

    flat, owner = [], []
    for si, s in enumerate(sessions):
        for p in paras.get(s["instruction"], []):
            flat.append(p)
            owner.append(si)
    uflat = sorted(set(flat))
    P, tok_p = embed_unique(client, args.model, uflat, args.batch, args.workers, str(out / "raw_para"))
    fpos = {t: i for i, t in enumerate(uflat)}
    pvec = P[[fpos[t] for t in flat]]
    owner = np.asarray(owner)
    sims = (pvec * keys[owner]).sum(1)
    theta = args.para_theta if args.para_theta > 0 else float(np.quantile(sims, 1 - args.para_keep_frac))
    keep = sims > theta
    pvec, owner = pvec[keep], owner[keep]
    count = np.bincount(owner, minlength=len(sessions)).astype(np.int32)
    offset = np.zeros(len(sessions), np.int64)
    offset[1:] = np.cumsum(count)[:-1]
    np.save(out / "para_keys.npy", pvec.astype(np.float16))
    np.savez(out / "para_index.npz", offset=offset, count=count)

    link = out / "text_vecs.npy"                                        # input features: Qwen3-0.6B, unchanged
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(paths.SHARED / "text_vecs.npy")

    tokens = tok_i + tok_p
    meta = {"model": args.model, "paraphrases": str(args.paraphrases or paths.SHARED / "paraphrases.json"), "dim": int(keys.shape[1]), "label_source": "openai", "input_text_vecs": "Qwen3-0.6B",
            "theta_para": theta, "para_total": int(len(sims)), "para_kept": int(keep.sum()),
            "para_sim_quantiles": {q: float(np.quantile(sims, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "sessions_with_paraphrase": int((count > 0).sum()), "api_tokens": int(tokens),
            "approx_cost_usd": round(tokens / 1e6 * PRICE_PER_MTOK.get(args.model, 0.13), 2)}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    # how well does each key space alone rank the trace-judged same-workflow pairs? (text-key oracle)
    import torch
    from .evaluate import _judged_retrieval, load_judged
    gmap = {g: i for i, g in enumerate(sorted({s["group"] for s in sessions}))}
    group = np.array([gmap[s["group"]] for s in sessions])
    split = np.array([s["split"] for s in sessions])
    apps = np.array([s["app"] for s in sessions])
    for name, K in (("qwen3-0.6B", np.load(paths.SHARED / "keys.npy")), (args.model, keys)):
        for sp in ("val", "test"):
            sidx = np.nonzero(split == sp)[0]
            rel = load_judged(sp, sidx, group[sidx])
            if rel is not None:
                m = _judged_retrieval(torch.from_numpy(K[sidx].astype(np.float32)), rel, apps[sidx], within_app=False)
                print(f"text-key oracle  {name:24s} {sp}:", {k: round(v, 4) for k, v in m.items()}, flush=True)


if __name__ == "__main__":
    main()
