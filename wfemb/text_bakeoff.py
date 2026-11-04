"""Which text embedder gives the best target space for workflows? (no training)

Embeds every session instruction with each candidate and scores how well instruction similarity ALONE ranks
the trace-judged session pairs (val + test): the "text-key oracle" for that key space. A key space that already
puts same-workflow sessions together is a better target for the recurrent model.

  python -m wfemb.text_bakeoff --models Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-8B ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import torch

from . import paths
from .evaluate import _judged_retrieval, load_judged

TASK_PROMPT = ("Instruct: Given the description of a task performed on a computer, represent the workflow it "
               "describes\nQuery: ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--prompt", choices=["none", "task", "both"], default="both",
                    help="task: the same instruction-style prompt on every text (symmetric similarity)")
    ap.add_argument("--out", default=str(paths.SHARED / "text_bakeoff.json"))
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    instr = [s["instruction"] for s in sessions]
    uniq = sorted(set(instr))
    pos = {t: i for i, t in enumerate(uniq)}
    gmap = {g: i for i, g in enumerate(sorted({s["group"] for s in sessions}))}
    group = np.array([gmap[s["group"]] for s in sessions])
    split = np.array([s["split"] for s in sessions])
    apps = np.array([s["app"] for s in sessions])
    results = json.load(open(args.out)) if __import__("os").path.exists(args.out) else {}
    variants = ["none", "task"] if args.prompt == "both" else [args.prompt]

    for name in args.models:
        t0 = time.time()
        try:
            model = SentenceTransformer(name, device="cuda", trust_remote_code=True,
                                        model_kwargs={"torch_dtype": torch.bfloat16})
        except Exception as e:                                           # record and move on
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(name, "LOAD ERROR", results[name]["error"], flush=True)
            continue
        model.max_seq_length = args.max_len
        load_s = time.time() - t0
        for v in variants:
            texts = [TASK_PROMPT + t for t in uniq] if v == "task" else uniq
            order = np.argsort([len(t) for t in texts])[::-1]
            t1 = time.time()
            vec = model.encode([texts[i] for i in order], batch_size=args.batch, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            U = np.empty_like(vec)
            U[order] = vec
            E = torch.from_numpy(U[[pos[t] for t in instr]].astype(np.float32))
            rec = {"dim": int(U.shape[1]), "load_s": round(load_s, 1), "encode_s": round(time.time() - t1, 1)}
            rng = np.random.default_rng(0)
            a, b = rng.integers(0, len(U), 20000), rng.integers(0, len(U), 20000)
            rec["median_cos_random"] = float(np.median((U[a] * U[b]).sum(1)))
            for sp in ("val", "test"):
                sidx = np.nonzero(split == sp)[0]
                rel = load_judged(sp, sidx, group[sidx])
                if rel is None:
                    continue
                for within in (False, True):
                    m = _judged_retrieval(E[sidx], rel, apps[sidx], within_app=within)
                    rec.update({f"{sp}/{k}": round(v_, 4) for k, v_ in m.items()})
            results[f"{name}|prompt={v}"] = rec
            print(name, v, json.dumps({k: rec[k] for k in rec if "nDCG" in k or "R@5_cross" in k or k == "dim"}),
                  flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
        del model
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
