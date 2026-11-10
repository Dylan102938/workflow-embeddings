"""Re-evaluate saved checkpoints (e.g. after new judged relevance labels arrive):

  python -m wfemb.eval_ckpt RUN [RUN ...] [--splits val test] [--finalize]

Writes runs/<RUN>/eval_<split>.json with the full evaluate() output (including the nuisance probe).
--finalize also writes final.json and the final log line exactly as train.py does at the end of a run (for runs that
were stopped after their best checkpoint, e.g. a stalled job): the evaluation is the same, only the tail of the val
curve after the stop is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import paths
from .data import SessionData
from .evaluate import evaluate
from .model import ModelConfig, build_model


def finalize(run, state, a, res):
    """final.json + final log record, in train.py's format."""
    import time
    rows = [json.loads(l) for l in open(paths.RUNS / run / "log.jsonl")]
    val = [r for r in rows if "val_loss" in r and not r.get("final")]
    best = {"loss": min(r["val_loss"] for r in val)}
    for k in ("key_R@5", "jsession_R@5_cross_app", "jsession_nDCG@10_cross_app"):
        xs = [r[f"val_{k}"] for r in val if f"val_{k}" in r and r[f"val_{k}"] == r[f"val_{k}"]]
        if xs:
            best[k] = max(xs)
    last = max(r["step"] for r in rows if "step" in r)
    final = {"best_step": state["step"], "best": best, "select_on": a.get("select_on", "loss"),
             "val": res["val"], "test": res["test"],
             "note": f"finalized by eval_ckpt: training stopped at step {last} (stalled job), after the best checkpoint"}
    with open(paths.RUNS / run / "final.json", "w") as f:
        json.dump(final, f, indent=1)
    rec = {"time": time.time(), "final": True, "finalized_by": "eval_ckpt", "stopped_at_step": last,
           **{f"{sp}_{k}": v for sp in ("val", "test") for k, v in res[sp].items() if not isinstance(v, list)}}
    with open(paths.RUNS / run / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(run, "finalized (best step", state["step"], "stopped at", last, ")", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--finalize", action="store_true", help="write final.json + final log line (needs val and test)")
    ap.add_argument("--frames_mode", default="pread", help="how to read the frame cache (default pread)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (evaluation needs no GPU; slower on CPU)")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True           # as in train.py, so scores match the in-training evals
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cpu":
        import os
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    data, data_key = None, None
    for run in args.runs:
        state = torch.load(paths.RUNS / run / args.ckpt, map_location=device, weights_only=False)
        a = state["args"]
        paths.KEYS = Path(a["keys_dir"]) if a.get("keys_dir") else paths.SHARED
        if not paths.KEYS.exists() and (paths.DATA / paths.KEYS.name).exists():   # e.g. evaluated off-cluster
            paths.KEYS = paths.DATA / paths.KEYS.name
        extras = [x for x in a.get("extra", "").split(",") if x] + (["effect"] if a.get("use_effect") else [])
        key = (bool(a.get("use_crop")), tuple(extras), a.get("frames_dir"), a.get("avg_grid", 0), a.get("keys_dir"))
        if key != data_key:                      # reload frames only when the token layout changes
            data, data_key = None, None
            torch.cuda.empty_cache()
            fdir = Path(a.get("frames_dir", paths.FRAMES))
            if not fdir.exists() and (paths.BULK / fdir.name).exists():  # a job-local copy that is gone
                fdir = paths.BULK / fdir.name
            mode = "gpu_fp8" if a.get("avg_grid", 0) else args.frames_mode
            data = SessionData(fdir, use_crop=key[0], frames_mode=mode, device=device, extras=list(extras),
                               avg_grid=a.get("avg_grid", 0))
            data_key = key
        if a.get("center_keys"):
            raise NotImplementedError("center_keys runs: re-derive the key transform before evaluating")
        train_idx = data.indices("train")
        U = torch.from_numpy(data.keys[train_idx]).to(device)
        model = build_model(ModelConfig(**state["cfg"]), torch.from_numpy(data.text_vecs)).to(device)
        model.load_state_dict(state["model"])
        res = {}
        for split in args.splits:
            m = evaluate(model, data, split, device, a.get("eval_theta_fn", 0.9), U, train_idx, full=True)
            res[split] = m
            with open(paths.RUNS / run / f"eval_{split}.json", "w") as f:
                json.dump(m, f, indent=1)
            brief = {k: round(v, 4) for k, v in m.items()
                     if isinstance(v, float) and (k in ("loss", "key_R@5", "key_MRR") or k.startswith("jsession"))}
            print(run, split, json.dumps(brief), flush=True)
        if args.finalize:
            finalize(run, state, a, res)
        del model
    np.random.seed(0)


if __name__ == "__main__":
    main()
