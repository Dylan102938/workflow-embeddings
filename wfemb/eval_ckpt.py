"""Re-evaluate saved checkpoints (e.g. after new judged relevance labels arrive):

  python -m wfemb.eval_ckpt RUN [RUN ...] [--splits val test]

Writes runs/<RUN>/eval_<split>.json with the full evaluate() output (including the nuisance probe).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    device = torch.device("cuda")
    data, data_key = None, None
    for run in args.runs:
        state = torch.load(paths.RUNS / run / args.ckpt, map_location=device, weights_only=False)
        a = state["args"]
        paths.KEYS = Path(a["keys_dir"]) if a.get("keys_dir") else paths.SHARED
        extras = [x for x in a.get("extra", "").split(",") if x] + (["effect"] if a.get("use_effect") else [])
        key = (bool(a.get("use_crop")), tuple(extras), a.get("frames_dir"), a.get("avg_grid", 0), a.get("keys_dir"))
        if key != data_key:                      # reload frames only when the token layout changes
            data, data_key = None, None
            torch.cuda.empty_cache()
            data = SessionData(Path(a.get("frames_dir", paths.FRAMES)), use_crop=key[0], frames_mode=a.get("frames_mode", "gpu_fp8"),
                               device=device, extras=list(extras), avg_grid=a.get("avg_grid", 0))
            data_key = key
        if a.get("center_keys"):
            raise NotImplementedError("center_keys runs: re-derive the key transform before evaluating")
        train_idx = data.indices("train")
        U = torch.from_numpy(data.keys[train_idx]).to(device)
        model = build_model(ModelConfig(**state["cfg"]), torch.from_numpy(data.text_vecs)).to(device)
        model.load_state_dict(state["model"])
        for split in args.splits:
            m = evaluate(model, data, split, device, a.get("eval_theta_fn", 0.9), U, train_idx, full=True)
            with open(paths.RUNS / run / f"eval_{split}.json", "w") as f:
                json.dump(m, f, indent=1)
            brief = {k: round(v, 4) for k, v in m.items()
                     if isinstance(v, float) and (k in ("loss", "key_R@5", "key_MRR") or k.startswith("jsession"))}
            print(run, split, json.dumps(brief), flush=True)
        del model
    np.random.seed(0)


if __name__ == "__main__":
    main()
