"""Checks on the teacher keys (pseudocode 1c / 1d):
  - literal check: top-10 neighbours of random train instructions (are they driven by literals?)
  - neighbour pairs by similarity band, to set THETA_FN
  - how many candidates the false-negative mask would remove at several thresholds
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

from . import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keys = torch.from_numpy(np.load(paths.KEYS / "keys.npy")).to(dev)
    tr = [i for i, s in enumerate(sessions) if s["split"] == "train"]
    U = keys[tr]
    instr = [sessions[i]["instruction"] for i in tr]
    group = np.array([sessions[i]["group"] for i in tr])
    rng = np.random.default_rng(args.seed)

    sims = U @ U.T
    sims.fill_diagonal_(-1)
    print("== cosine stats between distinct train keys")
    sample = sims[torch.randint(0, len(tr), (2000,))].flatten()
    sample = sample[sample > -1]
    for q in (0.5, 0.9, 0.99, 0.999):
        print(f"  q{q}: {torch.quantile(sample[:1_000_000].float(), q).item():.3f}")

    print("\n== literal check: top-10 neighbours")
    for qi in rng.choice(len(tr), args.n_queries, replace=False):
        v, ix = sims[qi].topk(10)
        print(f"\n# {instr[qi]}")
        for s, j in zip(v.tolist(), ix.tolist()):
            print(f"   {s:.3f}  {instr[j]}")

    print("\n== pairs by band (for THETA_FN)")
    for lo, hi in ((0.95, 1.01), (0.9, 0.95), (0.85, 0.9), (0.8, 0.85), (0.75, 0.8)):
        m = (sims >= lo) & (sims < hi)
        idx = m.nonzero()
        idx = idx[idx[:, 0] < idx[:, 1]]
        print(f"\n-- [{lo}, {hi}): {len(idx)} pairs")
        for k in rng.choice(len(idx), min(8, len(idx)), replace=False):
            a, b = idx[k].tolist()
            same = "SAMEGROUP " if group[a] == group[b] else ""
            print(f"   {sims[a, b].item():.3f} {same}| {instr[a]}  ||  {instr[b]}")

    print("\n== fn-mask size per session (train candidates removed)")
    for th in (0.75, 0.8, 0.85, 0.9, 0.95):
        c = (sims > th).sum(1).float()
        print(f"  theta={th}: mean {c.mean():.2f}  p90 {torch.quantile(c, 0.9):.0f}  max {c.max():.0f}"
              f"  frac sessions with >=1: {(c > 0).float().mean():.3f}")


if __name__ == "__main__":
    main()
