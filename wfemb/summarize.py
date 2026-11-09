"""Compact comparison table across runs:  python -m wfemb.summarize [run-name-prefix ...]"""
from __future__ import annotations

import json
import sys

from . import paths

COLS = [("loss", "val_loss", "{:.3f}"), ("loss_last", "val_loss_last", "{:.3f}"),
        ("keyR@1", "val_key_R@1", "{:.3f}"), ("keyR@5", "val_key_R@5", "{:.3f}"),
        ("keyMRR", "val_key_MRR", "{:.3f}"), ("jR@5", "val_jsession_R@5_cross_app", "{:.3f}"),
        ("jnDCG", "val_jsession_nDCG@10_cross_app", "{:.3f}"), ("κ0", "val_kappa_first", "{:.0f}"),
        ("κT", "val_kappa_last", "{:.0f}")]


def load(run):
    rows = [json.loads(l) for l in open(paths.RUNS / run / "log.jsonl")]
    val = [r for r in rows if "val_loss" in r and "final" not in r]
    tr = [r for r in rows if "train_loss" in r]
    final = None
    fp = paths.RUNS / run / "final.json"
    if fp.exists():
        final = json.load(open(fp))
    return val, tr, final


def main():
    prefixes = sys.argv[1:] or [""]
    runs = sorted(p.name for p in paths.RUNS.iterdir() if p.is_dir() and any(p.name.startswith(x) for x in prefixes)
                  and (p / "log.jsonl").exists())
    hdr = f"{'run':24s} {'best@step':>9s} {'train':>6s} " + " ".join(f"{c[0]:>8s}" for c in COLS) + \
          f" | {'test_loss':>9s} {'t_keyR@5':>8s} {'t_jR@5':>6s} {'t_jnDCG':>7s} | {'it/s':>5s} {'memGB':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for run in runs:
        val, tr, final = load(run)
        if not val:
            print(f"{run:24s} (no eval yet)")
            continue
        best = min(val, key=lambda r: r["val_loss"])
        near = [t for t in tr if t["step"] <= best["step"]][-5:]
        trl = sum(t["train_loss"] for t in near) / max(1, len(near))
        cells = " ".join(f"{(c[2].format(best[c[1]]) if c[1] in best else '-'):>8s}" for c in COLS)
        tail = ""
        if final:
            t = final["test"]
            tail = (f" | {t['loss']:9.3f} {t['key_R@5']:8.3f} {t.get('jsession_R@5_cross_app', float('nan')):6.3f}"
                    f" {t.get('jsession_nDCG@10_cross_app', float('nan')):7.3f}")
        else:
            tail = " | " + " " * 34
        sps = sorted(t["steps_per_s"] for t in tr if "steps_per_s" in t)
        mem = max((r.get("max_mem_gb", 0) for r in val), default=0)
        tail += f" | {sps[len(sps) // 2] if sps else 0:5.1f} {mem:5.1f}"
        done = "" if final else "*"
        print(f"{run + done:24s} {best['step']:9d} {trl:6.3f} {cells}{tail}")
    print("\n* = still running. Columns are the val metrics at the best-val-loss eval; train = mean train loss "
          "over the 5 logs before it.")


if __name__ == "__main__":
    main()
