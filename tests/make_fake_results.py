"""Write a fake results tree (docs/EXPERIMENTS.md layout) and a jobs/ queue folder, for testing the dashboard.

    python tests/make_fake_results.py OUT_DIR [--steps 14000] [--seed 0]
    python dashboard/server.py --results OUT_DIR/results --jobs OUT_DIR/jobs

13 runs: the default config with seeds 0-2, arch transf (2 seeds) and gru, d_model 384 (2 seeds), dropout 0.1,
qkv sep-q, beta 2 still running (no final.json, files touched now), a run that died (stale: no final.json, files two
days old), and one imported legacy run (meta.legacy, steps.jsonl every 50 steps, no rigid_loss / avg_dist /
*_avg_log_dist_100NN). Curves: training loss ~9 -> ~3 with noise, val_loss with its minimum around step 6000.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

VARS = [  # key, default (docs/EXPERIMENTS.md order)
    ("arch", "late-history-transf"), ("dm", 512), ("dh", 64), ("nl", 4), ("nh", 8), ("drop", 0.3), ("smooth", 0.2),
    ("patch", 576), ("lr", 5e-4), ("bs", 64), ("ep", 45), ("beta", 0), ("dupmask", 0.8), ("qkv", "shared"), ("seed", 0),
]


def canon(v) -> str:
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        if abs(v) < 0.01:
            m, e = f"{v:e}".split("e")
            return f"{m.rstrip('0').rstrip('.')}e{int(e)}"
        return f"{v:g}"
    return str(v)


def run_id(exp: dict) -> str:
    return "_".join(f"{k}-{canon(exp[k])}" for k, _ in VARS)


# Runs: overrides of the defaults, plus how good the config is (lower = better final loss) and what state it is in.
RUNS = [
    ({}, 0.00, "done"), ({"seed": 1}, 0.00, "done"), ({"seed": 2}, 0.00, "done"),
    ({"arch": "transf"}, 0.12, "done"), ({"arch": "transf", "seed": 1}, 0.12, "done"),
    ({"arch": "gru"}, 0.35, "done"),
    ({"dm": 384}, 0.08, "done"), ({"dm": 384, "seed": 1}, 0.08, "done"),
    ({"drop": 0.1}, 0.20, "done"),
    ({"qkv": "sep-q"}, 0.04, "done"),
    ({"beta": 2}, -0.03, "running"),
    ({"lr": 1.5e-3}, 0.25, "stale"),
    ({"dm": 384, "drop": 0.1}, 0.15, "legacy"),
]


def curves(rng: random.Random, total: int, offset: float, beta: float, lr_peak: float):
    """Per-step rows and per-eval rows over `total` steps."""
    warm = 500
    steps, evals = [], []
    t = 0.0
    spe = math.ceil(total / 45)
    for s in range(1, total + 1):
        lr = lr_peak * s / warm if s <= warm else lr_peak * 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm)))
        base = 3.0 + offset + 6.0 * math.exp(-s / 2200) + 0.4 * math.exp(-s / 9000)
        loss = base + rng.gauss(0, 0.32) * (0.6 + 0.4 * math.exp(-s / 4000))
        if rng.random() < 0.002:
            loss += rng.uniform(0.5, 2.0)                          # occasional spikes
        rigid = loss * 0.93 + rng.gauss(0, 0.05)
        avg_dist = 0.98 - 0.25 * math.exp(-s / 3000) + rng.gauss(0, 0.012)
        step_s = abs(rng.gauss(0.21, 0.03))
        data_s = abs(rng.gauss(0.02, 0.01))
        t += step_s + data_s
        b = beta * min(1.0, s / (0.3 * total))
        steps.append({"step": s, "epoch": (s - 1) // spe, "time": round(t, 2), "loss": loss, "rigid_loss": rigid,
                      "avg_dist": avg_dist, "lr": lr, "grad_norm": abs(rng.gauss(1.2 * math.exp(-s / 5000) + 0.35, 0.12)),
                      "beta": b, "data_s": data_s, "step_s": step_s})
        if s % 250 == 0 or s == total:
            over = 0.02 * max(0.0, (s - 5200) / 1000) ** 2
            vl = 4.1 + offset + 4.6 * math.exp(-s / 1500) + over + rng.gauss(0, 0.015)
            q = max(0.0, min(1.0, 1.0 - (vl - 3.6) / 6.0))       # 1 = good
            evals.append({
                "step": s, "epoch": (s - 1) // spe, "time": round(t + 40, 2), "val_loss": vl,
                "val_r@1": 0.62 * q ** 2 + rng.gauss(0, 0.006), "val_r@5": 0.86 * q ** 1.3 + rng.gauss(0, 0.006),
                "val_avg_log_dist_100NN": -0.55 - 0.5 * q + rng.gauss(0, 0.004),
                "val_judged_r@1": 0.41 * q ** 1.5 + rng.gauss(0, 0.006),
                "val_judged_r@5": 0.70 * q ** 1.2 + rng.gauss(0, 0.006),
                "val_judged_ndcg@10": 0.55 * q + rng.gauss(0, 0.006),
                "max_mem_gb": round(31.5 + rng.random(), 2),
            })
            t += 40
    return steps, evals


def r6(row: dict) -> dict:
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}


def write_jsonl(p: Path, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r6(r)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=14000, help="total steps of a full run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    root = out / "results"
    if root.exists():
        shutil.rmtree(root)
    rng = random.Random(a.seed)
    ids = []
    now = time.time()
    for over, offset, kind in RUNS:
        exp = {k: v for k, v in VARS} | over
        rid = run_id(exp)
        ids.append(rid)
        d = root.joinpath(*rid.split("_"))
        d.mkdir(parents=True)
        total = a.steps
        spe = math.ceil(total / exp["ep"])
        steps, evals = curves(random.Random(rng.random()), total, offset + 0.01 * exp["seed"], exp["beta"], exp["lr"])
        legacy = kind == "legacy"
        if kind == "running":
            cut = int(total * 0.64)
            steps = steps[:cut]
            evals = [e for e in evals if e["step"] <= cut]
        elif kind == "stale":
            steps = steps[:4120]
            evals = [e for e in evals if e["step"] <= 4120]
        if legacy:
            steps = [{k: v for k, v in r.items() if k not in ("rigid_loss", "avg_dist")} for r in steps if r["step"] % 50 == 0]
            evals = [{k: v for k, v in r.items() if "avg_log_dist" not in k} for r in evals]
        meta = {"git_commit": "abd7a6a", "git_dirty": False, "host": "rnn-a100-3" if exp["seed"] != 2 else "baseten",
                "launcher": "jobs/fake.txt",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now - 4 * 3600)),
                "n_params": 21_345_792 if exp["dm"] == 512 else 12_402_304, "steps_per_epoch": spe, "total_steps": total}
        if legacy:
            meta |= {"legacy": True, "legacy_name": "r20_xf4_d384_p01"}
        cfg = {
            "id": rid, "experiment": exp,
            "train": {"warmup_steps": 500, "weight_decay": 0.05, "ema": 0.999, "max_frames": 4096 if exp["patch"] == 576 else 8192,
                      "eval_steps": 250, "grad_clip": 1.0, "amp": "bf16", "checkpointing": True},
            "eval": {"dupmask": 0.8, "judged": "keys/judged_val.jsonl", "batch_sessions": 128},
            "data": {"root": "/nas/ucb/dfeng/data/workflow-embeddings", "keys": "keys_openai", "frames": "",
                     "prefetch_threads": 4, "prefetch_depth": 4},
            "model": {"d_model": exp["dm"], "n_layers": exp["nl"], "n_heads": exp["nh"], "d_head": exp["dh"],
                      "qkv": exp["qkv"], "dropout": exp["drop"], "d_key": 3072},
            "meta": meta,
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        write_jsonl(d / "steps.jsonl", steps)
        write_jsonl(d / "evals.jsonl", evals)
        if kind in ("done", "legacy"):
            best = min(evals, key=lambda e: e["val_loss"])
            final = {"best_step": best["step"], "selected_on": "val_loss"}
            for k, v in best.items():
                if k.startswith("val_"):
                    final[k] = v
            for k, v in best.items():
                if k.startswith("val_"):
                    final["test_" + k[4:]] = v + rng.gauss(0.02 if k.endswith("loss") else -0.01, 0.01)
            final["train_minutes"] = round(steps[-1]["time"] / 60, 1)
            (d / "final.json").write_text(json.dumps(r6(final), indent=2))
        if kind == "stale":
            old = now - 2 * 86400
            for p in d.iterdir():
                os.utime(p, (old, old))

    # a queue: done, running, stale and never-started ids, overrides, comments, one malformed line
    jobs = out / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    default = {k: v for k, v in VARS}
    todo = [run_id(default | {"arch": "gru", "seed": 1}), run_id(default | {"nl": 6}), run_id(default | {"dupmask": "off"}),
            run_id(default | {"qkv": "sep-qkv"})]
    (jobs / "r26.txt").write_text(
        "# Round 26 (fake): architecture and per-type projections.\n"
        "# Same recipe as the best model; 24x24 SigLIP2 frames.\n\n"
        + "\n".join(ids[:6]) + "\n"
        + ids[9] + "   # per-type W_Q\n"
        + todo[3] + " train.max_frames=2048\n"
        + todo[0] + "\n"
        + "\n# hard negatives\n"
        + ids[10] + " train.max_frames=2048 train.eval_steps=500\n"
    )
    (jobs / "r27.txt").write_text(
        "# Round 27 (fake): width, depth, masking ablations.\n"
        + "\n".join([ids[6], ids[7], ids[8], ids[11], todo[1], todo[2]]) + "\n"
        + "arch-gru_dm-512_foo-1_seed-0   # malformed: unknown variable\n"
    )
    print(f"wrote {len(ids)} runs to {root} and 2 queue files to {jobs}")
    for rid in ids:
        print("  " + rid)


if __name__ == "__main__":
    main()
