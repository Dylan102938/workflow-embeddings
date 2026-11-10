"""Import runs made before the experiment-id refactor (runs/<name>/{config.json, log.jsonl, final.json}) into the
results tree, for the runs the experiment variables can express (transformer archs on the fixed recipe); the rest
(pool model, GQA, Qwen2.5-VL frames, other recipes) are listed as skipped. Imported runs are marked meta.legacy: their
steps.jsonl rows are 50-step means of the training loss (no rigid_loss / avg_dist), and they lack the kNN metric.

  python runners/slurm/import_legacy.py fetch      # rnn:runs/* and every Baseten job's runs/* (json/jsonl) -> legacy_runs/
  python runners/slurm/import_legacy.py convert    # legacy_runs/* -> results/
  python runners/slurm/import_legacy.py ckpt OLD_RUN_DIR/best.pt   # an old checkpoint -> results/<run folder>/best.pt
"""
from __future__ import annotations

import argparse
import re
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import chz

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from wfemb.config import DataConfig, EvalConfig, Experiment, TrainConfig, default_results  # noqa: E402
from wfemb.storage.fetch import baseten_jobs  # noqa: E402
from wfemb.storage.results import clean, dumps  # noqa: E402

LEGACY = REPO / "legacy_runs"
FILES = ("config.json", "log.jsonl", "final.json")
PATCH = {"frames_dense": 256, "frames_dense_g24": 576}
# the fixed recipe (TrainConfig / EvalConfig / DataConfig defaults) in the old argument names
RECIPE = {"para_negatives": True, "tok_drop": 0.4, "wd": 0.1, "step_drop": 0.4, "k_max": 50.0, "k_min": 1.0,
          "ema": 0.999, "eval_theta_fn": 0.8, "warmup": 500, "mlp_ratio": 4, "smooth_tau": 0.05,
          "no_ev_pos": False, "step_drop_anneal": 0.0, "center_keys": False, "no_para": False, "max_len": 0}


def fetch():
    LEGACY.mkdir(exist_ok=True)
    inc = ["--include=*/"] + [f"--include={f}" for f in FILES]
    subprocess.run(["rsync", "-a", "--prune-empty-dirs", *inc, "--exclude=*",
                    "rnn:/nas/ucb/dfeng/data/workflow-embeddings/runs/", f"{LEGACY}/"], check=True)
    for j in baseten_jobs():
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["truss", "train", "get_checkpoint_urls", "--job-id", j["id"]], cwd=tmp,
                           capture_output=True, text=True)
            files = list(Path(tmp).glob("*.json"))
            arts = json.loads(files[0].read_text()).get("checkpoint_artifacts", []) if files else []
        n = 0
        for a in arts:
            parts = a["relative_file_name"].split("/")
            if len(parts) != 3 or parts[0] != "runs" or parts[2] not in FILES:
                continue
            dst = LEGACY / parts[1] / parts[2]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(a["url"], timeout=300) as r:     # Baseten copies are newer (finalized runs)
                dst.write_bytes(r.read())
            n += 1
        print(f"baseten {j['id']} {j.get('name') or ''}: {n} files")


def experiment(a: dict) -> tuple[Experiment | None, str]:
    arch, nl, hj = a.get("arch", "pool_gru"), int(a.get("n_layers", 4)), int(a.get("h_join", 0))
    if arch == "xf" and hj == 0:
        new_arch = "transf"
    elif arch == "xf" and hj % nl == nl - 1:
        new_arch = "late-history-transf"
    elif arch == "xf_gru" and hj == 0:
        new_arch, nl = "gru", nl + 1                                  # old n_layers counted the transformer blocks
    else:
        return None, f"arch {arch} h_join {hj}"
    if a.get("n_kv_heads", 0) not in (0, a.get("n_heads")):
        return None, "grouped-query attention"
    frames = Path(a.get("frames_dir", "")).name
    if frames not in PATCH:
        return None, f"frames {frames}"
    if not str(a.get("keys_dir", "")).rstrip("/").endswith("keys_openai"):
        return None, f"keys {a.get('keys_dir')}"
    off = {k: a.get(k) for k, v in RECIPE.items() if k in a and a[k] != v}
    if off:
        return None, f"recipe {off}"
    if float(a.get("theta_fn", 0.9)) != 0.8:
        return None, f"near-duplicate mask {a.get('theta_fn')} (fixed at 0.8)"
    if a.get("qkv_mode", "shared") != "shared":
        return None, f"qkv_mode {a.get('qkv_mode')} (removed)"
    try:
        return Experiment(arch=new_arch, dm=int(a["d"]), dh=int(a.get("d_head", 64)), nl=nl, nh=int(a["n_heads"]),
                          drop=float(a["dropout"]), smooth=float(a.get("label_smooth", 0.0)), patch=PATCH[frames],
                          lr=float(a["lr"]), bs=int(a.get("batch", 64)), ep=int(a["epochs"]),
                          beta=float(a.get("hard_neg", 0.0)), seed=int(a.get("seed", 0))), ""
    except Exception as e:
        return None, str(e)


EVAL_KEYS = {"loss": "loss", "key_R@1": "r@1", "key_R@5": "r@5", "jsession_R@1_cross_app": "judged_r@1",
             "jsession_R@5_cross_app": "judged_r@5", "jsession_nDCG@10_cross_app": "judged_ndcg@10"}


def convert_run(src: Path, root: Path) -> str:
    c = json.loads((src / "config.json").read_text())
    exp, why = experiment(c["args"])
    if exp is None:
        return f"skip   {src.name}: {why}"
    dst = root / exp.path
    if (dst / "config.json").exists() and not json.loads((dst / "config.json").read_text())["meta"].get("legacy"):
        return f"skip   {src.name}: {exp.id} already has a non-legacy run"
    rows = [json.loads(l) for l in (src / "log.jsonl").read_text().splitlines() if l.strip()]
    t0 = rows[0]["time"] if rows else 0
    steps = [{"step": r["step"], "epoch": r.get("epoch"), "time": round(r["time"] - t0, 3), "loss": r["train_loss"],
              "lr": r.get("lr"), "grad_norm": r.get("grad_norm")} for r in rows if "train_loss" in r]
    evals = [{"step": r["step"], "epoch": r.get("epoch"), "time": round(r["time"] - t0, 3),
              **{f"val_{n}": r.get(f"val_{o}") for o, n in EVAL_KEYS.items()}, "max_mem_gb": r.get("max_mem_gb")}
             for r in rows if "val_loss" in r and not r.get("final")]
    a = c["args"]
    train = {**chz.asdict(TrainConfig()), "max_frames": a.get("max_frames", 4096),
             "activation_checkpointing": not a.get("no_xf_ckpt", False)}
    cfg = {"id": exp.id, "experiment": {k: getattr(exp, k) for k in Experiment.keys()}, "train": train,
           "eval": {**chz.asdict(EvalConfig()), "eval_steps": a.get("eval_every", 500)},
           "data": {**chz.asdict(DataConfig()), "keys": "keys_openai"}, "model": c.get("model"),
           "meta": {"legacy": True, "legacy_name": src.name, "train_log_every": a.get("log_every", 50),
                    "launcher": f"slurm:{c['slurm_job']}" if c.get("slurm_job") else "legacy",
                    "n_params": c.get("n_params"), "steps_per_epoch": c.get("steps_per_epoch"),
                    "total_steps": c.get("total_steps")}}
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("steps.jsonl", "evals.jsonl", "final.json"):
        (dst / f).unlink(missing_ok=True)
    (dst / "config.json").write_text(json.dumps(clean(cfg), indent=1))
    (dst / "steps.jsonl").write_text("".join(dumps(r) + "\n" for r in steps))
    (dst / "evals.jsonl").write_text("".join(dumps(r) + "\n" for r in evals))
    if (src / "final.json").exists():
        f = json.loads((src / "final.json").read_text())
        out = {"best_step": f["best_step"], "selected_on": "val_" + f.get("select_on", "loss")}
        for split in ("val", "test"):
            out.update({f"{split}_{n}": f.get(split, {}).get(o) for o, n in EVAL_KEYS.items()})
        (dst / "final.json").write_text(json.dumps(clean(out), indent=1))
    return f"import {src.name} -> {exp.id}{'' if (src / 'final.json').exists() else '  (no final.json)'}"


def convert_ckpt(path: Path, root: Path) -> str:
    """An old checkpoint ({"model", "cfg", "args", "step"}) -> the new format ({"model", "model_cfg", "id", ...}).
    Q/K/V projections are plain Linear layers now (were 1-element ModuleLists); the GRU final block moved into
    model/gru.py."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    exp, why = experiment(state["args"])
    if exp is None:
        return f"skip   {path}: {why}"
    c = state["cfg"]
    model_cfg = {"arch": exp.arch, "d": exp.dm, "d_head": exp.dh, "n_layers": exp.nl, "n_heads": exp.nh,
                 "dropout": exp.drop, "d_key": c["d_key"], "d_vis": c["d_vis"], "n_tok": c["n_tok"],
                 "n_key_tokens": c["n_keys"], "d_text": c["d_text"], "e_max": c["e_max"], "tok_drop": c["tok_drop"],
                 "mlp_ratio": c["mlp_ratio"], "k_min": c["k_min"], "k_max": c["k_max"], "k_init": c["k_init"],
                 "checkpointing": c.get("xf_ckpt", True)}
    ren = {"gru_in.": "gru.norm_in.", "h_norm.": "gru.norm_out."} if exp.arch == "gru" else {}
    sd = {}
    for k, v in state["model"].items():
        k = re.sub(r"\.(wq|wk|wv)\.0\.", r".\1.", k)                 # Q/K/V were 1-element ModuleLists
        if exp.arch == "gru" and k.startswith("gru.") and not k.startswith("gru.norm"):
            k = "gru.cell." + k[len("gru."):]
        for a, b in ren.items():
            k = b + k[len(a):] if k.startswith(a) else k
        sd[k] = v
    dst = root / exp.path / "best.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "model_cfg": model_cfg, "id": exp.id,
                "experiment": {k: getattr(exp, k) for k in Experiment.keys()}, "step": state["step"]}, dst)
    return f"ckpt   {path} -> {dst}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "convert", "ckpt"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--root", default=default_results())
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch()
        return
    if a.cmd == "ckpt":
        for p in a.paths:
            print(convert_ckpt(Path(p), Path(a.root)))
        return
    for src in sorted(p for p in LEGACY.iterdir() if (p / "config.json").exists() and (p / "log.jsonl").exists()):
        print(convert_run(src, Path(a.root)))


if __name__ == "__main__":
    main()
