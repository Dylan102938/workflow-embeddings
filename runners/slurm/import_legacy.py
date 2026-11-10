"""Imports runs made before the experiment-id refactor into the results tree.

Only runs the experiment variables can express are imported; the rest are listed as skipped. Imported runs carry
meta.legacy in config.json: their steps.jsonl has one row per 50 steps and only the training loss.

  python runners/slurm/import_legacy.py fetch                  # old run folders from rnn and Baseten -> legacy_runs/
  python runners/slurm/import_legacy.py convert                # legacy_runs/* -> results/
  python runners/slurm/import_legacy.py checkpoint <best.pt>   # an old checkpoint -> results/<run folder>/best.pt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import chz

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from wfemb.config import DataConfig, EvalConfig, Experiment, TrainConfig, default_results_root  # noqa: E402
from wfemb.storage.fetch import baseten_job_files, baseten_jobs, download  # noqa: E402
from wfemb.storage.results import to_json  # noqa: E402

LEGACY_DIR = REPO / "legacy_runs"
RUN_FILES = ("config.json", "log.jsonl", "final.json")
PATCHES_OF_FRAME_CACHE = {"frames_dense": 256, "frames_dense_g24": 576}

# Old arguments that must equal today's fixed recipe for a run to be comparable.
RECIPE_ARGUMENTS = {
    "para_negatives": True,
    "tok_drop": 0.4,
    "wd": 0.1,
    "step_drop": 0.4,
    "k_max": 50.0,
    "k_min": 1.0,
    "ema": 0.999,
    "theta_fn": 0.8,
    "eval_theta_fn": 0.8,
    "warmup": 500,
    "mlp_ratio": 4,
    "smooth_tau": 0.05,
    "qkv_mode": "shared",
    "no_ev_pos": False,
    "step_drop_anneal": 0.0,
    "center_keys": False,
    "no_para": False,
    "max_len": 0,
}

# Old metric name -> new metric name (both get the split as prefix).
METRIC_NAMES = {
    "loss": "loss",
    "key_R@1": "r@1",
    "key_R@5": "r@5",
    "jsession_R@1_cross_app": "judged_r@1",
    "jsession_R@5_cross_app": "judged_r@5",
    "jsession_nDCG@10_cross_app": "judged_ndcg@10",
}

# Old parameter name pattern -> new name, applied in order to checkpoint keys.
PARAMETER_RENAMES = [
    (r"^event_enc\.type_emb\.", "event_encoder.type_embedding."),
    (r"^event_enc\.xy_proj\.", "event_encoder.point_projection."),
    (r"^event_enc\.scroll_proj\.", "event_encoder.scroll_projection."),
    (r"^event_enc\.key_emb\.", "event_encoder.key_embedding."),
    (r"^event_enc\.text_proj\.", "event_encoder.text_projection."),
    (r"^event_enc\.", "event_encoder."),
    (r"^no_event$", "no_event_token"),
    (r"^ev_pos$", "event_positions"),
    (r"^scr_norm\.", "screen_norm."),
    (r"^scr_proj\.", "screen_projection."),
    (r"^scr_pos$", "screen_positions"),
    (r"^type_emb$", "token_type_embedding"),
    (r"^(blocks\.\d+)\.wq\.0\.", r"\1.query."),
    (r"^(blocks\.\d+)\.wk\.0\.", r"\1.key."),
    (r"^(blocks\.\d+)\.wv\.0\.", r"\1.value."),
    (r"^(blocks\.\d+)\.wo\.", r"\1.output."),
    (r"^(blocks\.\d+)\.ln1\.", r"\1.attention_norm."),
    (r"^(blocks\.\d+)\.ln2\.", r"\1.mlp_norm."),
    (r"^h0$", "initial_state"),
    (r"^dir_head\.", "direction_head."),
    (r"^conf_head\.", "confidence_head."),
]
GRU_PARAMETER_RENAMES = [
    (r"^gru_in\.", "gru.input_norm."),
    (r"^h_norm\.", "gru.output_norm."),
    (r"^gru\.(weight|bias)_", r"gru.cell.\1_"),
]
NON_GRU_PARAMETER_RENAMES = [(r"^h_norm\.", "state_norm.")]


def fetch():
    LEGACY_DIR.mkdir(exist_ok=True)
    includes = ["--include=*/"] + [f"--include={name}" for name in RUN_FILES]
    subprocess.run(
        [
            "rsync",
            "-a",
            "--prune-empty-dirs",
            *includes,
            "--exclude=*",
            "rnn:/nas/ucb/dfeng/data/workflow-embeddings/runs/",
            f"{LEGACY_DIR}/",
        ],
        check=True,
    )
    for job in baseten_jobs():  # Baseten copies are newer where both exist
        n_files = 0
        for file in baseten_job_files(job["id"]):
            parts = file["relative_file_name"].split("/")
            if len(parts) == 3 and parts[0] == "runs" and parts[2] in RUN_FILES:
                download(file["url"], LEGACY_DIR / parts[1] / parts[2])
                n_files += 1
        print(f"baseten {job['id']} {job.get('name') or ''}: {n_files} files")


def experiment_of(arguments: dict) -> Experiment:
    """The experiment an old run's arguments describe. Raises ValueError if the variables cannot express it."""
    architecture = arguments.get("arch", "pool_gru")
    n_layers = int(arguments.get("n_layers", 4))
    history_joins_at = int(arguments.get("h_join", 0)) % n_layers
    if architecture == "xf" and history_joins_at == 0:
        architecture = "transf"
    elif architecture == "xf" and history_joins_at == n_layers - 1:
        architecture = "late-history-transf"
    elif architecture == "xf_gru" and history_joins_at == 0:
        architecture, n_layers = "gru", n_layers + 1  # the GRU now counts as a layer
    else:
        raise ValueError(f"architecture {architecture} with h_join {arguments.get('h_join')}")
    if arguments.get("n_kv_heads", 0) not in (0, arguments.get("n_heads")):
        raise ValueError("grouped-query attention")
    frame_cache = Path(arguments.get("frames_dir", "")).name
    if frame_cache not in PATCHES_OF_FRAME_CACHE:
        raise ValueError(f"frame cache {frame_cache}")
    if Path(str(arguments.get("keys_dir", ""))).name != "keys_openai":
        raise ValueError(f"label set {arguments.get('keys_dir')}")
    off_recipe = {
        name: arguments[name]
        for name, value in RECIPE_ARGUMENTS.items()
        if name in arguments and arguments[name] != value
    }
    if off_recipe:
        raise ValueError(f"off-recipe settings {off_recipe}")
    return Experiment(
        architecture=architecture,
        d_model=int(arguments["d"]),
        d_head=int(arguments.get("d_head", 64)),
        n_layers=n_layers,
        n_heads=int(arguments["n_heads"]),
        dropout=float(arguments["dropout"]),
        label_smoothing=float(arguments.get("label_smooth", 0.0)),
        n_patches=PATCHES_OF_FRAME_CACHE[frame_cache],
        learning_rate=float(arguments["lr"]),
        batch_size=int(arguments.get("batch", 64)),
        n_epochs=int(arguments["epochs"]),
        hard_negative_beta=float(arguments.get("hard_neg", 0.0)),
        seed=int(arguments.get("seed", 0)),
    )


def convert_run(source: Path, results_root: Path) -> str:
    old_config = json.loads((source / "config.json").read_text())
    arguments = old_config["args"]
    try:
        experiment = experiment_of(arguments)
    except ValueError as reason:
        return f"skip   {source.name}: {reason}"
    run_dir = results_root / experiment.relative_dir
    existing = run_dir / "config.json"
    if existing.exists() and not json.loads(existing.read_text())["meta"].get("legacy"):
        return f"skip   {source.name}: {experiment.id} already has a run from the current code"

    log = [json.loads(line) for line in (source / "log.jsonl").read_text().splitlines() if line.strip()]
    started = log[0]["time"] if log else 0
    steps = [
        {
            "step": row["step"],
            "epoch": row.get("epoch"),
            "time": round(row["time"] - started, 3),
            "loss": row["train_loss"],
            "lr": row.get("lr"),
            "grad_norm": row.get("grad_norm"),
        }
        for row in log
        if "train_loss" in row
    ]
    evals = [
        {
            "step": row["step"],
            "epoch": row.get("epoch"),
            "time": round(row["time"] - started, 3),
            **{f"val_{new}": row.get(f"val_{old}") for old, new in METRIC_NAMES.items()},
            "max_mem_gb": row.get("max_mem_gb"),
        }
        for row in log
        if "val_loss" in row and not row.get("final")
    ]
    config = {
        "id": experiment.id,
        "experiment": chz.asdict(experiment),
        "train": {
            **chz.asdict(TrainConfig()),
            "max_frames_per_batch": arguments.get("max_frames", 4096),
            "activation_checkpointing": not arguments.get("no_xf_ckpt", False),
        },
        "eval": {**chz.asdict(EvalConfig()), "eval_steps": arguments.get("eval_every", 500)},
        "data": chz.asdict(DataConfig()),
        "meta": {
            "legacy": True,
            "legacy_name": source.name,
            "train_log_every": arguments.get("log_every", 50),
            "launcher": f"slurm:{old_config['slurm_job']}" if old_config.get("slurm_job") else "legacy",
            "n_params": old_config.get("n_params"),
            "steps_per_epoch": old_config.get("steps_per_epoch"),
            "total_steps": old_config.get("total_steps"),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(to_json(config, indent=1))
    (run_dir / "steps.jsonl").write_text("".join(to_json(row) + "\n" for row in steps))
    (run_dir / "evals.jsonl").write_text("".join(to_json(row) + "\n" for row in evals))
    old_final_path, final_path = source / "final.json", run_dir / "final.json"
    if old_final_path.exists():
        old_final = json.loads(old_final_path.read_text())
        final = {"best_step": old_final["best_step"], "selected_on": "val_loss"}
        for split in ("val", "test"):
            final.update({f"{split}_{new}": old_final.get(split, {}).get(old) for old, new in METRIC_NAMES.items()})
        final_path.write_text(to_json(final, indent=1))
    note = "" if final_path.exists() else "  (no final.json)"
    return f"import {source.name} -> {experiment.id}{note}"


def convert_checkpoint(path: Path, results_root: Path) -> str:
    import torch

    old = torch.load(path, map_location="cpu", weights_only=False)
    try:
        experiment = experiment_of(old["args"])
    except ValueError as reason:
        return f"skip   {path}: {reason}"
    old_model = old["cfg"]
    model_config = {
        "architecture": experiment.architecture,
        "d_model": experiment.d_model,
        "d_head": experiment.d_head,
        "n_layers": experiment.n_layers,
        "n_heads": experiment.n_heads,
        "dropout": experiment.dropout,
        "d_label": old_model["d_key"],
        "d_screen": old_model["d_vis"],
        "n_patches": old_model["n_tok"],
        "n_key_tokens": old_model["n_keys"],
        "d_text": old_model["d_text"],
        "max_events": old_model["e_max"],
        "screen_token_dropout": old_model["tok_drop"],
        "mlp_ratio": old_model["mlp_ratio"],
        "confidence_min": old_model["k_min"],
        "confidence_max": old_model["k_max"],
        "confidence_init": old_model["k_init"],
        "activation_checkpointing": old_model.get("xf_ckpt", True),
    }
    is_gru = experiment.architecture == "gru"
    renames = PARAMETER_RENAMES + (GRU_PARAMETER_RENAMES if is_gru else NON_GRU_PARAMETER_RENAMES)
    parameters = {}
    for name, tensor in old["model"].items():
        for pattern, replacement in renames:
            name, n_replaced = re.subn(pattern, replacement, name)
            if n_replaced:
                break
        parameters[name] = tensor
    destination = results_root / experiment.relative_dir / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": parameters,
            "model_config": model_config,
            "id": experiment.id,
            "experiment": chz.asdict(experiment),
            "step": old["step"],
        },
        destination,
    )
    return f"checkpoint {path} -> {destination}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "convert", "checkpoint"])
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--root", default=default_results_root())
    arguments = parser.parse_args()
    results_root = Path(arguments.root)
    if arguments.command == "fetch":
        fetch()
    elif arguments.command == "checkpoint":
        for path in arguments.paths:
            print(convert_checkpoint(Path(path), results_root))
    else:
        for source in sorted(LEGACY_DIR.iterdir()):
            if (source / "config.json").exists() and (source / "log.jsonl").exists():
                print(convert_run(source, results_root))


if __name__ == "__main__":
    main()
