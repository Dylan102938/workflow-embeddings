"""Baseten job: the runs of one section of jobs.txt. Push from the repo root:
  WFEMB_JOB=<section> WFEMB_GPUS=2 truss train push runners/baseten/train_config.py --job-name <section>
WFEMB_CMD replaces the start command (debugging)."""

import os
from pathlib import Path

from truss.base.truss_config import AcceleratorSpec
from truss_train import (
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    SecretReference,
    TrainingJob,
    TrainingProject,
    Workspace,
)

ROOT = Path(__file__).resolve().parents[2]
JOB = os.environ["WFEMB_JOB"]
assert "WFEMB_CMD" in os.environ or f"[{JOB}]" in (ROOT / "jobs.txt").read_text().split(), (
    f"no section [{JOB}] in jobs.txt"
)
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=int(os.environ.get("WFEMB_GPUS", "2")))),
        runtime=Runtime(
            start_commands=[os.environ.get("WFEMB_CMD", f"bash runners/baseten/train.sh {JOB}")],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        ),
        # top-level folders only; results/ and dashboard/ (demo bundles) can be large
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
