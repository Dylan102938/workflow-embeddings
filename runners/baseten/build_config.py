"""Baseten job: build frame caches (see runners/baseten/build.sh). Push from the repo root:
WFEMB_SPECS="siglip2:32 siglip2:16:0" WFEMB_GPUS=2 truss train push runners/baseten/build_config.py --job-name build"""

import os
from pathlib import Path

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject, Workspace

ROOT = Path(__file__).resolve().parents[2]
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=int(os.environ.get("WFEMB_GPUS", "4")))),
        runtime=Runtime(
            start_commands=[f'bash runners/baseten/build.sh "{os.environ["WFEMB_SPECS"]}"'],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
        ),
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
