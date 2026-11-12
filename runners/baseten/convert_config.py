"""Baseten CPU-only job: download and convert the added datasets (see runners/baseten/convert.sh). From the repo root:
WFEMB_DATASETS="videocua scalecua gui360" truss train push runners/baseten/convert_config.py --job-name convert"""

import os
from pathlib import Path

from truss_train import CacheConfig, Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject, Workspace

ROOT = Path(__file__).resolve().parents[2]
DATASETS = os.environ.get("WFEMB_DATASETS", "videocua scalecua gui360")
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(
            cpu_count=int(os.environ.get("WFEMB_CPUS", "16")), memory=os.environ.get("WFEMB_MEMORY", "64Gi")
        ),
        runtime=Runtime(
            start_commands=[f'bash runners/baseten/convert.sh "{DATASETS}"'],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
        ),
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
