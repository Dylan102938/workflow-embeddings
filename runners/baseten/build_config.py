"""Baseten job: build frame caches (see runners/baseten/build.sh). Push from the repo root:
truss train push runners/baseten/build_config.py --job-name build"""

from pathlib import Path

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject, Workspace

ROOT = Path(__file__).resolve().parents[2]
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=4)),  # platform picks CPU / memory
        runtime=Runtime(
            start_commands=['bash runners/baseten/build.sh "siglip2:16 siglip2:24"'],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
        ),
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
