"""Baseten CPU-only job: download inputs to the team cache. From the repo root:
truss train push runners/baseten/download_config.py --job-name download"""

from pathlib import Path

from truss_train import CacheConfig, Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject, Workspace

ROOT = Path(__file__).resolve().parents[2]
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(cpu_count=8, memory="32Gi"),
        runtime=Runtime(
            start_commands=["bash runners/baseten/download.sh"],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
        ),
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
