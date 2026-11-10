"""Baseten job: demo inputs for finished runs (test embeddings, screenshot cache; see runners/baseten/demo.sh):
  WFEMB_DEMO_IDS="<run id> ..." [WFEMB_DEMO_LEGACY="<old run name> ..."] \
      truss train push runners/baseten/demo_config.py --job-name demo"""
import os
import shlex
from pathlib import Path

from truss.base.truss_config import AcceleratorSpec
from truss_train import (CacheConfig, CheckpointingConfig, Compute, Image, Runtime, SecretReference, TrainingJob,
                         TrainingProject, Workspace)

ROOT = Path(__file__).resolve().parents[2]
IDS, LEGACY = os.environ["WFEMB_DEMO_IDS"], os.environ.get("WFEMB_DEMO_LEGACY", "")
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
        runtime=Runtime(
            start_commands=[f"bash runners/baseten/demo.sh {shlex.quote(IDS)} {shlex.quote(LEGACY)}"],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        ),
        workspace=Workspace(workspace_root=str(ROOT), exclude_dirs=[str(ROOT / d) for d in (
            "docs", ".git", ".venv", "results", "dashboard", "legacy_runs")]),
    ),
)
