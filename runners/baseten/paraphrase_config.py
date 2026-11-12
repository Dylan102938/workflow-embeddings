"""Baseten job: paraphrase a prepared set's instructions (see runners/baseten/paraphrase.sh). From the repo root:
WFEMB_SET=added truss train push runners/baseten/paraphrase_config.py --job-name paraphrase"""

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
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
        runtime=Runtime(
            start_commands=[f"bash runners/baseten/paraphrase.sh {os.environ.get('WFEMB_SET', 'added')}"],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        ),
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
