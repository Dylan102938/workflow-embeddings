"""Baseten job: redo the final evaluation of finished runs (see runners/baseten/evaluate.sh):
  WFEMB_RUN_IDS="<run id> ..." [WFEMB_OLD_RUN_NAMES="<old run name> ..."] \
      truss train push runners/baseten/evaluate_config.py --job-name evaluate"""

import os
import shlex
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
RUN_IDS, OLD_RUN_NAMES = os.environ["WFEMB_RUN_IDS"], os.environ.get("WFEMB_OLD_RUN_NAMES", "")
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
        runtime=Runtime(
            start_commands=[f"bash runners/baseten/evaluate.sh {shlex.quote(RUN_IDS)} {shlex.quote(OLD_RUN_NAMES)}"],
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
