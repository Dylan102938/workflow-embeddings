"""Baseten job: one packed group of wfemb runs. Push from the repo root:
  WFEMB_GROUP=<group> WFEMB_GPUS=2 truss train push baseten/train_config.py --job-name <group>
where baseten/jobs/<group>.txt holds the run lines."""
import os
from pathlib import Path

from truss.base.truss_config import AcceleratorSpec
from truss_train import (CacheConfig, CheckpointingConfig, Compute, Image, Runtime, SecretReference, TrainingJob,
                         TrainingProject, Workspace)

ROOT = Path(__file__).resolve().parent.parent
GROUP = os.environ["WFEMB_GROUP"]
assert "WFEMB_CMD" in os.environ or (ROOT / "baseten" / "jobs" / f"{GROUP}.txt").exists(), GROUP
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=int(os.environ.get("WFEMB_GPUS", "2")))),
        runtime=Runtime(
            start_commands=[os.environ.get("WFEMB_CMD", f"bash baseten/train.sh baseten/jobs/{GROUP}.txt")],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        ),
        workspace=Workspace(workspace_root=str(ROOT), exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "sweeps")]),
    ),
)
