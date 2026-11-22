"""Baseten job: build frame caches. Push from the repo root:
WFEMB_SPECS="siglip2:24 siglip2:32" WFEMB_GPUS=2 truss train push runners/baseten/build_config.py --job-name build
WFEMB_TOWERS="qwen3.8:576 inkling2:576" ... builds caches of towers' own tokens instead (build_towers.sh).
WFEMB_SET=added builds the caches of the added datasets."""

import os
from pathlib import Path

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject, Workspace

ROOT = Path(__file__).resolve().parents[2]
DATA_SET = os.environ.get("WFEMB_SET", "agentnet")
if "WFEMB_TOWERS" in os.environ:  # the towers need transformers 5, which needs torch >= 2.6
    BASE_IMAGE = "pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime"
    COMMAND = f'bash runners/baseten/build_towers.sh "{os.environ["WFEMB_TOWERS"]}" {DATA_SET}'
else:
    BASE_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    COMMAND = f'bash runners/baseten/build.sh "{os.environ["WFEMB_SPECS"]}" {DATA_SET}'
project = TrainingProject(
    name="wfemb",
    job=TrainingJob(
        image=Image(base_image=BASE_IMAGE),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=int(os.environ.get("WFEMB_GPUS", "4")))),
        runtime=Runtime(
            start_commands=[COMMAND],
            environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")},
            cache_config=CacheConfig(enabled=True),
        ),
        workspace=Workspace(
            workspace_root=str(ROOT),
            exclude_dirs=[str(ROOT / d) for d in ("docs", ".git", ".venv", "results", "dashboard", "legacy_runs")],
        ),
    ),
)
