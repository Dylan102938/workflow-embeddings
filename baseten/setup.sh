#!/bin/bash
# Common environment for wfemb jobs on Baseten (base image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime, Python 3.11).
# Persistent team cache: pip downloads, HF models, built frame caches and run outputs survive across jobs.
set -euo pipefail
export TEAM=${BT_TEAM_CACHE_DIR:?}/wfemb
export PIP_CACHE_DIR=$TEAM/pip-cache HF_HOME=$TEAM/hf-home HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TEAM" "$PIP_CACHE_DIR" "$HF_HOME"
pip install -q --root-user-action=ignore "transformers==4.53.2" "numpy>=1.26,<2.3" "pillow>=10.0" "tqdm>=4.66" \
    "scikit-learn>=1.4" "pyyaml>=6.0" "einops>=0.8" "huggingface_hub[hf_xet]"
if [ "${NEED_FLASH_ATTN:-0}" = "1" ] && ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    pip install -q --root-user-action=ignore --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
fi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
