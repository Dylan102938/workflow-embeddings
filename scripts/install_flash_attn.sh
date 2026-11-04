#!/bin/bash
# flash-attn prebuilt wheel matching torch 2.5.1+cu121 (pre-cxx11 ABI), installed WITHOUT dependencies so uv
# cannot swap in a different torch. `uv sync` removes it (not in the lockfile); this script is idempotent.
set -euo pipefail
cd /nas/ucb/dfeng/code/workflow-embeddings
if ! .venv/bin/python -c "import flash_attn_2_cuda" 2>/dev/null; then
    "${UV:-/nas/ucb/dfeng/bin/uv}" pip install --python .venv/bin/python --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
fi
.venv/bin/python -c "import flash_attn, einops; print('flash_attn', flash_attn.__version__)"
