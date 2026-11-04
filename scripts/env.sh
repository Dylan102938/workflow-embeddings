# Common environment for all wfemb jobs on rnn. Everything persistent lives on NAS.
export WFEMB_DATA=/nas/ucb/dfeng/data/workflow-embeddings
export HF_HOME=/nas/ucb/dfeng/huggingface
export UV_CACHE_DIR=/nas/ucb/dfeng/.cache/uv
export UV_PYTHON_INSTALL_DIR=/nas/ucb/dfeng/uv/python
export TMPDIR=/nas/ucb/dfeng/tmp
export TRITON_CACHE_DIR=/nas/ucb/dfeng/.cache/triton
export UV=/nas/ucb/dfeng/bin/uv
export REPO=/nas/ucb/dfeng/code/workflow-embeddings
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WFEMB_DATA/logs" "$TMPDIR"
cd "$REPO"
