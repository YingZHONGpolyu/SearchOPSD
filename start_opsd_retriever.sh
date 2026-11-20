#!/usr/bin/env bash
# Foreground service: the user can inspect its output and stop it normally.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${OPSD_RETRIEVER_GPU:-6}"
export PYTHONPATH="$ROOT/verl-official"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
# Keep any dataset preprocessing caches inside the requested project directory.
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/cache/huggingface/datasets"
export HF_HUB_OFFLINE=1
exec /data/work/YingZhong/miniconda3/envs/retriever/bin/python \
  -m recipe.opsd.retriever_server "$@"
