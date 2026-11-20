#!/usr/bin/env bash
# Run OPSD with the server's conda verl Python, using host paths throughout.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${OPSD_CONDA_ENV:-/data/work/YingZhong/miniconda3/envs/verl}"
PYTHON="$CONDA_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing conda Python: $PYTHON" >&2
    exit 1
fi
# Keep an explicit GPU assignment; default to the GPU used for installation checks.
export CUDA_VISIBLE_DEVICES="${OPSD_GPUS:-${CUDA_VISIBLE_DEVICES:-7}}"
export PYTHONPATH="$ROOT/verl-official${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPSD_SMOKE_MODEL="${OPSD_SMOKE_MODEL:-/data/work/YingZhong/models/Qwen2.5-3B}"
export OPSD_SMOKE_OUTPUT="${OPSD_SMOKE_OUTPUT:-$ROOT/outputs/opsd/conda_gpu_smoke}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NPROC="${OPSD_NPROC:-1}"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo "OPSD_NPROC must be a positive integer" >&2
    exit 2
fi
if [[ "$NPROC" -gt 1 && "${1:-}" != train ]]; then
    echo "Multiple processes are supported only for train" >&2
    exit 2
fi
cd "$ROOT/verl-official"
if [[ "${1:-}" == retriever ]]; then
    # Use the same conda environment for the existing E5/FAISS retrieval service.
    shift
    exec "$PYTHON" -m recipe.opsd.retriever_server "$@"
fi
EXTRA=()
case "${1:-}" in
    collect|train|evaluate)
        # Port 8000 belongs to another service; explicit user arguments win below.
        EXTRA=(--retriever-url "${OPSD_RETRIEVER_URL:-http://127.0.0.1:8123/retrieve}")
        ;;
esac
if [[ "$NPROC" -gt 1 ]]; then
    # torchrun comes from the selected interpreter, avoiding another env's binary.
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        -m recipe.opsd "${1}" "${EXTRA[@]}" "${@:2}"
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    exec "$PYTHON" -m recipe.opsd "${1}" "${EXTRA[@]}" "${@:2}"
fi
exec "$PYTHON" -m recipe.opsd "$@"
