#!/usr/bin/env bash
# Run Search-E1 OPSD inside the already installed veRL image and virtualenv.
# External assets are read-only; all new results stay in this task's directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Pin the verified installed digest so a later tag update cannot change the runtime.
IMAGE="verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:53aa78ecb8c86637d41636a21e88bdd8b18bfb9a753e812dbe321fd6f922deb5"
GPU_SPEC="${OPSD_GPUS:-device=7}"
if [[ "$GPU_SPEC" == device=* && "$GPU_SPEC" == *,* ]]; then
  GPU_SPEC="\"$GPU_SPEC\""
fi
mkdir -p "$ROOT/outputs/opsd" "$ROOT/logs/opsd"
# A multi-GPU invocation uses torchrun; callers can explicitly choose NPROC.
NPROC="${OPSD_NPROC:-1}"
COMMAND=(/workspace/verl/.venv/bin/python -m recipe.opsd.compare_checkpoints)
if [[ "$NPROC" -gt 1 ]]; then
  if [[ "${1:-}" != train ]]; then
    echo 'OPSD_NPROC>1 is supported only for train; collection and evaluation are single-process.' >&2
    exit 2
  fi
  COMMAND=(/workspace/verl/.venv/bin/torchrun --standalone --nproc_per_node="$NPROC" -m recipe.opsd.compare_checkpoints)
fi
# Match host ownership so generated checkpoints are readable outside Docker too.
exec docker run --rm --user "$(id -u):$(id -g)" --gpus "$GPU_SPEC" --ipc=host --network=host \
  --ulimit nofile=65535:65535 \
  -e PYTHONPATH=/workspace/verl -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/workspace/cache/huggingface \
  -v "$ROOT/verl-official:/workspace/verl" \
  -v "$ROOT/outputs:/workspace/outputs" -v "$ROOT/logs:/workspace/logs" \
  -v "$ROOT/data:/workspace/data" -v "$ROOT/models:/workspace/models" \
  -v "$ROOT/cache:/workspace/cache" \
  -v /data/work/YingZhong/models:/external/models:ro \
  -v /data/work/YingZhong/data:/external/data:ro \
  -v /data/work/YingZhong/code/Search-R1/verl_checkpoints:/external/grpo:ro \
  -w /workspace/verl "$IMAGE" "${COMMAND[@]}" "$@"
