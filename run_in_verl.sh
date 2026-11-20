#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/work/YingZhong/code/on_policy_distillation_verl"
IMAGE="verl-ci-cn-beijing.cr.volces.com/verlai/verl:uv.cu130"
GPU_SPEC="${VERL_GPUS:-all}"

if [[ "${GPU_SPEC}" == device=* && "${GPU_SPEC}" == *,* ]]; then
  GPU_SPEC="\"${GPU_SPEC}\""
fi

mkdir -p \
  "${ROOT}/cache/huggingface" \
  "${ROOT}/cache/modelscope" \
  "${ROOT}/data" \
  "${ROOT}/logs" \
  "${ROOT}/models" \
  "${ROOT}/outputs"

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

if [[ $# -eq 0 ]]; then
  set -- /bin/bash
fi

exec docker run --rm "${tty_args[@]}" \
  --gpus "${GPU_SPEC}" \
  --ipc=host \
  --network=host \
  --ulimit nofile=65535:65535 \
  -e HF_HOME=/workspace/cache/huggingface \
  -e MODELSCOPE_CACHE=/workspace/cache/modelscope \
  -e PYTHONPATH=/workspace/verl:/workspace/Megatron-LM \
  -e TOKENIZERS_PARALLELISM=false \
  -v "${ROOT}/verl-official:/workspace/verl" \
  -v "${ROOT}/deps/Megatron-LM:/workspace/Megatron-LM:ro" \
  -v "${ROOT}/cache:/workspace/cache" \
  -v "${ROOT}/data:/workspace/data" \
  -v "${ROOT}/logs:/workspace/logs" \
  -v "${ROOT}/models:/workspace/models" \
  -v "${ROOT}/outputs:/workspace/outputs" \
  -v "${ROOT}/verify_verl_env.py:/workspace/verify_verl_env.py:ro" \
  -v "${ROOT}/check_distributed.py:/workspace/check_distributed.py:ro" \
  -v "${ROOT}/check_ray.py:/workspace/check_ray.py:ro" \
  -w /workspace/verl \
  "${IMAGE}" "$@"
