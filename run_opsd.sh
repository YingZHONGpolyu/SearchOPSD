#!/usr/bin/env bash
# -- OPSD Docker 启动脚本 --
# 在 veRL 容器内运行 recipe.opsd，支持单卡/多卡
set -euo pipefail

# ---- 基础配置 ----
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:53aa78ecb8c86637d41636a21e88bdd8b18bfb9a753e812dbe321fd6f922deb5"  # 固定 digest，避免镜像更新导致不可复现

# GPU: 默认 device=7；多卡写成 device=1,2,3
GPU_SPEC="${OPSD_GPUS:-device=7}"
[[ "$GPU_SPEC" == device=*,* ]] && GPU_SPEC="\"$GPU_SPEC\""

mkdir -p "$ROOT/outputs/opsd" "$ROOT/logs/opsd"

# ---- 选择执行方式：单卡 (python) / 多卡 (torchrun) ----
# 多卡仅支持 train 模式
NPROC="${OPSD_NPROC:-1}"
if [[ "$NPROC" -gt 1 ]]; then
  [[ "${1:-}" != train ]] && { echo 'OPSD_NPROC>1 only supports train' >&2; exit 2; }
  CMD=(/workspace/verl/.venv/bin/torchrun --standalone --nproc_per_node="$NPROC" -m recipe.opsd)
else
  CMD=(/workspace/verl/.venv/bin/python -m recipe.opsd)
fi

# ---- 启动容器 ----
exec docker run --rm --user "$(id -u):$(id -g)" --gpus "$GPU_SPEC" \
  --ipc=host --network=host --ulimit nofile=65535:65535 \
  -w /workspace/verl \
  # 环境变量
  -e PYTHONPATH=/workspace/verl \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/workspace/cache/huggingface \
  # 项目目录挂载（可写）
  -v "$ROOT/verl-official:/workspace/verl" \
  -v "$ROOT/outputs:/workspace/outputs" \
  -v "$ROOT/logs:/workspace/logs" \
  -v "$ROOT/data:/workspace/data" \
  -v "$ROOT/models:/workspace/models" \
  -v "$ROOT/cache:/workspace/cache" \
  # 外部数据与模型挂载（只读）
  -v /data/work/YingZhong/models:/external/models:ro \
  -v /data/work/YingZhong/data:/external/data:ro \
  -v /data/work/YingZhong/code/Search-R1/verl_checkpoints:/external/grpo:ro \
  "$IMAGE" "${CMD[@]}" "$@"