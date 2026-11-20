#!/usr/bin/env bash
# 设置严格模式：出错即退出(-e)、未定义变量报错(-u)、管道中任意命令失败则整个管道失败(-o pipefail)
set -euo pipefail

# 项目根目录，用于挂载到 Docker 容器中
ROOT="/data/work/YingZhong/code/on_policy_distillation_verl"

# Docker 镜像地址（内部 Harbor 仓库），预装了 Python 3.x + CUDA 13.0 环境
IMAGE="verl-ci-cn-beijing.cr.volces.com/verlai/verl:uv.cu130"

# 必填参数：从命令行第一个参数读取 ModelScope 上的模型 ID
# 若未提供则打印用法并退出
MODEL_ID="${1:?Usage: download_model.sh MODEL_ID [DESTINATION_NAME]}"

# 可选参数：本地存储的目录名称，默认为 MODEL_ID 本身
DESTINATION_NAME="${2:-${MODEL_ID}}"

# 提前创建目标目录和 ModelScope 缓存目录，防止容器内无写入权限
mkdir -p "${ROOT}/models/${DESTINATION_NAME}" "${ROOT}/cache/modelscope"

# 使用 Docker 运行 modelscope 命令下载模型
exec docker run --rm \                          # 容器退出后自动删除
  --network=host \                               # 使用宿主机网络，便于访问外部 ModelScope 服务
  -e MODELSCOPE_CACHE=/workspace/cache/modelscope \  # 容器内缓存路径，指向挂载卷
  -v "${ROOT}:/workspace" \                    # 将项目根目录挂载到容器 /workspace
  -w /workspace \                                # 容器内工作目录设为 /workspace
  --entrypoint /workspace/tools/modelscope-venv/bin/modelscope \  # 覆盖默认入口，直接执行 modelscope CLI
  "${IMAGE}" \                                   # 使用的 Docker 镜像
  download --model "${MODEL_ID}" \               # modelscope 子命令：下载指定模型
  --local_dir "/workspace/models/${DESTINATION_NAME}"  # 下载到的本地目录（容器内路径，即宿主机 ROOT/models/...）