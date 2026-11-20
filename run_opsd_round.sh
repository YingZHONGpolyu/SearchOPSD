#!/usr/bin/env bash
# =============================================================================
# 一轮完整的 OPSD (On-Policy Self-Distillation) 流水线脚本
# 流程：prepare → collect → mine → train → merge
# 每完成一轮后，可用产出的 merged 模型替换 POLICY 开始下一轮
# =============================================================================
# -e: 任何命令失败立即退出；-u: 未定义变量报错；-o pipefail: 管道中任何命令失败则整个管道失败
set -euo pipefail

# 定位脚本所在目录，后续所有相对路径都以此为基准
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# -------------------- 核心环境变量 --------------------
# POLICY: 初始策略模型路径 (GRPO 训练后的 checkpoint)
# 这里用 step_250 的 Search-R1 作为起点，每轮 OPSD 训练完后替换为 merged 模型
POLICY="${OPSD_POLICY:-/external/grpo/nq-hotpotqa-search-r1-grpo-qwen2.5-3b-em/actor/global_step_250}"

# RUN: 本轮运行名称，必填！用于区分输出目录，格式示例 round1_20261019
RUN="${OPSD_RUN:?Set OPSD_RUN to a new name, for example round1_20261019}"

# 校验 RUN 只能包含字母、数字、下划线、连字符（与 Python 端保持一致）
if [[ ! "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'OPSD_RUN may contain letters, numbers, underscores, and hyphens only.' >&2
  exit 2
fi

# BASE: 容器内输出路径；HOST_BASE: 宿主机对应路径（run_opsd.sh 内部会做映射）
BASE="/workspace/outputs/opsd/$RUN"
HOST_BASE="$ROOT/outputs/opsd/$RUN"
# -------------------- 运行模式控制 --------------------
# RESUME: 0=全新开始, 1=断点续跑（仅支持从 collect 阶段继续，prepare 已完成）
# COLLECT_ONLY: 0=完整流水线, 1=只跑收集阶段（用于预先生成 pool 再统一训练）
RESUME="${OPSD_RESUME:-0}"
COLLECT_ONLY="${OPSD_COLLECT_ONLY:-0}"

# 二值变量合法性校验
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]] || [[ "$COLLECT_ONLY" != 0 && "$COLLECT_ONLY" != 1 ]]; then
  echo 'OPSD_RESUME and OPSD_COLLECT_ONLY must be 0 or 1.' >&2
  exit 2
fi

# 保护：输出目录已存在且未声明续跑 → 拒绝覆盖
if [[ -e "$HOST_BASE" && "$RESUME" != 1 ]]; then
  echo "Output already exists: $HOST_BASE" >&2
  exit 2
fi

# 续跑模式的前置检查：必须有完整 QA split，且只能续在 collect 阶段
# （pairs / train / merged 已存在说明下游已跑，续跑意义不大）
if [[ -e "$HOST_BASE" ]]; then
  if [[ ! -f "$HOST_BASE/qa/train.jsonl" || ! -f "$HOST_BASE/qa/validation.jsonl" ]]; then
    echo 'Resume requires the original complete QA split.' >&2
    exit 2
  fi
  if [[ -e "$HOST_BASE/pairs.jsonl" || -e "$HOST_BASE/train" || -e "$HOST_BASE/merged" ]]; then
    echo 'Resume is restricted to collection; downstream outputs already exist.' >&2
    exit 2
  fi
fi


# -------------------- 各阶段参数 --------------------
URL="${OPSD_RETRIEVER_URL:-http://127.0.0.1:8123/retrieve}"  # 检索服务地址
STEPS="${OPSD_MAX_STEPS:-0}"              # 训练最大步数，0 表示由 epochs 决定
QUESTION_LIMIT="${OPSD_QUESTION_LIMIT:-0}"  # 收集阶段限制问题数，0 表示全部
MAX_LENGTH="${OPSD_MAX_LENGTH:-8192}"      # 序列最大长度
MAX_NEW="${OPSD_MAX_NEW_TOKENS:-2048}"     # 新生成 token 上限
FINAL_RESERVE="${OPSD_FINAL_ANSWER_RESERVE:-128}"  # 为最终答案预留的 token 空间
NPROC="${OPSD_NPROC:-1}"                   # 训练阶段 DDP 进程数（GPU 数量）


# 收集阶段强制单进程（每个 collect worker 独占一卡，多卡并行由外层调度）
export OPSD_NPROC=1

# ======================== 阶段 1: Prepare ========================
# 将 parquet 格式的 QA 数据切分为 train.jsonl + validation.jsonl
# 仅切训练集，无独立测试集用于选择 adapter
if [[ ! -e "$HOST_BASE" ]]; then
  ./run_opsd.sh prepare --questions /external/data/nq_hotpotqa_train/train.parquet --output "$BASE/qa"
fi

# 为 collect 阶段准备参数
collect_args=()
if [[ "$RESUME" == 1 ]]; then
  collect_args+=(--resume)  # 续跑时跳过已收集的问题
fi


# ======================== 阶段 2: Collect ========================
# 使用当前 POLICY 模型对每个问题采样 k=5 条带检索的推理轨迹
# 参数: --k 每个问题的候选轨迹数; --temperature 采样温度; --max-searches 每轨迹最多检索次数
./run_opsd.sh collect --model "$POLICY" --questions "$BASE/qa/train.jsonl" \
  --output "$BASE/pool.jsonl" --retriever-url "$URL" --k 5 --temperature 1.0 \
  --max-searches 4 --limit "$QUESTION_LIMIT" --max-length "$MAX_LENGTH" --max-new-tokens "$MAX_NEW" \
  --final-answer-reserve "$FINAL_RESERVE" "${collect_args[@]}"

# COLLECT_ONLY=1 时提前退出，pool.jsonl 已生成但不继续训练
# 场景：先批量收集，再统一用不同参数训练
if [[ "$COLLECT_ONLY" == 1 ]]; then
  echo "Collection complete: $HOST_BASE/pool.jsonl; no training requested."
  exit 0
fi

# ======================== 阶段 3: Mine ========================
# 从收集到的轨迹池中挖掘 preference pairs
# 逻辑：同一问题的多条轨迹两两比较，质量好的 → chosen，差的 → rejected
./run_opsd.sh mine --pool "$BASE/pool.jsonl" --output "$BASE/pairs.jsonl"

# ======================== 阶段 4: Train ========================
# 用 preference pairs 训练 LoRA adapter (GRPO/DPO 方式)
# OPSD_NPROC 此时恢复为用户设定值，启动对应数量的 DDP 进程
# best.json 会在每个 checkpoint 跑完完整推理评估后更新
OPSD_NPROC="$NPROC" ./run_opsd.sh train --model "$POLICY" --pairs "$BASE/pairs.jsonl" \
  --output "$BASE/train" --validation "$BASE/qa/validation.jsonl" \
  --validation-limit "${OPSD_VALIDATION_LIMIT:-64}" --retriever-url "$URL" \
  --max-steps "$STEPS" --epochs "${OPSD_EPOCHS:-1}" --clip 10.0 \
  --max-length "$MAX_LENGTH" --eval-max-new-tokens "$MAX_NEW" --final-answer-reserve "$FINAL_RESERVE"


# ======================== 阶段 5: Merge ========================
# 从 best.json 中提取最优 LoRA adapter 路径，合并回基座模型得到完整权重
# best.json 只有在完整推理 + EM 评估完成后才会写入，保证 adapter 质量
ADAPTER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"])' "$HOST_BASE/train/best.json")"
./run_opsd.sh merge --model "$POLICY" --adapter "$ADAPTER" --output "$BASE/merged"
echo "Merged policy: $HOST_BASE/merged"