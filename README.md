# SearchOPSD

**面向检索增强问答的 On-Policy Self-Distillation（OPSD，策略内自蒸馏）实验项目。** 从已有的 Search-R1 GRPO 检查点出发，让模型采集带检索的推理轨迹，再利用同一问题中的优质轨迹指导自身训练。

当前实验配置以 **Qwen2.5-3B Base 的 GRPO `global_step_250` 检查点**为起点，基于仓库内的 veRL OPSD recipe 完成一轮训练。`250` 是上游 GRPO 检查点编号，并非本项目的蒸馏训练步数。仓库提供代码与运行脚本；模型权重、检索索引、外部问答数据、虚拟环境和训练产物需要自行准备。

## 方法与流程

```text
已有 GRPO 策略检查点 + 问答数据 + 检索服务
                 │
              prepare  划分训练集和留出验证集
                 │
              collect  冻结策略，每题采样多条检索推理轨迹
                 │
                mine   在同一问题内构造参考轨迹与学生轨迹
                 │
               train   LoRA 自蒸馏，并用验证集 EM 选择 adapter
                 │
               merge   将选中的 adapter 合并回底座
```

- **轨迹配对**：优先从正确轨迹中选择检索次数少、长度短的参考轨迹，再选择与其差异较大的学生轨迹，优先使用错误轨迹；没有合适配对的问题跳过。
- **蒸馏方式**：教师使用冻结底座，并额外看到参考轨迹；学生只看到原问题，通过 LoRA 更新。两者在学生轨迹的生成位置上计算全词表前向 KL 损失，检索返回的文本仅作为上下文，不计入损失。
- **模型选择**：使用留出验证集的答案精确匹配率（Exact Match，EM）选择 adapter。测试集应留给最终评估；本 README 不声明性能提升。

实现细节见 [OPSD recipe](verl-official/recipe/opsd/README.md)。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `run_opsd_step250.py` | 当前服务器的七卡采样、训练控制器，连接同一个在线 W&B 实验 |
| `run_opsd_conda.sh` | 使用 conda 环境执行 OPSD 子命令，支持多卡训练 |
| `requirements-opsd-conda.txt` | 当前实验环境的依赖版本记录 |
| `verl-official/recipe/opsd/` | 数据准备、采样、配对、蒸馏、评估与合并实现 |
| `deps/Megatron-LM/` | 随项目保留的上游依赖源码 |
| `README_OPSD.md` | 原有服务器运行说明 |
| `run_opsd.sh`、`run_opsd_round.sh` | Docker 启动及分阶段流水线脚本，使用前需核对路径与配置 |
| `outputs/opsd/<run-name>/` | 本地实验产物，已从 Git 排除 |

## 运行前准备

### 环境与外部资源

当前默认控制器面向 **8 张可见的 NVIDIA GPU**：GPU 0 运行 E5 检索与索引，GPU 1–7 执行采样、配对评分和分布式训练。显存需求取决于模型、上下文长度、采样并发和索引规模；请先确认这些 GPU 可用。

需要准备：

1. 可供 Transformers 加载的本地策略模型目录，包含模型权重、配置和 tokenizer。
2. NQ / HotpotQA 训练问答数据，当前配置使用 `train.parquet`。
3. Search-R1 的 `retrieval_server.py`、E5 模型、FAISS 索引及与索引匹配的 Wikipedia 语料。
4. 包含 PyTorch、Transformers、PEFT、vLLM、检索依赖及 W&B 的 Python 环境。依赖版本见 `requirements-opsd-conda.txt`；安装时需要匹配实际 CUDA / PyTorch 软件源，不能假定这些固定版本适用于所有机器。
5. W&B 账户及网络连接。默认控制器显式使用 `mode='online'`。

现有服务器使用以下路径：

| 资源 | 默认路径 |
| --- | --- |
| 项目根目录 | `/data/work/YingZhong/code/on_policy_distillation_verl` |
| conda 环境 | `/data/work/YingZhong/miniconda3/envs/verl` |
| 策略检查点 | `/data/work/YingZhong/code/Search-R1/verl_checkpoints/nq-hotpotqa-search-r1-grpo-qwen2.5-3b-em/actor/global_step_250` |
| 问答训练数据 | `/data/work/YingZhong/data/nq_hotpotqa_train/train.parquet` |
| 检索服务源码 | `/data/work/YingZhong/code/Search-R1/search_r1/search/retrieval_server.py` |
| FAISS 索引 | `/data/work/YingZhong/data/search_data/e5_Flat.index` |
| Wikipedia 语料 | `/data/work/YingZhong/data/search_data/wiki-18.jsonl` |
| E5 模型 | `/data/work/YingZhong/models/e5-base-v2` |

脚本默认启用 Hugging Face 离线模式，启动时不会自动下载缺失模型。

## 在现有服务器运行

以下命令均在服务器的项目根目录执行。启动检索服务和控制器需要两个终端。

### 1. 检查入口并登录 W&B

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
export OPSD_CONDA_ENV=/data/work/YingZhong/miniconda3/envs/verl

bash run_opsd_conda.sh --help
"$OPSD_CONDA_ENV/bin/python" -m wandb login
```

### 2. 启动检索服务

终端 A：如果端口 8123 上尚无可用的 OPSD 检索服务，执行：

```bash
OPSD_GPUS=0 bash run_opsd_conda.sh retriever --port 8123
```

服务绑定 `127.0.0.1:8123`，检索接口为 `/retrieve`。在另一个终端检查实际检索响应：

```bash
curl --fail --show-error http://127.0.0.1:8123/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["What is the capital of France?"],"topk":1,"return_scores":true}'
```

应返回包含非空检索结果的 `result` 字段。索引加载可能需要时间，控制器也会在开始采样前检查该接口。

### 3. 启动完整一轮实验

终端 B：每次新实验使用一个新的运行名称。

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl

/data/work/YingZhong/miniconda3/envs/verl/bin/python run_opsd_step250.py \
  --run-name step250_round1 \
  --sampler vllm \
  --concurrency 128
```

控制器会依次准备数据、启动七个采样 worker、合并并校验轨迹完整性、挖掘配对、执行七卡训练，再合并最佳 adapter。它不会自动启动检索服务。输出目录已存在时，普通启动会拒绝覆盖。

| 参数 / 配置 | 当前默认值 |
| --- | --- |
| `--run-name` | 必填，只允许字母、数字、下划线和连字符 |
| `--sampler` | `vllm`，也支持 `transformers` |
| `--concurrency` | vLLM 每卡并发上限 128 |
| `--project` | W&B 项目 `opsd-qwen2.5-3b` |
| 每题轨迹数 / 温度 | 5 / 1.0 |
| 最大检索次数 | 4 |
| 上下文 / 新生成 token 上限 | 8192 / 2048 |
| vLLM 显存预算比例 | 0.88，在控制器中设置 |
| 训练 epoch / 全局 batch | 1 / 560 对 |
| DDP 进程数 / 每卡 micro-batch | 7 / 1 对 |
| 完整 batch 的梯度累积次数 | 80 |
| checkpoint 保存间隔 | 每 10 个优化器 step，并保留最终权重 |
| checkpoint 验证问题上限 | 64 |

有效配对数少于 560 时，默认控制器会停止。训练总步数为 `ceil(usable_pairs / 560)`，其中 `usable_pairs` 是编码和过滤后的可训练配对数，尾批也执行更新。

### 4. 查看进度与结果

```bash
cat outputs/opsd/step250_round1/status.json
tail -f outputs/opsd/step250_round1/collect_gpu_1.log
# 进入训练阶段后：
tail -f outputs/opsd/step250_round1/train.log
```

| 产物 | 内容 |
| --- | --- |
| `status.json` | 当前阶段、进程信息与采样进度 |
| `qa/` | 训练 / 验证划分及划分统计 |
| `collect_gpu_*.log` | 各卡采样日志 |
| `pool_gpu_*.jsonl`、`pool_gpu_*.vllm.jsonl` | 各卡轨迹 |
| `pool.jsonl` | 合并后的完整轨迹池 |
| `pairs.jsonl`、`pairs.jsonl.stats.json` | 蒸馏配对与挖掘统计 |
| `train.log`、`train/` | 训练日志、LoRA adapter、验证记录 |
| `train/best.json` | 验证选中的 adapter 路径 |
| `merged/` | 最佳 adapter 与底座合并后的模型 |
| `wandb_run.json` | W&B 实验 ID 和链接 |

### 5. 采样中断后续跑

先确认旧控制器和采样 worker 已退出，然后使用原运行名称与相同采样配置：

```bash
/data/work/YingZhong/miniconda3/envs/verl/bin/python run_opsd_step250.py \
  --run-name step250_round1 \
  --sampler vllm \
  --concurrency 128 \
  --resume
```

`--resume` 用于复用已有数据划分、W&B 记录并补齐缺失轨迹，**不会恢复训练优化器状态**。如果已经生成 `pool.jsonl`、配对或训练产物，不应把它当作任意阶段的一键恢复命令。

## 在其他机器或使用不同配置

```bash
git clone https://github.com/YingZHONGpolyu/SearchOPSD.git
cd SearchOPSD
export OPSD_CONDA_ENV=/absolute/path/to/your/conda/env
bash run_opsd_conda.sh --help
```

克隆代码不会同时获取外部模型、索引和训练数据。`run_opsd_conda.sh` 根据自身位置定位项目，并支持 `OPSD_CONDA_ENV`、`OPSD_GPUS`、`OPSD_NPROC` 和 `OPSD_RETRIEVER_URL` 环境变量；但 `run_opsd_step250.py` 的项目路径、模型路径、问答数据路径、Python 路径、检索地址和七卡配置仍有硬编码，迁移时必须修改这些位置。

也可以直接通过分阶段 CLI 指定自己的路径，避免使用默认控制器：

```bash
# 查看各阶段支持的参数
bash run_opsd_conda.sh prepare --help
bash run_opsd_conda.sh collect --help
bash run_opsd_conda.sh mine --help
bash run_opsd_conda.sh train --help
bash run_opsd_conda.sh evaluate --help
bash run_opsd_conda.sh merge --help
```

检索服务可显式配置外部资源路径：

```bash
OPSD_GPUS=0 bash run_opsd_conda.sh retriever \
  --source /absolute/path/to/Search-R1/search_r1/search/retrieval_server.py \
  --index /absolute/path/to/e5_Flat.index \
  --corpus /absolute/path/to/wiki-18.jsonl \
  --model /absolute/path/to/e5-base-v2 \
  --port 8123
```

多卡训练通过 `OPSD_GPUS=1,2,3 OPSD_NPROC=3 bash run_opsd_conda.sh train ...` 启动；其他阶段不支持该包装脚本的多进程模式。单独训练时如需记录 W&B，需要提供 `--wandb-project` 等参数。

## 常见问题

- **检索服务不可用**：检查索引、语料、E5 模型和 Search-R1 源码路径，查看终端 A 日志，并用上面的 POST 请求检查 `/retrieve`。
- **CUDA 显存不足**：vLLM 采样可先降低 `--concurrency`；显存预算 0.88、序列长度与其他资源配置需要在对应入口中调整。`--index-device cpu` 只将检索索引放到 CPU，E5 编码器仍使用 GPU。
- **模型或 tokenizer 缺失**：默认离线运行，请提前准备完整模型目录；原始分片检查点需先转换成可加载的模型格式。
- **W&B 登录失败**：用运行控制器的 Python 环境登录，并确认网络可用。默认控制器不提供离线模式参数。
- **没有足够的蒸馏对**：查看 `pairs.jsonl.stats.json` 和采样日志，检查正确轨迹比例、数据与检索结果；默认控制器至少需要 560 对。

## 上游代码与许可

本项目包含 veRL 和 Megatron-LM 上游源码，并依赖外部 Search-R1 检索实现。相关许可证和版权声明保留在各自目录中，使用或分发时请遵守对应条款。
