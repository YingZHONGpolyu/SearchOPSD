# OPSD：Search-E1 自蒸馏运行指南

本项目在 Search-R1 的 GRPO checkpoint 上执行一轮 OPSD：用冻结策略采集检索轨迹，从同一问题的轨迹中构造蒸馏对，训练 LoRA，再按验证集 Exact Match（EM）选择 adapter 并合并权重。当前入口使用 Qwen2.5-3B **Base** 的 `global_step_250`；这里的 250 是上游 checkpoint 编号。

## 在 11_Server_Ugreen 上运行

从项目根目录执行，使用已有的 `verl` conda 环境和本地模型、数据。启动前确认 GPU 0–7 可用，且 W&B 已登录（控制器使用在线模式）。

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
# 如果 8123 上还没有 OPSD 检索服务，在单独终端启动：
OPSD_GPUS=0 bash run_opsd_conda.sh retriever --port 8123

# 另开终端，在同一目录启动；每次新实验使用新的 run name：
/data/work/YingZhong/miniconda3/envs/verl/bin/python run_opsd_step250.py \
  --run-name step250_my_run --sampler vllm --concurrency 128
```

控制器使用 GPU 1–7，每卡一个采样进程；默认 vLLM，每卡并发上限 128，显存预算参数为 0.88。显存不足时可降低 `--concurrency`；0.88 在控制器中设置。每题采 5 条轨迹，温度 1.0，最多检索 4 次，上下文上限 8192，新生成 token 上限 2048。

如采样中断，确认原控制器及 worker 已退出，再用原 run name 和相同采样参数追加 `--resume`。vLLM 续采会读取旧轨迹文件并补齐缺失项；这个选项不从训练 checkpoint 恢复优化器。

## 训练步数与 checkpoint

训练为 1 个 epoch，配置全局 batch 为 560 对：7 卡、每卡 micro-batch 为 1，完整 batch 累积 80 次。最后不足 560 对的 batch 仍执行一次更新。总训练步数为 `ceil(usable_pairs / 560)`，其中 `usable_pairs` 是编码和过滤后实际可训练的配对数，可在 `train.log` 中查看。

每 **10 个优化器 step** 保存一次 checkpoint，并保存最终权重。训练结束后按验证集 EM 选优，控制器读取 `train/best.json` 并将选中的 adapter 合并到 `merged/`。

## 查看进度和产物

在 `outputs/opsd/<run-name>/` 下查看：

- `status.json`：当前阶段；采样时记录已完成和目标轨迹数。
- `collect_gpu_*.log` 和 `pool_gpu_*.jsonl`：各卡采样日志和轨迹；vLLM 轨迹以 `.vllm.jsonl` 结尾。
- `pool.jsonl`、`pairs.jsonl`：合并后的轨迹和挖掘得到的蒸馏对。
- `train.log`、`train/`、`merged/`：训练日志、adapter 与验证记录、合并模型。
- `wandb_run.json`：对应的 W&B 实验地址。

入口是 `run_opsd_step250.py`，环境包装脚本是 `run_opsd_conda.sh`。配对规则、检索 token 掩码和蒸馏损失见 [recipe 实现说明](verl-official/recipe/opsd/README.md)。
