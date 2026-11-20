"""Run a native OPSD round: seven collection workers, mining, seven-rank training.

The frozen checkpoint is unchanged during collection. All stages keep durable
local logs, and collection/training metrics share a single online W&B run.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import wandb

ROOT = Path('/data/work/YingZhong/code/on_policy_distillation_verl')
MODEL = '/data/work/YingZhong/code/Search-R1/verl_checkpoints/nq-hotpotqa-search-r1-grpo-qwen2.5-3b-em/actor/global_step_250'
URL = 'http://127.0.0.1:8123/retrieve'


def main():
    """Start a fresh run and advance only after each preceding stage succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--project', default='opsd-qwen2.5-3b')
    # Resume the durable question split and W&B run when changing the sampler backend.
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sampler', choices=('vllm', 'transformers'), default='vllm')
    parser.add_argument('--concurrency', type=int, default=128)
    args = parser.parse_args()
    if not args.run_name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Run name must contain only letters, numbers, underscores and hyphens')
    base = ROOT / 'outputs/opsd' / args.run_name
    base.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and not (base / 'qa/train.jsonl').exists():
        raise ValueError('Resume requires the existing complete QA split')
    launcher = str(ROOT / 'run_opsd_conda.sh')
    children = []
    run = None

    def status(phase, **extra):
        # Replace status atomically so monitoring never sees half-written JSON.
        record = dict(phase=phase, updated_at=time.time(), controller_pid=os.getpid(), **extra)
        temporary = base / 'status.tmp'
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(base / 'status.json')
        print(json.dumps(record), flush=True)

    def command(stage, command_args, gpu='1', nproc=1):
        # Each subprocess inherits the conda launcher; no container paths are used.
        with (base / f'{stage}.log').open('ab') as log:
            process = subprocess.Popen([launcher, *command_args], cwd=ROOT,
                                       env=dict(os.environ, OPSD_GPUS=gpu, OPSD_NPROC=str(nproc)),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(process)
            code = process.wait()
        if code:
            raise RuntimeError(f'{stage} failed with exit code {code}; see {base / (stage + ".log")}')

    try:
        # Connect before launching expensive model workers; authentication failures stop here.
        previous_run = json.loads((base / 'wandb_run.json').read_text()) if args.resume else {}
        run = wandb.init(project=args.project, name=args.run_name, dir=str(base), mode='online',
                         id=previous_run.get('id'), entity=previous_run.get('entity'),
                         resume='allow' if args.resume else None,
                         config=dict(model=MODEL, global_batch_size=560, world_size=7,
                                     micro_batch_size_per_gpu=1, effective_gradient_accumulation_steps=80,
                                     collection_k=5, collection_temperature=1.0, epochs=1,
                                     save_steps=10, save_final_checkpoint=True,
                                     max_length=8192, max_new_tokens=2048, retriever_gpu=0,
                                     training_gpus=list(range(1, 8))),
                         settings=wandb.Settings(init_timeout=120))
        # Log a backend change explicitly; trajectories retain their per-record provenance.
        run.config.update({'collection_backend': args.sampler, 'collection_concurrency_per_gpu': args.concurrency,
                           'retriever_index_device': 'cuda', 'retriever_index_dtype': 'float32'}, allow_val_change=True)
        run_id, entity = run.id, run.entity
        (base / 'wandb_run.json').write_text(json.dumps({'id': run_id, 'url': run.url, 'entity': entity}, indent=2))
        status('waiting_for_retriever', wandb_url=run.url)
        # Require a real retrieval response; an open socket alone is insufficient.
        for attempt in range(120):
            try:
                request = urllib.request.Request(URL, data=json.dumps({
                    'queries': ['What is the capital of France?'], 'topk': 1, 'return_scores': True,
                }).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    result = json.load(response)
                assert result['result'][0]
                break
            except (OSError, AssertionError):
                if attempt == 119:
                    raise RuntimeError('Retriever did not become ready')
                time.sleep(10)
        if not args.resume:
            command('prepare', ['prepare', '--questions', '/data/work/YingZhong/data/nq_hotpotqa_train/train.parquet',
                                '--output', str(base / 'qa')])
        # Shard whole questions, keeping all five siblings together on one worker.
        shard_paths = [base / f'questions_gpu_{gpu}.jsonl' for gpu in range(1, 8)]
        counts = [0] * 7
        streams = [] if args.resume else [path.open('x') for path in shard_paths]
        try:
            with (base / 'qa/train.jsonl').open() as source:
                for index, line in enumerate(source):
                    if streams:
                        streams[index % 7].write(line)
                    counts[index % 7] += 1
        finally:
            for stream in streams:
                stream.close()
        # ======================== 并行数据收集 (核心阶段) ========================
        # 启动 7 个并行 worker，每个 GPU 一个，各自处理一个 shard 的问题
        # 每个问题采样 k=5 条轨迹（siblings），用于后续挖掘偏好对
        workers, log_readers = [], []
        for gpu, path in enumerate(shard_paths, 1):
            log_path = base / f'collect_gpu_{gpu}.log'
            if args.sampler == 'vllm':
                # vLLM 后端：高性能异步采样
                # 旧 pool (pool_gpu_{i}.jsonl) 保持不变，新后端输出写入 .vllm.jsonl
                # 这样支持从 transformers 后端切换到 vLLM 时复用已有结果
                worker_args = [
                    '/data/work/YingZhong/miniconda3/envs/verl/bin/python', '-m', 'recipe.opsd.rollout_vllm',
                    '--model', MODEL, '--questions', str(path),
                    '--output', str(base / f'pool_gpu_{gpu}.vllm.jsonl'),
                    '--existing-pool', str(base / f'pool_gpu_{gpu}.jsonl'),  # 已有的 pool，跳过重复问题
                    '--concurrency', str(args.concurrency), '--gpu-memory-utilization', '0.88',
                ]
                if args.resume:
                    worker_args.append('--resume')
            else:
                # transformers 后端：兼容性好，参数更细
                worker_args = [
                    launcher, 'collect', '--model', MODEL, '--questions', str(path),
                    '--output', str(base / f'pool_gpu_{gpu}.jsonl'), '--k', '5',  # 每问题 5 条轨迹
                    '--temperature', '1.0', '--max-searches', '4', '--max-length', '8192',
                    '--max-new-tokens', '2048', '--final-answer-reserve', '128',
                ]
                if args.resume:
                    worker_args.append('--resume')
            # 启动子进程（非阻塞），日志追加写入 collect_gpu_{i}.log
            # CUDA_VISIBLE_DEVICES 绑定单个 GPU，离线模式避免联网下载模型
            with log_path.open('ab') as log:
                process = subprocess.Popen(worker_args, cwd=ROOT / 'verl-official',
                    env=dict(os.environ, OPSD_GPUS=str(gpu), OPSD_NPROC='1', CUDA_VISIBLE_DEVICES=str(gpu),
                             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                             OMP_NUM_THREADS='1', VLLM_WORKER_MULTIPROC_METHOD='spawn'),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(process)
            workers.append(process)
            log_readers.append(log_path.open())  # 打开日志文件用于后续读取进度
        completed = [0] * 7  # 记录每个 worker 已完成的轨迹数
        status('collecting', question_count=sum(counts), target_trajectories=sum(counts) * 5,
               worker_pids=[worker.pid for worker in workers])
        run.config.update({'collection_question_count': sum(counts)})

        # 进度监控循环：每 30 秒检查一次
        # 采用 tail -f 模式：只读取日志文件中新增的行，不重新扫描整个 pool
        while True:
            for index, reader in enumerate(log_readers):
                while True:
                    position = reader.tell()
                    line = reader.readline()
                    if not line:
                        break  # 没有新行
                    if not line.endswith('\n'):
                        reader.seek(position)  # 半写入的行，下轮再读
                        break
                    try:
                        record = json.loads(line)
                        completed[index] = record.get('completed', completed[index])
                    except (ValueError, AttributeError):
                        pass  # 非 JSON 行忽略（如普通日志输出）
            # 检查是否有 worker 提前退出且失败
            codes = [worker.poll() for worker in workers]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'Collection worker failed: exit codes {codes}')
            # 记录进度到 W&B
            run.log({'collection/completed_trajectories': sum(completed),
                     'collection/fraction': sum(completed) / (sum(counts) * 5),
                     **{f'collection/gpu_{i+1}_completed': count for i, count in enumerate(completed)}})
            status('collecting', completed_trajectories=sum(completed), target_trajectories=sum(counts) * 5,
                   worker_pids=[worker.pid for worker in workers])
            if all(code == 0 for code in codes):
                break  # 全部 worker 正常完成
            time.sleep(30)
        for reader in log_readers:
            reader.close()
        # ======================== 合并轨迹池 ========================
        # 必须先验证完整性和去重，不允许部分数据静默流入训练
        # 同时支持合并两种后端的输出文件 (transformers 旧池 + vLLM 新池)
        with (base / 'pool.jsonl').open('x') as merged:  # 'x' 模式：文件已存在则报错
            for gpu, count in enumerate(counts, 1):
                actual = 0
                seen = set()  # 去重 key: (question_index, sibling)
                for pool in (base / f'pool_gpu_{gpu}.jsonl', base / f'pool_gpu_{gpu}.vllm.jsonl'):
                    if not pool.exists():
                        continue
                    with pool.open() as source:
                        for line in source:
                            row = json.loads(line)
                            key = (row['question_index'], row['sibling'])
                            if key in seen:
                                raise ValueError(f'Duplicate trajectory {key} on GPU {gpu}')
                            seen.add(key)
                            merged.write(line)
                            actual += 1
                # 严格校验：每个 GPU 必须恰好产出 count × 5 条轨迹
                if actual != count * 5:
                    raise RuntimeError(f'Incomplete collection on GPU {gpu}: {actual} != {count * 5}')
        status('mining')
        # ======================== 偏好挖掘 (Mining) ========================
        # 使用全部 7 张 GPU 从轨迹池中挖掘 preference pairs
        # 挖掘逻辑：同一问题的多条轨迹之间两两比较，质量好的 chosen vs 差的 rejected
        command('mine', ['mine', '--backend', 'gpu', '--pool', str(base / 'pool.jsonl'),
                         '--output', str(base / 'pairs.jsonl')], gpu='1,2,3,4,5,6,7')
        stats = json.loads((base / 'pairs.jsonl.stats.json').read_text())
        run.log({f'mining/{key}': value for key, value in stats.items()})
        # 最低保障：pair 数量必须 >= 560（一个 global batch 的要求）
        if stats['pairs'] < 560:
            raise RuntimeError(f'Only {stats["pairs"]} pairs; cannot form the requested batch of 560')
        # ======================== 分布式训练 (Training) ========================
        # 关键设计：先关闭 collection 阶段的 W&B run，再由训练 rank 0 用相同 ID 重新连接
        # 这样避免了多进程（7卡 DDP + 本控制器）同时写入同一 W&B run 的冲突
        run.finish()
        run = None
        status('training', global_batch_size=560, world_size=7, accumulation_steps=80)
        command('train', [
            'train', '--model', MODEL, '--pairs', str(base / 'pairs.jsonl'),
            '--output', str(base / 'train'), '--validation', str(base / 'qa/validation.jsonl'),
            '--validation-limit', '64',     # 验证集取 64 条快速评估
            '--batch-size', '560',           # 全局 batch size = 7 GPU × 80 累积步
            '--epochs', '1',
            '--save-steps', '10',            # 每 10 步保存 checkpoint + 最终权重
            '--max-length', '8192', '--eval-max-new-tokens', '2048', '--clip', '10',
            '--wandb-project', args.project, '--wandb-entity', entity,
            '--wandb-run-name', args.run_name, '--wandb-run-id', run_id,  # 复用同一 W&B run
        ], gpu='1,2,3,4,5,6,7', nproc=7)  # nproc=7 启动 7 个 DDP 进程
        # ======================== LoRA 适配器合并 ========================
        # 训练产出的 LoRA adapter 需要合并回基座模型，得到完整可部署的模型
        status('merging')
        adapter = json.loads((base / 'train/best.json').read_text())['adapter']
        command('merge', ['merge', '--model', MODEL, '--adapter', adapter, '--output', str(base / 'merged')])
        status('complete', merged_model=str(base / 'merged'))

    # ======================== 全局异常处理 ========================
    except BaseException as error:
        # 捕获所有异常（包括 KeyboardInterrupt），确保子进程被清理
        # 只终止本控制器启动的子进程，所有日志文件保留供诊断
        for process in children:
            if process.poll() is None:
                process.terminate()
        status('failed', error=str(error))
        if run:
            run.finish(exit_code=1)  # 标记 W&B run 为失败
        raise  # 重新抛出异常，让上层调用者也能感知到


if __name__ == '__main__':
    main()