"""Evaluate one disjoint test shard with continuous batching and real retrieval."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from transformers import AutoTokenizer
from . import rollout
from .core import exact_match, read_jsonl
from .rollout_vllm import BatchedRetriever, VLLMModel


async def evaluate(args):
    """Generate once per original row; answer aliases are used only after inference."""
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    questions = read_jsonl(args.questions)
    if not questions:
        raise ValueError('Empty evaluation shard')
    # Exclusive output creation prevents accidental reuse of a different checkpoint's results.
    with Path(args.output).open('x', encoding='utf-8') as stream:
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, tokenizer=args.model, dtype='bfloat16', tensor_parallel_size=1,
            gpu_memory_utilization=0.88, max_model_len=8192, max_num_seqs=args.concurrency,
            max_num_batched_tokens=16384, enable_prefix_caching=True,
        ))
        loop = asyncio.get_running_loop()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        adapter = VLLMModel(engine, tokenizer, loop)
        retriever = BatchedRetriever(loop, args.retriever_url)
        rollout.retrieve = retriever.retrieve
        retrieval_task = asyncio.create_task(retriever.serve())
        pending = iter(questions)
        executor = ThreadPoolExecutor(max_workers=args.concurrency)
        completed, correct, started = 0, 0, time.monotonic()

        def generate(item):
            # Only the question enters the student prompt; each row gets a unique request ID.
            adapter.begin(42, f"test-{item['row_index']}")
            result = rollout.generate_trajectory(adapter, tokenizer, item['question'],
                retriever_url=args.retriever_url, temperature=0.0, max_searches=4,
                max_new_tokens=2048, max_length=8192, final_answer_reserve=128)
            # Token arrays are training artifacts; keep full readable trajectories for test auditing.
            for key in ('input_ids', 'response_ids', 'policy_mask', 'prompt_ids'):
                result.pop(key, None)
            result.update(item, reward=exact_match(result['trajectory'], item['aliases']))
            return result

        async def worker():
            nonlocal completed, correct
            while True:
                item = next(pending, None)
                if item is None:
                    return
                result = await loop.run_in_executor(executor, generate, item)
                # Writes occur on the event loop so concurrent requests cannot interleave JSON lines.
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                completed += 1
                correct += result['reward']
                if completed % 50 == 0 or completed == len(questions):
                    progress = dict(completed=completed, total=len(questions), correct=correct,
                                    elapsed_seconds=time.monotonic() - started)
                    temporary = Path(args.output + '.progress.tmp')
                    temporary.write_text(json.dumps(progress))
                    temporary.replace(args.output + '.progress.json')
                    print(json.dumps(progress), flush=True)

        workers = [asyncio.create_task(worker()) for _ in range(min(args.concurrency, len(questions)))]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                task.cancel()
            retrieval_task.cancel()
            engine.shutdown()
            executor.shutdown(wait=False, cancel_futures=True)


def main():
    """Expose only evaluation inputs, without sampling or optimizer configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'questions', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--concurrency', type=int, default=128)
    parser.add_argument('--retriever-url', default='http://127.0.0.1:8123/retrieve')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('concurrency must be positive')
    asyncio.run(evaluate(args))


if __name__ == '__main__':
    main()
