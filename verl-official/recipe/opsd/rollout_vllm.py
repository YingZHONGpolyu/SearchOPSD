"""Continuous-batch OPSD collection, reusing the tested search/mask state machine.

Each trajectory runs its CPU orchestration independently. vLLM schedules their
generation requests together; retrieval queries are combined into small batches.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

import torch
from transformers import AutoTokenizer

from . import rollout
from .collection import collection_manifest, collection_output, persist_trajectory, recover_completed, trajectory_seed
from .core import exact_match

# Use vLLM's seeded Triton sampler; the host nvcc is older than its FlashInfer JIT requirements.
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')


class BatchedRetriever:
    """Combine concurrent search actions, without blocking the inference event loop."""

    def __init__(self, loop, url, batch_size=32):
        self.loop, self.url, self.batch_size = loop, url, batch_size
        self.queue = asyncio.Queue()

    def retrieve(self, url, query, topk=3):
        if url != self.url or topk != 3:
            raise ValueError('Unexpected retrieval protocol')
        return asyncio.run_coroutine_threadsafe(self.submit(query), self.loop).result(timeout=900)

    async def submit(self, query):
        future = self.loop.create_future()
        await self.queue.put((query, future))
        return await future

    def request(self, queries):
        payload = json.dumps({'queries': queries, 'topk': 3, 'return_scores': True}).encode()
        request = urllib.request.Request(self.url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            rows = json.load(response)['result']
        if len(rows) != len(queries):
            raise ValueError('Retrieval batch response length differs from the request')
        output = []
        for docs in rows:
            formatted = []
            for index, hit in enumerate(docs[:3], 1):
                doc = hit.get('document', hit)
                contents = doc.get('contents', doc.get('text', ''))
                if not contents:
                    raise ValueError('Retriever returned an empty document')
                contents = contents.replace('<', '&lt;').replace('>', '&gt;')
                formatted.append(f'Doc {index}: {contents}')
            output.append('\n'.join(formatted))
        return output

    async def serve(self):
        while True:
            batch = [await self.queue.get()]
            await asyncio.sleep(0.01)
            while len(batch) < self.batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            try:
                outputs = await asyncio.to_thread(self.request, [item[0] for item in batch])
                for (_, future), output in zip(batch, outputs):
                    if not future.done():
                        future.set_result(output)
            except Exception as error:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)


class VLLMModel(torch.nn.Module):
    """Adapt HF-style action calls to the asynchronous, continuous-batch engine."""

    def __init__(self, engine, tokenizer, loop):
        super().__init__()
        self.engine, self.tokenizer, self.loop = engine, tokenizer, loop
        # The orchestration only needs a device marker, not another copy of model weights.
        self.marker = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.local = threading.local()

    def begin(self, sample_seed, key):
        self.local.seed, self.local.action, self.local.key = sample_seed, 0, key

    async def action(self, ids, kwargs, seed, request_id):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        allow_search = kwargs['stopping_criteria'][0].allow_search
        parameters = SamplingParams(
            temperature=kwargs.get('temperature', 0.0), top_p=1.0, top_k=-1,
            max_tokens=kwargs['max_new_tokens'], seed=seed,
            stop=['</answer>', '</search>'] if allow_search else ['</answer>'],
            include_stop_str_in_output=True,
            bad_words=['<search>', ' <search>', '\n<search>'] if not allow_search else None,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        final = None
        async for result in self.engine.generate({'prompt_token_ids': ids}, parameters, request_id):
            final = result
        if final is None:
            raise RuntimeError('vLLM produced no final output')
        return list(final.outputs[0].token_ids)

    def generate(self, **kwargs):
        ids = kwargs['input_ids'][0].tolist()
        action = self.local.action
        self.local.action += 1
        # Each action has an explicit seed; concurrent scheduling cannot consume another sample's RNG.
        seed = int.from_bytes(hashlib.sha256(f'{self.local.seed}:{action}'.encode()).digest()[:4], 'big')
        tokens = asyncio.run_coroutine_threadsafe(
            self.action(ids, kwargs, seed, f'{self.local.key}-{action}'), self.loop,
        ).result(timeout=900)
        return torch.tensor([ids + tokens])


@contextmanager
def legacy_pool(path, model, questions, k, seed, parameters):
    """Validate and lock the original pool; never rewrite its provenance manifest."""
    if path is None or not path.exists():
        yield set(), None
        return
    with Path(str(path) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(Path(str(path) + '.meta.json').read_text())
        if manifest != collection_manifest(model, questions, k, seed, parameters):
            raise ValueError('Legacy checkpoint/questions/sampler protocol mismatch')
        completed = recover_completed(path, questions, k, seed, str(Path(model).resolve()))
        yield completed, hashlib.sha256(path.read_bytes()).hexdigest()


async def collect(args):
    """Keep up to concurrency trajectories active and persist every completed result."""
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    import vllm

    questions = rollout.load_questions(args.questions, args.limit)
    parameters = dict(retriever_url=args.retriever_url, max_searches=args.max_searches,
                      max_new_tokens=args.max_new_tokens, max_length=args.max_length,
                      final_answer_reserve=args.final_answer_reserve, temperature=args.temperature)
    with legacy_pool(args.existing_pool, args.model, questions, args.k, args.seed, parameters) as (previous, digest):
        protocol = dict(parameters, backend='vllm', backend_version=vllm.__version__,
                        backend_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        predecessor_sha256=digest)
        with collection_output(args.output, args.model, questions, args.k, args.seed, protocol, args.resume) as state:
            stream, completed, model_path = state
            if completed & previous:
                raise ValueError('New and legacy pools contain duplicate trajectory IDs')
            done = completed | previous
            pending = iter((index, sibling) for index in range(len(questions))
                           for sibling in range(args.k) if (index, sibling) not in done)
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                model=args.model, tokenizer=args.model, dtype='bfloat16', tensor_parallel_size=1,
                gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_length,
                max_num_seqs=args.concurrency, max_num_batched_tokens=16384,
                enable_prefix_caching=True, enforce_eager=args.enforce_eager,
            ))
            loop = asyncio.get_running_loop()
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
            adapter = VLLMModel(engine, tokenizer, loop)
            retriever = BatchedRetriever(loop, args.retriever_url)
            rollout.retrieve = retriever.retrieve
            retrieval_task = asyncio.create_task(retriever.serve())
            workers = []
            started, initial = time.monotonic(), len(done)

            def generate(index, sibling):
                item = questions[index]
                sample_seed = trajectory_seed(args.seed, index, item['question'], sibling)
                adapter.begin(sample_seed, f'{index}-{sibling}')
                result = rollout.generate_trajectory(adapter, tokenizer, item['question'], **parameters)
                result.update(item, sibling=sibling, question_index=index, policy_model=model_path,
                              sampling_seed=sample_seed, sampler_backend='vllm')
                result['reward'] = exact_match(result['trajectory'], item['aliases'])
                return result

            executor = ThreadPoolExecutor(max_workers=args.concurrency)
            async def worker():
                while True:
                    key = next(pending, None)
                    if key is None:
                        return
                    result = await loop.run_in_executor(executor, generate, *key)
                    persist_trajectory(stream, result)
                    done.add(key)
                    print(json.dumps({'completed': len(done), 'total': len(questions) * args.k,
                                      'reward': result['reward'], 'policy_tokens': result['policy_tokens'],
                                      'trajectories_per_second': (len(done) - initial) / (time.monotonic() - started)}),
                          flush=True)
            try:
                workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    task.cancel()
                retrieval_task.cancel()
                engine.shutdown()
                executor.shutdown(wait=False, cancel_futures=True)


def main():
    """Expose inference concurrency separately from the optimizer's global batch size."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'questions', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--existing-pool', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--final-answer-reserve', type=int, default=128)
    parser.add_argument('--retriever-url', default='http://127.0.0.1:8123/retrieve')
    parser.add_argument('--concurrency', type=int, default=128)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.88)
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args()
    if args.concurrency <= 0 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('Invalid concurrency or GPU memory utilization')
    asyncio.run(collect(args))


if __name__ == '__main__':
    main()
