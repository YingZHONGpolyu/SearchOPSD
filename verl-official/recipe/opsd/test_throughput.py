"""Check concurrent adapter semantics and preservation of existing trajectory IDs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import rollout
from .collection import collection_output, persist_trajectory, trajectory_seed
from .rollout_vllm import BatchedRetriever, VLLMModel, legacy_pool


class CharTokenizer:
    """Small tokenizer makes policy/environment boundaries visible to assertions."""
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class FakeEngine:
    """Return a search action then an answer for each independently seeded request."""
    def __init__(self):
        self.requests = []

    async def generate(self, prompt, parameters, request_id):
        self.requests.append((parameters, request_id))
        text = '<search>France capital</search>' if request_id.endswith('-0') else '<answer>Paris</answer>'
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(map(ord, text)))])


class ThroughputContracts(unittest.TestCase):
    def test_adapter_reuses_budget_mask_and_final_phase(self):
        async def exercise():
            loop = asyncio.get_running_loop()
            engine = FakeEngine()
            adapter = VLLMModel(engine, CharTokenizer(), loop)
            def generate():
                adapter.begin(42, 'sample')
                return rollout.generate_trajectory(adapter, adapter.tokenizer, 'capital?', 'unused',
                                                   max_searches=1, max_length=4096, max_new_tokens=256)
            with patch.object(rollout, 'retrieve', return_value='Paris <capital>'):
                with ThreadPoolExecutor(1) as executor:
                    result = await loop.run_in_executor(executor, generate)
            self.assertEqual(result['search_calls'], 1)
            self.assertEqual(result['termination_reason'], 'answer')
            self.assertIn(0, result['policy_mask'])
            self.assertEqual(len(result['response_ids']), len(result['policy_mask']))
            self.assertLessEqual(result['policy_tokens'], 256)
            self.assertNotEqual(engine.requests[0][0].seed, engine.requests[1][0].seed)
            self.assertIn('<search>', engine.requests[1][0].bad_words)
        asyncio.run(exercise())

    def test_retrieval_queries_are_batched(self):
        async def exercise():
            batcher = BatchedRetriever(asyncio.get_running_loop(), 'unused')
            observed = []
            def request(queries):
                observed.append(queries)
                return ['evidence:' + query for query in queries]
            with patch.object(batcher, 'request', request):
                server = asyncio.create_task(batcher.serve())
                values = await asyncio.gather(*(batcher.submit(str(i)) for i in range(16)))
                server.cancel()
            self.assertEqual(len(observed), 1)
            self.assertEqual(values, ['evidence:' + str(i) for i in range(16)])
        asyncio.run(exercise())

    def test_legacy_pool_ids_remain_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model'
            model.mkdir()
            path = Path(tmp) / 'pool.jsonl'
            questions = [{'question': 'q', 'aliases': ['a']}]
            params = {'temperature': 1.0}
            with collection_output(path, str(model), questions, 5, 42, params, False) as state:
                stream, _, policy = state
                persist_trajectory(stream, dict(questions[0], question_index=0, sibling=2,
                    sampling_seed=trajectory_seed(42, 0, 'q', 2), policy_model=policy,
                    reward=1, trajectory='a', response_ids=[1], policy_mask=[1]))
            before = path.read_bytes()
            with legacy_pool(path, str(model), questions, 5, 42, params) as (completed, digest):
                self.assertEqual(completed, {(0, 2)})
                self.assertEqual(len(digest), 64)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                with legacy_pool(path, str(model), questions, 5, 42, {'temperature': 0.5}):
                    pass


if __name__ == '__main__':
    unittest.main()
