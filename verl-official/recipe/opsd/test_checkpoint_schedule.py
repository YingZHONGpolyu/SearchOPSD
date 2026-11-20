"""Exercise the real optimizer loop on CPU without disturbing active GPU jobs."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .__main__ import parser
from . import train as training


class TinyPolicy(torch.nn.Module):
    """Replace only the large model, recording every real checkpoint write."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.saved = []

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def enable_input_require_grads(self):
        pass

    def save_pretrained(self, path):
        self.saved.append(path.name)
        path.mkdir(exist_ok=False)
        torch.save(self.state_dict(), path / 'test_weights.pt')


class TinyTokenizer:
    """Retain the tokenizer-write contract using a small marker file."""

    def save_pretrained(self, path):
        (path / 'tokenizer.json').write_text('{}')


class TinyObjective(torch.nn.Module):
    """Use a differentiable scalar loss with the unchanged AdamW training loop."""

    def __init__(self, policy, clip, chunk_size):
        super().__init__()
        self.policy = policy

    def forward(self, example):
        return (self.policy.weight - 1.0).square()


class CheckpointSchedule(unittest.TestCase):
    def run_case(self, examples, expected, max_steps=0):
        """Patch device/model boundaries; keep step counting and disk writes intact."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pairs = base / 'pairs.jsonl'
            pairs.write_text(''.join(json.dumps({'question': str(i)}) + '\n' for i in range(examples)))
            args = parser().parse_args([
                'train', '--model', 'test-model', '--pairs', str(pairs), '--output', str(base / 'train'),
                '--accumulation-steps', '1', '--max-steps', str(max_steps), '--smoke',
            ])
            self.assertEqual(args.save_steps, 10)
            policy = TinyPolicy()
            real_tensor, real_zeros = torch.tensor, torch.zeros

            def cpu_factory(factory):
                # Force temporary metric tensors onto CPU while preserving their values.
                def create(*values, **kwargs):
                    kwargs.pop('device', None)
                    return factory(*values, **kwargs)
                return create

            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0'}))
                stack.enter_context(patch.object(training, 'load_model', return_value=(policy, TinyTokenizer())))
                stack.enter_context(patch.object(training, 'get_peft_model', side_effect=lambda model, config: model))
                stack.enter_context(patch.object(training, 'OPSDObjective', TinyObjective))
                stack.enter_context(patch.object(training, 'encode_pair', return_value={
                    'student_positions': [0], 'student_ids': [1], 'teacher_ids': [1],
                }))
                stack.enter_context(patch.object(torch, 'tensor', side_effect=cpu_factory(real_tensor)))
                stack.enter_context(patch.object(torch, 'zeros', side_effect=cpu_factory(real_zeros)))
                for name in ('set_device', 'synchronize', 'reset_peak_memory_stats'):
                    stack.enter_context(patch.object(torch.cuda, name))
                for name in ('max_memory_allocated', 'max_memory_reserved'):
                    stack.enter_context(patch.object(torch.cuda, name, return_value=0))
                training.train(args)
            self.assertEqual(policy.saved, [f'step_{step}' for step in expected])
            self.assertEqual(json.loads((base / 'train/last_checkpoint.json').read_text())['step'], expected[-1])
            for step in expected:
                self.assertTrue((base / f'train/step_{step}/tokenizer.json').exists())

    def test_periodic_and_nonmultiple_final(self):
        self.run_case(23, [10, 20, 23])

    def test_multiple_final_is_not_saved_twice(self):
        self.run_case(20, [10, 20])

    def test_short_run_still_saves_final(self):
        self.run_case(3, [3])

    def test_max_steps_retains_final(self):
        self.run_case(23, [10, 11], max_steps=11)


if __name__ == '__main__':
    unittest.main()
