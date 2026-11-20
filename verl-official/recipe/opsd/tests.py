"""Numerical and scientific-contract tests; no model download or GPU required."""

import unittest
from unittest.mock import patch

import torch

from .core import encode_pair, exact_match, mine_pairs, response_tokens
from .loss import clipped_forward_kl, hidden_forward_kl


class CharacterTokenizer:
    """A transparent tokenizer makes prefix alignment and masking expectations exact."""

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ScientificContracts(unittest.TestCase):
    """Test paper decisions whose mistakes would silently change the algorithm."""

    def test_mining_prioritizes_search_then_length_and_wrong_sibling(self):
        rows = [
            {"question": "q", "trajectory": "correct but long", "reward": 1, "search_calls": 0},
            {"question": "q", "trajectory": "ref", "reward": 1, "search_calls": 0},
            {"question": "q", "trajectory": "r", "reward": 1, "search_calls": 1},
            {"question": "q", "trajectory": "wrong and different", "reward": 0, "search_calls": 0},
            {"question": "bad", "trajectory": "wrong", "reward": 0},
        ]
        pairs, stats = mine_pairs(rows)
        self.assertEqual(pairs[0]["reference"], "ref")
        self.assertEqual(pairs[0]["student"], "wrong and different")
        self.assertEqual(stats["no_correct"], 1)

    def test_all_correct_fallback_and_singleton(self):
        pairs, stats = mine_pairs([
            {"question": "q", "trajectory": "a", "reward": 1},
            {"question": "q", "trajectory": "bbb", "reward": 1},
            {"question": "alone", "trajectory": "a", "reward": 1},
        ])
        self.assertEqual(pairs[0]["student"], "bbb")
        self.assertEqual(stats["no_sibling"], 1)

    def test_largest_difference_among_wrong_siblings(self):
        pairs, _ = mine_pairs([
            {"question": "q", "trajectory": "aaaa", "reward": 1},
            {"question": "q", "trajectory": "aaab", "reward": 0},
            {"question": "q", "trajectory": "zzzz", "reward": 0},
        ])
        self.assertEqual(pairs[0]["student"], "zzzz")

    def test_environment_exclusion_and_causal_alignment(self):
        text = "<think>x</think><information>secret</information><answer>y</answer>"
        tokenizer = CharacterTokenizer()
        pair = {"question": "q", "reference": "reference", "student": text}
        ids, mask = response_tokens(tokenizer, pair)
        self.assertEqual("".join(chr(i) for i, active in zip(ids, mask) if active),
                         "<think>x</think><answer>y</answer>")
        encoded = encode_pair(tokenizer, pair, 10000)
        for s, t in zip(encoded["student_positions"], encoded["teacher_positions"]):
            self.assertEqual(encoded["student_ids"][s + 1], encoded["teacher_ids"][t + 1])
        self.assertIsNone(encode_pair(tokenizer, pair, 10))

    def test_exact_sampled_ids_and_open_information_span(self):
        pair = {"student": "ignored", "response_ids": [9, 8, 7], "policy_mask": [1, 0, 1]}
        self.assertEqual(response_tokens(CharacterTokenizer(), pair), ([9, 8, 7], [1, 0, 1]))
        _, mask = response_tokens(CharacterTokenizer(), {"student": "x<information>hidden"})
        self.assertEqual(sum(mask), 1)

    def test_em_aliases_last_answer_and_no_format_bonus(self):
        self.assertEqual(exact_match("<answer>bad</answer><answer>The Paris!</answer>", ["Paris"]), 1)
        self.assertEqual(exact_match("Paris", ["Paris"]), 0)
        self.assertEqual(exact_match("<answer>London</answer>", ["Paris"]), 0)

    def test_forward_direction_pointwise_clip_and_stop_gradient(self):
        student = torch.tensor([[-20.0, 0.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[4.0, 1.0, 0.0]], requires_grad=True)
        clip = 0.25
        lp, lq = student.log_softmax(-1), teacher.detach().log_softmax(-1)
        expected = (lq.exp() * (lq - lp)).clamp(max=clip).sum(-1).mean()
        result = clipped_forward_kl(student, teacher, clip)
        self.assertTrue(torch.allclose(result, expected))
        self.assertFalse(torch.allclose(result, (lq.exp() * (lq - lp)).sum(-1).clamp(max=clip).mean()))
        result.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_chunked_loss_and_gradient_equal_dense_full_vocabulary(self):
        torch.manual_seed(8)
        head = torch.nn.Linear(7, 31, bias=False).requires_grad_(False)
        student = torch.randn(11, 7, requires_grad=True)
        teacher = torch.randn(11, 7)
        dense = clipped_forward_kl(head(student), head(teacher), clip=0.1)
        dense_gradient, = torch.autograd.grad(dense, student)
        chunked = hidden_forward_kl(student, teacher, head, clip=0.1, chunk_size=3)
        chunked.backward()
        self.assertTrue(torch.allclose(dense, chunked, atol=1e-6))
        self.assertTrue(torch.allclose(dense_gradient, student.grad, atol=1e-6))

    def test_rollout_search_cap_and_environment_mask(self):
        from .rollout import generate_trajectory

        class StubModel(torch.nn.Module):
            # Real generation is covered separately by the GPU integration smoke test.
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def generate(self, input_ids, **kwargs):
                text = "<answer>done</answer>" if "bad_words_ids" in kwargs else "<search>q</search>"
                suffix = torch.tensor([list(map(ord, text))])
                return torch.cat((input_ids, suffix), dim=1)

        class DecodeTokenizer(CharacterTokenizer):
            eos_token_id = 0

            def decode(self, tokens, **kwargs):
                return "".join(map(chr, tokens))

        with patch("recipe.opsd.rollout.retrieve", return_value="evidence") as retriever:
            result = generate_trajectory(StubModel(), DecodeTokenizer(), "q", "unused", max_searches=1)
        self.assertEqual(retriever.call_count, 1)
        self.assertEqual(result["search_calls"], 1)
        self.assertIn(0, result["policy_mask"])
        self.assertEqual(len(result["policy_mask"]), len(result["response_ids"]))


if __name__ == "__main__":
    unittest.main()
