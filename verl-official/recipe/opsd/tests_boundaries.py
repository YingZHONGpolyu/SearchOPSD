"""CPU regression checks for validation, finalization, budgets and durable collection."""

import fcntl
import json
import os
import random
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import NoBadWordsLogitsProcessor

from .__main__ import main, parser
from .core import student_prompt
from .data import prepare
from .rollout import EndOfAction, collect, generate_trajectory, load_questions, stratified_questions


class Tokenizer:
    """Use byte-sized character IDs for exact context and policy-budget assertions."""
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i != self.eos_token_id)


class ScriptedModel(torch.nn.Module):
    """Supply controlled generations while respecting the requested token budget."""
    def __init__(self, outputs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.outputs = iter(outputs)
        self.requests = []

    def generate(self, input_ids, **kwargs):
        self.requests.append(kwargs)
        text = next(self.outputs)
        ids = Tokenizer().encode(text)[:kwargs["max_new_tokens"]]
        return torch.cat((input_ids, torch.tensor([ids], dtype=torch.long)), dim=1)


def answer_result(text="Paris"):
    """A complete synthetic rollout; no model training or retrieval is involved."""
    trajectory = f"<answer>{text}</answer>"
    ids = Tokenizer().encode(trajectory)
    return {"trajectory": trajectory, "response_ids": ids, "policy_mask": [1] * len(ids),
            "search_calls": 0, "context_limit": False, "termination_reason": "answer"}


class ValidationBoundaries(unittest.TestCase):
    """Checkpoint selection must cover both source datasets, independent of file order."""
    def questions(self):
        return [{"question": f"{source}-{i}", "aliases": [str(i)], "data_source": source}
                for source in ("nq", "hotpotqa") for i in range(100)]

    def test_sorted_input_first64_is_balanced_and_order_independent(self):
        rows = self.questions()
        selected = stratified_questions(rows, 64, 42)
        self.assertEqual(Counter(x["data_source"] for x in selected), {"nq": 32, "hotpotqa": 32})
        self.assertEqual(selected, stratified_questions(list(reversed(rows)), 64, 42))
        self.assertNotEqual(selected, stratified_questions(rows, 64, 43))

    def test_load_preserves_source_and_applies_stratification(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "qa.jsonl"
            p.write_text("".join(json.dumps(x) + "\n" for x in self.questions()))
            selected = load_questions(p, 64, stratified=True)
        self.assertEqual(Counter(x["data_source"] for x in selected), {"nq": 32, "hotpotqa": 32})

    def test_legacy_unlabelled_validation_fails_instead_of_silently_biasing(self):
        with self.assertRaisesRegex(ValueError, "data_source"):
            stratified_questions([{"question": "q", "aliases": ["a"]}], 64)

    def test_small_limit_all_data_and_imbalanced_strata(self):
        rows = self.questions()
        with self.assertRaises(ValueError):
            stratified_questions(rows, 1)
        self.assertEqual(len(stratified_questions(rows, 0)), 200)
        tiny = [x for x in rows if x["data_source"] == "nq"] + [rows[-1]]
        chosen = stratified_questions(tiny, 64)
        self.assertEqual(Counter(x["data_source"] for x in chosen), {"nq": 63, "hotpotqa": 1})

    def test_prepare_retains_source_and_prevents_question_split_leakage(self):
        from argparse import Namespace
        rows = self.questions() * 2
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "qa"
            with patch("recipe.opsd.data.load_questions", return_value=rows):
                prepare(Namespace(questions="unused", output=str(out), validation_fraction=.2, seed=42))
            train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
            val = [json.loads(x) for x in (out / "validation.jsonl").read_text().splitlines()]
            self.assertFalse({x["question"] for x in train} & {x["question"] for x in val})
            self.assertEqual({x["data_source"] for x in val}, {"nq", "hotpotqa"})
            self.assertEqual(len(load_questions(out / "validation.jsonl", 16, stratified=True)), 16)


class RolloutBoundaries(unittest.TestCase):
    """Use synthetic actions to exercise failure paths without a model checkpoint."""
    def rollout(self, outputs, evidence="evidence", **kwargs):
        model = ScriptedModel(outputs)
        with patch("recipe.opsd.rollout.retrieve", return_value=evidence) as retriever:
            result = generate_trajectory(model, Tokenizer(), "q", "unused", **kwargs)
        self.assertEqual(len(result["response_ids"]), len(result["policy_mask"]))
        self.assertLessEqual(len(result["response_ids"]) + len(Tokenizer().encode(student_prompt("q"))),
                             kwargs.get("max_length", 8192))
        self.assertLessEqual(sum(result["policy_mask"]), kwargs.get("max_new_tokens", 2048))
        return model, retriever, result

    def test_four_searches_enter_terminal_phase_and_do_not_end_on_rejected_search(self):
        model, retriever, result = self.rollout(
            ["<search>q</search>"] * 5 + ["<answer>Paris</answer>"], max_searches=4)
        self.assertEqual(retriever.call_count, 4)
        self.assertEqual(result["search_calls"], 4)
        self.assertEqual(result["termination_reason"], "answer")
        self.assertTrue(result["forced_final"])
        final_request = model.requests[4]
        search = torch.tensor([Tokenizer().encode("<search>q</search>")])
        answer = torch.tensor([Tokenizer().encode("<answer>Paris</answer>")])
        # Construct a final-phase stopping criterion at suffix offset zero.
        stop = EndOfAction(Tokenizer(), 0, allow_search=False)
        self.assertFalse(stop(search, None))
        self.assertTrue(stop(answer, None))
        self.assertIn(Tokenizer().encode("<search>"), final_request["bad_words_ids"])
        self.assertIn(0, result["policy_mask"])

    def test_final_phase_bad_words_really_block_search_tag_completion(self):
        model, _, _ = self.rollout(["<search>q</search>", "<answer>Paris</answer>"], max_searches=1)
        processor = NoBadWordsLogitsProcessor(model.requests[1]["bad_words_ids"], eos_token_id=0)
        ids = torch.tensor([Tokenizer().encode("context<search")])
        scores = processor(ids, torch.zeros((1, 256)))
        self.assertTrue(torch.isneginf(scores[0, ord(">")]))
        self.assertTrue(torch.isfinite(scores[0, ord("a")]))

    def test_oversized_observation_is_rejected_whole_and_flagged(self):
        _, retriever, result = self.rollout(["<search>x</search>"], evidence="x" * 2000,
                                           max_length=1024, max_new_tokens=64, final_answer_reserve=0)
        self.assertEqual(retriever.call_count, 1)
        self.assertTrue(result["context_limit"])
        self.assertEqual(result["termination_reason"], "observation_context_limit")
        self.assertNotIn("x" * 100, result["trajectory"])

    def test_last_policy_token_cannot_append_oversized_observation_unflagged(self):
        _, _, result = self.rollout(["<search>x</search>"], evidence="x" * 2000,
                                    max_length=1024, max_new_tokens=18, final_answer_reserve=0)
        self.assertTrue(result["context_limit"])
        self.assertEqual(result["termination_reason"], "observation_context_limit")

    def test_final_answer_reserve_does_not_expand_total_token_budget(self):
        model, _, result = self.rollout(["x" * 200, "y" * 200], max_new_tokens=128)
        self.assertEqual(result["policy_tokens"], 128)
        self.assertEqual(sum(result["policy_mask"]), 128)
        self.assertEqual(result["termination_reason"], "policy_token_limit")
        self.assertIn("bad_words_ids", model.requests[-1])
        self.assertNotIn("<answer>", result["trajectory"])

    def test_eos_without_answer_is_explicit_failure(self):
        _, _, result = self.rollout(["\x00"])
        self.assertEqual(result["termination_reason"], "eos_without_answer")

    def test_prompt_too_long_does_not_call_generator(self):
        model = ScriptedModel([])
        result = generate_trajectory(model, Tokenizer(), "q", "unused", max_length=1)
        self.assertTrue(result["context_limit"])
        self.assertEqual(result["termination_reason"], "context_limit")
        self.assertFalse(model.requests)

    def test_notice_cannot_overrun_context(self):
        maximum = len(Tokenizer().encode(student_prompt("q"))) + 1
        model = ScriptedModel([])
        result = generate_trajectory(model, Tokenizer(), "q", "unused", max_length=maximum)
        self.assertTrue(result["context_limit"])
        self.assertEqual(result["termination_reason"], "final_prompt_context_limit")
        self.assertFalse(model.requests)


class CollectionBoundaries(unittest.TestCase):
    """Interrupted pools must resume deterministically and reject incompatible inputs."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path = self.folder / "pool.jsonl"
        self.questions = [{"question": "q", "aliases": ["Paris"], "data_source": "nq"}]
        self.kwargs = dict(retriever_url="unused", temperature=1., max_new_tokens=2048,
                           max_length=8192, max_searches=4, final_answer_reserve=128)

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *, output=None, questions=None, model="model", k=3, seed=42, resume=False, **overrides):
        params = {**self.kwargs, **overrides}
        return collect(None, Tokenizer(), questions or self.questions, output or self.path,
                       model, k=k, seed=seed, resume=resume, **params)

    def interrupted(self):
        with patch("recipe.opsd.rollout.generate_trajectory", side_effect=[answer_result(), TimeoutError("timeout")]):
            with self.assertRaises(TimeoutError):
                self.invoke()
        self.assertEqual(len(self.path.read_text().splitlines()), 1)

    def test_interruption_resume_matches_uninterrupted_rng_and_skips_completed(self):
        def random_answer(*args, **kwargs):
            return answer_result(str(random.random()) + ":" + str(torch.rand(()).item()))
        full = self.folder / "full.jsonl"
        with patch("recipe.opsd.rollout.generate_trajectory", side_effect=random_answer):
            self.invoke(output=full)
        calls = 0
        def failing(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                random.random()
                raise TimeoutError("temporary")
            return random_answer()
        with patch("recipe.opsd.rollout.generate_trajectory", side_effect=failing):
            with self.assertRaises(TimeoutError):
                self.invoke()
        with patch("recipe.opsd.rollout.generate_trajectory", side_effect=random_answer) as generator:
            self.invoke(resume=True)
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(self.path.read_text(), full.read_text())
        with patch("recipe.opsd.rollout.generate_trajectory") as generator:
            self.invoke(resume=True)
            generator.assert_not_called()

    def test_resume_requires_opt_in(self):
        self.interrupted()
        with self.assertRaises(FileExistsError):
            self.invoke()

    def test_changed_checkpoint_questions_seed_k_or_generation_parameters_fail(self):
        self.interrupted()
        cases = [dict(model="other_model"), dict(k=4), dict(seed=43),
                 dict(questions=[{"question": "other", "aliases": ["Paris"]}]),
                 dict(temperature=.5), dict(max_length=4096), dict(max_new_tokens=1024),
                 dict(max_searches=3), dict(retriever_url="other"), dict(final_answer_reserve=64)]
        for override in cases:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "protocol mismatch"):
                self.invoke(resume=True, **override)

    def test_changed_weights_metadata_at_same_path_is_rejected(self):
        model = self.folder / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model":"before"}')
        with patch("recipe.opsd.rollout.generate_trajectory", return_value=answer_result()):
            self.invoke(model=str(model))
        (model / "config.json").write_text('{"model":"after"}')
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            self.invoke(model=str(model), resume=True)

    def test_incomplete_trailing_fragment_is_recovered(self):
        self.interrupted()
        with self.path.open("ab") as stream:
            stream.write(b'{"incomplete":')
        with patch("recipe.opsd.rollout.generate_trajectory", return_value=answer_result()):
            self.invoke(resume=True)
        rows = [json.loads(x) for x in self.path.read_text().splitlines()]
        self.assertEqual([x["sibling"] for x in rows], [0, 1, 2])

    def test_complete_last_record_without_newline_is_not_regenerated(self):
        self.interrupted()
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        with patch("recipe.opsd.rollout.generate_trajectory", return_value=answer_result()) as generator:
            self.invoke(resume=True)
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(len(self.path.read_text().splitlines()), 3)

    def test_committed_corruption_and_duplicates_are_not_silently_discarded(self):
        self.interrupted()
        saved = self.path.read_bytes()
        for tail in (b'not-json\n', saved):
            self.path.write_bytes(saved + tail)
            with self.subTest(tail=tail[:20]), self.assertRaises(ValueError):
                self.invoke(resume=True)

    def test_legacy_pool_without_manifest_is_not_adopted(self):
        self.path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, "Legacy pool"):
            self.invoke(resume=True)

    def test_second_writer_is_rejected(self):
        lock = self.path.with_name(self.path.name + ".lock")
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another collector"):
                self.invoke()

    def test_cli_resume_reaches_collector_and_keeps_default_sampling_parameters(self):
        qa = self.folder / "qa.jsonl"
        qa.write_text(json.dumps(self.questions[0]) + "\n")
        model = ScriptedModel(["<answer>Paris</answer>"])
        argv = ["opsd", "collect", "--model", "model", "--questions", str(qa),
                "--output", str(self.path), "--k", "1", "--retriever-url", "unused"]
        with patch("recipe.opsd.train.load_model", return_value=(model, Tokenizer())):
            with patch("sys.argv", argv):
                main()
            with patch("sys.argv", argv + ["--resume"]):
                main()
        self.assertEqual(len(model.requests), 1)
        self.assertTrue(parser().parse_args(argv[1:] + ["--resume"]).resume)


class LauncherBoundaries(unittest.TestCase):
    """Exercise shell orchestration with a fake runner that cannot train any model."""
    def test_collect_only_and_resume_never_invoke_training_and_default_to_step250(self):
        launcher = Path(os.environ.get("OPSD_TEST_LAUNCHER", Path(__file__).parents[3] / "run_opsd_round.sh"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "run_opsd_round.sh"
            target.write_text(launcher.read_text())
            runner = root / "run_opsd.sh"
            runner.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
root=Path(__file__).parent
args=sys.argv[1:]
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps(args)+'\\n')
assert args[0] in ('prepare','collect'), 'Training is forbidden in this fixture'
out=args[args.index('--output')+1].replace('/workspace/outputs/',str(root/'outputs')+'/')
if args[0]=='prepare':
 p=Path(out);p.mkdir(parents=True);(p/'train.jsonl').write_text('{}\\n');(p/'validation.jsonl').write_text('{}\\n')
else:
 Path(out).write_text('partial\\n')
 if os.environ.get('FAIL_COLLECT')=='1':sys.exit(7)
''')
            runner.chmod(0o755)
            env = os.environ.copy()
            for name in list(env):
                if name.startswith("OPSD_"):
                    env.pop(name)
            env.update(OPSD_RUN="boundary", OPSD_COLLECT_ONLY="1", FAIL_COLLECT="1")
            first = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 7, first.stderr)
            env.update(OPSD_RESUME="1", FAIL_COLLECT="0")
            second = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            calls = [json.loads(x) for x in (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual([x[0] for x in calls], ["prepare", "collect", "collect"])
            self.assertIn("--resume", calls[-1])
            self.assertTrue(calls[-1][calls[-1].index("--model") + 1].endswith("global_step_250"))
            self.assertIn("no training requested", second.stdout)
            (root / "outputs/opsd/boundary/train").mkdir()
            rejected = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("downstream outputs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
