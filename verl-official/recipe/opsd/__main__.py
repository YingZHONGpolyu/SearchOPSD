"""Command-line stages: collect -> mine -> train -> evaluate -> merge."""

import argparse
import json
from pathlib import Path

from .core import mine_pairs, read_jsonl, write_jsonl


def parser():
    """Expose paper settings separately from unspecified reproduction defaults."""
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("gpu-smoke", help="Synthetic real-3B integration check, not a training result")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--questions", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--validation-fraction", type=float, default=0.02)
    prepare.add_argument("--seed", type=int, default=42)
    mine = sub.add_parser("mine")
    mine.add_argument("--pool", required=True)
    mine.add_argument("--output", required=True)
    # Production pair scoring uses all GPUs exposed by CUDA_VISIBLE_DEVICES.
    mine.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    mine.add_argument("--gpu-batch-size", type=int, default=1024)
    for name in ("collect", "train", "evaluate", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="Local Base checkpoint, normally after GRPO")
        p.add_argument("--output", required=True)
        if name in ("collect", "evaluate"):
            p.add_argument("--questions", required=True)
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--offset", type=int, default=0)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--max-new-tokens", type=int, default=2048)
        if name in ("evaluate", "merge"):
            p.add_argument("--adapter")
        if name in ("collect", "train", "evaluate"):
            p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
            p.add_argument("--max-searches", type=int, default=4)
            p.add_argument("--max-length", type=int, default=8192)
            p.add_argument("--final-answer-reserve", type=int, default=128)
        if name == "collect":
            p.add_argument("--k", type=int, default=5)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--resume", action="store_true",
                           help="Resume only a pool with matching provenance and sampling parameters")
        if name == "train":
            p.add_argument("--pairs", required=True)
            p.add_argument("--validation")
            p.add_argument("--validation-limit", type=int, default=64)
            p.add_argument("--eval-max-new-tokens", type=int, default=2048)
            # Paper reports clipping=10 but not these LoRA/optimization parameters.
            p.add_argument("--clip", type=float, default=10.0)
            p.add_argument("--lora-rank", type=int, default=16)
            p.add_argument("--lora-alpha", type=int, default=32)
            p.add_argument("--lr", type=float, default=1e-5)
            p.add_argument("--epochs", type=int, default=1)
            p.add_argument("--max-steps", type=int, default=0)
            # Count optimizer updates, not per-GPU micro-steps, for checkpoint retention.
            p.add_argument("--save-steps", type=int, default=10,
                           help="Save every N optimizer steps; always retain the final checkpoint")
            p.add_argument("--accumulation-steps", type=int, default=4)
            # Global pair batch: with one pair per GPU, 560 / 7 means 80 accumulations.
            p.add_argument("--batch-size", type=int, help="Global pair batch; must divide by WORLD_SIZE")
            # Only rank zero writes experiment metrics to the selected W&B run.
            p.add_argument("--wandb-project")
            p.add_argument("--wandb-entity")
            p.add_argument("--wandb-run-name")
            p.add_argument("--wandb-run-id")
            p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
            p.add_argument("--logit-chunk-size", type=int, default=32)
            p.add_argument("--max-grad-norm", type=float, default=1.0)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--smoke", action="store_true", help="Permit no EM selection for integration tests")
    return root


def main():
    """Import GPU dependencies only for commands that need them."""
    args = parser().parse_args()
    if args.command == "gpu-smoke":
        from .gpu_smoke import main as gpu_smoke

        gpu_smoke()
        return
    if args.command == "prepare":
        from .data import prepare

        prepare(args)
        return
    for field in ("k", "max_searches", "max_length", "max_new_tokens", "epochs", "lora_rank",
                  "lora_alpha", "accumulation_steps", "logit_chunk_size", "lr", "clip", "max_grad_norm", "save_steps"):
        if hasattr(args, field) and getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if hasattr(args, "final_answer_reserve") and args.final_answer_reserve < 0:
        raise ValueError("--final-answer-reserve must be nonnegative")
    if args.command == "mine":
        if args.backend == "gpu":
            from .gpu_mining import mine_file

            mine_file(args.pool, args.output, batch_size=args.gpu_batch_size)
            return
        pairs, stats = mine_pairs(read_jsonl(args.pool))
        if not pairs:
            raise ValueError(f"No usable siblings: {stats}")
        write_jsonl(args.output, pairs)
        Path(args.output + ".stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats))
        return
    if args.command == "train":
        from .train import train

        train(args)
        return
    # Fail before loading large weights when an output already exists.
    if Path(args.output).exists() and not (args.command == "collect" and args.resume):
        raise FileExistsError(args.output)
    from peft import PeftModel
    from transformers import set_seed

    from .rollout import collect, evaluate, load_questions
    from .train import load_model

    set_seed(getattr(args, "seed", 42))
    model, tokenizer = load_model(args.model)
    if getattr(args, "adapter", None):
        # Reject adapters trained against another frozen checkpoint.
        config = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        if Path(config["base_model_name_or_path"]).resolve() != Path(args.model).resolve():
            raise ValueError("Adapter base checkpoint differs from --model")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    if args.command == "merge":
        if not args.adapter:
            raise ValueError("Merging requires --adapter")
        Path(args.output).mkdir(parents=True)
        model = model.merge_and_unload()
        model.config.use_cache = True
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        return
    questions = load_questions(args.questions, args.limit, args.offset)
    kwargs = dict(retriever_url=args.retriever_url, max_searches=args.max_searches,
                  max_new_tokens=args.max_new_tokens, max_length=args.max_length,
                  final_answer_reserve=args.final_answer_reserve)
    if args.command == "collect":
        collect(model, tokenizer, questions, args.output, args.model, k=args.k,
                temperature=args.temperature, resume=args.resume, seed=args.seed, **kwargs)
    else:
        em, predictions = evaluate(model, tokenizer, questions, **kwargs)
        write_jsonl(args.output, predictions)
        print(json.dumps({"exact_match": em, "questions": len(predictions)}))


if __name__ == "__main__":
    main()
