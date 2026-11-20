"""Single-GPU/DDP LoRA OPSD training in the existing veRL runtime.

This intentionally bypasses veRL's external-teacher OPD actor: Search-E1 needs
two different prefixes, the SAME suffix and an adapter-disabled frozen teacher.
No GRPO internals or existing environment dependencies are modified.
"""

import json
import math
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .core import encode_pair, read_jsonl, write_jsonl
from .loss import hidden_forward_kl
from .rollout import evaluate, load_questions


def load_model(path):
    """Use local Qwen2.5 Base weights and a fast tokenizer without remote code."""
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))))
    model.config.use_cache = False
    return model, tokenizer


class OPSDObjective(torch.nn.Module):
    """Shared-base teacher and student with independently aligned causal positions."""

    def __init__(self, model, clip, chunk_size):
        super().__init__()
        self.policy = model
        self.clip = clip
        self.chunk_size = chunk_size

    def forward(self, example):
        device = next(self.policy.parameters()).device
        student_ids = torch.tensor([example["student_ids"]], device=device)
        teacher_ids = torch.tensor([example["teacher_ids"]], device=device)
        # Qwen2ForCausalLM.model returns hidden states without allocating full logits.
        backbone = self.policy.get_base_model().model
        head = self.policy.get_output_embeddings()
        self.policy.eval()
        with torch.no_grad(), self.policy.disable_adapter():
            teacher_hidden = backbone(input_ids=teacher_ids, use_cache=False).last_hidden_state
            teacher_hidden = teacher_hidden[0, example["teacher_positions"]].detach()
        # Train only the LoRA path. Qwen2.5's attention dropout and LoRA dropout are zero.
        self.policy.train()
        student_hidden = backbone(input_ids=student_ids, use_cache=False).last_hidden_state
        student_hidden = student_hidden[0, example["student_positions"]]
        return hidden_forward_kl(student_hidden, teacher_hidden, head, self.clip, self.chunk_size, with_entropy=True)


def train(args):
    """Average Eq. (7) over pairs; select adapters by held-out generation EM."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    # Also validate programmatic callers that bypass the CLI's positive-value checks.
    if args.save_steps <= 0:
        raise ValueError("--save-steps must be positive")
    # Interpret batch_size as a global pair count, preserving the existing micro-batch of one.
    if args.batch_size is not None:
        if args.batch_size <= 0 or args.batch_size % world:
            raise ValueError("--batch-size must be positive and divisible by WORLD_SIZE")
        args.accumulation_steps = args.batch_size // world
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        # Rank-zero search evaluation may take much longer than one training step.
        dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()),
                                timeout=timedelta(hours=24))
    set_seed(args.seed)
    output = Path(args.output)
    # Every process checks the same fresh path before rank zero creates it.
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if world > 1:
        dist.barrier()
    if rank == 0:
        output.mkdir(parents=True)
        (output / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    # A single writer avoids seven duplicate runs and duplicated metric points.
    tracker = None
    if rank == 0 and args.wandb_project:
        from .tracking import start_run

        tracker = start_run(args, output, world)
    model, tokenizer = load_model(args.model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pairs = read_jsonl(args.pairs)
    if not pairs:
        raise ValueError("No mined pairs: collect a competent frozen GRPO policy first")
    encoded = []
    skipped = 0
    for pair in pairs:
        # Pair provenance prevents accidental off-policy training against another checkpoint.
        provenance = pair.get("policy_model")
        if not provenance and not args.smoke:
            raise ValueError("Scientific runs require policy_model provenance from collect/mine")
        if provenance and Path(provenance).resolve() != Path(args.model).resolve():
            raise ValueError("Mined pool policy_model differs from the frozen teacher checkpoint")
        item = encode_pair(tokenizer, pair, args.max_length)
        if item is None:
            skipped += 1
        else:
            encoded.append(item)
    if not encoded:
        raise ValueError("No usable policy tokens; inspect masks or increase --max-length")
    validation = load_questions(args.validation, args.validation_limit,
                                stratified=True, seed=args.seed) if args.validation else None
    if validation and rank == 0:
        counts = {}
        for item in validation:
            source = item["data_source"]
            counts[source] = counts.get(source, 0) + 1
        (output / "validation_selection.json").write_text(json.dumps({
            "seed": args.seed, "source_counts": counts, "questions": validation,
        }, ensure_ascii=False, indent=2))
    def normalized_question(text):
        # Match the split's normalization even when custom QA inputs are supplied.
        return " ".join(text.casefold().split())

    if validation and {normalized_question(x["question"]) for x in validation} & {
        normalized_question(x["question"]) for x in pairs
    }:
        raise ValueError("Validation questions overlap OPSD training pairs")
    if not validation and not args.smoke:
        raise ValueError("Scientific runs require --validation; --smoke permits unselected checkpoints")
    objective = OPSDObjective(model, args.clip, args.logit_chunk_size)
    runner = DistributedDataParallel(objective, device_ids=[torch.cuda.current_device()]) if world > 1 else objective
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    last_saved_step = None

    def save_checkpoint(current_step):
        """Save once per optimizer step and keep every completed checkpoint directory."""
        nonlocal last_saved_step
        checkpoint_path = output / f"step_{current_step}"
        # The final update may also be a periodic save; never write it twice.
        if last_saved_step == current_step:
            return checkpoint_path
        if world > 1:
            dist.barrier()
        if rank == 0:
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            # Publish the latest pointer only after both model and tokenizer are saved.
            metadata = {"adapter": str(checkpoint_path), "step": current_step}
            temporary = output / "last_checkpoint.tmp"
            temporary.write_text(json.dumps(metadata, indent=2))
            temporary.replace(output / "last_checkpoint.json")
            print(json.dumps({"checkpoint_saved": str(checkpoint_path), "step": current_step}), flush=True)
            if tracker:
                tracker.summary["last_checkpoint"] = str(checkpoint_path)
                tracker.summary["last_checkpoint_step"] = current_step
        if world > 1:
            dist.barrier()
        last_saved_step = current_step
        return checkpoint_path

    if rank == 0:
        print(json.dumps({"usable_pairs": len(encoded), "skipped_pairs": skipped,
                          "trainable_parameters": sum(x.numel() for x in trainable), "world_size": world}), flush=True)
    step, best_em = 0, -1.0
    stop = False
    for epoch in range(args.epochs):
        indices = list(range(len(encoded)))
        random.Random(args.seed + epoch).shuffle(indices)
        # No duplicated DistributedSampler tail: each example appears once per epoch.
        batch_width = world * args.accumulation_steps
        for start in range(0, len(indices), batch_width):
            group = indices[start:start + batch_width]
            local = [encoded[i] for i in group[rank::world]]
            token_count = torch.tensor(sum(len(x["student_positions"]) for x in local), device="cuda", dtype=torch.float32)
            if world > 1:
                dist.all_reduce(token_count)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            metric = torch.zeros((), device="cuda")
            entropy_sum = torch.zeros_like(metric)
            # DDP ranks must execute the same number of forwards and backwards.
            # A zero-weight placeholder for a missing tail item does not duplicate data.
            forwards = math.ceil(len(group) / world)
            for micro in range(forwards):
                real = micro < len(local)
                example = local[micro] if real else encoded[group[0]]
                # Synchronize only the final micro-step; all ranks execute the same count.
                sync = runner.no_sync() if world > 1 and micro < forwards - 1 else nullcontext()
                with sync:
                    loss, entropy = runner(example)
                    # Eq. (7) averages positions, then pairs; compensate DDP's rank mean.
                    (loss * (int(real) * world / len(group))).backward()
                metric += loss.detach() * int(real)
                if real:
                    entropy_sum += entropy.detach() * len(example["student_positions"])
            if world > 1:
                dist.all_reduce(metric)
                dist.all_reduce(entropy_sum)
            norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite OPSD gradient")
            optimizer.step()
            torch.cuda.synchronize()
            memory_record = {
                "step": step + 1, "rank": rank,
                "micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": args.accumulation_steps,
                "actual_global_batch_size": len(group),
                "configured_global_batch_size": batch_width,
                "max_student_length": max(len(x["student_ids"]) for x in local) if local else 0,
                "max_teacher_length": max(len(x["teacher_ids"]) for x in local) if local else 0,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            with (output / f"memory_rank_{rank}.jsonl").open("a") as memory_stream:
                memory_stream.write(json.dumps(memory_record) + "\n")
            step += 1
            if rank == 0:
                record = {"step": step, "epoch": epoch + 1, "loss": (metric / len(group)).item(),
                          "grad_norm": norm.item(), "policy_tokens": int(token_count.item()),
                          "actual_global_batch_size": len(group), "configured_global_batch_size": batch_width}
                # These are fixed training trajectories, not fresh policy rollouts.
                stats = [encoded[i].get("action_stats") for i in group]
                stats = [x for x in stats if x is not None and x["turns"] > 0]
                extra = {"actor/entropy_loss": (entropy_sum / token_count).item(),
                         "env/action_stats_coverage": len(stats) / len(group)}
                if stats:
                    extra.update({
                        "env/ratio_of_valid_action": sum(x["valid_actions"] / x["turns"] for x in stats) / len(stats),
                        "env/number_of_valid_action": sum(x["valid_actions"] for x in stats) / len(stats),
                        "env/number_of_valid_search": sum(x["valid_searches"] for x in stats) / len(stats),
                    })
                print(json.dumps({**record, **extra}), flush=True)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({**record, **extra}) + "\n")
                if tracker:
                    # Record the actual tail batch size separately from the configured batch.
                    tracker.log({**extra, "train/step": step, **{f"train/{key}": value for key, value in record.items()},
                                 "train/lr": optimizer.param_groups[0]["lr"],
                                 "memory/rank0_peak_allocated_gib": memory_record["peak_allocated_gib"]})
            # Periodic retention is independent of the epoch-end generation evaluation.
            if step % args.save_steps == 0:
                save_checkpoint(step)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        # Save once, then let all DDP ranks generate their own held-out question shard.
        # Retain the final/epoch-end weights even when the step is not divisible by N.
        checkpoint_path = save_checkpoint(step)
        predictions = None
        if validation:
            from .evaluation import distributed_evaluate

            em, predictions = distributed_evaluate(
                model, tokenizer, validation, output=output, retriever_url=args.retriever_url,
                max_searches=args.max_searches, max_new_tokens=args.eval_max_new_tokens,
                max_length=args.max_length, final_answer_reserve=args.final_answer_reserve,
            )
        if rank == 0:
            if validation:
                write_jsonl(output / f"validation_step_{step}.jsonl", predictions)
                by_source = {}
                for source in sorted({item["data_source"] for item in predictions}):
                    subset = [item for item in predictions if item["data_source"] == source]
                    by_source[source] = sum(item["reward"] for item in subset) / len(subset)
                print(json.dumps({"step": step, "validation_em": em,
                                  "validation_em_by_source": by_source}), flush=True)
                if tracker:
                    tracker.log({**extra, "train/step": step, "validation/em": em,
                                 **{f"validation/em_{source}": score for source, score in by_source.items()}})
                    tracker.summary["last_checkpoint"] = str(checkpoint_path)
                if em > best_em:
                    best_em = em
                    # Tie-breaking is deterministic: keep the earliest best-EM adapter.
                    (output / "best.json").write_text(json.dumps({
                        "adapter": str(checkpoint_path), "step": step, "validation_em": em,
                        "validation_em_by_source": by_source,
                    }, indent=2))
            else:
                (output / "smoke_only.json").write_text(json.dumps({
                    "adapter": str(checkpoint_path), "selected_by_validation": False,
                }, indent=2))
        if world > 1:
            dist.barrier()
        if stop:
            break
    if world > 1:
        dist.destroy_process_group()
    if tracker:
        tracker.finish()
