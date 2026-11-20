"""Real Qwen2.5-3B backward/update test with explicitly synthetic sibling text.

This is a software integration check, never evidence of on-policy training or EM.
Run it before expensive collection to verify the frozen-teacher implementation.
"""

import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .core import encode_pair, write_jsonl
from .train import OPSDObjective, load_model


def main():
    """Check nonzero LoRA updates, unchanged teacher logits, and finite exact KL."""
    model_path = os.environ.get("OPSD_SMOKE_MODEL", "/external/models/Qwen2.5-3B")
    model, tokenizer = load_model(model_path)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pair = {
        "question": "What is the capital of France?", "aliases": ["Paris"],
        "reference": "<think>The retrieved passage gives the capital.</think><answer>Paris</answer>",
        "student": "<think>I should check.</think><search>France capital</search>"
                   "<information>Paris is the capital of France.</information><answer>Lyon</answer>",
    }
    example = encode_pair(tokenizer, pair, 2048)
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    ids = torch.tensor([example["teacher_ids"]], device="cuda")
    # Probe one vocabulary distribution rather than allocating all teacher logits.
    def teacher_probe():
        with torch.no_grad(), model.disable_adapter():
            model.eval()
            hidden = model.get_base_model().model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
            return model.get_output_embeddings()(hidden).detach().clone()

    teacher_before = teacher_probe()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    loss = OPSDObjective(model, clip=10.0, chunk_size=8)(example)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    teacher_after = teacher_probe()
    changed = sum(not torch.equal(a, b) for a, b in zip(before, trainable))
    assert torch.isfinite(loss) and torch.isfinite(gradient_norm)
    assert changed > 0, "No LoRA parameters changed"
    assert torch.equal(teacher_before, teacher_after), "Adapter-disabled teacher changed"
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
    report = {
        "synthetic_fixture": True, "model": model_path, "loss": loss.item(),
        "gradient_norm": gradient_norm.item(), "changed_lora_tensors": changed,
        "teacher_unchanged": True, "policy_tokens": len(example["student_positions"]),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps(report, indent=2), flush=True)
    # The conda launcher supplies a host path; keep the Docker default compatible.
    output = Path(os.environ.get("OPSD_SMOKE_OUTPUT", "/workspace/outputs/opsd/integration"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "gpu_report.json").write_text(json.dumps(report, indent=2))
    # Separate fixture supports a short DDP CLI exercise; never label it mined data.
    fixture = output / "synthetic_pairs.jsonl"
    if not fixture.exists():
        write_jsonl(fixture, [pair, dict(pair, question="Which city is France's capital?")])


if __name__ == "__main__":
    main()
