"""Deterministic held-out split of training QA; test benchmarks stay untouched."""

import hashlib
import json
from pathlib import Path

from .core import write_jsonl
from .rollout import load_questions


def prepare(args):
    """Hash normalized questions so duplicated rows cannot leak across the split."""
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    training, validation = [], []
    for item in load_questions(args.questions):
        normalized = " ".join(item["question"].casefold().split())
        digest = hashlib.sha256(f"{args.seed}:{normalized}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        (validation if fraction < args.validation_fraction else training).append(item)
    if not training or not validation:
        raise ValueError("Empty split; increase input size or adjust validation_fraction")
    write_jsonl(output / "train.jsonl", training)
    write_jsonl(output / "validation.jsonl", validation)
    report = {"source": args.questions, "seed": args.seed,
              "validation_fraction": args.validation_fraction,
              "training_questions": len(training), "validation_questions": len(validation)}
    report["validation_source_counts"] = {
        source: sum(item.get("data_source", "unknown") == source for item in validation)
        for source in sorted({item.get("data_source", "unknown") for item in validation})
    }
    (output / "split.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
