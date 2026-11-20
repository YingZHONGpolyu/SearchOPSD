"""Distributed held-out generation evaluation using each rank's existing model."""

import json
from pathlib import Path

import torch.distributed as dist

from .rollout import evaluate


def merge_predictions(shards, expected_count):
    """Restore validation order and reject duplicate or missing question indices."""
    indexed = [item for shard in shards for item in shard]
    if sorted(index for index, _ in indexed) != list(range(expected_count)):
        raise ValueError('Parallel validation contains missing or duplicate indices')
    predictions = [row for _, row in sorted(indexed, key=lambda item: item[0])]
    em = sum(row['reward'] for row in predictions) / len(predictions)
    return em, predictions


def distributed_evaluate(model, tokenizer, questions, *, output=None, **kwargs):
    """Shard questions, generate on every GPU, and gather predictions before EM scoring."""
    distributed = dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    indices = list(range(rank, len(questions), world))
    local = []
    for position, index in enumerate(indices, 1):
        _, rows = evaluate(model, tokenizer, [questions[index]], **kwargs)
        local.append((index, rows[0]))
        record = {'validation_rank': rank, 'completed': position, 'total': len(indices)}
        print(json.dumps(record), flush=True)
        if output is not None:
            # Each rank owns its progress file; the shared validation file is rank-zero only.
            path = Path(output) / f'validation_progress_rank_{rank}.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record))
            temporary.replace(path)
    shards = [None] * world
    if distributed:
        dist.all_gather_object(shards, local)
    else:
        shards[0] = local
    return merge_predictions(shards, len(questions))
