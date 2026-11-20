"""Exercise seven real CPU collectives, including validation subsets smaller than world size."""
import json
import os
from unittest.mock import patch

import torch.distributed as dist

from .evaluation import distributed_evaluate, merge_predictions


def main():
    """Check exact coverage/order and EM aggregation while GPU training continues."""
    dist.init_process_group('gloo')
    for count in (64, 3):
        questions = [dict(question=str(index), reward=index % 2,
                          data_source='nq' if index % 2 else 'hotpotqa') for index in range(count)]
        def fake_evaluate(model, tokenizer, subset, **kwargs):
            return subset[0]['reward'], [dict(subset[0])]
        with patch('recipe.opsd.evaluation.evaluate', fake_evaluate):
            em, rows = distributed_evaluate(None, None, questions)
        assert rows == questions
        assert em == sum(row['reward'] for row in questions) / count
    # A repeated question index cannot silently increase the reported denominator.
    try:
        merge_predictions([[(0, {'reward': 1}), (0, {'reward': 1})]], 2)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate validation index accepted')
    if dist.get_rank() == 0:
        print(json.dumps({'passed': True, 'world_size': dist.get_world_size(),
                          'tested_question_counts': [64, 3], 'exact_order_and_em': True,
                          'duplicate_detection': True}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
