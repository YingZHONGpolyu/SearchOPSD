"""Verify complete selected pairs and statistics against the original CPU miner."""
from collections import defaultdict
import json
from pathlib import Path
import tempfile

from .core import mine_pairs, read_jsonl
from .gpu_mining import mine_file


def main():
    rows = []
    # Cover wrong-sibling priority, all-correct fallback, ties and no-correct exclusion.
    for question in range(40):
        for sibling in range(5):
            text = '<think>' + ('中文abc ' * (sibling + 1)) + f'{question % 3}</think><answer>{sibling}</answer>'
            rows.append(dict(question=f'q{question}', trajectory=text, aliases=['a'],
                             reward=int(question % 4 != 0 and (sibling < 2 or question % 3 == 0)),
                             search_calls=sibling % 2, response_ids=[1] * len(text),
                             policy_mask=[1] * len(text), policy_model='same-policy'))
    # Also verify real short trajectories from the completed production pool.
    groups = defaultdict(list)
    path = Path('/data/work/YingZhong/code/on_policy_distillation_verl/outputs/opsd/step250_bs560_20261021_120743/pool.jsonl')
    with path.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            groups[row['question']].append(row)
            if index >= 999:
                break
    for siblings in sorted(groups.values(), key=lambda items: max(len(row['trajectory']) for row in items))[:2]:
        rows.extend(siblings)
    expected, stats = mine_pairs(rows)
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / 'pool.jsonl', Path(tmp) / 'pairs.jsonl'
        source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        actual_stats = mine_file(source, output, devices=list(range(1, 8)), batch_size=32)
        actual = read_jsonl(output)
        assert actual == expected
        assert actual_stats == stats
    report = {'input_trajectories': len(rows), 'selected_pairs': len(expected),
              'exact_selected_pairs_match': True, 'exact_statistics_match': True}
    destination = path.parent / 'gpu_pair_selection_tests.json'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
