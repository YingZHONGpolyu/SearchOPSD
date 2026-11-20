"""Compare GPU matching scores to the original algorithm on all seven free GPUs."""
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
from pathlib import Path
import random

from .gpu_sequence_matcher import GPUSequenceMatcher, build_kernel


def main():
    rng = random.Random(42)
    cases = [('', ''), ('a', ''), ('', 'b'), ('abc', 'abc'), ('abxcd', 'abcd'),
             ('你好🧪世界', '你好世界🧪'), ('abcabcXabc', 'abcYabcabc'),
             ('<information>' * 100, '<information>' * 99 + 'x')]
    for _ in range(1000):
        cases.append((''.join(rng.choices(' abcd<>中文', k=rng.randrange(100))),
                      ''.join(rng.choices(' abcd<>中文', k=rng.randrange(100)))))
    expected = [1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() for a, b in cases]
    binary = build_kernel()
    def check(device):
        scores = GPUSequenceMatcher(device, binary).distances(cases)
        assert scores == expected, next((i, scores[i], expected[i]) for i in range(len(scores)) if scores[i] != expected[i])
        return {'gpu': device, 'pairs': len(cases), 'exact_scores_match': True}
    with ThreadPoolExecutor(7) as executor:
        reports = list(executor.map(check, range(1, 8)))
    output = Path('/data/work/YingZhong/code/on_policy_distillation_verl/outputs/opsd/throughput_verification')
    (output / 'gpu_sequence_matcher_tests.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
