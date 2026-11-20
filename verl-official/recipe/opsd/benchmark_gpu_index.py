"""Check the full GPU index against FAISS before switching the live retriever."""
import json
from pathlib import Path
import time

import faiss
import numpy as np
import torch

from .gpu_flat_index import GPUFlatIPIndex


def main():
    """Use real corpus vectors as reproducible test queries and record both timings."""
    faiss.omp_set_num_threads(8)
    index = faiss.read_index('/data/work/YingZhong/data/search_data/e5_Flat.index')
    # Query actual index vectors with a small perturbation; test several positions in the corpus.
    vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
    rows = np.linspace(0, index.ntotal - 1, 16, dtype=np.int64)
    queries = vectors[rows].copy()
    start = time.monotonic()
    expected_scores, expected_ids = index.search(queries, 3)
    cpu_seconds = time.monotonic() - start
    gpu = GPUFlatIPIndex(index)
    gpu.search(queries[:1], 3)
    start = time.monotonic()
    actual_scores, actual_ids = gpu.search(queries, 3)
    gpu_seconds = time.monotonic() - start
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_allclose(actual_scores, expected_scores, atol=2e-5, rtol=2e-5)
    report = dict(ntotal=index.ntotal, dimension=index.d, query_count=len(queries),
                  matching_top3=True, cpu_seconds=cpu_seconds, gpu_seconds=gpu_seconds,
                  speedup=cpu_seconds / gpu_seconds,
                  index_gib=gpu.vectors.numel() * 4 / 2**30,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    output = Path('/data/work/YingZhong/code/on_policy_distillation_verl/outputs/opsd/throughput_verification')
    output.mkdir(exist_ok=True)
    (output / 'gpu_index_benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
