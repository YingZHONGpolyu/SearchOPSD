"""Start the existing Search-R1 E5/Wikipedia retriever on a separate local port.

Run with the configured `verl` conda environment or the pre-existing `retriever`
environment. The upstream retriever implementation is loaded read-only.
"""

import argparse
import importlib.util
from pathlib import Path


def main():
    """Load local retrieval assets and use a port separate from production services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/data/work/YingZhong/code/Search-R1/search_r1/search/retrieval_server.py")
    parser.add_argument("--index", default="/data/work/YingZhong/data/search_data/e5_Flat.index")
    parser.add_argument("--corpus", default="/data/work/YingZhong/data/search_data/wiki-18.jsonl")
    parser.add_argument("--model", default="/data/work/YingZhong/models/e5-base-v2")
    parser.add_argument("--port", type=int, default=8123)
    # Keep the full-precision flat index on the retrieval GPU when memory permits.
    parser.add_argument("--index-device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    for path in (args.source, args.index, args.corpus):
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError(f"Missing or empty retrieval asset: {path}")
    spec = importlib.util.spec_from_file_location("opsd_search_r1_retrieval", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.config = module.Config(
        retrieval_method="e5", retrieval_topk=3, index_path=args.index,
        corpus_path=args.corpus, faiss_gpu=False, retrieval_model_path=args.model,
        retrieval_pooling_method="mean", retrieval_query_max_length=256,
        retrieval_use_fp16=True, retrieval_batch_size=32,
    )
    # E5 remains on GPU; the index backend preserves the original float32 vectors.
    module.retriever = module.get_retriever(module.config)
    if args.index_device == "cuda":
        from .gpu_flat_index import GPUFlatIPIndex

        module.retriever.index = GPUFlatIPIndex(module.retriever.index)
        print("OPSD retrieval: float32 index resident on GPU, batch size 32", flush=True)
    module.uvicorn.run(module.app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
