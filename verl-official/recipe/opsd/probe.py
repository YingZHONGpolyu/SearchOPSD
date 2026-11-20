"""Read-only environment probe; run through the veRL Python interpreter."""

import json
import urllib.request

import pandas as pd
import peft
import torch
import transformers


if __name__ == "__main__":
    # Report versions and schema, avoiding a dump of the full QA training corpus.
    print(torch.__version__, transformers.__version__, peft.__version__)
    frame = pd.read_parquet("/external/data/nq_hotpotqa_train/train.parquet")
    print("rows", len(frame), "columns", list(frame.columns))
    print(frame.iloc[0].to_json(force_ascii=False))
    for port in (8000, 8003, 8010, 8011):
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/retrieve",
                data=json.dumps({"queries": ["capital of France"], "topk": 3, "return_scores": True}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                print(port, str(json.load(response))[:400])
        except Exception as error:
            print(port, type(error).__name__, str(error)[:100])
