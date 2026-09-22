#!/usr/bin/env python3
"""Benchmark candidate scoring models on this machine (time per document)."""
import os
import sys
import time

import conformal as C

TEXT = open("/tmp/human_text.txt").read()
CANDIDATES = sys.argv[1:] or ["gpt2-medium", "Qwen/Qwen2.5-1.5B", "EleutherAI/pythia-1.4b"]


def main():
    C.DEVICE = C.get_device("auto")
    C.CACHE_DIR = os.path.expanduser("~/.cache")
    for model in CANDIDATES:
        print(f"\n=== {model} ===", flush=True)
        C.BASE_MODEL = model
        C._model = None
        C._tokenizer = None
        t0 = time.time()
        try:
            pt = C.per_token_scores(TEXT)
            load = time.time() - t0
            t1 = time.time()
            for _ in range(2):
                pt = C.per_token_scores(TEXT)
            per_doc = (time.time() - t1) / 2
            n = len(TEXT.split())
            est_cal = per_doc * 250 / 60
            print(f"  load: {load:.1f}s | score/doc: {per_doc:.1f}s "
                  f"({n} words) | calibration m=200+50: ~{est_cal:.0f} min", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()