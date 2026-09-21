#!/usr/bin/env bash
# CPU smoke test -- proves the pipeline runs with no GPU/data/weights.
python3 tests/test_metrics_smoke.py && \
python3 -m frpure.eval.runner --synthetic --device cpu --out results/synthetic.csv