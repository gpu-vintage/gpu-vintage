#!/usr/bin/env python3
"""Measure prefill and decode throughput and GPU energy of one model on one GPU.

    python scripts/bench.py --model Qwen/Qwen3-1.7B --dtype fp16 \
        --batches 1 4 16 64 256 --run-dir out/runs/<name>

The model must already be in the local Hugging Face cache (download it first,
with `huggingface-cli download <repo> --revision <commit>`): runs never reach
the network. Rerunning with the same --run-dir skips the batch sizes already
measured. See src/gpuvintage/bench.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpuvintage.bench import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face repo id (cached) or local directory")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--min-seconds", type=float, default=10.0, help="minimum length of a measurement window")
    ap.add_argument("--warmup-seconds", type=float, default=2.0)
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--prefill-max-tokens", type=int, default=8192, help="prompt tokens per prefill forward pass")
    ap.add_argument("--bucket", type=int, default=64, help="attention span bucket of the decode graphs")
    ap.add_argument("--no-graphs", action="store_true", help="eager decoding (no CUDA graphs)")
    ap.add_argument("--gpu-tag", default="", help="short GPU label recorded in config.json")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    args.batches = sorted(set(args.batches))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
