#!/usr/bin/env python3
"""
reproduce_table1_zoo_sweep.py

Reproduce Section 4 & Table 1 of the paper:
Pretrained-model sweep across 30 models (124M to 3B parameters),
evaluating super weight formation via activation-guided detection.

Usage:
    # Fast sanity check on 4 representative models (~5 minutes):
    uv run python scripts/reproduce_table1_zoo_sweep.py --quick-check

    # Full 30-model sweep (~1.1 GPU hours):
    uv run python scripts/reproduce_table1_zoo_sweep.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swx.cli import main as cli_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1: Pretrained-model sweep across 30 models"
    )
    parser.add_argument(
        "--quick-check", action="store_true",
        help="Run fast check on 4 representative models (SmolLM2-135M, GPT-2 124M, SmolLM2-1.7B, Pythia-1B)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--allow-fp16", action="store_true")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()

    cli_args = ["g1", "--device", args.device, "--dtype", args.dtype]
    if args.quick_check:
        cli_args.append("--quick-check")
    if args.allow_fp16:
        cli_args.append("--allow-fp16")
    if args.runs_dir:
        cli_args.extend(["--runs-dir", args.runs_dir])

    log.info("Executing Table 1 reproduction: swx %s", " ".join(cli_args))
    return cli_main(cli_args)


if __name__ == "__main__":
    sys.exit(main())
