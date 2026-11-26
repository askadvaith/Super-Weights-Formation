#!/usr/bin/env python3
"""
reproduce_training_twins.py

Reproduce Section 5, Tables 2-4, Figure 1, Figure 2, and Appendices B-C:
Controlled 124M parameter-matched Transformer pretraining across:
  - 6 activation functions: SwiGLU, GeGLU, ReGLU, GELU, SiLU, ReLU^2
  - 3 pretraining corpora: FineWeb-Edu, C4, SlimPajama
  - 3 random seeds: 42, 52, 62

Usage:
    # Quick verification smoke test (100 steps on 1 model, ~1 minute):
    uv run python scripts/reproduce_training_twins.py --quick-test --variants swiglu --datasets fineweb-edu --seeds 42

    # Single full run (20,000 steps, ~95 minutes on RTX 3080):
    uv run python scripts/reproduce_training_twins.py --variants swiglu --datasets fineweb-edu --seeds 42

    # Full pretraining matrix (54 runs across all variants, datasets, and seeds):
    uv run python scripts/reproduce_training_twins.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swx.cli import main as cli_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALL_VARIANTS = ["swiglu", "geglu", "reglu", "gelu_std", "silu_std", "relu2_std"]
ALL_DATASETS = ["fineweb-edu", "c4", "slimpajama"]
ALL_SEEDS = [42, 52, 62]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Section 5: Controlled activation family pretraining grid"
    )
    parser.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS,
        help=f"Subset of activation variants to train (default: {ALL_VARIANTS})",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Subset of datasets to train on (default: {ALL_DATASETS})",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=ALL_SEEDS,
        help=f"Random seeds for initialisation (default: {ALL_SEEDS})",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20000,
        help="Training step budget (paper default: 20000)",
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run in smoke-test mode with 100 steps per run for fast pipeline verification",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--runs-dir", default="runs/training_twins")
    args = parser.parse_args()

    max_steps = 100 if args.quick_test else args.max_steps
    log_every = 20 if args.quick_test else 100

    total_runs = len(args.datasets) * len(args.variants) * len(args.seeds)
    run_idx = 0

    log.info(
        "Beginning Training Twins grid: %d runs (steps=%d, bs=%d, lr=%.1e)",
        total_runs, max_steps, args.batch_size, args.lr,
    )

    for ds in args.datasets:
        for var in args.variants:
            for seed in args.seeds:
                run_idx += 1
                out_dir = Path(args.runs_dir) / ds / f"{var}_seed{seed}"
                log.info(
                    "[%d/%d] Training %s on %s (seed=%d) -> %s ...",
                    run_idx, total_runs, var, ds, seed, out_dir,
                )

                cli_args = [
                    "train-twins",
                    "--variant", var,
                    "--dataset", ds,
                    "--seed", str(seed),
                    "--max-steps", str(max_steps),
                    "--log-every", str(log_every),
                    "--batch-size", str(args.batch_size),
                    "--micro-batch-size", str(args.micro_batch_size),
                    "--lr", str(args.lr),
                    "--device", args.device,
                    "--dtype", args.dtype,
                    "--output-dir", str(out_dir),
                ]
                ret = cli_main(cli_args)
                if ret != 0:
                    log.error("Run failed with return code %d: %s", ret, cli_args)
                    return ret

    log.info("All %d Training Twins runs completed successfully!", total_runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
