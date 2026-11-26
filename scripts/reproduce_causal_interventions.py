#!/usr/bin/env python3
"""
reproduce_causal_interventions.py

Reproduce Section 6, Tables 5-7, Figure 3, and Appendices D-E:
Training-time causal interventions on SwiGLU (FineWeb-Edu, seeds 42, 52, 62):
  1. Unperturbed Baseline
  2. Hard Activation Clipping (AHC, clamp to [-12, 12])
  3. Soft Activation Clipping (ASC, 12 * tanh(h / 12))
  4. Weight Cap (WC, clamp W_d to [-0.35, 0.35])
  5. Early Inoculation (EI, step 200, +2 mean|W_d| to L7 [100, 500])
  6. Mid Inoculation (MI, step 1000, +2 mean|W_d| to L7 [100, 500])

Usage:
    # Fast sanity smoke test (100 steps on AHC, ~1 minute):
    uv run python scripts/reproduce_causal_interventions.py --quick-test --conditions AHC --seeds 42

    # Run full set of interventions (15 runs, ~24.6 GPU hours on RTX 3080):
    uv run python scripts/reproduce_causal_interventions.py
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

INTERVENTIONS: dict[str, str | None] = {
    "baseline": None,
    "AHC": "conditions/A1_hard_clamp.json",
    "ASC": "conditions/A2_soft_clamp.json",
    "WC": "conditions/W1_weight_cap.json",
    "EI": "conditions/I1_early_inoc.json",
    "MI": "conditions/I2_mid_inoc.json",
}
ALL_SEEDS = [42, 52, 62]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Section 6: SwiGLU training-time causal interventions"
    )
    parser.add_argument(
        "--conditions", nargs="+", default=list(INTERVENTIONS.keys()),
        choices=list(INTERVENTIONS.keys()),
        help=f"Subset of conditions to evaluate (default: {list(INTERVENTIONS.keys())})",
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
        help="Run in smoke-test mode with 100 steps per run for fast verification",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--runs-dir", default="runs/training_twins/interventions")
    args = parser.parse_args()

    max_steps = 100 if args.quick_test else args.max_steps
    log_every = 20 if args.quick_test else 100

    total_runs = len(args.conditions) * len(args.seeds)
    run_idx = 0

    log.info(
        "Beginning Causal Interventions: %d runs (steps=%d, bs=%d, lr=%.1e)",
        total_runs, max_steps, args.batch_size, args.lr,
    )

    for cond_key in args.conditions:
        manifest_rel = INTERVENTIONS[cond_key]
        for seed in args.seeds:
            run_idx += 1
            out_dir = Path(args.runs_dir) / f"swiglu_{cond_key}_seed{seed}"
            log.info(
                "[%d/%d] Running SwiGLU with %s (seed=%d) -> %s ...",
                run_idx, total_runs, cond_key, seed, out_dir,
            )

            cli_args = [
                "train-twins",
                "--variant", "swiglu",
                "--dataset", "fineweb-edu",
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
            if manifest_rel:
                cli_args.extend(["--condition-manifest", str(ROOT / manifest_rel)])

            ret = cli_main(cli_args)
            if ret != 0:
                log.error("Run failed with return code %d: %s", ret, cli_args)
                return ret

    log.info("All %d Intervention runs completed successfully!", total_runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
