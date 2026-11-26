#!/usr/bin/env bash
# ==============================================================================
# Master reproduction script for NeurIPS 2026 Camera Ready
# Paper: "Tracing the Formation of Super Weights Through Neural Network Artifacts and Dynamics"
# ==============================================================================
set -euo pipefail

MODE="${1:---smoke-test}"

echo "=============================================================================="
echo " Super Weights NeurIPS 2026 Replication Suite"
echo " Mode: ${MODE}"
echo "=============================================================================="

if [[ "${MODE}" == "--smoke-test" ]]; then
    echo "[1/4] Running model-free unit tests (173 test cases)..."
    uv run pytest tests/

    echo "[2/4] Running Table 1 Zoo Sweep smoke test (4 representative models)..."
    uv run python scripts/reproduce_table1_zoo_sweep.py --quick-check

    echo "[3/4] Running Section 5 Pretraining Twins smoke test (100 steps)..."
    uv run python scripts/reproduce_training_twins.py --quick-test --variants swiglu geglu --datasets fineweb-edu --seeds 42

    echo "[4/4] Running Section 6 Causal Intervention smoke test (100 steps)..."
    uv run python scripts/reproduce_causal_interventions.py --quick-test --conditions AHC ASC --seeds 42

    echo "=============================================================================="
    echo " Smoke test completed successfully! All pipelines verified."
    echo "=============================================================================="
elif [[ "${MODE}" == "--full" ]]; then
    echo "[1/5] Running full unit tests..."
    uv run pytest tests/

    echo "[2/5] Reproducing Table 1 (30-model pretrained sweep, ~1.1 GPU hours)..."
    uv run python scripts/reproduce_table1_zoo_sweep.py

    echo "[3/5] Reproducing Section 5 (6 variants x 3 datasets x 3 seeds, ~84.6 GPU hours)..."
    uv run python scripts/reproduce_training_twins.py

    echo "[4/5] Reproducing Section 6 (SwiGLU interventions across 3 seeds, ~24.6 GPU hours)..."
    uv run python scripts/reproduce_causal_interventions.py

    echo "[5/5] Generating trajectory and multi-seed lag dynamics plots..."
    uv run swx plot-trajectories --run-dir runs/training_twins/fineweb-edu --dataset fineweb-edu
    uv run swx plot-trajectories --run-dir runs/training_twins/c4 --dataset c4
    uv run swx plot-trajectories --run-dir runs/training_twins/slimpajama --dataset slimpajama
    uv run swx plot-trajectories --run-dir runs/training_twins/interventions --baseline-dir runs/training_twins/fineweb-edu
    uv run swx plot-lag --runs-dir runs/training_twins --out-dir runs/training_twins/plots

    echo "=============================================================================="
    echo " Full reproduction completed! Figures and summaries are in runs/training_twins/"
    echo "=============================================================================="
else
    echo "Unknown option: ${MODE}. Use --smoke-test or --full."
    exit 1
fi
