# ==============================================================================
# Master reproduction script for NeurIPS 2026 Camera Ready (PowerShell)
# Paper: "Tracing the Formation of Super Weights Through Neural Network Artifacts and Dynamics"
# ==============================================================================
param(
    [string]$Mode = "--smoke-test"
)

$ErrorActionPreference = "Stop"

Write-Host "=============================================================================="
Write-Host " Super Weights NeurIPS 2026 Replication Suite"
Write-Host " Mode: $Mode"
Write-Host "=============================================================================="

if ($Mode -eq "--smoke-test") {
    Write-Host "[1/4] Running model-free unit tests (173 test cases)..."
    uv run pytest tests/

    Write-Host "[2/4] Running Table 1 Zoo Sweep smoke test (4 representative models)..."
    uv run python scripts/reproduce_table1_zoo_sweep.py --quick-check

    Write-Host "[3/4] Running Section 5 Pretraining Twins smoke test (100 steps)..."
    uv run python scripts/reproduce_training_twins.py --quick-test --variants swiglu geglu --datasets fineweb-edu --seeds 42

    Write-Host "[4/4] Running Section 6 Causal Intervention smoke test (100 steps)..."
    uv run python scripts/reproduce_causal_interventions.py --quick-test --conditions AHC ASC --seeds 42

    Write-Host "=============================================================================="
    Write-Host " Smoke test completed successfully! All pipelines verified."
    Write-Host "=============================================================================="
} elseif ($Mode -eq "--full") {
    Write-Host "[1/5] Running full unit tests..."
    uv run pytest tests/

    Write-Host "[2/5] Reproducing Table 1 (30-model pretrained sweep, ~1.1 GPU hours)..."
    uv run python scripts/reproduce_table1_zoo_sweep.py

    Write-Host "[3/5] Reproducing Section 5 (6 variants x 3 datasets x 3 seeds, ~84.6 GPU hours)..."
    uv run python scripts/reproduce_training_twins.py

    Write-Host "[4/5] Reproducing Section 6 (SwiGLU interventions across 3 seeds, ~24.6 GPU hours)..."
    uv run python scripts/reproduce_causal_interventions.py

    Write-Host "[5/5] Generating trajectory and multi-seed lag dynamics plots..."
    uv run swx plot-trajectories --run-dir runs/training_twins/fineweb-edu --dataset fineweb-edu
    uv run swx plot-trajectories --run-dir runs/training_twins/c4 --dataset c4
    uv run swx plot-trajectories --run-dir runs/training_twins/slimpajama --dataset slimpajama
    uv run swx plot-trajectories --run-dir runs/training_twins/interventions --baseline-dir runs/training_twins/fineweb-edu
    uv run swx plot-lag --runs-dir runs/training_twins --out-dir runs/training_twins/plots

    Write-Host "=============================================================================="
    Write-Host " Full reproduction completed! Figures and summaries are in runs/training_twins/"
    Write-Host "=============================================================================="
} else {
    Write-Host "Unknown option: $Mode. Use --smoke-test or --full."
    exit 1
}
