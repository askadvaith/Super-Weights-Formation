# Tracing the Formation of Super Weights Through Neural Network Artifacts and Dynamics

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official camera-ready code release for reproducing all experimental results, figures, and tables in the paper:

> **Tracing the Formation of Super Weights Through Neural Network Artifacts and Dynamics**  
> *NeurIPS 2026 Workshop on Neural Network Artifacts as a New Data Modality*

---

## Table of Contents
1. [Overview & Paper Contributions](#1-overview--paper-contributions)
2. [Environment & Installation](#2-environment--installation)
3. [Data Access and Preparation](#3-data-access-and-preparation)
4. [Reproducing Paper Results](#4-reproducing-paper-results)
   - [Section 4 / Table 1: Pretrained-Model Zoo Sweep](#a-section-4--table-1-pretrained-model-zoo-sweep)
   - [Section 5 / Tables 2–4 & Figures 1–2: Controlled Activation-Family Training](#b-section-5--tables-24--figures-12-controlled-activation-family-training)
   - [Section 6 / Tables 5–7 & Figure 3: SwiGLU Causal Interventions](#c-section-6--tables-57--figure-3-swiglu-causal-interventions)
   - [Appendix C / Figure 4: Complete Multi-Seed Lag Dynamics Grid](#d-appendix-c--figure-4-complete-multi-seed-lag-dynamics-grid)
   - [Trajectory Plots & Summary Metrics](#e-trajectory-plots--summary-metrics)
5. [Interactive Inspection & REPL](#5-interactive-inspection--repl)
6. [Directory Structure](#6-directory-structure)


---

## 1. Overview & Paper Contributions

Super weights are individual parameters whose single-scalar ablation catastrophically degrades a language model. This codebase provides the complete computational framework to investigate and reproduce the architectural and dynamical conditions governing their formation:

1. **Observational Evidence (Section 4, Table 1):** Activation-guided detection across 30 open-source pretrained models (124M to 3B parameters). Super weights are found in 9 of 12 gated models (SwiGLU / GeGLU) and 0 of 18 non-gated models (GELU / ReLU).
2. **Causal Evidence (Section 5, Tables 2–4, Figures 1–2, Appendix B & C):** Controlled pretraining of 124M parameter-matched Transformers across 3 pretraining datasets (FineWeb-Edu, C4, SlimPajama) and 3 random seeds (42, 52, 62). Models with second-order activation growth (SwiGLU, GeGLU, ReGLU, non-gated ReLU$^2$) reliably develop high dominance ($D > 0.40$), while standard GELU and SiLU remain low ($D < 0.25$) despite comparable weight outlier ratios ($R_W \approx 16$--$23$).
3. **Intervention Evidence (Section 6, Tables 5–7, Figure 3, Appendix D & E):** Training-time causal interventions on SwiGLU demonstrating that constraining hidden activations (AHC, ASC) prevents super weight formation at $<0.012$ loss penalty, whereas weight caps (WC) or coordinate inoculations (EI, MI) merely relocate super weights to new coordinates without eliminating them.

---

## 2. Environment & Installation

### Requirements
- **OS:** Linux (Ubuntu 20.04+ recommended) or Windows 10/11
- **Python:** Version `3.11`
- **Hardware:** NVIDIA GPU with $\ge 10\text{ GB}$ VRAM (e.g., RTX 3080, RTX 3090, A100). All telemetry tracking, plotting, and unit tests also run on CPU.
- **CUDA:** 12.4+ (matches pre-configured PyTorch wheels)

### Setup via `uv` (Recommended)
[`uv`](https://github.com/astral-sh/uv) provides fast, reproducible virtual environment and dependency management:

```bash
# 1. Create a virtual environment using Python 3.11
uv venv --python 3.11

# 2. Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Synchronize dependencies (installs PyTorch with CUDA 12.4 support)
uv sync

# 4. Verify environment & CUDA availability
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Setup via `pip` (Alternative)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

---

## 3. Data Access and Preparation

### Raw Data Sources
All datasets used in the paper are publicly hosted and downloaded via the Hugging Face `datasets` library:
- **FineWeb-Edu:** [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (subset: `sample-10BT`)
- **C4:** [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) (subset: `en`)
- **SlimPajama:** [`DKYoon/SlimPajama-6B`](https://huggingface.co/datasets/DKYoon/SlimPajama-6B)
- **LAMBADA & WikiText-2:** Used for detection validation probes (auto-cached).

### Deterministic Tokenization and Local Caching
To eliminate non-determinism across runs, random seeds, and distributed workers:
1. When a pretraining run begins, the data loader pulls a contiguous token stream and tokenizes it using the GPT-2 BPE tokenizer.
2. The stream is saved locally as a contiguous `uint16` binary file (`.bin`) under `runs/training_twins/data/` (e.g., `fineweb-edu_tokens_seed42.bin`).
3. All subsequent runs on that corpus memory-map this file directly, guaranteeing **100% byte-identical token presentation orders across seeds and model variants**.
4. Cached binary files are automatically created on demand upon first run, or can be pre-cached independently.

### Generated Telemetry Data
During training, telemetry snapshots are recorded every 100 steps through step 2,000 and every 500 steps thereafter:
- Input/output activation extrema across all layers
- Peak dominance ratio: $D(t) = \frac{|X_{ik} W_{jk}|}{|Y_{ij}|}$
- Weight outlier ratio: $R_W(t) = \frac{\max |W_d|}{\text{mean} |W_d|}$
- Input spike magnitude $|X_{ik^*}|$ vs. dominant weight magnitude $|W_{j^*k^*}|$ (lag dynamics)
- Persistent watchlist tracking: tracks the specific winning coordinate retrospectively from initialization.

---

## 4. Reproducing Paper Results

### A. Section 4 / Table 1: Pretrained-Model Zoo Sweep
Evaluates 30 pretrained models (18 non-gated, 12 gated) using activation-guided detection.

```bash
# Fast check (4 models: SmolLM2-135M, GPT-2 124M, SmolLM2-1.7B, Pythia-1B, ~5 mins):
uv run python scripts/reproduce_table1_zoo_sweep.py --quick-check

# Full 30-model sweep:
uv run python scripts/reproduce_table1_zoo_sweep.py

# Alternatively, run detection on a single model directly via CLI:
uv run swx detect --model olmo-1b
uv run swx verify-directory --model olmo-1b
```

### B. Section 5 / Tables 2–4 & Figures 1–2: Controlled Activation-Family Training
Trains parameter-matched 124M Transformers (12 layers, $d_{\text{model}}=768$, 12 heads, context 1024) with:
- Gated variants: SwiGLU, GeGLU, ReGLU ($d_{\text{ff}} = 2048$, 3 matrices per MLP)
- Non-gated variants: GELU, SiLU, ReLU$^2$ ($d_{\text{ff}} = 3072$, 2 matrices per MLP)
- Exact parameter parity: 4,718,592 MLP params/layer; 123,551,232 total trainable params.

```bash
# Train a single model (e.g. SwiGLU on FineWeb-Edu, seed 42):
uv run swx train-twins --variant swiglu --dataset fineweb-edu --seed 42 --max-steps 20000 --batch-size 8

# Run full pretraining matrix across all 6 variants, 3 datasets, and 3 seeds (54 runs):
uv run python scripts/reproduce_training_twins.py
```

### C. Section 6 / Tables 5–7 & Figure 3: SwiGLU Causal Interventions
Evaluates 5 training-time interventions on SwiGLU (FineWeb-Edu, seeds 42, 52, 62):
1. **Hard Activation Clipping (AHC):** Bounding down-proj input $h' = \text{clamp}(h, -12, 12)$ (`conditions/A1_hard_clamp.json`)
2. **Soft Activation Clipping (ASC):** Differentiable bound $h' = 12 \tanh(h / 12)$ (`conditions/A2_soft_clamp.json`)
3. **Weight Cap (WC):** Projection $W_d \leftarrow \text{clamp}(W_d, -0.35, 0.35)$ after each optimizer step (`conditions/W1_weight_cap.json`)
4. **Early Inoculation (EI):** Coordinate perturbation at step 200 by $+2 \text{mean}|W_d|$ (`conditions/I1_early_inoc.json`)
5. **Mid Inoculation (MI):** Coordinate perturbation at step 1000 by $+2 \text{mean}|W_d|$ (`conditions/I2_mid_inoc.json`)
6. **Unperturbed Baseline:** Seed-matched SwiGLU control (`conditions/C0_noop.json`)

```bash
# Run a single intervention directly via CLI:
uv run swx train-twins --variant swiglu --dataset fineweb-edu --seed 42 \
  --condition-manifest conditions/A1_hard_clamp.json

# Run all interventions across all seeds:
uv run python scripts/reproduce_causal_interventions.py
```

### D. Appendix C / Figure 4: Complete Multi-Seed Lag Dynamics Grid
Generates the comprehensive $6 \times 3$ grid of instantaneous dominant-coordinate dynamics across all 6 activation variants (rows) and all 3 corpora (columns) for $N=3$ random seeds:

```bash
uv run swx plot-lag --runs-dir runs/training_twins --out-dir runs/training_twins/plots
```

### E. Trajectory Plots & Summary Metrics
After completing training runs, generate per-dataset trajectory plots, intervention comparisons, and `trajectory_summary.json` files via:

```bash
uv run swx plot-trajectories --run-dir runs/training_twins/fineweb-edu --dataset fineweb-edu
uv run swx plot-trajectories --run-dir runs/training_twins/c4 --dataset c4
uv run swx plot-trajectories --run-dir runs/training_twins/slimpajama --dataset slimpajama
uv run swx plot-trajectories --run-dir runs/training_twins/interventions --baseline-dir runs/training_twins/fineweb-edu
```

Combined with `uv run swx plot-lag`, this produces:
- **Multi-seed lag dynamics & composite $6 \times 3$ grid:** `composite_lag_dynamics_grid.png`, `swiglu_cross_dataset_lag_dynamics.png`, `swiglu_cross_dataset_dominance.png`
- **Cross-variant & intervention plots:** `cross_variant_dominance.png`, `cross_variant_outlier_ratio.png`, `cross_variant_lag_dynamics.png`, `cross_variant_loss.png`, plus per-dataset/intervention `trajectory_summary.json` summaries.

---

## 5. Interactive Inspection & REPL

Verify the catastrophic degradation caused by super weight ablation interactively:

```bash
# Compare intact vs. ablated generation live in your terminal:
uv run swx chat --model olmo-1b --compare

# Show model specifications and tabulated super weight coordinates:
uv run swx list-models
```

---

## 6. Directory Structure

```
paper_artifacts/
├── LICENSE                     # Apache 2.0 license
├── pyproject.toml              # Build specification, dependencies, and entrypoints
├── README.md                   # This camera-ready documentation
├── .gitignore                  # Exclusion rules for local runs, checkpoints, and caches
├── conditions/                 # Intervention manifests
│   ├── A1_hard_clamp.json      # Hard activation clipping manifest (bound = 12.0)
│   ├── A2_soft_clamp.json      # Soft activation clipping manifest (tanh saturation)
│   ├── W1_weight_cap.json      # Weight projection manifest (cap = 0.35)
│   ├── I1_early_inoc.json      # Early coordinate inoculation manifest (step 200)
│   ├── I2_mid_inoc.json        # Mid coordinate inoculation manifest (step 1000)
│   └── C0_noop.json            # Unperturbed control baseline manifest
├── scripts/                    # Reproduction and visualization scripts
│   ├── reproduce_all.sh        # Master shell runner (--smoke-test or --full)
│   ├── reproduce_all.ps1       # Master PowerShell runner (--smoke-test or --full)
│   ├── reproduce_table1_zoo_sweep.py      # Section 4 Table 1 runner
│   ├── reproduce_training_twins.py        # Section 5 Pretraining grid runner
│   ├── reproduce_causal_interventions.py  # Section 6 SwiGLU interventions runner
│   └── plot_multi_seed_lag_dynamics.py    # Multi-seed lag dynamics and composite grid plotter
├── src/swx/                    # Source code package
│   ├── cli.py                  # CLI entrypoint for all paper workflows
│   ├── registry.py             # Experiment registry
│   ├── chat.py                 # Interactive REPL for live intact vs ablated generation
│   ├── core/                   # Core inference and detection infrastructure
│   │   ├── args.py             # Configuration dataclasses and CLI parsing
│   │   ├── checkpoints.py      # Remote weight tensor range slicing via HTTP
│   │   ├── data.py             # Evaluation prompts and data helpers
│   │   ├── detection.py        # Activation-guided super weight detection & dominance
│   │   ├── logging_utils.py    # Structured run logging and table formatting
│   │   ├── metrics.py          # Numerical stability and high-precision metrics
│   │   ├── modeling.py         # Model loading, activation hooks, weight zeroing
│   │   └── sw_directory.py     # Registry of 30 models and tabulated coordinates
│   └── s46_formation/          # Formation, pretraining, and intervention logic
│       ├── _common.py          # Multi-model sweep helpers and metrics
│       ├── g1_activation_family.py   # Table 1 30-model sweep harness
│       ├── train_activation_twins.py # 124M param-matched pretraining loop & interventions
│       └── plot_trajectories.py      # Telemetry aggregation and trajectory plotting
└── tests/                      # Unit test suite (173 passing test cases)
    ├── test_chat.py            # Live ablation generation tests
    ├── test_dataset_config.py  # Dataset resolution and token cache tests
    ├── test_detection.py       # Activation-guided detection & dominance tests
    ├── test_directory.py       # Model zoo architecture and coordinate checks
    ├── test_metrics.py         # Numerical precision and metric invariant tests
    ├── test_plot_trajectories.py      # Multi-seed trajectory envelope and summary tests
    └── test_training_interventions.py # Forward/backward passes with causal interventions
```

---

