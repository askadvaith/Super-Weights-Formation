#!/usr/bin/env python3
"""
plot_multi_seed_lag_dynamics.py

Generates consolidated multi-seed lag dynamics plots
(|X_ik*| activation spike vs. |W_j*k*| dominant weight magnitude)
for each activation function on each dataset across all random seeds.

Output is written to a dedicated directory inside runs/training_twins.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Ensure src is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swx.s46_formation.plot_trajectories import (
    RunData,
    _load_runs,
    aggregate_metric_trajectories,
    dataset_display_name,
    variant_display_name,
    ENVELOPE_ALPHA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATASET_DIRS = ["fineweb-edu", "c4", "slimpajama"]
VARIANT_ORDER = ["swiglu", "geglu", "reglu", "relu2_std", "gelu_std", "silu_std"]


def _get_dominant_weight(r: dict[str, Any]) -> float:
    """Extract |W[j*, k*]| at the dominant coordinate from snapshot record."""
    layer_idx = r.get("top_dominance_layer", -1)
    if layer_idx >= 0 and layer_idx < len(r.get("per_layer", [])):
        pl = r["per_layer"][layer_idx]
        dw = pl.get("dominant_weight_value")
        if dw is not None:
            return abs(dw)
    return abs(r.get("top_weight_value", 0.0))


def plot_single_variant_multi_seed_lag(
    dataset_name: str,
    variant: str,
    runs: list[RunData],
    out_path: Path,
    dpi: int = 150,
) -> bool:
    """Plot dual-axis lag dynamics across seeds for a single variant on a dataset."""
    if not runs:
        log.warning("No runs provided for %s / %s", dataset_name, variant)
        return False

    # Aggregate input activation spike |X_ik*|
    agg_act = aggregate_metric_trajectories(
        runs,
        lambda r: r.get("top_input_spike") if r.get("top_dominance", 0) > 0 else None,
    )

    # Aggregate dominant weight magnitude |W_j*k*|
    agg_wt = aggregate_metric_trajectories(
        runs,
        lambda r: _get_dominant_weight(r) if r.get("top_dominance", 0) > 0 else None,
    )

    if not agg_act or not agg_wt:
        log.warning("Failed to aggregate lag metrics for %s / %s", dataset_name, variant)
        return False

    v_disp = variant_display_name(variant)
    n_seeds = len(runs)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    act_color = "#e63946"  # Red
    wt_color = "#457b9d"   # Steel Blue

    # 1. Plot individual seed series faintly
    for seed_k, (s_steps, s_vals) in sorted(agg_act.seed_series.items(), key=lambda x: str(x[0])):
        ax1.plot(s_steps, s_vals, color=act_color, alpha=0.30, linewidth=1.0, linestyle=":",
                 label=f"|X_ik*| (Seed {seed_k})" if n_seeds <= 3 else None)

    for seed_k, (s_steps, s_vals) in sorted(agg_wt.seed_series.items(), key=lambda x: str(x[0])):
        ax2.plot(s_steps, s_vals, color=wt_color, alpha=0.30, linewidth=1.0, linestyle=":",
                 label=f"|W_j*k*| (Seed {seed_k})" if n_seeds <= 3 else None)

    # 2. Plot mean lines and min-max envelopes
    ln1 = ax1.plot(
        agg_act.steps, agg_act.mean, color=act_color, linewidth=2.2,
        label=f"|X_ik*| mean (activation at dominant k*, N={n_seeds})",
    )
    ax1.fill_between(
        agg_act.steps, agg_act.lower, agg_act.upper,
        color=act_color, alpha=ENVELOPE_ALPHA,
    )

    ln2 = ax2.plot(
        agg_wt.steps, agg_wt.mean, color=wt_color, linewidth=2.2, linestyle="--",
        label=f"|W_j*k*| mean (weight at dominant coord, N={n_seeds})",
    )
    ax2.fill_between(
        agg_wt.steps, agg_wt.lower, agg_wt.upper,
        color=wt_color, alpha=ENVELOPE_ALPHA,
    )

    # Formatting axes and labels
    ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel("|X_ik*| (mean down_proj input at dominant k*)", color=act_color, fontsize=11, fontweight="bold")
    ax2.set_ylabel("|W_j*k*| (weight at dominant coordinate)", color=wt_color, fontsize=11, fontweight="bold")

    ax1.tick_params(axis="y", labelcolor=act_color)
    ax2.tick_params(axis="y", labelcolor=wt_color)
    ax1.tick_params(axis="x")

    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title(
        f"{dataset_name}: {v_disp} Instantaneous Dominant-Candidate Dynamics (N={n_seeds} seeds)",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", framealpha=0.9, fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)
    return True


def generate_composite_grid(
    dataset_runs: dict[str, dict[str, list[RunData]]],
    out_path: Path,
    dpi: int = 200,
) -> None:
    """Generate a clean 6 (variants) x 3 (datasets) multi-seed lag dynamics grid."""
    n_rows = len(VARIANT_ORDER)
    n_cols = len(DATASET_DIRS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 18), sharex=True)

    act_color = "#e63946"
    wt_color = "#457b9d"

    for row_idx, var in enumerate(VARIANT_ORDER):
        v_disp = variant_display_name(var)
        for col_idx, ds_key in enumerate(DATASET_DIRS):
            ax1 = axes[row_idx, col_idx]
            ax2 = ax1.twinx()

            ds_disp = dataset_display_name(ds_key)
            runs = dataset_runs.get(ds_key, {}).get(var, [])
            if not runs:
                ax1.text(0.5, 0.5, "No Data", ha="center", va="center")
                continue

            agg_act = aggregate_metric_trajectories(
                runs, lambda r: r.get("top_input_spike") if r.get("top_dominance", 0) > 0 else None
            )
            agg_wt = aggregate_metric_trajectories(
                runs, lambda r: _get_dominant_weight(r) if r.get("top_dominance", 0) > 0 else None
            )

            if agg_act and agg_wt:
                ax1.plot(agg_act.steps, agg_act.mean, color=act_color, linewidth=1.8)
                ax1.fill_between(agg_act.steps, agg_act.lower, agg_act.upper, color=act_color, alpha=0.15)
                ax2.plot(agg_wt.steps, agg_wt.mean, color=wt_color, linewidth=1.8, linestyle="--")
                ax2.fill_between(agg_wt.steps, agg_wt.lower, agg_wt.upper, color=wt_color, alpha=0.15)

            if row_idx == 0:
                ax1.set_title(ds_disp, fontsize=13, fontweight="bold", pad=8)
            if col_idx == 0:
                ax1.set_ylabel(f"{v_disp}\n|X_ik*|", color=act_color, fontsize=10, fontweight="bold")
            if col_idx == n_cols - 1:
                ax2.set_ylabel(f"|W_j*k*|", color=wt_color, fontsize=10, fontweight="bold")

            ax1.tick_params(axis="y", labelcolor=act_color, labelsize=8)
            ax2.tick_params(axis="y", labelcolor=wt_color, labelsize=8)
            ax1.tick_params(axis="x", labelsize=8)
            ax1.grid(True, linestyle="--", alpha=0.25)

            if row_idx == n_rows - 1:
                ax1.set_xlabel("Step", fontsize=9)

    fig.suptitle(
        "Instantaneous Dominant-Candidate Dynamics across Variants and Datasets (N=3 seeds)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.96)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved composite grid: %s", out_path)


def format_k_ticks(x: float, pos: Any) -> str:
    """Format step numbers as clean thousands: 0, 5k, 10k, etc."""
    if x == 0:
        return "0"
    return f"{int(x / 1000)}k"


def plot_swiglu_cross_dataset_lag_dynamics(
    dataset_runs: dict[str, dict[str, list[RunData]]],
    out_path_1x3: Path,
    out_path_2row: Path,
    dpi: int = 300,
) -> None:
    """Generate consolidated SwiGLU lag dynamics comparing FineWeb-Edu, C4, and SlimPajama.

    Produces both a 1x3 panel and a 2-row layout with shared y-limits for 1:1 comparability.
    """
    act_color = "#e63946"
    wt_color = "#457b9d"

    extracted: dict[str, tuple[Any, Any]] = {}
    for ds_key in DATASET_DIRS:
        runs = dataset_runs.get(ds_key, {}).get("swiglu", [])
        if not runs:
            continue
        agg_act = aggregate_metric_trajectories(
            runs, lambda r: r.get("top_input_spike") if r.get("top_dominance", 0) > 0 else None
        )
        agg_wt = aggregate_metric_trajectories(
            runs, lambda r: _get_dominant_weight(r) if r.get("top_dominance", 0) > 0 else None
        )
        if agg_act and agg_wt:
            extracted[ds_key] = (agg_act, agg_wt)

    if len(extracted) < len(DATASET_DIRS):
        log.warning("Incomplete dataset runs for SwiGLU cross-dataset lag plot: %d found", len(extracted))
        return

    # 1. 1x3 Panel Layout (tuned for NeurIPS page width and margins)
    fig1, axes1 = plt.subplots(1, 3, figsize=(7.2, 2.75))
    ln_act, ln_wt, ln_seed = None, None, None

    for i, ds_key in enumerate(DATASET_DIRS):
        ax1 = axes1[i]
        ax2 = ax1.twinx()
        ds_disp = dataset_display_name(ds_key)
        agg_act, agg_wt = extracted[ds_key]

        for seed_k, (s_steps, s_vals) in sorted(agg_act.seed_series.items(), key=lambda x: str(x[0])):
            s_line = ax1.plot(s_steps, s_vals, color=act_color, alpha=0.25, linewidth=0.7, linestyle=":")[0]
            if ln_seed is None:
                ln_seed = s_line
        for seed_k, (s_steps, s_vals) in sorted(agg_wt.seed_series.items(), key=lambda x: str(x[0])):
            ax2.plot(s_steps, s_vals, color=wt_color, alpha=0.25, linewidth=0.7, linestyle=":")

        l1 = ax1.plot(agg_act.steps, agg_act.mean, color=act_color, linewidth=2.0)[0]
        ax1.fill_between(agg_act.steps, agg_act.lower, agg_act.upper, color=act_color, alpha=0.18)
        if ln_act is None:
            ln_act = l1

        l2 = ax2.plot(agg_wt.steps, agg_wt.mean, color=wt_color, linewidth=2.0, linestyle="--")[0]
        ax2.fill_between(agg_wt.steps, agg_wt.lower, agg_wt.upper, color=wt_color, alpha=0.18)
        if ln_wt is None:
            ln_wt = l2

        ax1.set_title(ds_disp, fontsize=12.5, fontweight="bold", pad=5)
        ax1.set_xlabel("Training Step", fontsize=10.5, fontweight="bold", labelpad=3)
        ax1.set_xlim(-200, 20200)
        ax1.set_xticks([0, 5000, 10000, 15000, 20000])
        ax1.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        ax1.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)

        ax1.set_ylim(-0.3, 10.5)
        ax2.set_ylim(0.0, 0.72)

        ax1.tick_params(axis="x", pad=2, length=3, labelsize=9.5)
        ax1.tick_params(axis="y", labelcolor=act_color, pad=2, length=3, labelsize=9.5)
        ax2.tick_params(axis="y", labelcolor=wt_color, pad=2, length=3, labelsize=9.5)

        if i == 0:
            ax1.set_ylabel("|X_ik*| (input surge)", color=act_color, fontsize=11, fontweight="bold", labelpad=2)
        else:
            ax1.set_ylabel("")

        if i == 2:
            ax2.set_ylabel("|W_j*k*| (dominant weight)", color=wt_color, fontsize=11, fontweight="bold", labelpad=2)
        else:
            ax2.set_ylabel("")

    fig1.legend(
        [ln_act, ln_wt, ln_seed],
        ["|X_ik*| mean (input surge)", "|W_j*k*| mean (dominant weight)", "Seeds (N=3)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        framealpha=0.95,
        fontsize=10.0,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.25,
    )
    plt.tight_layout()
    fig1.subplots_adjust(top=0.86, wspace=0.32)
    out_path_1x3.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_path_1x3, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig1)
    log.info("Saved SwiGLU 1x3 composite: %s", out_path_1x3)

    # 2. 2-Row Layout (2 Top, 1 Bottom Centered)
    fig2 = plt.figure(figsize=(11, 7.5))
    gs = fig2.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.28)
    ax_tl = fig2.add_subplot(gs[0, 0])
    ax_tr = fig2.add_subplot(gs[0, 1])
    gs_b = gs[1, :].subgridspec(1, 3, width_ratios=[0.25, 1, 0.25])
    ax_b = fig2.add_subplot(gs_b[0, 1])
    axes2 = [ax_tl, ax_tr, ax_b]

    for i, ds_key in enumerate(DATASET_DIRS):
        ax1 = axes2[i]
        ax2 = ax1.twinx()
        ds_disp = dataset_display_name(ds_key)
        agg_act, agg_wt = extracted[ds_key]

        for seed_k, (s_steps, s_vals) in sorted(agg_act.seed_series.items(), key=lambda x: str(x[0])):
            ax1.plot(s_steps, s_vals, color=act_color, alpha=0.25, linewidth=0.9, linestyle=":")
        for seed_k, (s_steps, s_vals) in sorted(agg_wt.seed_series.items(), key=lambda x: str(x[0])):
            ax2.plot(s_steps, s_vals, color=wt_color, alpha=0.25, linewidth=0.9, linestyle=":")

        ax1.plot(agg_act.steps, agg_act.mean, color=act_color, linewidth=2.4)
        ax1.fill_between(agg_act.steps, agg_act.lower, agg_act.upper, color=act_color, alpha=ENVELOPE_ALPHA)

        ax2.plot(agg_wt.steps, agg_wt.mean, color=wt_color, linewidth=2.4, linestyle="--")
        ax2.fill_between(agg_wt.steps, agg_wt.lower, agg_wt.upper, color=wt_color, alpha=ENVELOPE_ALPHA)

        ax1.set_title(ds_disp, fontsize=13, fontweight="bold", pad=8)
        ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
        ax1.set_xlim(-200, 20200)
        ax1.set_xticks([0, 5000, 10000, 15000, 20000])
        ax1.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        ax1.grid(True, linestyle="--", alpha=0.3)

        ax1.set_ylim(-0.3, 10.5)
        ax2.set_ylim(0.0, 0.72)

        ax1.tick_params(axis="y", labelcolor=act_color, labelsize=10)
        ax2.tick_params(axis="y", labelcolor=wt_color, labelsize=10)
        ax1.tick_params(axis="x", labelsize=10)

        ax1.set_ylabel("|X_ik*| (input activation)", color=act_color, fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("|W_j*k*| (dominant weight)", color=wt_color, fontsize=10.5, fontweight="bold")

    fig2.legend(
        [ln_act, ln_wt, ln_seed],
        ["|X_ik*| mean (input activation surge)", "|W_j*k*| mean (dominant parameter weight)", "Individual seed runs (N=3)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        framealpha=0.95,
        fontsize=11,
    )
    fig2.subplots_adjust(top=0.92)
    out_path_2row.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_path_2row, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig2)
    log.info("Saved SwiGLU 2-row composite: %s", out_path_2row)


def plot_swiglu_cross_dataset_dominance(
    dataset_runs: dict[str, dict[str, list[RunData]]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot SwiGLU dominance trajectory comparison across all 3 datasets."""
    colors = {"fineweb-edu": "#1f77b4", "c4": "#ff7f0e", "slimpajama": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for ds_key in DATASET_DIRS:
        runs = dataset_runs.get(ds_key, {}).get("swiglu", [])
        if not runs:
            continue
        ds_disp = dataset_display_name(ds_key)
        agg = aggregate_metric_trajectories(
            runs, lambda r: r.get("top_dominance") if r.get("top_dominance", 0) > 0 else None
        )
        if not agg:
            continue
        c = colors.get(ds_key, "#333333")
        ax.plot(agg.steps, agg.mean, color=c, linewidth=2.4, label=f"{ds_disp} (N=3)")
        ax.fill_between(agg.steps, agg.lower, agg.upper, color=c, alpha=ENVELOPE_ALPHA)

    ax.axhline(y=0.4, color="#4B5563", linestyle=":", linewidth=1.8, label="D = 0.4 Threshold")
    ax.set_title("SwiGLU Dominance Trajectories across Datasets (N=3 seeds)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dominance D(t)", fontsize=11, fontweight="bold")
    ax.set_xlim(-200, 20200)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved SwiGLU dominance comparison: %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot multi-seed lag dynamics for each activation and dataset.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT_DIR / "runs" / "training_twins",
        help="Base directory containing dataset runs (e.g. fineweb-edu, c4, slimpajama)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Defaults to runs-dir / multi_seed_lag_dynamics",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")

    args = parser.parse_args()
    runs_dir: Path = args.runs_dir
    out_dir: Path = args.out_dir or (runs_dir / "multi_seed_lag_dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Runs base directory: %s", runs_dir)
    log.info("Output directory: %s", out_dir)

    dataset_runs: dict[str, dict[str, list[RunData]]] = {}
    total_plots = 0

    for ds_key in DATASET_DIRS:
        ds_path = runs_dir / ds_key
        if not ds_path.exists():
            log.warning("Dataset directory not found: %s", ds_path)
            continue

        ds_disp = dataset_display_name(ds_key)
        runs = _load_runs(ds_path)
        runs_by_var: dict[str, list[RunData]] = {}
        for r in runs:
            if not r.is_intervention:
                runs_by_var.setdefault(r.base_variant, []).append(r)

        dataset_runs[ds_key] = runs_by_var

        for var in VARIANT_ORDER:
            v_runs = runs_by_var.get(var, [])
            if not v_runs:
                log.warning("No runs found for %s / %s", ds_key, var)
                continue

            # Save in dataset subdirectory
            sub_path = out_dir / ds_key / f"{var}_lag_dynamics.png"
            # Also save flattened filename in out_dir root
            flat_path = out_dir / f"{ds_key}_{var}_lag_dynamics.png"

            success = plot_single_variant_multi_seed_lag(
                dataset_name=ds_disp,
                variant=var,
                runs=v_runs,
                out_path=sub_path,
                dpi=args.dpi,
            )
            if success:
                # Mirror flat path
                import shutil
                shutil.copyfile(sub_path, flat_path)
                total_plots += 1

    # Generate composite grid
    composite_path = out_dir / "composite_lag_dynamics_grid.png"
    generate_composite_grid(dataset_runs, composite_path, dpi=200)

    # Generate SwiGLU cross-dataset plots
    swiglu_1x3_path = out_dir / "swiglu_cross_dataset_lag_dynamics.png"
    swiglu_2row_path = out_dir / "swiglu_cross_dataset_lag_dynamics_2row.png"
    plot_swiglu_cross_dataset_lag_dynamics(dataset_runs, swiglu_1x3_path, swiglu_2row_path, dpi=300)

    swiglu_dom_path = out_dir / "swiglu_cross_dataset_dominance.png"
    plot_swiglu_cross_dataset_dominance(dataset_runs, swiglu_dom_path, dpi=300)

    log.info("Successfully generated %d multi-seed lag dynamics plots in %s", total_plots, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
