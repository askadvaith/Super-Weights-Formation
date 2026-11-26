"""
plot_trajectories -- Telemetry analysis and visualisation for the
training-twins experiment.

Reads the telemetry.jsonl produced by `train_activation_twins.train()` and
generates:

  1. Parameter trajectory: top-weight magnitude over training steps.
  2. Outlier ratio evolution: |W_max| / mean|W| per layer over time.
  3. Kurtosis evolution: weight tensor kurtosis over training.
  4. Adam moment analysis: m_t and v_t for the top candidate.
  5. Cross-variant comparison: overlay trajectories from all 6 variants.

Output is a set of PNG plots and a summary JSON in the run directory.

Usage:
    swx plot-trajectories --run-dir runs/training_twins/m1_swiglu_20260811-050104
    swx plot-trajectories --run-dir runs/training_twins   # compares all variants
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


# Intervention plots use this run as their default SwiGLU control.  Leave the
# default sentinel in place to select the newest ordinary SwiGLU run under the
# runs directory, or replace it with a concrete run-directory name (for
# example, "m1_swiglu_20260827-151555") to pin every comparison to that run.
DEFAULT_BASELINE_RUN = "m1_swiglu_20260827-151555"

VARIANT_DISPLAY_NAMES = {
    "m1_swiglu": "SwiGLU",
    "swiglu": "SwiGLU",
    "m2_geglu": "GeGLU",
    "geglu": "GeGLU",
    "m3_reglu": "ReGLU",
    "reglu": "ReGLU",
    "m4_gelu_std": "GELU",
    "gelu_std": "GELU",
    "m5_silu_std": "SiLU",
    "silu_std": "SiLU",
    "m6_relu2_std": "ReLU²",
    "relu2_std": "ReLU²",
}

# These labels deliberately describe the experimental manipulation rather than
# exposing manifest IDs such as A1 or I2 in plot legends.
CONDITION_SHORT_LABELS = {
    "a1_hard_clamp": "AHC",
    "a2_soft_clamp": "ASC",
    "i1_early_inoc": "EI",
    "i2_mid_inoc": "MI",
    "w1_weight_cap": "WC",
    "o1_adam_cap": "UC",
}

CONDITION_DISPLAY_NAMES = {
    "a1_hard_clamp": "Activation hard clamp",
    "a2_soft_clamp": "Activation soft clamp",
    "i1_early_inoc": "Early inoculation",
    "i2_mid_inoc": "Mid inoculation",
    "w1_weight_cap": "Weight cap",
    "o1_adam_cap": "Update cap",
}

INTERVENTION_COLOR_MAP = {
    "AHC": "#2563EB",  # Cobalt / Royal Blue
    "ASC": "#0D9488",  # Teal / Cyan
    "EI": "#EA580C",   # Vivid Orange
    "MI": "#9333EA",   # Purple
    "WC": "#0284C7",   # Sky Blue
    "UC": "#B45309",   # Warm Amber / Brown
}

ENVELOPE_ALPHA = 0.18


@dataclass
class RunData:
    """One telemetry run plus the config metadata needed for plotting."""

    run_id: str
    telemetry_path: Path
    records: list[dict[str, Any]]
    base_variant: str
    condition_name: str | None
    intervention: dict[str, Any] | None
    seed: int | None = None
    dataset: str | None = None

    @property
    def is_intervention(self) -> bool:
        return bool(self.condition_name or self.intervention)

    @property
    def short_label(self) -> str:
        if not self.is_intervention:
            return "Base"
        key = (self.condition_name or "").lower()
        if key in CONDITION_SHORT_LABELS:
            return CONDITION_SHORT_LABELS[key]
        kind = (self.intervention or {}).get("kind", "intervention")
        return str(kind).replace("_", " ").title()

    @property
    def display_name(self) -> str:
        base = variant_display_name(self.base_variant)
        if not self.is_intervention:
            return base
        key = (self.condition_name or "").lower()
        condition = CONDITION_DISPLAY_NAMES.get(key, self.condition_name or "Intervention")
        return f"{base} — {condition} ({self.short_label})"

    @property
    def file_stem(self) -> str:
        if not self.is_intervention:
            return self.base_variant
        return f"{self.base_variant}_{self.short_label.lower()}"


DATASET_DISPLAY_NAMES = {
    "c4": "C4",
    "allenai/c4": "C4",
    "slimpajama": "SlimPajama",
    "slimpajama-6b": "SlimPajama",
    "dkyoon/slimpajama-6b": "SlimPajama",
    "fineweb-edu": "FineWeb-Edu",
    "fineweb_edu": "FineWeb-Edu",
    "huggingfacefw/fineweb-edu": "FineWeb-Edu",
    "seeds_intervention": "FineWeb-Edu",
}


def dataset_display_name(dataset: str | None) -> str:
    """Return a readable display name for a training dataset."""
    if not dataset:
        return "Training Twins"
    key = str(dataset).lower().strip()
    return DATASET_DISPLAY_NAMES.get(key, str(dataset))


def resolve_dataset_name(
    run_dir: Path,
    runs: list[RunData] | None = None,
    explicit_name: str | None = None,
) -> str:
    """Determine the user-facing dataset label for plot titles."""
    if explicit_name:
        return dataset_display_name(explicit_name)

    dir_name = run_dir.name.lower()
    if dir_name in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[dir_name]

    if runs:
        for r in runs:
            if r.dataset:
                name = dataset_display_name(r.dataset)
                if name != "Training Twins":
                    return name

    cleaned = run_dir.name.replace("_", " ").title()
    return cleaned if cleaned else "Training Twins"


@dataclass
class AggregatedSeries:
    """Aligned time series with statistics aggregated across multiple seeds."""

    steps: list[int]
    mean: list[float]
    lower: list[float]   # min across seeds for envelope
    upper: list[float]   # max across seeds for envelope
    std: list[float]
    n_seeds: int
    seed_series: dict[int | str, tuple[list[int], list[float]]]


def aggregate_metric_trajectories(
    runs: list[RunData],
    field_extractor: Any,
) -> AggregatedSeries | None:
    """Aggregate a metric across multiple seed runs, aligned by step.

    Parameters
    ----------
    runs : list[RunData]
        List of runs for the same condition/variant.
    field_extractor : str or Callable[[dict], float | None]
        Either a string key in telemetry record dict, or a callable.

    Returns
    -------
    AggregatedSeries or None if no valid data points exist.
    """
    if not runs:
        return None

    if isinstance(field_extractor, str):
        key = field_extractor
        extractor = lambda r: r.get(key)
    else:
        extractor = field_extractor

    seed_series: dict[int | str, tuple[list[int], list[float]]] = {}
    per_run_data: list[dict[int, float]] = []

    for r in runs:
        run_key = r.seed if r.seed is not None else r.run_id
        if run_key in seed_series:
            run_key = f"{run_key}_{r.run_id}"

        r_steps: list[int] = []
        r_vals: list[float] = []
        step_dict: dict[int, float] = {}

        for rec in r.records:
            val = extractor(rec)
            if val is not None:
                try:
                    fval = float(val)
                    if not (np.isnan(fval) or np.isinf(fval)):
                        s = rec["step"]
                        r_steps.append(s)
                        r_vals.append(fval)
                        step_dict[s] = fval
                except (ValueError, TypeError):
                    continue

        if r_steps:
            seed_series[run_key] = (r_steps, r_vals)
            per_run_data.append(step_dict)

    if not per_run_data:
        return None

    all_steps = sorted(set().union(*[d.keys() for d in per_run_data]))
    steps: list[int] = []
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    stds: list[float] = []

    for s in all_steps:
        vals = [d[s] for d in per_run_data if s in d]
        if vals:
            steps.append(s)
            means.append(float(np.mean(vals)))
            lowers.append(float(np.min(vals)))
            uppers.append(float(np.max(vals)))
            stds.append(float(np.std(vals)))

    if not steps:
        return None

    return AggregatedSeries(
        steps=steps,
        mean=means,
        lower=lowers,
        upper=uppers,
        std=stds,
        n_seeds=len(seed_series),
        seed_series=seed_series,
    )


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


def _read_telemetry(path: Path) -> list[dict[str, Any]]:
    """Read a telemetry.jsonl file into a list of snapshot dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def variant_display_name(variant: str) -> str:
    """Return a readable name for a canonical architecture variant."""
    return VARIANT_DISPLAY_NAMES.get(variant.lower(), variant)


def _extract_variant_name(dir_path: Path) -> str:
    """Extract full variant name from config.json or directory name."""
    config_path = dir_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.loads(f.read())
                if "variant" in cfg and cfg["variant"]:
                    return cfg["variant"]
        except Exception:
            pass

    name = dir_path.name
    known_keys = [
        "m1_swiglu", "m2_geglu", "m3_reglu", "m4_gelu_std", "m5_silu_std", "m6_relu2_std",
        "gelu_std", "silu_std", "relu2_std", "swiglu", "geglu", "reglu",
    ]
    for key in known_keys:
        if name.startswith(key):
            return key

    return name.split("_")[0] if "_" in name else name


def _read_run_metadata(dir_path: Path, run_id: str) -> RunData:
    """Load one run's plotting metadata without conflating it with its key."""
    config: dict[str, Any] = {}
    config_path = dir_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read config for %s: %s", dir_path, exc)

    base_variant = str(config.get("variant") or _extract_variant_name(dir_path))
    condition_name = config.get("condition_name") or None
    intervention = config.get("intervention")
    if intervention is not None and not isinstance(intervention, dict):
        log.warning("Ignoring malformed intervention metadata in %s", config_path)
        intervention = None

    seed = config.get("seed")
    if seed is None:
        m = re.search(r"seed(\d+)", dir_path.name)
        if m:
            seed = int(m.group(1))

    dataset = config.get("dataset_alias") or config.get("dataset")
    if not dataset:
        for candidate in [dir_path.name, dir_path.parent.name]:
            c_lower = candidate.lower()
            if "c4" in c_lower:
                dataset = "c4"
                break
            elif "slimpajama" in c_lower:
                dataset = "slimpajama"
                break
            elif "fineweb" in c_lower:
                dataset = "fineweb-edu"
                break

    return RunData(
        run_id=run_id,
        telemetry_path=dir_path / "telemetry.jsonl",
        records=[],
        base_variant=base_variant,
        condition_name=str(condition_name) if condition_name else None,
        intervention=intervention,
        seed=seed,
        dataset=dataset,
    )


def _load_runs(run_dir: Path) -> list[RunData]:
    """Discover telemetry files and attach their immutable run metadata."""
    runs: list[RunData] = []
    for run_id, telemetry_path in _find_telemetry_files(run_dir).items():
        run = _read_run_metadata(telemetry_path.parent, run_id)
        run.records = _read_telemetry(telemetry_path)
        runs.append(run)
    return runs


def _find_telemetry_files(run_dir: Path) -> dict[str, Path]:
    """Find all telemetry.jsonl files under a run directory.

    If run_dir itself contains telemetry.jsonl, returns {variant: path}.
    If run_dir contains subdirectories with telemetry.jsonl, returns all.
    """
    results: dict[str, Path] = {}

    direct = run_dir / "telemetry.jsonl"
    if direct.exists():
        variant = _extract_variant_name(run_dir)
        results[variant] = direct
        return results

    # Look in subdirectories
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir():
            tel = sub / "telemetry.jsonl"
            if tel.exists():
                results[sub.name] = tel

    return results



# =============================================================================
# Plot generation
# =============================================================================


def plot_single_variant(
    records: list[dict], variant: str, out_dir: Path,
    *, display_name: str | None = None, file_stem: str | None = None,
    dataset_name: str | None = None,
) -> list[Path]:
    """Generate all plots for a single run's telemetry."""
    if not _HAVE_MPL:
        log.warning("matplotlib not installed; skipping plots.")
        return []

    plots: list[Path] = []
    display_name = display_name or variant
    file_stem = file_stem or variant
    prefix = f"{dataset_name} — " if dataset_name else ""
    steps = [r["step"] for r in records]

    # 1. Top weight ratio over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ratios = [r["top_weight_ratio"] for r in records]
    ax.plot(steps, ratios, "b-", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("|W_max| / mean|W|")
    ax.set_title(f"{prefix}{display_name}: Top Weight Outlier Ratio")
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{file_stem}_outlier_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    # 2. Top weight absolute value over time
    fig, ax = plt.subplots(figsize=(10, 5))
    values = [abs(r["top_weight_value"]) for r in records]
    ax.plot(steps, values, "r-", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("|W_max|")
    ax.set_title(f"{prefix}{display_name}: Top Weight Magnitude")
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{file_stem}_weight_magnitude.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    # 3. Dominance score D(t) evolution (Metric 2 from the plan)
    dom_records = [r for r in records if "top_dominance" in r and r["top_dominance"] > 0]
    if dom_records:
        dom_steps = [r["step"] for r in dom_records]
        dominances = [r["top_dominance"] for r in dom_records]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dom_steps, dominances, color="#8b0000", linewidth=1.5)
        ax.axhline(y=0.4, color="lightgray", linestyle="--", alpha=0.6, label="D=0.4 (non-gated ceiling, fixed probe)")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Dominance D(t)")
        ax.set_title(f"{prefix}{display_name}: Dominance Score Evolution D(t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{file_stem}_dominance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

        def _get_dominant_weight(r: dict) -> float:
            """Extract |W[j*, k*]| at the dominant coordinate from per-layer data."""
            layer_idx = r.get("top_dominance_layer", -1)
            if layer_idx >= 0 and layer_idx < len(r.get("per_layer", [])):
                pl = r["per_layer"][layer_idx]
                dw = pl.get("dominant_weight_value")
                if dw is not None:
                    return abs(dw)
            return abs(r.get("top_weight_value", 0.0))

        input_spikes = [r["top_input_spike"] for r in dom_records]
        weight_mags = [_get_dominant_weight(r) for r in dom_records]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ln1 = ax1.plot(dom_steps, input_spikes, color="#e63946", linewidth=1.5, label="|X_ik*| (activation at dominant k*)")
        ln2 = ax2.plot(dom_steps, weight_mags, color="#457b9d", linewidth=1.5, linestyle="--", label="|W_j*k*| (weight at dominant coord)")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("|X_ik*| (mean down_proj input at dominant k*)", color="#e63946")
        ax2.set_ylabel("|W_j*k*| (weight at dominant coordinate)", color="#457b9d")
        ax1.tick_params(axis="y", labelcolor="#e63946")
        ax2.tick_params(axis="y", labelcolor="#457b9d")
        lns = ln1 + ln2
        ax1.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=8)
        ax1.set_title(f"{prefix}{display_name}: Instantaneous Dominant-Candidate Dynamics")
        ax1.grid(True, alpha=0.3)
        path = out_dir / f"{file_stem}_lag_dynamics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # Fixed coordinates traces
    trace_by_coord: dict[tuple[int, int, int], list[tuple[int, dict]]] = {}
    for record in records:
        for trace in record.get("fixed_coord_traces", []):
            coord = (trace["layer"], trace["j"], trace["k"])
            trace_by_coord.setdefault(coord, []).append((record["step"], trace))
    for (layer, j, k), points in trace_by_coord.items():
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        trace_steps = [step for step, _trace in points]
        activations = [trace["max_abs_x"] for _step, trace in points]
        weights = [abs(trace["weight_value"]) for _step, trace in points]
        ln1 = ax1.plot(trace_steps, activations, color="#e63946", linewidth=1.7,
                       label="max |X_k| on fixed probe")
        ln2 = ax2.plot(trace_steps, weights, color="#457b9d", linewidth=1.7,
                       linestyle="--", label="|W[j,k]|")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("max |X_k|", color="#e63946")
        ax2.set_ylabel("|W[j,k]|", color="#457b9d")
        ax1.tick_params(axis="y", labelcolor="#e63946")
        ax2.tick_params(axis="y", labelcolor="#457b9d")
        lns = ln1 + ln2
        ax1.legend(lns, [line.get_label() for line in lns], loc="upper left", fontsize=8)
        ax1.set_title(f"{prefix}{display_name}: Fixed Coordinate L{layer}[{j},{k}]")
        ax1.grid(True, alpha=0.3)
        path = out_dir / f"{file_stem}_fixed_L{layer}_{j}_{k}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # 4 (combined). Weight and activation kurtosis over time (Metric 5)
    fig, ax = plt.subplots(figsize=(10, 5))
    weight_kurtosis = [r["weight_kurtosis_max"] for r in records]
    ax.plot(steps, weight_kurtosis, "g-", linewidth=1.5, label="Weight kurtosis")
    act_kurtosis_records = [r for r in records if r.get("activation_kurtosis_max", 0) > 0]
    if act_kurtosis_records:
        act_steps = [r["step"] for r in act_kurtosis_records]
        act_kurtosis = [r["activation_kurtosis_max"] for r in act_kurtosis_records]
        ax.plot(act_steps, act_kurtosis, color="#f4a261", linewidth=1.5, label="Activation kurtosis")
    ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="Gaussian (κ=3)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Kurtosis (max across layers)")
    ax.set_title(f"{prefix}{display_name}: Weight and Activation Tensor Kurtosis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{file_stem}_kurtosis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    # 5. Per-layer outlier ratio heatmap (if enough data)
    if records and records[0].get("per_layer"):
        n_layers = len(records[0]["per_layer"])
        fig, ax = plt.subplots(figsize=(12, 6))
        data = []
        for r in records:
            row = [ld["weight_ratio"] for ld in r["per_layer"]]
            data.append(row)

        data_arr = np.array(data).T  # [n_layers, n_steps]
        im = ax.imshow(
            data_arr, aspect="auto", cmap="hot", origin="lower",
            extent=[steps[0], steps[-1], 0, n_layers],
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Layer")
        ax.set_title(f"{prefix}{display_name}: Per-Layer Weight Outlier Ratio")
        plt.colorbar(im, ax=ax, label="|W_max| / mean|W|")
        path = out_dir / f"{file_stem}_per_layer_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

        # Per-layer dominance heatmap (when activation probe data present)
        if records[0]["per_layer"][0].get("dominance") is not None:
            dom_data = []
            for r in records:
                row = [ld.get("dominance", 0.0) for ld in r["per_layer"]]
                dom_data.append(row)
            dom_arr = np.array(dom_data).T
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(
                dom_arr, aspect="auto", cmap="plasma", origin="lower",
                vmin=0.0, vmax=1.0,
                extent=[steps[0], steps[-1], 0, n_layers],
            )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Layer")
            ax.set_title(f"{prefix}{display_name}: Per-Layer Dominance D(t)")
            plt.colorbar(im, ax=ax, label="D(t)")
            path = out_dir / f"{file_stem}_per_layer_dominance.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots.append(path)

    # 5b. Retrospective super-weight trajectory from watchlist data.
    has_watchlist = any(r.get("watched_coords") for r in records)
    if has_watchlist and records:
        final = records[-1]
        target_layer = final.get("top_dominance_layer", -1)
        target_j = final.get("top_dominance_j", -1)
        target_k = final.get("top_dominance_k", -1)

        if target_layer < 0 or target_j < 0:
            target_layer = final.get("top_weight_layer", 0)
            if final.get("per_layer"):
                pl = final["per_layer"][target_layer]
                coord = pl.get("max_weight_coord", [0, 0])
                target_j, target_k = coord[0], coord[1]

        if target_layer >= 0 and target_j >= 0 and target_k >= 0:
            traj_steps: list[int] = []
            traj_values: list[float] = []
            envelope_steps: list[int] = []
            envelope_values: list[float] = []

            for r in records:
                s = r["step"]
                envelope_steps.append(s)
                envelope_values.append(abs(r.get("top_weight_value", 0.0)))

                for entry in r.get("watched_coords", []):
                    if (entry.get("layer") == target_layer and
                            entry.get("j") == target_j and
                            entry.get("k") == target_k):
                        traj_steps.append(s)
                        traj_values.append(abs(entry["weight_value"]))
                        break

            if traj_steps:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(envelope_steps, envelope_values, color="lightgray",
                        linewidth=1.0, linestyle="--", alpha=0.7,
                        label="argmax|W| envelope (may jump coords)")
                ax.plot(traj_steps, [abs(v) for v in traj_values],
                        color="#8b0000", linewidth=2.0,
                        label=f"L{target_layer}[{target_j},{target_k}] (final super weight)")
                ax.set_xlabel("Training Step")
                ax.set_ylabel("|W|")
                ax.set_title(f"{prefix}{display_name}: Super Weight Trajectory "
                             f"L{target_layer}[{target_j},{target_k}]")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                path = out_dir / f"{file_stem}_sw_trajectory.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                plots.append(path)

    # 6. Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    losses = [r["loss"] for r in records]
    ax.plot(steps, losses, "k-", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{prefix}{display_name}: Training Loss")
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{file_stem}_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    log.info("Generated %d plots for %s in %s", len(plots), display_name, out_dir)
    return plots


def plot_multi_seed_variant(
    variant: str,
    runs: list[RunData],
    out_dir: Path,
    dataset_name: str = "Training Twins",
) -> list[Path]:
    """Generate consolidated multi-seed plots for a single variant."""
    if not _HAVE_MPL or len(runs) <= 1:
        return []

    plots: list[Path] = []
    v_name = variant_display_name(variant)
    c = get_variant_color(variant)
    n_seeds = len(runs)

    # 1. Outlier ratio
    agg = aggregate_metric_trajectories(runs, lambda r: r.get("top_weight_ratio"))
    if agg:
        fig, ax = plt.subplots(figsize=(10, 5))
        for seed_k, (s_steps, s_vals) in sorted(agg.seed_series.items(), key=lambda x: str(x[0])):
            ax.plot(s_steps, s_vals, color=c, alpha=0.35, linewidth=1.0, linestyle=":",
                    label=f"Seed {seed_k}")
        ax.plot(agg.steps, agg.mean, color=c, linewidth=2.2, label=f"Mean (N={n_seeds})")
        ax.fill_between(agg.steps, agg.lower, agg.upper, color=c, alpha=ENVELOPE_ALPHA, label="Min–Max envelope")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("|W_max| / mean|W|")
        ax.set_title(f"{dataset_name} — {v_name}: Top Weight Outlier Ratio (N={n_seeds} seeds)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{variant}_multi_seed_outlier_ratio.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # 2. Weight magnitude
    agg_w = aggregate_metric_trajectories(runs, lambda r: abs(r.get("top_weight_value", 0.0)))
    if agg_w:
        fig, ax = plt.subplots(figsize=(10, 5))
        for seed_k, (s_steps, s_vals) in sorted(agg_w.seed_series.items(), key=lambda x: str(x[0])):
            ax.plot(s_steps, s_vals, color=c, alpha=0.35, linewidth=1.0, linestyle=":",
                    label=f"Seed {seed_k}")
        ax.plot(agg_w.steps, agg_w.mean, color=c, linewidth=2.2, label=f"Mean (N={n_seeds})")
        ax.fill_between(agg_w.steps, agg_w.lower, agg_w.upper, color=c, alpha=ENVELOPE_ALPHA, label="Min–Max envelope")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("|W_max|")
        ax.set_title(f"{dataset_name} — {v_name}: Top Weight Magnitude (N={n_seeds} seeds)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{variant}_multi_seed_weight_magnitude.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # 3. Dominance D(t)
    agg_d = aggregate_metric_trajectories(runs, lambda r: r.get("top_dominance") if r.get("top_dominance", 0) > 0 else None)
    if agg_d:
        fig, ax = plt.subplots(figsize=(10, 5))
        for seed_k, (s_steps, s_vals) in sorted(agg_d.seed_series.items(), key=lambda x: str(x[0])):
            ax.plot(s_steps, s_vals, color=c, alpha=0.35, linewidth=1.0, linestyle=":",
                    label=f"Seed {seed_k}")
        ax.plot(agg_d.steps, agg_d.mean, color=c, linewidth=2.2, label=f"Mean (N={n_seeds})")
        ax.fill_between(agg_d.steps, agg_d.lower, agg_d.upper, color=c, alpha=ENVELOPE_ALPHA, label="Min–Max envelope")
        ax.axhline(y=0.4, color="lightgray", linestyle="--", alpha=0.6, label="D=0.4 (non-gated ceiling)")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Dominance D(t)")
        ax.set_title(f"{dataset_name} — {v_name}: Dominance Score Evolution D(t) (N={n_seeds} seeds)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{variant}_multi_seed_dominance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # 4. Kurtosis
    agg_k = aggregate_metric_trajectories(runs, lambda r: r.get("weight_kurtosis_max"))
    if agg_k:
        fig, ax = plt.subplots(figsize=(10, 5))
        for seed_k, (s_steps, s_vals) in sorted(agg_k.seed_series.items(), key=lambda x: str(x[0])):
            ax.plot(s_steps, s_vals, color=c, alpha=0.35, linewidth=1.0, linestyle=":",
                    label=f"Seed {seed_k}")
        ax.plot(agg_k.steps, agg_k.mean, color=c, linewidth=2.2, label=f"Mean (N={n_seeds})")
        ax.fill_between(agg_k.steps, agg_k.lower, agg_k.upper, color=c, alpha=ENVELOPE_ALPHA, label="Min–Max envelope")
        ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="Gaussian (κ=3)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Kurtosis (max across layers)")
        ax.set_title(f"{dataset_name} — {v_name}: Weight Tensor Kurtosis (N={n_seeds} seeds)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{variant}_multi_seed_kurtosis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # 5. Loss
    agg_l = aggregate_metric_trajectories(runs, lambda r: r.get("loss"))
    if agg_l:
        fig, ax = plt.subplots(figsize=(10, 5))
        for seed_k, (s_steps, s_vals) in sorted(agg_l.seed_series.items(), key=lambda x: str(x[0])):
            ax.plot(s_steps, s_vals, color=c, alpha=0.35, linewidth=1.0, linestyle=":",
                    label=f"Seed {seed_k}")
        ax.plot(agg_l.steps, agg_l.mean, color=c, linewidth=2.2, label=f"Mean (N={n_seeds})")
        ax.fill_between(agg_l.steps, agg_l.lower, agg_l.upper, color=c, alpha=ENVELOPE_ALPHA, label="Min–Max envelope")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{dataset_name} — {v_name}: Training Loss (N={n_seeds} seeds)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{variant}_multi_seed_loss.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    return plots


# Color palette: High-contrast, colorblind-safe distinct hues across the color wheel
VARIANT_COLOR_MAP = {
    # Canonical 6 variants
    "m1_swiglu": "#D62728",    # Crimson Red
    "m2_geglu": "#1F77B4",     # Deep Cobalt / Royal Blue
    "m3_reglu": "#E69F00",     # Amber / Golden Orange
    "m4_gelu_std": "#2CA02C",   # Emerald Green
    "m5_silu_std": "#334155",   # Dark Slate / Charcoal
    "m6_relu2_std": "#9467BD",  # Royal / Vivid Purple
    # Aliases
    "swiglu": "#D62728",
    "geglu": "#1F77B4",
    "reglu": "#E69F00",
    "gelu_std": "#2CA02C",
    "silu_std": "#334155",
    "relu2_std": "#9467BD",
    "m1": "#D62728",
    "m2": "#1F77B4",
    "m3": "#E69F00",
    "m4": "#2CA02C",
    "m5": "#334155",
    "m6": "#9467BD",
}

FALLBACK_COLORS = [
    "#D62728", "#1F77B4", "#E69F00", "#2CA02C", "#334155", "#9467BD",
    "#E76F51", "#2A9D8F", "#E9C46A", "#17BECF", "#D55E00", "#CC79A7",
]


def get_variant_color(variant: str, all_variants: list[str] | None = None) -> str:
    """Return a distinct, high-contrast color for a given variant name.

    Tries exact lookup, canonical alias mapping, substring matching,
    and dynamic palette fallback so no curve is ever rendered in generic gray.
    """
    v_lower = variant.lower()

    if v_lower in VARIANT_COLOR_MAP:
        return VARIANT_COLOR_MAP[v_lower]

    for key, color in VARIANT_COLOR_MAP.items():
        if key in v_lower:
            return color

    if all_variants:
        try:
            idx = sorted(all_variants).index(variant)
            return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
        except ValueError:
            pass

    idx = sum(ord(c) for c in variant)
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def get_variant_linestyle(variant: str) -> str:
    """Return solid '-' for gated variants and dashed '--' for non-gated variants."""
    v_lower = variant.lower()
    non_gated_keys = [
        "m4", "m5", "m6",
        "gelu", "silu", "relu2", "relu_2", "relu^2",
        "standard", "std",
    ]
    if any(ng in v_lower for ng in non_gated_keys):
        return "--"
    return "-"


def _legend_sorted_by_final_value(ax: Any, **kwargs: Any) -> Any:
    """Sort legend handles and labels in descending order of their final trajectory y-value.

    Ensures that the top-to-bottom order of legend entries matches the
    top-to-bottom order of curves on the right side of the plot.
    Reference and threshold lines (e.g., Gaussian, D=0.4) are placed at the bottom.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    def _get_final_y(handle: Any) -> float:
        if hasattr(handle, "get_ydata"):
            ydata = handle.get_ydata()
            if len(ydata) > 0:
                for val in reversed(ydata):
                    try:
                        fval = float(val)
                        if not (np.isnan(fval) or np.isinf(fval)):
                            return fval
                    except (TypeError, ValueError):
                        continue
        return -float("inf")

    data_items: list[tuple[Any, str, float]] = []
    ref_items: list[tuple[Any, str, float]] = []
    seen_labels: set[str] = set()

    for h, l in zip(handles, labels):
        if l in seen_labels:
            continue
        seen_labels.add(l)
        y_val = _get_final_y(h)
        l_lower = l.lower()
        if any(term in l_lower for term in ("threshold", "gaussian", "ceiling", "reference")):
            ref_items.append((h, l, y_val))
        else:
            data_items.append((h, l, y_val))

    # Sort data curves by final y-value descending
    data_items.sort(key=lambda item: item[2], reverse=True)

    sorted_handles = [h for h, _l, _y in data_items] + [h for h, _l, _y in ref_items]
    sorted_labels = [l for _h, l, _y in data_items] + [l for _h, l, _y in ref_items]

    return ax.legend(sorted_handles, sorted_labels, **kwargs)


def plot_cross_variant(
    all_records: dict[str, list[RunData]] | dict[str, list[dict]],
    out_dir: Path,
    dataset_name: str | None = None,
) -> list[Path]:
    """Generate comparison plots across all variants with multi-seed aggregation."""
    if not _HAVE_MPL:
        log.warning("matplotlib not installed; skipping cross-variant plots.")
        return []

    plots: list[Path] = []
    all_var_names = list(all_records.keys())

    # Normalize input to dict[str, list[RunData]]
    grouped_runs: dict[str, list[RunData]] = {}
    for variant, val in all_records.items():
        if not val:
            continue
        if isinstance(val[0], RunData):
            grouped_runs[variant] = val
        elif isinstance(val[0], dict):
            grouped_runs[variant] = [RunData(
                run_id=variant,
                telemetry_path=Path("dummy"),
                records=val,
                base_variant=variant,
                condition_name=None,
                intervention=None,
                seed=42,
            )]

    if dataset_name is None:
        flat_runs = [r for r_list in grouped_runs.values() for r in r_list if str(r.telemetry_path) != "dummy"]
        dataset_name = resolve_dataset_name(out_dir.parent, flat_runs)

    # 1. Outlier ratio comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant, v_runs in sorted(grouped_runs.items()):
        agg = aggregate_metric_trajectories(v_runs, "top_weight_ratio")
        if not agg:
            continue
        c = get_variant_color(variant, all_var_names)
        ls = get_variant_linestyle(variant)
        label = f"{variant_display_name(variant)} (N={agg.n_seeds})" if agg.n_seeds > 1 else variant_display_name(variant)
        ax.plot(agg.steps, agg.mean, color=c, linestyle=ls, linewidth=2.0, label=label)
        if agg.n_seeds > 1:
            ax.fill_between(agg.steps, agg.lower, agg.upper, color=c, alpha=ENVELOPE_ALPHA)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("|W_max| / mean|W|", fontsize=11)
    ax.set_title(f"{dataset_name} — Cross-Variant: Weight Outlier Ratio", fontsize=13, fontweight="bold")
    _legend_sorted_by_final_value(ax, loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / "cross_variant_outlier_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    # 2. Kurtosis comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant, v_runs in sorted(grouped_runs.items()):
        agg = aggregate_metric_trajectories(v_runs, "weight_kurtosis_max")
        if not agg:
            continue
        c = get_variant_color(variant, all_var_names)
        ls = get_variant_linestyle(variant)
        label = f"{variant_display_name(variant)} (N={agg.n_seeds})" if agg.n_seeds > 1 else variant_display_name(variant)
        ax.plot(agg.steps, agg.mean, color=c, linestyle=ls, linewidth=2.0, label=label)
        if agg.n_seeds > 1:
            ax.fill_between(agg.steps, agg.lower, agg.upper, color=c, alpha=ENVELOPE_ALPHA)
    ax.axhline(y=3.0, color="gray", linestyle=":", alpha=0.7, label="Gaussian (κ=3)")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Max Weight Kurtosis", fontsize=11)
    ax.set_title(f"{dataset_name} — Cross-Variant: Weight Tensor Kurtosis", fontsize=13, fontweight="bold")
    _legend_sorted_by_final_value(ax, loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / "cross_variant_kurtosis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    # 3. Dominance D(t) comparison
    dom_variants: dict[str, list[RunData]] = {}
    for v, v_runs in grouped_runs.items():
        if any(any(r.get("top_dominance", 0) > 0 for r in run.records) for run in v_runs):
            dom_variants[v] = v_runs

    if dom_variants:
        dom_names = list(dom_variants.keys())
        fig, ax = plt.subplots(figsize=(12, 6))
        for variant, v_runs in sorted(dom_variants.items()):
            agg = aggregate_metric_trajectories(
                v_runs,
                lambda r: r.get("top_dominance") if r.get("top_dominance", 0) > 0 else None,
            )
            if not agg:
                continue
            c = get_variant_color(variant, dom_names)
            ls = get_variant_linestyle(variant)
            label = f"{variant_display_name(variant)} (N={agg.n_seeds})" if agg.n_seeds > 1 else variant_display_name(variant)
            ax.plot(agg.steps, agg.mean, color=c, linestyle=ls, linewidth=2.0, label=label)
            if agg.n_seeds > 1:
                ax.fill_between(agg.steps, agg.lower, agg.upper, color=c, alpha=ENVELOPE_ALPHA)
        ax.axhline(y=0.4, color="gray", linestyle=":", alpha=0.7, label="D=0.4 threshold (non-gated, fixed probe)")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Dominance D(t)", fontsize=11)
        ax.set_title(f"{dataset_name} — Cross-Variant: Dominance Score D(t)", fontsize=13, fontweight="bold")
        _legend_sorted_by_final_value(ax, loc="upper left", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        path = out_dir / "cross_variant_dominance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

        # 4. Activation spike vs weight magnitude lag comparison
        def _get_dominant_weight_cv(r: dict) -> float:
            layer_idx = r.get("top_dominance_layer", -1)
            if layer_idx >= 0 and layer_idx < len(r.get("per_layer", [])):
                pl = r["per_layer"][layer_idx]
                dw = pl.get("dominant_weight_value")
                if dw is not None:
                    return abs(dw)
            return abs(r.get("top_weight_value", 0.0))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        ax_x, ax_w = axes
        for variant, v_runs in sorted(dom_variants.items()):
            agg_x = aggregate_metric_trajectories(
                v_runs,
                lambda r: r.get("top_input_spike") if r.get("top_dominance", 0) > 0 else None,
            )
            agg_w = aggregate_metric_trajectories(
                v_runs,
                lambda r: _get_dominant_weight_cv(r) if r.get("top_dominance", 0) > 0 else None,
            )
            if not agg_x or not agg_w:
                continue
            c = get_variant_color(variant, dom_names)
            ls = get_variant_linestyle(variant)
            label = f"{variant_display_name(variant)} (N={agg_x.n_seeds})" if agg_x.n_seeds > 1 else variant_display_name(variant)
            ax_x.plot(agg_x.steps, agg_x.mean, color=c, linestyle=ls, linewidth=1.8, label=label)
            if agg_x.n_seeds > 1:
                ax_x.fill_between(agg_x.steps, agg_x.lower, agg_x.upper, color=c, alpha=ENVELOPE_ALPHA)
            ax_w.plot(agg_w.steps, agg_w.mean, color=c, linestyle=ls, linewidth=1.8, label=label)
            if agg_w.n_seeds > 1:
                ax_w.fill_between(agg_w.steps, agg_w.lower, agg_w.upper, color=c, alpha=ENVELOPE_ALPHA)
        ax_x.set_xlabel("Training Step", fontsize=11)
        ax_x.set_ylabel("|X_ik*| (mean activation at dominant k*)", fontsize=11)
        ax_x.set_title("Cross-Variant: Input Activation Spike |X_ik*|", fontsize=12, fontweight="bold")
        _legend_sorted_by_final_value(ax_x, loc="upper left", framealpha=0.9, fontsize=8)
        ax_x.grid(True, alpha=0.3)
        ax_w.set_xlabel("Training Step", fontsize=11)
        ax_w.set_ylabel("|W_j*k*| (weight at dominant coordinate)", fontsize=11)
        ax_w.set_title("Cross-Variant: Weight at Dominant Coordinate |W_j*k*|", fontsize=12, fontweight="bold")
        _legend_sorted_by_final_value(ax_w, loc="upper left", framealpha=0.9, fontsize=8)
        ax_w.grid(True, alpha=0.3)
        fig.suptitle(f"{dataset_name} — Cross-Variant: Activation Spike vs Weight Dynamics", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = out_dir / "cross_variant_lag_dynamics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    # 5. Loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant, v_runs in sorted(grouped_runs.items()):
        agg = aggregate_metric_trajectories(v_runs, "loss")
        if not agg:
            continue
        c = get_variant_color(variant, all_var_names)
        ls = get_variant_linestyle(variant)
        label = f"{variant_display_name(variant)} (N={agg.n_seeds})" if agg.n_seeds > 1 else variant_display_name(variant)
        ax.plot(agg.steps, agg.mean, color=c, linestyle=ls, linewidth=1.5, alpha=0.85, label=label)
        if agg.n_seeds > 1:
            ax.fill_between(agg.steps, agg.lower, agg.upper, color=c, alpha=ENVELOPE_ALPHA)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"{dataset_name} — Cross-Variant: Training Loss", fontsize=13, fontweight="bold")
    _legend_sorted_by_final_value(ax, loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / "cross_variant_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    log.info("Generated %d cross-variant plots in %s", len(plots), out_dir)
    return plots


def _run_sort_key(run: RunData) -> tuple[str, str]:
    """Sort runs by their timestamped directory name, then full path."""
    return (run.telemetry_path.parent.name, str(run.telemetry_path.parent))


def _latest_runs_by_variant(runs: list[RunData]) -> dict[str, RunData]:
    """Choose one current ordinary run per architecture for single-seed fallback."""
    latest: dict[str, RunData] = {}
    for run in runs:
        if run.is_intervention:
            continue
        current = latest.get(run.base_variant)
        if current is None or _run_sort_key(run) > _run_sort_key(current):
            latest[run.base_variant] = run
    return latest


def _single_plot_identifiers(runs: list[RunData]) -> dict[str, tuple[str, str]]:
    """Return collision-free display/file identifiers for per-run plots."""
    base_stems: dict[str, int] = {}
    for run in runs:
        base_stems[run.file_stem] = base_stems.get(run.file_stem, 0) + 1

    identifiers: dict[str, tuple[str, str]] = {}
    for run in runs:
        if base_stems[run.file_stem] == 1:
            display_name = run.display_name
            if run.seed is not None:
                display_name = f"{run.display_name} (Seed {run.seed})"
            identifiers[run.run_id] = (display_name, run.file_stem)
            continue

        directory = run.telemetry_path.parent.name
        seed_label = f"Seed {run.seed}" if run.seed is not None else directory
        file_suffix = f"seed{run.seed}" if run.seed is not None else directory
        identifiers[run.run_id] = (
            f"{run.display_name} ({seed_label})",
            f"{run.file_stem}_{file_suffix}",
        )
    return identifiers


def _find_baseline_candidates(
    run_dir: Path,
    runs: list[RunData],
    baseline_dir: str | Path | None = None,
) -> list[RunData]:
    """Discover ordinary baseline runs for intervention comparison."""
    ordinary = [r for r in runs if not r.is_intervention]
    if ordinary:
        return ordinary

    if baseline_dir:
        b_path = Path(baseline_dir)
        if b_path.exists():
            log.info("Loading baseline runs from specified directory: %s", b_path)
            b_runs = _load_runs(b_path)
            ordinary = [r for r in b_runs if not r.is_intervention]
            if ordinary:
                return ordinary

    # Sibling 'fineweb-edu' auto-discovery
    sibling_fineweb = run_dir.parent / "fineweb-edu"
    if sibling_fineweb.exists() and sibling_fineweb != run_dir:
        log.info("Auto-discovering baseline runs from sibling directory: %s", sibling_fineweb)
        b_runs = _load_runs(sibling_fineweb)
        ordinary = [r for r in b_runs if not r.is_intervention]
        if ordinary:
            return ordinary

    if run_dir.parent.exists() and run_dir.parent != run_dir:
        p_runs = _load_runs(run_dir.parent)
        ordinary = [r for r in p_runs if not r.is_intervention]
        if ordinary:
            return ordinary

    return []


def _resolve_baseline(
    base_variant: str, candidates: list[RunData],
) -> RunData | None:
    """Resolve the configured control run for one intervention family."""
    ordinary = [
        run for run in candidates
        if not run.is_intervention and run.base_variant == base_variant
    ]
    if not ordinary:
        log.warning("No ordinary %s baseline found for intervention comparison.", base_variant)
        return None

    configured = DEFAULT_BASELINE_RUN
    if configured and configured != f"latest_{base_variant}":
        configured_matches = [
            run for run in ordinary
            if run.telemetry_path.parent.name == configured
            or str(run.telemetry_path.parent) == configured
        ]
        if configured_matches:
            return max(configured_matches, key=_run_sort_key)
        log.warning(
            "Configured baseline %r was not found among ordinary %s runs; "
            "using the latest available baseline.",
            configured, base_variant,
        )

    return max(ordinary, key=_run_sort_key)


def _resolve_all_baselines(
    base_variant: str, candidates: list[RunData],
) -> list[RunData]:
    """Resolve all ordinary control runs for one intervention family."""
    ordinary = [
        run for run in candidates
        if not run.is_intervention and run.base_variant == base_variant
    ]
    if not ordinary:
        return []

    configured = DEFAULT_BASELINE_RUN
    if configured == f"latest_{base_variant}":
        return [max(ordinary, key=_run_sort_key)]

    # If distinct seeds are available, pick latest run for each seed
    distinct_seeds = {r.seed for r in ordinary if r.seed is not None}
    if len(distinct_seeds) > 1:
        by_seed: dict[Any, RunData] = {}
        for run in ordinary:
            s = run.seed
            if s not in by_seed or _run_sort_key(run) > _run_sort_key(by_seed[s]):
                by_seed[s] = run
        return list(by_seed.values())

    if configured:
        configured_matches = [
            run for run in ordinary
            if run.telemetry_path.parent.name == configured
            or str(run.telemetry_path.parent) == configured
        ]
        if configured_matches:
            return [max(configured_matches, key=_run_sort_key)]

    return [max(ordinary, key=_run_sort_key)]


def _intervention_colors(
    baseline: RunData | list[RunData], interventions: list[RunData],
) -> dict[str, str]:
    """Assign distinct intervention colours without reusing the base colour."""
    base_runs = [baseline] if isinstance(baseline, RunData) else list(baseline)
    base_variant = base_runs[0].base_variant if base_runs else "swiglu"
    base_color = get_variant_color(base_variant)

    colours: dict[str, str] = {"Base": base_color}
    for b in base_runs:
        colours[b.run_id] = base_color

    used = {base_color}
    fallback = [color for color in FALLBACK_COLORS if color not in used]
    fallback_idx = 0

    for run in sorted(interventions, key=lambda item: (item.short_label, _run_sort_key(item))):
        colour = INTERVENTION_COLOR_MAP.get(run.short_label)
        if colour in used:
            if run.short_label in colours:
                colour = colours[run.short_label]
            else:
                colour = None
        if colour is None:
            while fallback_idx < len(fallback) and fallback[fallback_idx] in used:
                fallback_idx += 1
            if fallback_idx < len(fallback):
                colour = fallback[fallback_idx]
                fallback_idx += 1
            else:
                colour = get_variant_color(run.run_id, list(colours))
        colours[run.short_label] = colour
        colours[run.run_id] = colour
        used.add(colour)
    return colours


def _dominant_weight(record: dict[str, Any]) -> float:
    """Extract the detected coordinate's weight magnitude from one snapshot."""
    layer_idx = record.get("top_dominance_layer", -1)
    if 0 <= layer_idx < len(record.get("per_layer", [])):
        value = record["per_layer"][layer_idx].get("dominant_weight_value")
        if value is not None:
            return abs(value)
    return abs(record.get("top_weight_value", 0.0))


def _plot_intervention_scalar(
    base_variant: str,
    baselines: list[RunData],
    interventions: list[RunData],
    colours: dict[str, str],
    out_dir: Path,
    *, field: str, filename_metric: str, ylabel: str, title: str,
    dominance: bool = False,
    dataset_name: str = "Training Twins",
) -> Path | None:
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    # 1. Plot Base
    base_agg = aggregate_metric_trajectories(baselines, field)
    if base_agg:
        base_col = colours.get("Base", get_variant_color(base_variant))
        base_label = f"Base (N={base_agg.n_seeds})" if base_agg.n_seeds > 1 else "Base"
        ax.plot(base_agg.steps, base_agg.mean, color=base_col, linestyle="-", linewidth=2.8, zorder=3, label=base_label)
        if base_agg.n_seeds > 1:
            ax.fill_between(base_agg.steps, base_agg.lower, base_agg.upper, color=base_col, alpha=ENVELOPE_ALPHA, zorder=2)
        plotted = True

    # 2. Group interventions by condition short_label
    by_condition: dict[str, list[RunData]] = {}
    for r in interventions:
        by_condition.setdefault(r.short_label, []).append(r)

    for label, cond_runs in sorted(by_condition.items()):
        if dominance and label == "UC":
            continue
        agg = aggregate_metric_trajectories(cond_runs, field)
        if not agg:
            continue
        col = colours.get(label, colours.get(cond_runs[0].run_id, "#333333"))
        cond_label = f"{label} (N={agg.n_seeds})" if agg.n_seeds > 1 else label
        ax.plot(agg.steps, agg.mean, color=col, linestyle="-", linewidth=1.6, zorder=2, label=cond_label)
        if agg.n_seeds > 1:
            ax.fill_between(agg.steps, agg.lower, agg.upper, color=col, alpha=ENVELOPE_ALPHA, zorder=1)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None
    if dominance:
        ax.axhline(0.4, color="gray", linewidth=1.0, alpha=0.55)
        ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{dataset_name} — {variant_display_name(base_variant)} Interventions: {title}", fontsize=13, fontweight="bold")
    _legend_sorted_by_final_value(ax, loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / f"cross_intervention_{base_variant}_{filename_metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_intervention_lag_dynamics(
    base_variant: str,
    baselines: list[RunData],
    interventions: list[RunData],
    colours: dict[str, str],
    out_dir: Path,
    dataset_name: str = "Training Twins",
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    ax_x, ax_w = axes
    plotted = False

    # Base
    agg_base_x = aggregate_metric_trajectories(
        baselines, lambda r: r.get("top_input_spike") if r.get("top_dominance", 0) > 0 else None
    )
    agg_base_w = aggregate_metric_trajectories(
        baselines, lambda r: _dominant_weight(r) if r.get("top_dominance", 0) > 0 else None
    )
    base_color = colours.get("Base", get_variant_color(base_variant))
    if agg_base_x and agg_base_w:
        base_label = f"Base (N={agg_base_x.n_seeds})" if agg_base_x.n_seeds > 1 else "Base"
        ax_x.plot(agg_base_x.steps, agg_base_x.mean, color=base_color, linestyle="-", linewidth=2.8, zorder=3, label=base_label)
        if agg_base_x.n_seeds > 1:
            ax_x.fill_between(agg_base_x.steps, agg_base_x.lower, agg_base_x.upper, color=base_color, alpha=ENVELOPE_ALPHA, zorder=2)
        ax_w.plot(agg_base_w.steps, agg_base_w.mean, color=base_color, linestyle="-", linewidth=2.8, zorder=3, label=base_label)
        if agg_base_w.n_seeds > 1:
            ax_w.fill_between(agg_base_w.steps, agg_base_w.lower, agg_base_w.upper, color=base_color, alpha=ENVELOPE_ALPHA, zorder=2)
        plotted = True

    # Group interventions
    by_condition: dict[str, list[RunData]] = {}
    for r in interventions:
        by_condition.setdefault(r.short_label, []).append(r)

    for label, cond_runs in sorted(by_condition.items()):
        agg_x = aggregate_metric_trajectories(
            cond_runs, lambda r: r.get("top_input_spike") if r.get("top_dominance", 0) > 0 else None
        )
        agg_w = aggregate_metric_trajectories(
            cond_runs, lambda r: _dominant_weight(r) if r.get("top_dominance", 0) > 0 else None
        )
        if not agg_x or not agg_w:
            continue
        col = colours.get(label, colours.get(cond_runs[0].run_id, "#333333"))
        cond_label = f"{label} (N={agg_x.n_seeds})" if agg_x.n_seeds > 1 else label
        ax_x.plot(agg_x.steps, agg_x.mean, color=col, linestyle="-", linewidth=1.6, zorder=2, label=cond_label)
        if agg_x.n_seeds > 1:
            ax_x.fill_between(agg_x.steps, agg_x.lower, agg_x.upper, color=col, alpha=ENVELOPE_ALPHA, zorder=1)
        ax_w.plot(agg_w.steps, agg_w.mean, color=col, linestyle="-", linewidth=1.6, zorder=2, label=cond_label)
        if agg_w.n_seeds > 1:
            ax_w.fill_between(agg_w.steps, agg_w.lower, agg_w.upper, color=col, alpha=ENVELOPE_ALPHA, zorder=1)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax_x.set_xlabel("Training Step", fontsize=11)
    ax_x.set_ylabel("|X_ik*| (mean activation at dominant k*)", fontsize=11)
    ax_x.set_title("Input activation spike", fontsize=12, fontweight="bold")
    _legend_sorted_by_final_value(ax_x, loc="best", framealpha=0.9, fontsize=8)
    ax_x.grid(True, alpha=0.3)
    ax_w.set_xlabel("Training Step", fontsize=11)
    ax_w.set_ylabel("|W_j*k*| (weight at dominant coordinate)", fontsize=11)
    ax_w.set_title("Weight at dominant coordinate", fontsize=12, fontweight="bold")
    _legend_sorted_by_final_value(ax_w, loc="best", framealpha=0.9, fontsize=8)
    ax_w.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset_name} — {variant_display_name(base_variant)} Interventions: Activation and Weight Dynamics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"cross_intervention_{base_variant}_lag_dynamics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_intervention_difference(
    base_variant: str,
    baselines: list[RunData],
    interventions: list[RunData],
    colours: dict[str, str],
    out_dir: Path,
    *, field: str, filename_metric: str, ylabel: str, title: str,
    dataset_name: str = "Training Twins",
) -> Path | None:
    # 1. Map baselines by seed
    base_by_seed: dict[int, dict[int, float]] = {}
    for r in baselines:
        if r.seed is not None:
            base_by_seed[r.seed] = {rec["step"]: rec[field] for rec in r.records if field in rec}

    # 2. Baseline mean by step as fallback
    base_agg = aggregate_metric_trajectories(baselines, field)
    if not base_agg:
        return None
    base_mean_by_step = dict(zip(base_agg.steps, base_agg.mean))

    # 3. Group interventions by short_label
    by_condition: dict[str, list[RunData]] = {}
    for r in interventions:
        by_condition.setdefault(r.short_label, []).append(r)

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    for label, cond_runs in sorted(by_condition.items()):
        if field == "top_dominance" and label == "UC":
            continue

        # Compute delta series per run with seed-matching
        delta_series_list: list[dict[int, float]] = []
        for irun in cond_runs:
            base_dict = base_by_seed.get(irun.seed, base_mean_by_step) if irun.seed is not None else base_mean_by_step
            run_delta: dict[int, float] = {}
            for rec in irun.records:
                if field in rec and rec["step"] in base_dict:
                    s = rec["step"]
                    run_delta[s] = rec[field] - base_dict[s]
            if run_delta:
                delta_series_list.append(run_delta)

        if not delta_series_list:
            continue

        all_steps = sorted(set().union(*[d.keys() for d in delta_series_list]))
        steps: list[int] = []
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for s in all_steps:
            vals = [d[s] for d in delta_series_list if s in d]
            if vals:
                steps.append(s)
                means.append(float(np.mean(vals)))
                lowers.append(float(np.min(vals)))
                uppers.append(float(np.max(vals)))

        if not steps:
            continue

        col = colours.get(label, colours.get(cond_runs[0].run_id, "#333333"))
        legend_label = f"{label} (N={len(delta_series_list)})" if len(delta_series_list) > 1 else label
        ax.plot(steps, means, color=col, linestyle="-", linewidth=1.8, label=legend_label)
        if len(delta_series_list) > 1:
            ax.fill_between(steps, lowers, uppers, color=col, alpha=ENVELOPE_ALPHA)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{dataset_name} — {variant_display_name(base_variant)} Intervention Effect: {title}",
                 fontsize=13, fontweight="bold")
    _legend_sorted_by_final_value(ax, loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / f"cross_intervention_{base_variant}_delta_{filename_metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_intervention_comparisons(
    base_variant: str,
    baseline: RunData | list[RunData],
    interventions: list[RunData],
    out_dir: Path,
    dataset_name: str | None = None,
) -> list[Path]:
    """Plot interventions only against their matched ordinary baseline."""
    if not _HAVE_MPL:
        log.warning("matplotlib not installed; skipping intervention comparison plots.")
        return []

    baselines = [baseline] if isinstance(baseline, RunData) else list(baseline)
    colours = _intervention_colors(baselines, interventions)
    plots: list[Path] = []

    if dataset_name is None:
        flat_runs = baselines + list(interventions)
        dataset_name = resolve_dataset_name(out_dir.parent, flat_runs)
    scalar_specs = [
        ("top_weight_ratio", "outlier_ratio", "|W_max| / mean|W|", "Weight Outlier Ratio", False),
        ("weight_kurtosis_max", "kurtosis", "Max Weight Kurtosis", "Weight Tensor Kurtosis", False),
        ("top_dominance", "dominance", "Dominance D(t)", "Dominance Score D(t)", True),
        ("loss", "loss", "Loss", "Training Loss", False),
    ]
    for field, metric, ylabel, title, is_dominance in scalar_specs:
        path = _plot_intervention_scalar(
            base_variant, baselines, interventions, colours, out_dir,
            field=field, filename_metric=metric, ylabel=ylabel, title=title,
            dominance=is_dominance, dataset_name=dataset_name,
        )
        if path is not None:
            plots.append(path)

    lag_path = _plot_intervention_lag_dynamics(
        base_variant, baselines, interventions, colours, out_dir,
        dataset_name=dataset_name,
    )
    if lag_path is not None:
        plots.append(lag_path)

    for field, metric, ylabel, title, _is_dominance in scalar_specs:
        path = _plot_intervention_difference(
            base_variant, baselines, interventions, colours, out_dir,
            field=field, filename_metric=metric,
            ylabel=f"Δ {ylabel} vs Base", title=f"Δ {title} vs Base",
            dataset_name=dataset_name,
        )
        if path is not None:
            plots.append(path)
    log.info("Generated %d intervention comparison plots for %s.", len(plots), base_variant)
    return plots


# =============================================================================
# Summary
# =============================================================================


def summarise(
    runs: list[RunData],
    baselines: dict[str, RunData | list[RunData]],
) -> dict[str, Any]:
    """Produce a summary dict including baseline pairing and multi-seed aggregates."""
    summary: dict[str, Any] = {}

    for run in sorted(runs, key=lambda item: item.run_id):
        records = run.records
        if not records:
            continue
        final = records[-1]
        item: dict[str, Any] = {
            "run_directory": run.telemetry_path.parent.name,
            "base_variant": run.base_variant,
            "seed": run.seed,
            "dataset": run.dataset,
            "is_intervention": run.is_intervention,
            "condition_name": run.condition_name,
            "condition_label": run.short_label if run.is_intervention else None,
            "final_step": final["step"],
            "final_loss": final["loss"],
            "final_top_weight_ratio": final["top_weight_ratio"],
            "final_top_weight_layer": final["top_weight_layer"],
            "final_top_weight_value": final["top_weight_value"],
            "final_kurtosis_max": final["weight_kurtosis_max"],
            "n_telemetry_snapshots": len(records),
        }
        if "top_dominance" in final:
            item["final_dominance"] = final["top_dominance"]

        if run.is_intervention and run.base_variant in baselines:
            b_val = baselines[run.base_variant]
            b_list = [b_val] if isinstance(b_val, RunData) else list(b_val)

            matched_base = None
            if run.seed is not None:
                for b in b_list:
                    if b.seed == run.seed:
                        matched_base = b
                        break
            if matched_base is None and b_list:
                matched_base = b_list[0]

            if matched_base and matched_base.records:
                b_final = matched_base.records[-1]
                item["baseline_run"] = matched_base.telemetry_path.parent.name
                for field, summary_key in [
                    ("loss", "final_loss_delta"),
                    ("top_weight_ratio", "final_top_weight_ratio_delta"),
                    ("weight_kurtosis_max", "final_kurtosis_max_delta"),
                    ("top_dominance", "final_dominance_delta"),
                ]:
                    if field in final and field in b_final:
                        item[summary_key] = final[field] - b_final[field]
            if "intervention" in final:
                item["final_intervention_metrics"] = final["intervention"]
        summary[run.run_id] = item

    # Aggregated statistics by condition / variant
    groups: dict[str, list[dict[str, Any]]] = {}
    for r_id, r_info in summary.items():
        if not isinstance(r_info, dict):
            continue
        key = r_info.get("condition_label") or r_info.get("base_variant")
        if key:
            groups.setdefault(key, []).append(r_info)

    aggregates: dict[str, Any] = {}
    for group_key, items in sorted(groups.items()):
        n = len(items)
        agg_item: dict[str, Any] = {"n_seeds": n}
        for metric in ["final_loss", "final_top_weight_ratio", "final_kurtosis_max", "final_dominance",
                       "final_loss_delta", "final_top_weight_ratio_delta", "final_kurtosis_max_delta", "final_dominance_delta"]:
            vals = [it[metric] for it in items if metric in it and it[metric] is not None]
            if vals:
                agg_item[f"{metric}_mean"] = float(np.mean(vals))
                agg_item[f"{metric}_std"] = float(np.std(vals))
                agg_item[f"{metric}_min"] = float(np.min(vals))
                agg_item[f"{metric}_max"] = float(np.max(vals))
        aggregates[group_key] = agg_item

    summary["_aggregates"] = aggregates
    return summary


# =============================================================================
# CLI entry point
# =============================================================================


def run_analysis(
    run_dir: str | Path,
    *,
    baseline_dir: str | Path | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Main entry point: find telemetry, generate plots, write summary."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    runs = _load_runs(run_dir)
    if not runs:
        raise FileNotFoundError(
            f"No telemetry.jsonl found in {run_dir} or its subdirectories."
        )

    log.info("Found telemetry for %d run(s): %s", len(runs),
             ", ".join(sorted(run.run_id for run in runs)))
    for run in runs:
        log.info("  %s: %d snapshots", run.display_name, len(run.records))

    dataset_name = resolve_dataset_name(run_dir, runs, explicit_name=dataset)
    log.info("Using dataset name for plot titles: %s", dataset_name)
    for run in runs:
        if run.dataset is None and dataset_name:
            run.dataset = dataset_name

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    all_plots: list[Path] = []

    # 1. Per-run individual plots
    single_plot_ids = _single_plot_identifiers(runs)
    for run in runs:
        display_name, file_stem = single_plot_ids[run.run_id]
        all_plots.extend(plot_single_variant(
            run.records, run.base_variant, plot_dir,
            display_name=display_name, file_stem=file_stem,
            dataset_name=dataset_name,
        ))

    # 2. Group ordinary runs by variant
    runs_by_variant: dict[str, list[RunData]] = {}
    for run in runs:
        if not run.is_intervention:
            runs_by_variant.setdefault(run.base_variant, []).append(run)

    # Multi-seed consolidated plots per variant (when N >= 2 seeds)
    for variant, var_runs in sorted(runs_by_variant.items()):
        if len(var_runs) > 1:
            all_plots.extend(plot_multi_seed_variant(
                variant, var_runs, plot_dir, dataset_name=dataset_name,
            ))

    # 3. Cross-variant comparison plots (when multiple variants present)
    if len(runs_by_variant) > 1:
        all_plots.extend(plot_cross_variant(
            runs_by_variant, plot_dir,
        ))

    # 4. Interventions
    interventions_by_variant: dict[str, list[RunData]] = {}
    for run in runs:
        if run.is_intervention:
            interventions_by_variant.setdefault(run.base_variant, []).append(run)

    selected_baselines: dict[str, list[RunData]] = {}
    if interventions_by_variant:
        baseline_candidates = _find_baseline_candidates(run_dir, runs, baseline_dir)
        for base_variant, interventions in sorted(interventions_by_variant.items()):
            baselines = _resolve_all_baselines(base_variant, baseline_candidates)
            if not baselines:
                log.warning("No %s baseline found for intervention comparison.", base_variant)
                continue
            selected_baselines[base_variant] = baselines
            baseline_arg = baselines[0] if len(baselines) == 1 else baselines
            all_plots.extend(plot_intervention_comparisons(
                base_variant, baseline_arg, interventions, plot_dir,
            ))

    # 5. Summary
    summary = summarise(runs, selected_baselines)
    summary_path = run_dir / "trajectory_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("Summary written to %s", summary_path)
    log.info("Total plots: %d", len(all_plots))

    return summary
