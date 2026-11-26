"""Regression tests for training-twins baseline/intervention plotting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swx.s46_formation import plot_trajectories as plots


def _records(loss_offset: float = 0.0) -> list[dict]:
    return [
        {
            "step": step,
            "loss": 5.0 + loss_offset - step / 1000,
            "top_weight_ratio": 8.0 + loss_offset + step / 1000,
            "top_weight_layer": 0,
            "top_weight_value": 0.1 + step / 1000,
            "weight_kurtosis_max": 3.0 + loss_offset + step / 1000,
            "top_dominance": 0.2 + loss_offset + step / 1000,
            "top_dominance_layer": 0,
            "top_input_spike": 0.5 + step / 1000,
            "per_layer": [{"dominant_weight_value": 0.1 + step / 1000}],
        }
        for step in (0, 100)
    ]


def _write_run(
    root: Path, name: str, *, variant: str, condition_name: str | None = None,
    intervention: dict | None = None, loss_offset: float = 0.0,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    config = {"variant": variant}
    if condition_name:
        config["condition_name"] = condition_name
    if intervention:
        config["intervention"] = intervention
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(record) for record in _records(loss_offset)) + "\n"
    )
    return directory


def test_run_analysis_keeps_interventions_out_of_normal_cross_variant_plots(tmp_path, monkeypatch):
    _write_run(tmp_path, "m1_swiglu_20260101-000000", variant="swiglu")
    newest = _write_run(tmp_path, "m1_swiglu_20260102-000000", variant="swiglu")
    _write_run(tmp_path, "m2_geglu_20260102-000000", variant="geglu")
    _write_run(
        tmp_path, "m1_swiglu_A1_hard_clamp_20260103-000000", variant="swiglu",
        condition_name="A1_hard_clamp",
        intervention={"kind": "activation_bound"}, loss_offset=1.0,
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(plots, "plot_single_variant", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        plots, "plot_cross_variant",
        lambda records, _out_dir: captured.setdefault("cross", records) and [],
    )
    monkeypatch.setattr(
        plots, "plot_intervention_comparisons",
        lambda base, baseline, interventions, _out_dir: captured.update(
            base=base, baseline=baseline, interventions=interventions,
        ) or [],
    )
    monkeypatch.setattr(plots, "DEFAULT_BASELINE_RUN", "latest_swiglu")

    plots.run_analysis(tmp_path)

    assert set(captured["cross"]) == {"swiglu", "geglu"}
    assert captured["baseline"].telemetry_path.parent == newest
    assert [run.short_label for run in captured["interventions"]] == ["AHC"]


def test_explicit_baseline_override_and_short_labels(tmp_path, monkeypatch):
    old = _write_run(tmp_path, "m1_swiglu_20260101-000000", variant="swiglu")
    _write_run(tmp_path, "m1_swiglu_20260102-000000", variant="swiglu")
    runs = plots._load_runs(tmp_path)
    monkeypatch.setattr(plots, "DEFAULT_BASELINE_RUN", old.name)

    selected = plots._resolve_baseline("swiglu", runs)

    assert selected is not None
    assert selected.telemetry_path.parent == old
    intervention = plots.RunData(
        run_id="i", telemetry_path=tmp_path / "i" / "telemetry.jsonl", records=[],
        base_variant="swiglu", condition_name="I2_mid_inoc",
        intervention={"kind": "inoculation"},
    )
    assert intervention.short_label == "MI"
    assert "Mid inoculation" in intervention.display_name


@pytest.mark.skipif(not plots._HAVE_MPL, reason="matplotlib is not installed")
def test_intervention_plots_include_absolute_and_difference_outputs(tmp_path):
    baseline = plots.RunData(
        run_id="base", telemetry_path=tmp_path / "m1_swiglu_20260101-000000" / "telemetry.jsonl",
        records=_records(), base_variant="swiglu", condition_name=None, intervention=None,
    )
    hard_clamp = plots.RunData(
        run_id="ahc", telemetry_path=tmp_path / "m1_swiglu_A1_20260102-000000" / "telemetry.jsonl",
        records=_records(1.0), base_variant="swiglu", condition_name="A1_hard_clamp",
        intervention={"kind": "activation_bound"},
    )

    outputs = plots.plot_intervention_comparisons("swiglu", baseline, [hard_clamp], tmp_path)

    names = {path.name for path in outputs}
    assert "cross_intervention_swiglu_loss.png" in names
    assert "cross_intervention_swiglu_delta_loss.png" in names
    assert "cross_intervention_swiglu_delta_dominance.png" in names
    assert all(path.exists() for path in outputs)
    colours = plots._intervention_colors(baseline, [hard_clamp])
    assert colours["base"] == plots.get_variant_color("swiglu")
    assert colours["ahc"] != colours["base"]


def test_aggregate_metric_trajectories():
    r1 = plots.RunData(
        run_id="r1", telemetry_path=Path("dummy1"), records=[
            {"step": 0, "loss": 1.0, "top_weight_ratio": 5.0},
            {"step": 100, "loss": 2.0, "top_weight_ratio": 6.0},
        ], base_variant="swiglu", condition_name=None, intervention=None, seed=42,
    )
    r2 = plots.RunData(
        run_id="r2", telemetry_path=Path("dummy2"), records=[
            {"step": 0, "loss": 2.0, "top_weight_ratio": 7.0},
            {"step": 100, "loss": 4.0, "top_weight_ratio": 8.0},
        ], base_variant="swiglu", condition_name=None, intervention=None, seed=52,
    )
    r3 = plots.RunData(
        run_id="r3", telemetry_path=Path("dummy3"), records=[
            {"step": 0, "loss": 3.0, "top_weight_ratio": 9.0},
            {"step": 100, "loss": 6.0, "top_weight_ratio": 10.0},
        ], base_variant="swiglu", condition_name=None, intervention=None, seed=62,
    )

    agg = plots.aggregate_metric_trajectories([r1, r2, r3], "loss")
    assert agg is not None
    assert agg.n_seeds == 3
    assert agg.steps == [0, 100]
    assert agg.mean == pytest.approx([2.0, 4.0])
    assert agg.lower == pytest.approx([1.0, 2.0])
    assert agg.upper == pytest.approx([3.0, 6.0])
    assert set(agg.seed_series.keys()) == {42, 52, 62}

    # Single-seed fallback
    single_agg = plots.aggregate_metric_trajectories([r1], "loss")
    assert single_agg is not None
    assert single_agg.n_seeds == 1
    assert single_agg.mean == single_agg.lower == single_agg.upper == [1.0, 2.0]
    assert single_agg.std == [0.0, 0.0]


def test_resolve_dataset_name():
    assert plots.resolve_dataset_name(Path("any"), explicit_name="c4") == "C4"
    assert plots.resolve_dataset_name(Path("any"), explicit_name="fineweb-edu") == "FineWeb-Edu"
    assert plots.resolve_dataset_name(Path("any"), explicit_name="slimpajama") == "SlimPajama"

    assert plots.resolve_dataset_name(Path("runs/training_twins/fineweb-edu")) == "FineWeb-Edu"
    assert plots.resolve_dataset_name(Path("runs/training_twins/c4")) == "C4"
    assert plots.resolve_dataset_name(Path("runs/training_twins/slimpajama")) == "SlimPajama"

    run = plots.RunData(
        run_id="test", telemetry_path=Path("dummy"), records=[],
        base_variant="swiglu", condition_name=None, intervention=None,
        dataset="c4",
    )
    assert plots.resolve_dataset_name(Path("runs/unknown"), [run]) == "C4"


@pytest.mark.skipif(not plots._HAVE_MPL, reason="matplotlib is not installed")
def test_multi_seed_cross_variant_plotting(tmp_path):
    swiglu_42 = plots.RunData(
        run_id="s42", telemetry_path=tmp_path / "s42" / "telemetry.jsonl",
        records=_records(0.0), base_variant="swiglu", condition_name=None,
        intervention=None, seed=42, dataset="fineweb-edu",
    )
    swiglu_52 = plots.RunData(
        run_id="s52", telemetry_path=tmp_path / "s52" / "telemetry.jsonl",
        records=_records(0.2), base_variant="swiglu", condition_name=None,
        intervention=None, seed=52, dataset="fineweb-edu",
    )
    geglu_42 = plots.RunData(
        run_id="g42", telemetry_path=tmp_path / "g42" / "telemetry.jsonl",
        records=_records(0.5), base_variant="geglu", condition_name=None,
        intervention=None, seed=42, dataset="fineweb-edu",
    )
    geglu_52 = plots.RunData(
        run_id="g52", telemetry_path=tmp_path / "g52" / "telemetry.jsonl",
        records=_records(0.7), base_variant="geglu", condition_name=None,
        intervention=None, seed=52, dataset="fineweb-edu",
    )

    grouped = {
        "swiglu": [swiglu_42, swiglu_52],
        "geglu": [geglu_42, geglu_52],
    }
    outputs = plots.plot_cross_variant(grouped, tmp_path, dataset_name="FineWeb-Edu")
    names = {p.name for p in outputs}
    assert "cross_variant_outlier_ratio.png" in names
    assert "cross_variant_kurtosis.png" in names
    assert "cross_variant_loss.png" in names
    assert "cross_variant_lag_dynamics.png" in names
    assert all(p.exists() for p in outputs)


@pytest.mark.skipif(not plots._HAVE_MPL, reason="matplotlib is not installed")
def test_seed_matched_delta_interventions(tmp_path):
    base_42 = plots.RunData(
        run_id="b42", telemetry_path=tmp_path / "b42" / "telemetry.jsonl",
        records=_records(0.0), base_variant="swiglu", condition_name=None,
        intervention=None, seed=42,
    )
    base_52 = plots.RunData(
        run_id="b52", telemetry_path=tmp_path / "b52" / "telemetry.jsonl",
        records=_records(0.1), base_variant="swiglu", condition_name=None,
        intervention=None, seed=52,
    )
    ahc_42 = plots.RunData(
        run_id="a42", telemetry_path=tmp_path / "a42" / "telemetry.jsonl",
        records=_records(1.0), base_variant="swiglu", condition_name="A1_hard_clamp",
        intervention={"kind": "activation_bound"}, seed=42,
    )
    ahc_52 = plots.RunData(
        run_id="a52", telemetry_path=tmp_path / "a52" / "telemetry.jsonl",
        records=_records(1.1), base_variant="swiglu", condition_name="A1_hard_clamp",
        intervention={"kind": "activation_bound"}, seed=52,
    )

    outputs = plots.plot_intervention_comparisons(
        "swiglu", [base_42, base_52], [ahc_42, ahc_52], tmp_path, dataset_name="FineWeb-Edu",
    )
    names = {p.name for p in outputs}
    assert "cross_intervention_swiglu_loss.png" in names
    assert "cross_intervention_swiglu_delta_loss.png" in names
    assert "cross_intervention_swiglu_delta_dominance.png" in names
    assert all(p.exists() for p in outputs)

    colours = plots._intervention_colors([base_42, base_52], [ahc_42, ahc_52])
    # Seeds of the same condition share the same color
    assert colours["a42"] == colours["a52"]
    assert colours["b42"] == colours["b52"]
    assert colours["a42"] != colours["b42"]


def test_sibling_baseline_autodiscovery(tmp_path):
    root = tmp_path / "training_twins"
    fineweb = root / "fineweb-edu"
    interventions = root / "seeds_intervention"

    base_dir = _write_run(fineweb, "m1_swiglu_seed42", variant="swiglu")
    int_dir = _write_run(
        interventions, "m1_swiglu_A1_seed42", variant="swiglu",
        condition_name="A1_hard_clamp", intervention={"kind": "clamp"},
    )

    int_runs = plots._load_runs(interventions)
    assert len(int_runs) == 1
    assert int_runs[0].is_intervention

    candidates = plots._find_baseline_candidates(interventions, int_runs)
    assert len(candidates) == 1
    assert candidates[0].base_variant == "swiglu"
    assert candidates[0].telemetry_path.parent == base_dir

