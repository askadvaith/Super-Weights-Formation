"""Focused tests for opt-in causal interventions in training_twins.

These tests intentionally use tiny local models: they validate intervention
boundaries, telemetry shape, and checkpoint-safe state without downloading data
or running the 124M training job.
"""

from __future__ import annotations

import json

import pytest
import torch

from swx.s46_formation.train_activation_twins import (
    InterventionConfig,
    TelemetrySnapshot,
    TrainingInterventionController,
    TwinTransformer,
    collect_telemetry,
    load_condition_manifest,
)


def _tiny_model(controller: TrainingInterventionController | None = None) -> TwinTransformer:
    return TwinTransformer(
        variant="swiglu", vocab_size=32, d_model=8, n_layers=1, n_heads=2,
        max_seq_len=8, intervention=controller,
    )


def test_baseline_snapshot_omits_optional_intervention_fields():
    snap = TelemetrySnapshot(step=0, tokens_seen=0, loss=1.0, lr=0.0, wall_seconds=0.0)
    payload = snap.as_dict()
    assert "intervention" not in payload
    assert "fixed_coord_traces" not in payload


def test_hard_and_soft_bounds_record_real_binding_metrics():
    x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    for mode in ("hard", "soft"):
        cfg = InterventionConfig(
            kind="activation_bound", scope_layers=(0,), activation_mode=mode,
            activation_threshold=1.0,
        )
        controller = TrainingInterventionController(cfg)
        controller.set_step(0)
        bounded = controller.apply_down_proj_input(0, x)
        assert torch.all(bounded.abs() <= 1.0)
        metrics = controller.snapshot_metrics()
        assert metrics is not None
        assert metrics["values_modified"] > 0
        assert metrics["fraction_modified"] > 0


def test_inoculation_is_additive_once_and_persists_state():
    cfg = InterventionConfig(
        kind="inoculation", scope_layers=(0,), start_step=3,
        target_coordinate=(0, 1, 2), inoculation_delta_scale=2.0,
    )
    controller = TrainingInterventionController(cfg)
    model = _tiny_model(controller)
    controller.validate(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weight = model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        weight.fill_(0.25)
    old = float(weight[1, 2])
    controller.after_optimizer_step(model, optimizer, 3)
    assert float(weight[1, 2]) == pytest.approx(old + 0.5)
    events = controller.pop_events()
    assert len(events) == 1
    assert events[0]["old_value"] == pytest.approx(old)
    controller.after_optimizer_step(model, optimizer, 3)
    assert float(weight[1, 2]) == pytest.approx(old + 0.5)
    assert controller.pop_events() == []


def test_weight_projection_only_changes_scoped_down_projection():
    cfg = InterventionConfig(
        kind="weight_projection", scope_layers=(0,), start_step=0, weight_cap=0.05,
    )
    controller = TrainingInterventionController(cfg)
    model = _tiny_model(controller)
    controller.validate(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weight = model.layers[0].mlp.down_proj.weight
    untouched = model.layers[0].mlp.up_proj.weight.detach().clone()
    with torch.no_grad():
        weight.fill_(0.2)
    controller.set_step(0)
    controller.after_optimizer_step(model, optimizer, 0)
    assert float(weight.abs().max()) == pytest.approx(0.05)
    assert torch.equal(model.layers[0].mlp.up_proj.weight, untouched)
    assert controller.snapshot_metrics()["parameters_modified"] == weight.numel()


def test_update_cap_limits_actual_post_optimizer_delta():
    cfg = InterventionConfig(
        kind="adam_update_cap", scope_layers=(0,), start_step=0, update_cap=0.01,
    )
    controller = TrainingInterventionController(cfg)
    model = _tiny_model(controller)
    controller.validate(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weight = model.layers[0].mlp.down_proj.weight
    before = weight.detach().clone()
    controller.set_step(0)
    controller.before_optimizer_step(model)
    with torch.no_grad():
        weight.add_(0.1)
    controller.after_optimizer_step(model, optimizer, 0)
    assert float((weight - before).abs().max()) == pytest.approx(0.01)
    metrics = controller.snapshot_metrics()
    assert metrics is not None
    assert metrics["parameters_modified"] == weight.numel()
    assert metrics["post_max_abs"] == pytest.approx(0.01)


def test_fixed_coordinate_trace_is_opt_in_and_uses_one_identity():
    cfg = InterventionConfig(
        kind="inoculation", scope_layers=(0,), start_step=4,
        target_coordinate=(0, 1, 2), inoculation_delta_scale=1.0,
    )
    controller = TrainingInterventionController(cfg)
    model = _tiny_model(controller)
    controller.validate(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    controller.set_step(0)
    snap = collect_telemetry(
        model, step=0, tokens_seen=0, loss=1.0, lr=1e-3, wall_seconds=0.0,
        optimizer=optimizer, probe_input=torch.tensor([[1, 2, 3, 4]]),
        intervention=controller, fixed_coordinates=controller.fixed_coordinates(),
    )
    payload = snap.as_dict()
    assert len(payload["fixed_coord_traces"]) == 1
    trace = payload["fixed_coord_traces"][0]
    assert (trace["layer"], trace["j"], trace["k"]) == (0, 1, 2)
    assert "intervention" not in payload  # inoculation is inactive until its event.


def test_condition_manifest_parses_explicit_coordinates(tmp_path):
    path = tmp_path / "condition.json"
    path.write_text(json.dumps({
        "name": "I1",
        "intervention": {
            "kind": "inoculation", "scope_layers": [0], "start_step": 2,
            "target_coordinate": [0, 1, 2], "tracked_coordinates": [[0, 2, 3]],
            "inoculation_delta_scale": 1.5,
        },
    }))
    name, cfg = load_condition_manifest(path)
    assert name == "I1"
    assert cfg is not None
    assert cfg.target_coordinate == (0, 1, 2)
    assert cfg.tracked_coordinates == ((0, 2, 3),)
