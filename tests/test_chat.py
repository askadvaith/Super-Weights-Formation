"""
Tests for the live ablation toggle behind `swx chat`.

The session mutates the loaded model in place, repeatedly, from user input.
Three things must hold or the tool quietly lies to you:

  * scalings must not COMPOUND -- `/scale 0.5` twice is still 0.5, not 0.25;
  * the model must be exactly as loaded when the session ends, including after
    an exception, so a chat session cannot poison a later experiment in the
    same process;
  * a coordinate that is in the pool but NOT selected must be untouched, and a
    coordinate dropped from the selection must be restored -- otherwise
    "/select 2" quietly leaves #1 zeroed and the next completion is attributed
    to the wrong scalar.
"""

from __future__ import annotations

import torch

from swx.chat import Ablation, Target, parse_selection
from swx.core.sw_directory import SuperWeight
from test_detection import _build


def _weights(loaded):
    return [b.mlp.down_proj.weight.clone() for b in loaded.model.model.layers]


def _value(loaded, layer, row, col) -> float:
    return float(loaded.model.model.layers[layer].mlp.down_proj.weight[row, col])


def _loaded():
    return _build(super_weights=[(1, 4, 7, 0.5), (2, 4, 9, 0.4)])


SWS = [SuperWeight(1, 4, 7)]


def test_ablate_then_restore_is_exact():
    loaded = _loaded()
    before = _weights(loaded)
    ablation = Ablation(loaded, SWS)

    ablation.set(0.0)
    assert _value(loaded, 1, 4, 7) == 0.0

    ablation.set(1.0)
    for layer, (b, a) in enumerate(zip(before, _weights(loaded))):
        assert torch.equal(b, a), f"layer {layer} not restored"


def test_scaling_does_not_compound():
    """`/scale 0.5` twice must leave 0.5x, not 0.25x."""
    loaded = _loaded()
    ablation = Ablation(loaded, SWS)

    ablation.set(0.5)
    assert _value(loaded, 1, 4, 7) == 0.25  # 0.5 * 0.5

    ablation.set(0.5)
    assert _value(loaded, 1, 4, 7) == 0.25, "scaling compounded across calls"

    ablation.set(2.0)
    assert _value(loaded, 1, 4, 7) == 1.0  # 0.5 * 2, not 0.25 * 2


def test_temporarily_returns_to_the_previous_factor():
    loaded = _loaded()
    ablation = Ablation(loaded, SWS)
    ablation.set(0.5)

    with ablation.temporarily(0.0):
        assert _value(loaded, 1, 4, 7) == 0.0
        assert ablation.label == "ABLATED"

    assert _value(loaded, 1, 4, 7) == 0.25
    assert ablation.factor == 0.5


def test_temporarily_restores_even_if_generation_raises():
    loaded = _loaded()
    ablation = Ablation(loaded, SWS)

    try:
        with ablation.temporarily(0.0):
            raise KeyboardInterrupt  # what Ctrl-C during generate() looks like
    except KeyboardInterrupt:
        pass

    assert _value(loaded, 1, 4, 7) == 0.5
    assert ablation.factor == 1.0


def test_close_restores_the_model_exactly():
    loaded = _loaded()
    before = _weights(loaded)
    ablation = Ablation(loaded, SWS)

    ablation.set(0.0)
    ablation.set(3.0)
    ablation.set(0.25)
    ablation.close()

    for layer, (b, a) in enumerate(zip(before, _weights(loaded))):
        assert torch.equal(b, a), f"layer {layer} not restored after close()"
    assert ablation.label == "INTACT"


def test_labels():
    loaded = _loaded()
    ablation = Ablation(loaded, SWS)
    assert ablation.label == "INTACT"
    ablation.set(0.0)
    assert ablation.label == "ABLATED"
    ablation.set(0.5)
    assert ablation.label == "SCALED x0.5"
