"""
Unit tests for the detection logic, on a tiny synthetic model.

These do not need a real LLM. What they pin down is the behaviour that was
wrong before and produced confident, incorrect coordinates on OLMo-1B:

  * picking the ONSET layer rather than the globally largest spike;
  * choosing the mlp channel from the WEIGHT (argmax_k |X_k W_jk|) rather than
    from argmax_k |X_k|;
  * refusing a candidate whose weight is not an outlier;
  * refusing a candidate whose spike is a sum of many terms;
  * restoring every weight it touched, exactly, even on repeat visits.
"""

from __future__ import annotations

import torch
from torch import nn

from swx.core.detection import detect
from swx.core.modeling import LoadedModel, weight_outlier_stats
from swx.core.sw_directory import ModelSpec, SuperWeight

D_MODEL, D_FF, N_LAYERS = 8, 16, 4


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)


class _Block(nn.Module):
    """A decoder block that ignores its input and emits a fixed activation.

    The point is to control the down_proj input exactly, so the test asserts on
    detection logic rather than on a model's behaviour.
    """

    def __init__(self, activation: torch.Tensor) -> None:
        super().__init__()
        self.mlp = _MLP()
        self.register_buffer("activation", activation)

    def forward(self, hidden, **kwargs):  # noqa: ANN001
        batch, seq, _ = hidden.shape
        x = self.activation.expand(batch, seq, D_FF)
        return (hidden + self.mlp.down_proj(x),)


class _Inner(nn.Module):
    def __init__(self, blocks: list[_Block]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(blocks)


class _Tiny(nn.Module):
    def __init__(self, blocks: list[_Block]) -> None:
        super().__init__()
        self.model = _Inner(blocks)

    def forward(self, input_ids, attention_mask=None, **kwargs):  # noqa: ANN001
        hidden = torch.zeros(*input_ids.shape, D_MODEL)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        return hidden


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "<pad>"

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):  # noqa: ANN001
        n = max(len(text.split()), 1)
        return {
            "input_ids": torch.zeros(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        }


def _build(super_weights=(), activations=None, row_overrides=()) -> LoadedModel:
    """A 4-layer toy whose down_proj weights are all tiny except where planted.

    `row_overrides` is a list of (layer, row, value) that fills a whole row,
    which is how you build a spike that no single weight explains.
    """
    torch.manual_seed(0)
    blocks = []
    for layer in range(N_LAYERS):
        activation = (
            activations[layer] if activations is not None else torch.zeros(D_FF)
        )
        block = _Block(activation)
        with torch.no_grad():
            block.mlp.down_proj.weight.copy_(
                torch.full((D_MODEL, D_FF), 1e-3)
            )
        blocks.append(block)

    with torch.no_grad():
        for layer, row, value in row_overrides:
            blocks[layer].mlp.down_proj.weight[row, :] = value
        for layer, row, col, value in super_weights:
            blocks[layer].mlp.down_proj.weight[row, col] = value

    spec = ModelSpec(
        key="tiny", hf_id="tiny/tiny",
        d_model=D_MODEL, d_ff=D_FF, n_layers=N_LAYERS,
    )
    return LoadedModel(
        model=_Tiny(blocks), tokenizer=_Tokenizer(), spec=spec,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _spike(channel: int, magnitude: float) -> torch.Tensor:
    x = torch.zeros(D_FF)
    x[channel] = magnitude
    return x


def test_picks_the_onset_layer_not_the_largest_spike():
    """
    Layer 3 has a strictly larger activation and a strictly larger weight than
    layer 1, but layer 1 comes first. This is the OLMo-1B failure exactly: the
    biggest down_proj output in that model is in the LAST layer, which cancels
    the super activation rather than creating it.
    """
    loaded = _build(
        super_weights=[(1, 4, 7, 0.5), (3, 4, 9, 0.9)],
        activations=[
            torch.zeros(D_FF),
            _spike(7, 100.0),
            torch.zeros(D_FF),
            _spike(9, 400.0),
        ],
    )
    result = detect(loaded, prompt="a b c", max_super_weights=1)
    assert [sw.as_tuple() for sw in result.candidates] == [(1, 4, 7)]


def test_column_comes_from_the_weight_not_from_the_loudest_input():
    """
    Channel 2 is the loudest down_proj input, but its weight is ordinary, so it
    contributes nothing to the output spike. Channel 7 carries the outlier
    weight and dominates the sum. `argmax|X|` returns 2; the correct answer is 7.
    """
    loaded = _build(
        super_weights=[(1, 4, 7, 0.9)],
        activations=[
            torch.zeros(D_FF),
            _spike(2, 500.0) + _spike(7, 100.0),
            torch.zeros(D_FF),
            torch.zeros(D_FF),
        ],
    )
    result = detect(loaded, prompt="a b c", max_super_weights=1)
    assert [sw.as_tuple() for sw in result.candidates] == [(1, 4, 7)]


def test_rejects_a_spike_with_no_outlier_weight():
    """An activation spike alone is not a super weight."""
    loaded = _build(
        super_weights=[],
        activations=[torch.zeros(D_FF), _spike(3, 900.0),
                     torch.zeros(D_FF), torch.zeros(D_FF)],
    )
    result = detect(loaded, prompt="a b c", max_super_weights=2)
    assert result.candidates == []
    assert result.rejected, "the rejection should be recorded, not silently dropped"
    assert "mean |W|" in result.rejected[0].rejected_because


def test_rejects_a_spike_that_is_a_sum_of_many_terms():
    """
    Every channel is active and the whole row carries substantial weight, so the
    output is large and [4, 7] IS the row's biggest entry -- but it accounts for
    only ~17% of the sum. §3.1's approximation does not hold, so there is no
    single weight to call the cause. This is the OLMo-1B L2[623, 2074] case,
    which the weight-magnitude gate alone lets through at 26x mean |W|.
    """
    loaded = _build(
        super_weights=[(1, 4, 7, 0.9)],
        row_overrides=[(1, 4, 0.3)],
        activations=[
            torch.zeros(D_FF),
            torch.full((D_FF,), 200.0),
            torch.zeros(D_FF),
            torch.zeros(D_FF),
        ],
    )
    result = detect(loaded, prompt="a b c", max_super_weights=1)
    assert result.candidates == []
    assert any("explains only" in c.rejected_because for c in result.rejected)


def test_restores_every_weight_it_touched():
    """
    detect() zeroes weights while searching. If restoration is wrong the model
    is silently damaged and every later experiment measures a broken model.
    """
    loaded = _build(
        super_weights=[(1, 4, 7, 0.5), (2, 4, 9, 0.4)],
        activations=[
            torch.zeros(D_FF),
            _spike(7, 300.0),
            _spike(9, 200.0),
            torch.zeros(D_FF),
        ],
    )
    before = [
        block.mlp.down_proj.weight.clone() for block in loaded.model.model.layers
    ]
    detect(loaded, prompt="a b c", max_super_weights=6)
    after = [block.mlp.down_proj.weight for block in loaded.model.model.layers]
    for layer, (b, a) in enumerate(zip(before, after)):
        assert torch.equal(b, a), f"layer {layer} was not restored exactly"


def test_reported_evidence_matches_the_planted_weight():
    """The magnitude and divergence-from-mean reported must be the real ones."""
    loaded = _build(
        super_weights=[(1, 4, 7, 0.5)],
        activations=[torch.zeros(D_FF), _spike(7, 300.0),
                     torch.zeros(D_FF), torch.zeros(D_FF)],
    )
    result = detect(loaded, prompt="a b c", max_super_weights=1)
    candidate = result.accepted[0]
    assert candidate.weight_stats.value == 0.5
    assert candidate.weight_stats.rank == 1
    assert candidate.weight_stats.ratio_to_mean > 20.0
    assert candidate.dominance > 0.9, "one weight should explain the whole spike"


def test_weight_outlier_stats_ranks_correctly():
    module = nn.Linear(D_FF, D_MODEL, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.full((D_MODEL, D_FF), 0.001))
        module.weight[2, 5] = 1.0
        module.weight[3, 6] = 0.5

    top = weight_outlier_stats(module, SuperWeight(0, 2, 5))
    assert top.rank == 1
    assert top.ratio_to_mean > 50
    assert top.is_outlier

    second = weight_outlier_stats(module, SuperWeight(0, 3, 6))
    assert second.rank == 2

    ordinary = weight_outlier_stats(module, SuperWeight(0, 0, 0))
    assert ordinary.rank > 2
    assert not ordinary.is_outlier
