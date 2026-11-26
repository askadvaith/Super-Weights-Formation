"""
Checks on the transcribed super-weight directory.

These cannot tell us whether a coordinate is CORRECT -- only `swx
verify-directory`, which re-derives it from the model, can do that. What they
can do is catch transcription typos and internal inconsistencies, which is
worth having given the directory was read off a two-column PDF table.
"""

from __future__ import annotations

import pytest

from swx.core.sw_directory import (
    DEFAULT_MODEL,
    SPECS,
    ModelSpec,
    SuperWeight,
    get_spec,
    residual_channels,
    validate_spec,
)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_every_coordinate_is_dimensionally_valid(key: str):
    """Row < d_model, col < d_ff, layer < n_layers for every tabulated entry."""
    problems = validate_spec(SPECS[key])
    assert not problems, f"{key}: " + "; ".join(problems)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_no_duplicate_coordinates(key: str):
    coords = [sw.as_tuple() for sw in SPECS[key].super_weights]
    assert len(coords) == len(set(coords)), f"{key} has duplicate coordinates"


def test_default_model_is_registered_and_has_super_weights():
    spec = get_spec(DEFAULT_MODEL)
    assert spec.super_weights, (
        "The default model must have published super weights so that a bare "
        "`swx e1` works without a detection step."
    )


def test_unknown_key_raises_with_a_helpful_message():
    with pytest.raises(KeyError) as exc:
        get_spec("definitely-not-a-model")
    assert "Available" in str(exc.value)


def test_olmo_1b_second_super_weight_is_at_layer_2_not_layer_1():
    """
    Table 2 prints BOTH OLMo-1B coordinates at layer 1. The table is wrong, and
    the weights of allenai/OLMo-1B-0724-hf say so unambiguously:

        layers[1].mlp.down_proj.weight[1764, 8041] = +0.00179
            0.4x mean |W|, rank 13,035,937 of 16,777,216 -- below median.
        layers[2].mlp.down_proj.weight[1764, 8041] = -0.59238
            117.5x mean |W|, RANK 1 of 16,777,216.

    Layer 2 is also where the down_proj input spikes at mlp channel 8041 and
    the output spikes at residual channel 1764, and one weight explains 90% of
    that spike.

    This test exists so nobody "fixes" the directory back to match the PDF.
    Reproduce with `swx verify-directory --model olmo-1b`.
    """
    spec = get_spec("olmo-1b")
    coords = {sw.as_tuple() for sw in spec.super_weights}
    assert coords == {(1, 1764, 1710), (2, 1764, 8041)}


def test_llama_7b_matches_the_papers_worked_example():
    """
    The paper's caption gives exactly one worked example:

        layers[2].mlp.down_proj.weight[3968, 7003]

    If this drifts, the whole indexing convention is wrong.
    """
    spec = get_spec("llama-7b")
    assert len(spec.super_weights) == 1
    sw = spec.super_weights[0]
    assert (sw.layer, sw.residual_channel, sw.mlp_channel) == (2, 3968, 7003)


def test_olmo_7b_has_a_late_layer_entry():
    """
    OLMo-7B has a tabulated super weight at layer 24 of 32, showing that
    super weights are not restricted to early layers.
    """
    spec = get_spec("olmo-7b")
    layers = sorted(sw.layer for sw in spec.super_weights)
    assert 24 in layers
    assert max(layers) > spec.n_layers // 2


def test_phi3_grid_structure():
    """
    Phi-3-mini's six super weights form a 3x2 grid: three residual channels
    crossed with two MLP channels. Recorded because it suggests a rank-1
    structure rather than six independent scalars -- something the paper does
    not comment on and worth a dedicated look.
    """
    spec = get_spec("phi-3-mini")
    rows = {sw.residual_channel for sw in spec.super_weights}
    cols = {sw.mlp_channel for sw in spec.super_weights}
    assert len(spec.super_weights) == 6
    assert len(rows) == 3 and len(cols) == 2
    assert len(rows) * len(cols) == len(spec.super_weights), "not a full grid"


def test_residual_channels_deduplicates():
    spec = ModelSpec(
        key="fake",
        hf_id="fake/fake",
        d_model=64,
        d_ff=128,
        n_layers=4,
        super_weights=(
            SuperWeight(0, 7, 11),
            SuperWeight(1, 7, 12),   # same residual channel, different layer
            SuperWeight(2, 9, 13),
        ),
    )
    assert residual_channels(spec) == [7, 9]


def test_validate_spec_catches_out_of_range():
    bad = ModelSpec(
        key="bad",
        hf_id="bad/bad",
        d_model=64,
        d_ff=128,
        n_layers=4,
        super_weights=(SuperWeight(layer=99, residual_channel=999, mlp_channel=999),),
    )
    problems = validate_spec(bad)
    assert len(problems) == 3  # layer, row and column are all out of range
