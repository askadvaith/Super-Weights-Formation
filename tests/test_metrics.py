"""
Sanity checks for metrics.py. No model download required -- these run in
seconds and catch the errors that would silently corrupt an experiment.

    uv run pytest
"""

from __future__ import annotations

import math

import pytest
import torch

from swx.core.metrics import (
    binned_logit_shift,
    centre_logits,
    entropy,
    js_divergence,
    kl_divergence,
    logit_delta,
    mean_pairwise_js,
    solve_temperature_for_entropy,
    temperature_match,
    to_probs,
    token_set_mass,
    top1_prob,
)


# -----------------------------------------------------------------------------
# The float64 contract
# -----------------------------------------------------------------------------
# Everything in metrics.py returns float64 regardless of input dtype, because
# the §4.5 argument turns on entropy differences of a fraction of a nat and
# probability masses of a few percent -- comparable to fp16/bf16 accumulation
# error. `torch.allclose` requires matching dtypes, so expected values in these
# tests must be float64 too. Pinned explicitly below so the contract is tested
# rather than assumed.


@pytest.mark.parametrize(
    "input_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_metrics_always_return_float64(input_dtype):
    logits = torch.randn(4, 64).to(input_dtype)
    probs = to_probs(logits)
    assert probs.dtype == torch.float64
    assert entropy(probs).dtype == torch.float64
    assert centre_logits(logits).dtype == torch.float64
    assert kl_divergence(probs, probs).dtype == torch.float64
    assert top1_prob(probs).dtype == torch.float64


def test_entropy_of_uniform_is_log_vocab():
    vocab = 128
    logits = torch.zeros(4, vocab)
    result = entropy(to_probs(logits))
    expected = torch.full((4,), math.log(vocab), dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-12)


def test_entropy_of_point_mass_is_zero():
    logits = torch.full((3, 64), -1e4)
    logits[:, 7] = 1e4
    expected = torch.zeros(3, dtype=torch.float64)
    assert torch.allclose(entropy(to_probs(logits)), expected, atol=1e-12)


def test_kl_of_identical_distributions_is_zero():
    probs = to_probs(torch.randn(6, 50))
    expected = torch.zeros(6, dtype=torch.float64)
    assert torch.allclose(kl_divergence(probs, probs), expected, atol=1e-12)


# -----------------------------------------------------------------------------
# Shift invariance
# -----------------------------------------------------------------------------
# These test a MATHEMATICAL property: centring removes the arbitrary additive
# constant that softmax is invariant to. The property is exact, so it is tested
# in float64 where the shift itself is exact.
#
# The earlier version of these tests built the shift in float32 and then asserted
# to atol=1e-9. That fails, but not because the property is false -- adding 50 to
# a value of order 1 in float32 discards low-order bits BEFORE metrics.py
# promotes to float64, so the two sides agree only to float32 epsilon (~1e-7).
# The float32 case is still worth covering, since real logits arrive as float32
# from the model, so it gets its own test with a tolerance that reflects what
# float32 can actually represent.


def test_centring_removes_an_arbitrary_shift_exactly():
    """In float64 the shift is exact, so centring must be exactly invariant."""
    torch.manual_seed(10)
    logits = torch.randn(5, 100, dtype=torch.float64)
    shifted = logits + torch.randn(5, 1, dtype=torch.float64) * 50

    assert torch.allclose(to_probs(logits), to_probs(shifted), atol=1e-12)
    assert torch.allclose(centre_logits(logits), centre_logits(shifted), atol=1e-12)


def test_centring_is_shift_invariant_within_float32_precision():
    """
    The realistic case: logits arrive as float32 from the model.

    Tolerance is set by float32 epsilon at the magnitude of the shift
    (~150 * 2^-24 ~= 1e-5), not by the property being approximate.
    """
    torch.manual_seed(11)
    logits = torch.randn(5, 100)
    shifted = logits + torch.randn(5, 1) * 50

    assert torch.allclose(centre_logits(logits), centre_logits(shifted), atol=1e-4)


def test_logit_delta_is_invariant_to_arbitrary_shifts():
    """
    The whole point of centring: an uncentred delta would pick up the arbitrary
    constant and make e2's regression meaningless.
    """
    torch.manual_seed(12)
    before = torch.randn(8, 200, dtype=torch.float64)
    after = torch.randn(8, 200, dtype=torch.float64)

    baseline = logit_delta(after, before)
    shifted = logit_delta(after + 17.0, before - 4.0)

    assert torch.allclose(baseline, shifted, atol=1e-12)


def test_logit_delta_shift_invariance_holds_for_float32_inputs():
    """Same property, float32 inputs, tolerance set by float32 epsilon at |shift| ~ 17."""
    torch.manual_seed(13)
    before = torch.randn(8, 200)
    after = torch.randn(8, 200)

    baseline = logit_delta(after, before)
    shifted = logit_delta(after + 17.0, before - 4.0)

    assert torch.allclose(baseline, shifted, atol=1e-4)


def test_uncentred_delta_would_be_shift_dependent():
    """
    Negative control for the two tests above: without centring, the same
    comparison fails badly. This is what makes them meaningful rather than
    vacuously true.
    """
    torch.manual_seed(14)
    before = torch.randn(8, 200, dtype=torch.float64)
    after = torch.randn(8, 200, dtype=torch.float64)

    naive_baseline = (after - before).mean(dim=0)
    naive_shifted = ((after + 17.0) - (before - 4.0)).mean(dim=0)

    # The uncentred delta is off by exactly the injected constant, 17 - (-4) = 21.
    assert torch.allclose(naive_shifted - naive_baseline, torch.full((200,), 21.0, dtype=torch.float64), atol=1e-9)
    assert not torch.allclose(naive_baseline, naive_shifted, atol=1e-4)


def test_kl_is_non_negative():
    p = to_probs(torch.randn(10, 80))
    q = to_probs(torch.randn(10, 80))
    assert (kl_divergence(p, q) >= -1e-9).all()


def test_js_is_symmetric_and_bounded_by_log2():
    p = to_probs(torch.randn(10, 80))
    q = to_probs(torch.randn(10, 80))
    forward = js_divergence(p, q)
    backward = js_divergence(q, p)
    assert torch.allclose(forward, backward, atol=1e-10)
    assert (forward <= math.log(2) + 1e-9).all()


def test_token_set_mass_sums_correctly():
    probs = to_probs(torch.randn(4, 32))
    ids = [0, 5, 9]
    expected = probs[:, ids].sum(dim=1)
    assert torch.allclose(token_set_mass(probs, ids), expected, atol=1e-12)


# -----------------------------------------------------------------------------
# Temperature matching -- the machinery e1's verdict depends on
# -----------------------------------------------------------------------------


def test_temperature_solver_hits_the_target_entropy():
    torch.manual_seed(0)
    logits = torch.randn(16, 500) * 3.0
    target = torch.full((16,), 3.0, dtype=torch.float64)

    temps = solve_temperature_for_entropy(logits, target)
    achieved = entropy(to_probs(centre_logits(logits) / temps.unsqueeze(-1)))

    assert torch.allclose(achieved, target, atol=1e-4), (
        f"max error {float((achieved - target).abs().max())}"
    )


def test_temperature_match_equalises_entropy():
    torch.manual_seed(1)
    sharp = torch.randn(12, 300) * 6.0   # low entropy
    flat = torch.randn(12, 300) * 0.5    # high entropy

    matched, temps = temperature_match(flat, sharp)

    assert torch.allclose(entropy(matched), entropy(to_probs(sharp)), atol=1e-4)
    # Flattening a flat distribution to match a sharp one requires T < 1.
    assert (temps < 1.0).all()


def test_temperature_matching_preserves_token_ranking():
    """
    The property that makes temperature the RIGHT control for e1: it changes
    confidence without reordering tokens. If rescaling could reorder them, a
    null result would be uninterpretable.
    """
    torch.manual_seed(2)
    logits = torch.randn(4, 200)
    matched, _ = temperature_match(logits, logits * 4.0)
    assert torch.equal(logits.argsort(dim=-1), matched.argsort(dim=-1))


# -----------------------------------------------------------------------------
# Frequency binning -- e2
# -----------------------------------------------------------------------------


def test_binned_logit_shift_recovers_a_planted_slope():
    """Construct a delta that is exactly linear in log frequency; check we find it."""
    torch.manual_seed(3)
    vocab = 5000
    log_freq = torch.linspace(-15.0, -3.0, vocab, dtype=torch.float64)
    true_slope = 0.35
    delta = true_slope * log_freq + 1.2

    result = binned_logit_shift(delta, log_freq, n_bins=20)

    assert result.slope == pytest.approx(true_slope, rel=1e-3)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.spearman_rho > 0.99
    assert sum(result.bin_token_count) == vocab


def test_binned_logit_shift_finds_nothing_in_noise():
    torch.manual_seed(4)
    vocab = 5000
    log_freq = torch.linspace(-15.0, -3.0, vocab, dtype=torch.float64)
    delta = torch.randn(vocab, dtype=torch.float64)

    result = binned_logit_shift(delta, log_freq, n_bins=20)

    assert abs(result.spearman_rho) < 0.2
    assert result.r_squared < 0.05


# -----------------------------------------------------------------------------
# Prompt independence -- e3
# -----------------------------------------------------------------------------


def test_pairwise_js_is_zero_for_identical_rows():
    """A perfectly collapsed model: every prompt gives the same distribution."""
    single = to_probs(torch.randn(1, 100))
    collapsed = single.expand(20, 100).contiguous()
    assert mean_pairwise_js(collapsed) == pytest.approx(0.0, abs=1e-9)


def test_pairwise_js_is_large_for_disjoint_point_masses():
    """A maximally prompt-sensitive model: every prompt picks a different token."""
    logits = torch.full((10, 100), -1e4)
    for i in range(10):
        logits[i, i] = 1e4
    assert mean_pairwise_js(to_probs(logits)) == pytest.approx(math.log(2), abs=1e-6)


def test_pairwise_js_ordering_matches_intuition():
    """A collapsed distribution must score below a varied one -- e3's core claim."""
    torch.manual_seed(5)
    varied = to_probs(torch.randn(24, 150) * 5.0)
    base = torch.randn(1, 150) * 5.0
    collapsed = to_probs(base + torch.randn(24, 150) * 0.01)
    assert mean_pairwise_js(collapsed) < mean_pairwise_js(varied)
