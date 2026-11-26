"""
Distributional metrics for the §4.5 re-analysis.

All arithmetic here is done in **float64**. That is not fussiness: the whole
argument turns on comparing probability masses of a few percent, and on
entropies that differ by fractions of a nat. fp16/bf16 accumulation error is of
comparable magnitude to the effects being measured.

------------------------------------------------------------------------------
THE ONE SUBTLETY THAT MATTERS MOST: LOGIT SHIFT-INVARIANCE
------------------------------------------------------------------------------
Softmax is invariant to adding a constant to every logit. So a raw difference
`logits_ablated - logits_original` contains an arbitrary additive constant and
is meaningless on its own -- you can make the "average logit shift" whatever
you like. Every function here that compares logits centres them first
(subtracting the per-row mean over the vocabulary). `centre_logits` is applied
inside `logit_delta`, but if you write new analyses, apply it yourself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


# =============================================================================
# Basic distributional quantities
# =============================================================================


def to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax in float64. Input [.., vocab] -> output [.., vocab]."""
    return torch.softmax(logits.double(), dim=-1)


def centre_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Remove the arbitrary additive constant from each row of logits.

    Softmax(z) == Softmax(z + c), so only centred logits are comparable across
    two forward passes. See the module docstring.
    """
    z = logits.double()
    return z - z.mean(dim=-1, keepdim=True)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in NATS, per row. Input [.., vocab] -> [..]."""
    p = probs.double()
    return -(p * torch.log(p.clamp_min(1e-300))).sum(dim=-1)


def perplexity_of_distribution(probs: torch.Tensor) -> torch.Tensor:
    """exp(entropy): the 'effective number of tokens' the model is choosing between."""
    return torch.exp(entropy(probs))


def top1_prob(probs: torch.Tensor) -> torch.Tensor:
    """Probability of the argmax token, per row."""
    return probs.max(dim=-1).values


def topk(probs: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (values, indices) of the top-k tokens, per row."""
    return torch.topk(probs, k=k, dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL(p || q) in nats, per row.

    Both inputs are probability distributions over the last axis. q is clamped
    away from zero; with a full-vocabulary softmax q is never exactly zero
    anyway, but the clamp guards against a passed-in empirical distribution.
    """
    p64, q64 = p.double(), q.double()
    return (p64 * (torch.log(p64.clamp_min(1e-300)) - torch.log(q64.clamp_min(1e-300)))).sum(-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence in nats, per row. Symmetric and bounded by ln 2.

    Used by e3 for pairwise prompt comparisons, where KL's asymmetry and
    unboundedness would make the numbers hard to read.
    """
    p64, q64 = p.double(), q.double()
    m = 0.5 * (p64 + q64)
    return 0.5 * kl_divergence(p64, m) + 0.5 * kl_divergence(q64, m)


def token_set_mass(probs: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """
    Total probability assigned to a set of token ids, per row.

    This is how "stopword probability" is operationalised throughout: the sum
    of probability over all stopword token ids, not the probability of a single
    hand-picked token. The paper reports single tokens ("the", ".", ","), which
    we also compute, but a set-level mass is the honest aggregate.
    """
    idx = torch.as_tensor(list(token_ids), dtype=torch.long, device=probs.device)
    return probs.double().index_select(dim=-1, index=idx).sum(dim=-1)


# =============================================================================
# Temperature matching -- the core of experiment e1
# =============================================================================


def entropy_at_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Entropy of softmax(logits / T), per row."""
    return entropy(to_probs(logits.double() / temperature))


def solve_temperature_for_entropy(
    logits: torch.Tensor,
    target_entropy: torch.Tensor,
    lo: float = 1e-3,
    hi: float = 1e3,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> torch.Tensor:
    """
    Per row, find T > 0 such that H(softmax(logits / T)) == target_entropy.

    Entropy is monotonically increasing in T (T -> 0 gives a point mass with
    H = 0; T -> inf gives the uniform distribution with H = log V), so plain
    bisection on log T converges reliably and needs no gradients.

    WHY THIS EXISTS
    ---------------
    This is the decisive control for §4.5. The claim under test is that
    "stopwords rise after ablation" is an artifact of the output distribution
    becoming flatter, not of stopwords being selectively promoted. If we
    re-sharpen the ablated distribution back to the ORIGINAL model's entropy
    and stopword mass returns to baseline, the stopword claim is fully
    explained by confidence collapse and nothing else.

    Args:
        logits: [batch, vocab], uncentred is fine (temperature scaling of a
            shifted logit vector is not the same as of an unshifted one, so we
            centre internally to make T well-defined).
        target_entropy: [batch], the entropy to match.

    Returns:
        [batch] tensor of temperatures. Rows whose target entropy is
        unreachable (outside [0, log V]) return the clamped bound.
    """
    z = centre_logits(logits)                       # [batch, vocab]
    target = target_entropy.double().to(z.device)   # [batch]

    lo_t = torch.full_like(target, lo)
    hi_t = torch.full_like(target, hi)

    for _ in range(max_iter):
        mid = torch.sqrt(lo_t * hi_t)               # geometric mean == bisect log T
        h_mid = entropy(to_probs(z / mid.unsqueeze(-1)))
        too_flat = h_mid > target
        # If the distribution at `mid` is flatter than the target, we need a
        # LOWER temperature; otherwise a higher one.
        hi_t = torch.where(too_flat, mid, hi_t)
        lo_t = torch.where(too_flat, lo_t, mid)
        if torch.all((hi_t - lo_t) / lo_t < tol):
            break

    return torch.sqrt(lo_t * hi_t)


def temperature_match(
    logits_to_rescale: torch.Tensor,
    reference_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rescale one distribution to have the same per-row entropy as another.

    Returns (matched_probs, temperatures).
    """
    reference_entropy = entropy(to_probs(reference_logits))
    temps = solve_temperature_for_entropy(logits_to_rescale, reference_entropy)
    matched = to_probs(centre_logits(logits_to_rescale) / temps.unsqueeze(-1))
    return matched, temps


# =============================================================================
# Frequency-binned logit shift -- experiment e2
# =============================================================================


@dataclass
class FrequencyBinResult:
    """Output of `binned_logit_shift`."""

    bin_edges: list[float]
    bin_mean_log_freq: list[float]
    bin_mean_delta: list[float]
    bin_sem_delta: list[float]
    bin_token_count: list[int]
    slope: float
    """OLS slope of mean centred-logit-delta on log unigram frequency."""
    intercept: float
    r_squared: float
    spearman_rho: float
    """Rank correlation -- robust to the exact shape of the relationship."""


def logit_delta(
    logits_after: torch.Tensor, logits_before: torch.Tensor
) -> torch.Tensor:
    """
    Centred logit difference, averaged over rows. Returns [vocab].

    Centring is essential -- see module docstring.
    """
    delta = centre_logits(logits_after) - centre_logits(logits_before)
    return delta.mean(dim=0)


def binned_logit_shift(
    delta: torch.Tensor,
    log_unigram_freq: torch.Tensor,
    n_bins: int = 20,
    min_log_freq: float = -20.0,
) -> FrequencyBinResult:
    """
    Bin the vocabulary by log unigram frequency and report the mean logit shift
    in each bin.

    THE POINT OF THIS EXPERIMENT
    ----------------------------
    If ablating the super weight moves the output toward the unigram prior,
    then the logit shift should increase monotonically with token frequency:
    common tokens gain, rare tokens lose. "Stopwords" would then simply be the
    highest-frequency bin, and the paper's framing ("super weights suppress
    stopword likelihood") would be a special case of a general frequency
    effect, stated in a way that implies a specific linguistic function it does
    not have.

    A strong positive slope with high R^2 supports that reading. A flat or
    non-monotone relationship with stopwords nonetheless elevated would
    genuinely support the paper.

    Args:
        delta: [vocab] mean centred logit change.
        log_unigram_freq: [vocab] log unigram probability.
        n_bins: number of equal-count (quantile) bins.
        min_log_freq: tokens rarer than this are dropped -- they are dominated
            by the add-k smoothing floor and would form a spurious spike.
    """
    d = delta.double().cpu()
    f = log_unigram_freq.double().cpu()
    keep = f > min_log_freq
    d, f = d[keep], f[keep]

    # Quantile bins give equal token counts per bin, which keeps the standard
    # errors comparable across bins. Frequency-uniform bins would put 90% of
    # the vocabulary in the rarest bin.
    quantiles = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    edges = torch.quantile(f, quantiles)
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    mean_freq, mean_delta, sem_delta, counts = [], [], [], []
    for i in range(n_bins):
        mask = (f > edges[i]) & (f <= edges[i + 1])
        n = int(mask.sum().item())
        counts.append(n)
        if n == 0:
            mean_freq.append(float("nan"))
            mean_delta.append(float("nan"))
            sem_delta.append(float("nan"))
            continue
        mean_freq.append(float(f[mask].mean()))
        mean_delta.append(float(d[mask].mean()))
        sem_delta.append(float(d[mask].std(unbiased=True) / math.sqrt(n)) if n > 1 else 0.0)

    slope, intercept, r2 = _ols(f, d)
    rho = _spearman(f, d)

    return FrequencyBinResult(
        bin_edges=edges.tolist(),
        bin_mean_log_freq=mean_freq,
        bin_mean_delta=mean_delta,
        bin_sem_delta=sem_delta,
        bin_token_count=counts,
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        spearman_rho=rho,
    )


def _ols(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    """Ordinary least squares fit of y on x. Returns (slope, intercept, R^2)."""
    x, y = x.double(), y.double()
    x_mean, y_mean = x.mean(), y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    slope = (cov / var) if var > 0 else torch.tensor(0.0, dtype=torch.float64)
    intercept = y_mean - slope * x_mean
    predicted = slope * x + intercept
    ss_res = ((y - predicted) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0, dtype=torch.float64)
    return float(slope), float(intercept), float(r2)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Spearman rank correlation: the Pearson correlation of the ranks.

    Reported alongside the OLS slope in e2 because it is robust to the exact
    functional form. The frequency effect need not be linear in log frequency
    for the "output moves toward the unigram prior" reading to hold -- it only
    needs to be monotone, which is what rho measures.
    """
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = (rx.norm() * ry.norm()).item()
    if denominator == 0:
        return float("nan")
    return float((rx @ ry).item() / denominator)


def _rank(v: torch.Tensor) -> torch.Tensor:
    """Ranks of a 1-D tensor (ties broken arbitrarily by argsort order)."""
    order = torch.argsort(v)
    ranks = torch.empty_like(v, dtype=torch.float64)
    ranks[order] = torch.arange(len(v), dtype=torch.float64, device=v.device)
    return ranks


# =============================================================================
# Prompt-independence -- experiment e3
# =============================================================================


def mean_pairwise_js(probs: torch.Tensor, max_pairs: int = 20_000, seed: int = 0) -> float:
    """
    Mean Jensen-Shannon divergence between the output distributions of
    different prompts.

    A context-conditioned model gives very different distributions for
    different prompts, so this is large. A model whose output has collapsed to
    a fixed prior gives nearly identical distributions regardless of input, so
    this approaches zero.

    This is the cleanest single number for "has the model stopped listening to
    the prompt?", and the anchor paper never measures it -- despite its own
    claim that super activations are prompt-independent, which makes the
    question obvious.

    Sub-samples pairs when the batch is large, since the full pairwise matrix
    is O(n^2 * vocab).
    """
    n = probs.shape[0]
    if n < 2:
        return float("nan")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    else:
        i_idx = torch.randint(0, n, (max_pairs,), generator=generator)
        j_idx = torch.randint(0, n, (max_pairs,), generator=generator)
        keep = i_idx != j_idx
        i_idx, j_idx = i_idx[keep], j_idx[keep]

    i_idx = i_idx.to(probs.device)
    j_idx = j_idx.to(probs.device)

    # Chunk to bound peak memory: [chunk, vocab] rather than [pairs, vocab].
    chunk, total, count = 4096, 0.0, 0
    for start in range(0, len(i_idx), chunk):
        a = probs.index_select(0, i_idx[start : start + chunk])
        b = probs.index_select(0, j_idx[start : start + chunk])
        vals = js_divergence(a, b)
        total += float(vals.sum())
        count += vals.numel()
    return total / max(count, 1)


def effective_distinct_distributions(probs: torch.Tensor, seed: int = 0) -> float:
    """
    A crude but readable summary: exp(mean pairwise JS) scaled to [1, n].

    Reported alongside `mean_pairwise_js` because a bare divergence in nats is
    hard to interpret; this expresses the same thing as "the model is behaving
    as if it had roughly N distinct responses".
    """
    mean_js = mean_pairwise_js(probs, seed=seed)
    if mean_js != mean_js:  # NaN
        return float("nan")
    # JS is bounded by ln 2; map [0, ln2] -> [1, n] monotonically.
    n = probs.shape[0]
    return 1.0 + (n - 1) * (mean_js / math.log(2))
