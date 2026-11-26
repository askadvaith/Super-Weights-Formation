"""
The anchor paper's own super-weight detection method (§3.1), reimplemented.

Two reasons this exists in a codebase nominally about §4.4 and §4.5:

  1. VERIFICATION. The coordinates in `sw_directory.py` were transcribed from a
     two-column PDF table. Every §4.5 experiment ablates one specific scalar, so
     a transcription error would silently invalidate all of them. `verify()`
     re-derives the coordinate independently and checks agreement.

  2. COVERAGE. Models outside the paper's directory (Pythia, anything from
     2025-26) have no published coordinates. `detect()` finds them.

------------------------------------------------------------------------------
THE METHOD
------------------------------------------------------------------------------
From §3.1: for down_proj weight W [D, H] and input X [L, H], with Y = X W^T,

    Y_ij = sum_k X_ik W_jk

If X_ik and W_jk are both large outliers, Y_ij is dominated by their product:

    Y_ij ~= X_ik * W_jk

So: profile the max-magnitude activation of every layer's down_proj INPUT and
OUTPUT in one forward pass. The output spike gives j (the residual channel);
the input spike marks the layer; the weight itself picks out k.

------------------------------------------------------------------------------
THREE THINGS THIS IMPLEMENTATION DOES THAT A NAIVE READING DOES NOT
------------------------------------------------------------------------------
All three were mistakes in an earlier version here, and each produced
confident, wrong coordinates on OLMo-1B.  They are worth stating explicitly
because the paper's prose does not.

  1. ONSET, NOT ARGMAX.  The super activation is created in one layer and then
     CARRIED DOWN THE RESIDUAL STREAM.  Later layers read it and can write
     large values into the very same channel, so the globally largest down_proj
     output is not necessarily where the phenomenon starts.  In OLMo-1B the
     largest down_proj output in the whole model is at layer 15 (|a| = 414),
     nearly twice the layer-1 spike (262) that actually creates the super
     activation -- layer 15's MLP is *cancelling* the super activation before
     the unembedding (residual channel 1764 goes -412 -> +16 across it).
     Taking an argmax over layers finds the eraser, not the source.  We scan
     layers in ASCENDING order and take the first that qualifies.

  2. THE COLUMN COMES FROM THE WEIGHT, NOT FROM argmax|X|.  Given the row j,
     the mlp channel is the k that dominates sum_k X_ik W_jk -- that is,
     argmax_k |X_ik * W_jk|.  Using argmax_k |X_ik| alone ignores W entirely
     and, at layers where some unrelated channel is loud, returns a coordinate
     whose weight is perfectly ordinary.

  3. THE WEIGHT MUST ACTUALLY BE AN OUTLIER.  The identity above requires
     W_jk to be a large outlier.  Every candidate is therefore gated on
     `weight_outlier_stats` and the numbers are reported, so a coordinate can
     never be accepted on activation evidence alone.

------------------------------------------------------------------------------
LIMITATIONS
------------------------------------------------------------------------------
Because the procedure is *derived from* the identity above, it can only ever
return a `down_proj` weight whose effect is a single dominant scalar product.
It is not a search over the full parameter space; it is a search over the one
family of weights the identity describes.

Note also that the paper's stopping rule -- "until the magnitudes of large
maximum activations are greatly suppressed" -- is a judgement call about a
plot, not a threshold.  `min_spike_ratio` and `min_weight_ratio` make it
explicit, and the number of super weights found is a function of both.  See
`DetectionResult.rejected` for what fell just outside.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .modeling import (
    LoadedModel,
    WeightStats,
    down_proj_row,
    down_proj_weight_view,
    get_decoder_layers,
    get_down_proj,
    sw_index,
    weight_outlier_stats,
)
from .sw_directory import SuperWeight

log = logging.getLogger(__name__)

# The paper stresses that detection needs only a single prompt. We use a short
# neutral one by default; `detect()` accepts any prompt so you can check the
# claim that the result is prompt-independent.
DEFAULT_DETECTION_PROMPT = "Apple Inc. is a worldwide tech company."

# A 512-token neutral, normal-looking expository text (no Lorem) used as the
# training-twins telemetry probe. With ~512 tokens the spike search has ~57x
# more position candidates than the 9-token default, which stabilises the
# watchlist coordinate (the (j*, k*) the watchlist traces through 50,000
# training steps) and tightens the std-error of mean_input_spike by ~7x.
# Truncated at max_length=512 by profile_down_proj().
TRAINING_TWINS_PROMPT = (
    "The history of the company began in 1976, when two college friends decided "
    "to start a small business in a garage in the suburbs. They had no real "
    "plan, only a shared interest in computers and a belief that ordinary "
    "people would eventually want one in their home. In the first year they "
    "sold a few dozen machines to local hobbyists, mostly by word of mouth. "
    "The founders were careful with money, paid themselves modest salaries, "
    "and reinvested almost everything back into the workshop. By the early "
    "1980s the team had grown to a dozen engineers, and the company moved "
    "into a converted warehouse on the edge of the city. They released their "
    "first consumer product in 1983, a simple desktop computer that became "
    "surprisingly popular with schools and small businesses. Sales doubled "
    "every year for nearly a decade, and the firm expanded into neighbouring "
    "countries, opening small offices in several European capitals. Critics at "
    "the time doubted the long-term prospects, arguing that the market for "
    "personal computers would soon be saturated. The leadership disagreed, "
    "and poured resources into research and development, hiring physicists, "
    "linguists, and designers alongside the original engineers. In 1995 the "
    "company shipped its first laptop, which was lighter and cheaper than "
    "anything else on the market, and within two years it had become the "
    "best-selling portable computer in the country. The early 2000s brought "
    "new challenges as global competition intensified, and the founders began "
    "to think seriously about succession. They appointed a chief executive "
    "from outside the company, a move that surprised the press but was "
    "ultimately well received by employees. Over the next decade the firm "
    "diversified into software, services, and eventually mobile devices. "
    "Each new line of business was treated as a separate division with its "
    "own budget, its own engineering team, and its own culture. The pattern "
    "of patient, long-term investment continued, even when quarterly results "
    "disappointed analysts. By 2015 the company employed more than thirty "
    "thousand people worldwide and reported revenues in the tens of billions. "
    "Its products could be found in nearly every office, school, and home. "
    "Looking back, the founders often said that the most important decision "
    "they ever made was to keep the company private for as long as they did. "
    "They believed that public markets rewarded short-term thinking, and they "
    "wanted the freedom to invest in projects that might not pay off for a "
    "decade. That discipline, more than any single product, is what made the "
    "firm what it is today. The board meets four times a year, usually in a "
    "quiet room above the main office, with a long table, white walls, and a "
    "single window overlooking the courtyard."
)

#: A layer's activation spike counts only if it is this many times the median
#: layer. The paper gives no number; 3.0 is loose enough to catch every
#: coordinate in its own Table 2 and tight enough to exclude flat layers.
DEFAULT_MIN_SPIKE_RATIO = 3.0

#: A candidate's weight must be at least this many times the mean |W| of its
#: own tensor. Genuine super weights in the paper's models run 30-140x; the
#: false positives the activation criteria alone let through run below 10x.
DEFAULT_MIN_WEIGHT_RATIO = 20.0

#: How many super weights a search will return before giving up. This is a
#: BUDGET, not a belief about how many exist: the paper never tabulates more
#: than six, but its stopping rule is a judgement call about a plot, and on
#: every model tried here detection keeps finding genuine outliers past the
#: sixth. A low cap makes the search stop early and report a count that is
#: really your parameter, so the default is set well above what any model here
#: has produced -- let `stopped_because == "exhausted"` be what ends the search.
DEFAULT_MAX_SUPER_WEIGHTS = 20

#: The single weight must explain at least this fraction of the output spike.
#: This is §3.1's own approximation Y_ij ~= X_ik * W_jk turned into a test
#: rather than an assumption. It is what separates a spike genuinely caused by
#: one scalar (OLMo-1B L1 scores 0.97, L2 0.90) from one that is the sum of
#: many medium-sized terms and merely happens to have an outlier weight
#: somewhere in the row (L2[623, 2074] scores 0.35).
DEFAULT_MIN_DOMINANCE = 0.5


@dataclass
class LayerSpikeProfile:
    """Per-layer max-magnitude activations at the down_proj boundary."""

    layer: int

    max_input_abs: float
    max_input_channel: int
    """Argmax over d_ff."""
    max_input_position: int
    """Token position of the input spike. The paper observes position 0."""

    max_output_abs: float
    max_output_channel: int
    """Argmax over d_model -- the super weight's ROW, if this layer qualifies."""
    max_output_position: int

    output_peak_value: float
    """Signed Y[pos, channel] at the output spike -- the denominator when we ask
    how much of the output one weight explains."""

    input_at_output_peak: torch.Tensor = field(repr=False)
    """The full down_proj input row [d_ff] at `max_output_position`. Needed to
    find which mlp channel dominates the output spike."""


@dataclass
class Candidate:
    """One coordinate plus every number that argues for or against it."""

    super_weight: SuperWeight
    weight_stats: WeightStats

    layer_input_abs: float
    layer_output_abs: float
    input_spike_ratio: float
    """Layer's max |down_proj input| relative to the median layer."""
    output_spike_ratio: float

    dominance: float
    """|X_ik * W_jk| / |Y_ij|: the fraction of the output spike explained by this
    single weight. The paper's rank-1 approximation is only valid near 1.0."""

    position: int
    """Token position the evidence was read at."""

    iteration: int
    rejected_because: str | None = None

    def as_dict(self) -> dict:
        return {
            "coordinate": str(self.super_weight),
            "huggingface_path": (
                f"layers[{self.super_weight.layer}].mlp.down_proj.weight"
                f"[{self.super_weight.residual_channel}, {self.super_weight.mlp_channel}]"
            ),
            "iteration": self.iteration,
            "token_position": self.position,
            "layer_max_input_abs": self.layer_input_abs,
            "layer_max_output_abs": self.layer_output_abs,
            "input_spike_ratio": self.input_spike_ratio,
            "output_spike_ratio": self.output_spike_ratio,
            "dominance": self.dominance,
            "rejected_because": self.rejected_because,
            **self.weight_stats.as_dict(),
        }


@dataclass
class DetectionResult:
    """Everything `detect()` learned, including the evidence for each choice."""

    profiles: list[LayerSpikeProfile]
    """The first pass -- this is the paper's Figure 3."""

    accepted: list[Candidate]
    rejected: list[Candidate]
    """Candidates that had an aligned activation spike but whose weight was not
    an outlier. Inspect these before lowering a threshold."""

    prompt: str
    min_spike_ratio: float
    min_weight_ratio: float
    min_dominance: float

    stopped_because: str = "exhausted"
    """Either "exhausted" (no layer still qualifies -- the count is a finding)
    or "hit max_super_weights" (the search was cut off -- the count is YOUR
    parameter, not a property of the model). Without this, "found 3" is
    unreadable: it could mean the model has three or that you asked for three."""

    @property
    def search_was_truncated(self) -> bool:
        return self.stopped_because != "exhausted"

    @property
    def candidates(self) -> list[SuperWeight]:
        return [c.super_weight for c in self.accepted]

    @property
    def weight_values(self) -> list[float]:
        return [c.weight_stats.value for c in self.accepted]

    @property
    def spike_ratios(self) -> list[float]:
        return [c.output_spike_ratio for c in self.accepted]


@torch.no_grad()
def profile_down_proj(
    loaded: LoadedModel,
    prompt: str = DEFAULT_DETECTION_PROMPT,
    max_length: int = 512,
) -> list[LayerSpikeProfile]:
    """
    One forward pass; record max |activation| at every layer's down_proj I/O.

    This reproduces Figures 3, 9 and 10 of the paper: plot `max_input_abs`
    against layer index and you get their figure directly.

    Unlike a plain per-channel max, this keeps the token POSITION of each spike
    and the full input row at the output spike, because both are needed to
    attribute the spike to a specific weight rather than guess at one.
    """
    tokenizer = loaded.tokenizer
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded["attention_mask"].to(loaded.device)

    n_layers = len(get_decoder_layers(loaded.model))
    captured: dict[int, dict[str, torch.Tensor]] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs, output):
            # inputs[0] is the down_proj input  [batch, seq, d_ff] or [seq, d_ff] (e.g. OPT flattens batch)
            # output    is the down_proj output [batch, seq, d_model] or [seq, d_model]
            inp_t = inputs[0].detach()
            if inp_t.ndim == 3:
                inp_t = inp_t[0]
            out_t = output.detach()
            if out_t.ndim == 3:
                out_t = out_t[0]

            captured[layer_idx] = {
                "input": inp_t.float().cpu(),
                "output": out_t.float().cpu(),
            }
        return hook

    try:
        for idx in range(n_layers):
            handles.append(
                get_down_proj(loaded.model, idx).register_forward_hook(make_hook(idx))
            )
        loaded.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    profiles: list[LayerSpikeProfile] = []
    for idx in range(n_layers):
        x = captured[idx]["input"]    # [seq, d_ff]
        y = captured[idx]["output"]   # [seq, d_model]

        in_val, in_flat = x.abs().flatten().max(dim=0)
        in_pos, in_ch = divmod(int(in_flat), x.shape[1])
        out_val, out_flat = y.abs().flatten().max(dim=0)
        out_pos, out_ch = divmod(int(out_flat), y.shape[1])

        profiles.append(
            LayerSpikeProfile(
                layer=idx,
                max_input_abs=float(in_val),
                max_input_channel=in_ch,
                max_input_position=in_pos,
                max_output_abs=float(out_val),
                max_output_channel=out_ch,
                max_output_position=out_pos,
                output_peak_value=float(y[out_pos, out_ch]),
                input_at_output_peak=x[out_pos].clone(),
            )
        )
    return profiles


def _dominant_mlp_channel(
    loaded: LoadedModel, layer: int, residual_channel: int, x_row: torch.Tensor
) -> tuple[int, float, float]:
    """
    Which mlp channel dominates sum_k X_k W_(residual_channel)k?

    Returns (channel, contribution, |contribution| / |full output|). The last
    number is the honest check on the paper's rank-1 approximation: if one
    weight really does explain the spike it is close to 1.0.
    """
    module = get_down_proj(loaded.model, layer)
    w_row = down_proj_row(module, residual_channel).detach().float().cpu()
    terms = x_row.float() * w_row
    channel = int(terms.abs().argmax())
    contribution = float(terms[channel])
    total = float(terms.sum())
    dominance = abs(contribution) / abs(total) if total else float("inf")
    return channel, contribution, dominance


def _median(values: Sequence[float]) -> float:
    return float(torch.tensor(list(values), dtype=torch.float64).median())


def detect(
    loaded: LoadedModel,
    prompt: str = DEFAULT_DETECTION_PROMPT,
    max_super_weights: int = DEFAULT_MAX_SUPER_WEIGHTS,
    min_spike_ratio: float = DEFAULT_MIN_SPIKE_RATIO,
    min_weight_ratio: float = DEFAULT_MIN_WEIGHT_RATIO,
    min_dominance: float = DEFAULT_MIN_DOMINANCE,
) -> DetectionResult:
    """
    Run the paper's iterative detection: find a spike, zero that weight, repeat.

    Args:
        max_super_weights: stop after this many. A budget, not an expectation:
            the paper tabulates at most six (Phi-3-mini), but this search
            routinely finds more, so the default is high enough that the
            thresholds -- not the cap -- normally decide where it stops. Check
            `stopped_because` before reading anything into the count.
        min_spike_ratio: a layer qualifies only if BOTH its down_proj input and
            its output exceed this multiple of the median layer. Requiring both
            is the paper's "alignment between input spikes and output spikes".
        min_weight_ratio: a candidate is accepted only if its weight is at least
            this many times the mean |W| of its own down_proj tensor.
        min_dominance: a candidate is accepted only if that one weight explains
            at least this fraction of the output spike.

    Layers are scanned in ascending order and the FIRST qualifying layer wins --
    see the module docstring on onset vs argmax.

    IMPORTANT: this mutates the model in place while searching, then restores
    every weight it touched before returning.
    """
    accepted: list[Candidate] = []
    rejected: list[Candidate] = []
    seen: set[tuple[int, int, int]] = set()
    rejected_seen: set[tuple[int, int, int]] = set()
    first_profiles: list[LayerSpikeProfile] = []
    stopped_because = "hit max_super_weights"

    # Keyed by coordinate, so a coordinate encountered twice cannot overwrite
    # its own saved original with the zero we just wrote.
    restore: dict[tuple[int, int, int], tuple[torch.nn.Module, int, int, torch.Tensor]] = {}

    try:
        for iteration in range(max_super_weights):
            profiles = profile_down_proj(loaded, prompt)
            if iteration == 0:
                first_profiles = profiles

            in_median = _median([p.max_input_abs for p in profiles])
            out_median = _median([p.max_output_abs for p in profiles])

            picked: Candidate | None = None
            for profile in profiles:  # ascending layer order == onset scan
                in_ratio = profile.max_input_abs / max(in_median, 1e-9)
                out_ratio = profile.max_output_abs / max(out_median, 1e-9)
                if in_ratio < min_spike_ratio or out_ratio < min_spike_ratio:
                    continue

                row = profile.max_output_channel
                col, _contribution, dominance = _dominant_mlp_channel(
                    loaded, profile.layer, row, profile.input_at_output_peak
                )
                sw = SuperWeight(
                    layer=profile.layer, residual_channel=row, mlp_channel=col
                )
                if sw.as_tuple() in seen:
                    continue  # already ablated; its spike is someone else's now

                module = get_down_proj(loaded.model, sw.layer)
                stats = weight_outlier_stats(module, sw)
                candidate = Candidate(
                    super_weight=sw,
                    weight_stats=stats,
                    layer_input_abs=profile.max_input_abs,
                    layer_output_abs=profile.max_output_abs,
                    input_spike_ratio=in_ratio,
                    output_spike_ratio=out_ratio,
                    dominance=dominance,
                    position=profile.max_output_position,
                    iteration=iteration,
                )

                if stats.ratio_to_mean < min_weight_ratio:
                    candidate.rejected_because = (
                        f"weight is only {stats.ratio_to_mean:.1f}x mean |W| "
                        f"(need {min_weight_ratio:.1f}x) -- the activation spike "
                        "at this layer is not produced by an outlier weight"
                    )
                elif dominance < min_dominance:
                    candidate.rejected_because = (
                        f"this weight explains only {dominance:.0%} of the output "
                        f"spike (need {min_dominance:.0%}) -- the spike is a sum of "
                        "many terms, so §3.1's Y_ij ~= X_ik * W_jk does not hold "
                        "here and there is no single weight to call the cause"
                    )

                if candidate.rejected_because:
                    # A rejected candidate is re-derived on every later
                    # iteration, since nothing about it changed. Record and log
                    # it once.
                    if sw.as_tuple() not in rejected_seen:
                        rejected_seen.add(sw.as_tuple())
                        rejected.append(candidate)
                        log.info(
                            "iter %d: %s rejected -- %s",
                            iteration, sw, candidate.rejected_because,
                        )
                    continue

                picked = candidate
                break

            if picked is None:
                stopped_because = "exhausted"
                log.info(
                    "iter %d: no layer has an aligned input+output spike above "
                    "%.1fx median, backed by a weight above %.1fx mean |W| that "
                    "explains >=%.0f%% of the spike. Stopping with %d super "
                    "weight(s).",
                    iteration, min_spike_ratio, min_weight_ratio,
                    100 * min_dominance, len(accepted),
                )
                break

            sw = picked.super_weight
            accepted.append(picked)
            seen.add(sw.as_tuple())
            log.info(
                "iter %d: %s  (onset layer; in %.1f = %.1fx median, "
                "out %.1f = %.1fx median, dominance %.3f)",
                iteration, sw, picked.layer_input_abs, picked.input_spike_ratio,
                picked.layer_output_abs, picked.output_spike_ratio, picked.dominance,
            )
            log.info("         %s", picked.weight_stats.describe())

            # Zero it and look again, as the paper describes.
            module = get_down_proj(loaded.model, sw.layer)
            weight, _ = down_proj_weight_view(module)
            i, j = sw_index(module, sw)
            restore[sw.as_tuple()] = (module, i, j, weight[i, j].detach().clone())
            with torch.no_grad():
                weight[i, j] = 0.0
        else:
            log.warning(
                "Stopped after %d super weight(s) because max_super_weights=%d "
                "was reached, NOT because the model ran out of candidates. The "
                "count is your parameter, not a property of %s -- raise "
                "--max-super-weights to see how many there really are.",
                len(accepted), max_super_weights, loaded.spec.key,
            )
    finally:
        with torch.no_grad():
            for module, i, j, original in restore.values():
                weight, _ = down_proj_weight_view(module)
                weight[i, j] = original

    return DetectionResult(
        profiles=first_profiles,
        accepted=accepted,
        rejected=rejected,
        prompt=prompt,
        min_spike_ratio=min_spike_ratio,
        min_weight_ratio=min_weight_ratio,
        min_dominance=min_dominance,
        stopped_because=stopped_because,
    )


@dataclass
class VerificationResult:
    """Agreement between the tabulated directory and independent detection."""

    tabulated: list[SuperWeight]
    detected: list[SuperWeight]
    tabulated_stats: dict[str, WeightStats] = field(default_factory=dict)
    """Weight magnitude / divergence-from-mean for each TABULATED coordinate.
    This alone settles most disputes: a tabulated coordinate holding an
    ordinary-magnitude scalar is wrong, whatever detection says."""

    detected_candidates: list[Candidate] = field(default_factory=list)

    exact_matches: list[SuperWeight] = field(default_factory=list)
    tabulated_only: list[SuperWeight] = field(default_factory=list)
    additional: list[SuperWeight] = field(default_factory=list)
    """Detected but not tabulated. Not a failure: the paper's stopping rule is a
    judgement call, so finding more is expected."""

    layer_mismatches: list[tuple[SuperWeight, SuperWeight]] = field(default_factory=list)
    """(tabulated, detected) pairs with the same (row, col) at different layers.
    Almost always a wrong layer number in the table rather than a real
    disagreement -- and the fastest thing to check, because the weight
    magnitudes at the two coordinates settle it outright."""

    @property
    def agrees(self) -> bool:
        """Every tabulated coordinate was independently reproduced."""
        return bool(self.tabulated) and not self.tabulated_only

    @property
    def bad_tabulated(self) -> list[SuperWeight]:
        """Tabulated coordinates whose scalar is not an outlier at all."""
        return [
            sw for sw in self.tabulated
            if str(sw) in self.tabulated_stats
            and not self.tabulated_stats[str(sw)].is_outlier
        ]

    @property
    def summary(self) -> str:
        if not self.tabulated:
            return f"No tabulated entries; detection found {len(self.detected)}."
        if self.agrees:
            extra = (
                f" Detection also found {len(self.additional)} coordinate(s) the "
                "table does not list."
                if self.additional else ""
            )
            return (
                f"Directory VERIFIED: all {len(self.exact_matches)} tabulated "
                f"coordinate(s) reproduced.{extra}"
            )
        return (
            f"Directory MISMATCH: {len(self.exact_matches)} of "
            f"{len(self.tabulated)} tabulated coordinate(s) reproduced, "
            f"{len(self.tabulated_only)} not found"
            + (
                f", {len(self.layer_mismatches)} found at a DIFFERENT LAYER"
                if self.layer_mismatches else ""
            )
            + "."
        )


def verify(
    loaded: LoadedModel,
    prompt: str = DEFAULT_DETECTION_PROMPT,
    max_super_weights: int | None = None,
    min_spike_ratio: float = DEFAULT_MIN_SPIKE_RATIO,
    min_weight_ratio: float = DEFAULT_MIN_WEIGHT_RATIO,
    min_dominance: float = DEFAULT_MIN_DOMINANCE,
) -> VerificationResult:
    """
    Check the transcribed directory against independent detection.

    Run this before trusting any §4.5 result on a model whose coordinates came
    from the PDF table. Records `weight_outlier_stats` at each tabulated
    coordinate, because a super weight is by definition a conspicuous outlier in
    its tensor -- a value near the tensor's typical magnitude is proof on its
    own that the coordinate is wrong, no detection required.
    """
    tabulated = list(loaded.spec.super_weights)
    budget = (
        max_super_weights
        if max_super_weights is not None
        else max(len(tabulated) + 4, DEFAULT_MAX_SUPER_WEIGHTS)
    )
    result = detect(
        loaded,
        prompt=prompt,
        max_super_weights=max(budget, 1),
        min_spike_ratio=min_spike_ratio,
        min_weight_ratio=min_weight_ratio,
        min_dominance=min_dominance,
    )
    detected = result.candidates

    tab_keys = {sw.as_tuple() for sw in tabulated}
    det_keys = {sw.as_tuple() for sw in detected}

    stats: dict[str, WeightStats] = {}
    for sw in tabulated:
        try:
            stats[str(sw)] = weight_outlier_stats(loaded.down_proj(sw.layer), sw)
        except Exception as exc:  # out-of-range index, wrong architecture, ...
            log.warning("Could not read %s: %s", sw, exc)

    missing = [sw for sw in tabulated if sw.as_tuple() not in det_keys]
    by_channel = {
        (sw.residual_channel, sw.mlp_channel): sw
        for sw in detected
        if sw.as_tuple() not in tab_keys
    }
    layer_mismatches = [
        (sw, by_channel[(sw.residual_channel, sw.mlp_channel)])
        for sw in missing
        if (sw.residual_channel, sw.mlp_channel) in by_channel
    ]

    return VerificationResult(
        tabulated=tabulated,
        detected=detected,
        tabulated_stats=stats,
        detected_candidates=result.accepted,
        exact_matches=[sw for sw in tabulated if sw.as_tuple() in det_keys],
        tabulated_only=missing,
        additional=[sw for sw in detected if sw.as_tuple() not in tab_keys],
        layer_mismatches=layer_mismatches,
    )


def resolve_super_weights(
    loaded: LoadedModel,
    override: Sequence[SuperWeight] | None = None,
    auto_detect: bool = True,
) -> tuple[list[SuperWeight], str]:
    """
    Decide which super weights an experiment should ablate.

    Precedence: explicit override > the published directory > live detection.
    Returns (super_weights, provenance_string) so provenance lands in the run
    config, and nobody has to guess later where a coordinate came from.
    """
    if override:
        return list(override), "override"
    if loaded.spec.super_weights:
        return list(loaded.spec.super_weights), "directory(Table 2, verified)"
    if not auto_detect:
        raise RuntimeError(
            f"{loaded.spec.key} has no published super weights and auto-detect "
            "is off. Pass --detect or supply --super-weight L,ROW,COL."
        )
    log.info("%s is not in the paper's directory -- detecting.", loaded.spec.key)
    result = detect(loaded)
    if not result.candidates:
        raise RuntimeError(
            f"Detection found no super weight in {loaded.spec.key}. "
            "Either this model has none, or the thresholds are too strict."
        )
    return result.candidates, f"detected(prompt={result.prompt!r})"
