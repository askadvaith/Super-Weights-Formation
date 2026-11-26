"""
g1 -- Cross-family activation zoo evaluation: does super weight localisation
      track the gated MLP form across 8+ model families?

------------------------------------------------------------------------------
PURPOSE
------------------------------------------------------------------------------
t0 established the gated-vs-non-gated split on a LIMITED set of models (the
Pythia ladder plus OLMo/SmolLM positive controls).  g1 scales this to
a COMPREHENSIVE zoo of 40+ pretrained models spanning:

    Gated (SwiGLU/GeGLU):
        SmolLM, SmolLM2, MobileLLM, Qwen2.5, Qwen3, OLMo, OLMo-2,
        TinyLlama, Llama-3.2, StableLM2, Gemma-2

    Non-Gated (GELU/ReLU):
        Pythia, GPT-Neo, GPT-2, OPT, BLOOM, Cerebras-GPT, Phi-1.5/2,
        Falcon-RW

The experiment runs the same detect-confirm-control pipeline as t0, but
across every model in the zoo and produces:

  1. A per-model summary table (detected, confirmed, dominance, damage).
  2. An architecture-grouped contingency table (gated vs non-gated).
  3. Scale-tier summaries showing whether the effect is consistent within
     each tier (~100M, ~1B, ~3B).

The output is what a paper table or chi-squared test would consume.

------------------------------------------------------------------------------
METHOD
------------------------------------------------------------------------------
For each model in the zoo:

  1. DETECT -- run the paper's §3.1 activation-spike search.
  2. CONFIRM CAUSALLY -- ablate the accepted coordinate and measure KL,
     entropy shift, and top-1 drop on a small probe set.
  3. CONTROL -- ablate a magnitude-matched ordinary weight from the same
     tensor and compare damage.

This is the same pipeline as t0._check_one, factored out for reuse.

The verdict groups results by MLP form and tests whether the gated/non-gated
split holds universally, partially, or not at all.
"""

from __future__ import annotations

from ..core.args import ExpArgs
from ..core.data import load_prompts
from ..core.detection import DEFAULT_DETECTION_PROMPT, detect
from ..core.logging_utils import Prediction, Run, Verdict
from ..core.metrics import entropy, kl_divergence, to_probs, top1_prob
from ..core.modeling import (
    LoadedModel,
    down_proj_weight_view,
    final_logits,
    scale_super_weights,
    weight_outlier_stats,
)
from ..core.sw_directory import SPECS, ModelSpec, SuperWeight, get_spec
from ._common import SweepItem, sweep_models

import torch

NAME = "g1_activation_family"

PREDICTION = Prediction(
    hypothesis=(
        "Super weight localisation is strictly driven by the multiplicative "
        "gating mechanism (SwiGLU/GeGLU/ReGLU): EVERY gated model will "
        "localise at least one super weight with dominance >= 0.5, while NO "
        "non-gated model (GELU/ReLU/SiLU without a gate) will localise one, "
        "regardless of scale, training data, or architectural details like "
        "positional encoding or normalisation type."
    ),
    if_true=(
        "A clean partition: all gated models confirm a super weight (high "
        "dominance, catastrophic ablation, beats control), and all non-gated "
        "models do not (low dominance, distributed spike, no single-point "
        "catastrophe). This would establish gating as a necessary and "
        "sufficient architectural condition for super weight formation."
    ),
    if_false=(
        "Either some non-gated model at large scale does produce a localisable "
        "super weight (invalidating 'necessary'), or some gated model does NOT "
        "(invalidating 'sufficient'). Either result narrows the mechanism "
        "further: is it ReLU² that crosses the threshold? Does ALiBi suppress "
        "super weights even in gated models? The contingency table pinpoints "
        "exactly which cell breaks the partition."
    ),
    key_metric="gated_vs_nongated_partition_clean",
)

#: Same thresholds as t0.
MIN_KL_FOR_CATASTROPHE = 1.0
MIN_DAMAGE_RATIO_VS_CONTROL = 5.0


# =============================================================================
# Zoo tier definitions
# =============================================================================

#: Default zoo tiers.  Each is a list of model keys.
#: Quick-check mode uses a minimal subset of each tier.

ZOO_TIERS: dict[str, dict[str, list[str]]] = {
    "small": {
        "gated": [
            "smollm2-135m", "smollm2-360m", "qwen2.5-0.5b",
            "mobilellm-125m", "mobilellm-350m",
        ],
        "non-gated": [
            "pythia-160m", "pythia-410m",
            "gpt-neo-125m", "gpt2", "gpt2-medium",
            "opt-125m", "opt-350m",
            "cerebras-gpt-111m", "cerebras-gpt-256m",
        ],
    },
    "medium": {
        "gated": [
            "olmo-1b", "olmo-2-1b", "qwen2.5-1.5b", "qwen3-0.6b",
            "tinyllama-1.1b", "llama-3.2-1b",
        ],
        "non-gated": [
            "phi-1.5", "pythia-1b", "pythia-1.4b",
            "gpt-neo-1.3b", "gpt2-large", "gpt2-xl",
            "opt-1.3b",
            "bloom-560m", "bloom-1b1",
            "cerebras-gpt-1.3b",
        ],
    },
    "large": {
        "gated": [
            "smollm2-1.7b", "qwen2.5-3b", "llama-3.2-3b", "gemma2-2b",
        ],
        "non-gated": [
            "phi-2", "pythia-2.8b",
            "gpt-neo-2.7b", "opt-2.7b",
            "bloom-3b",
            "cerebras-gpt-2.7b",
        ],
    },
}

QUICK_CHECK_MODELS: list[str] = [
    # One small gated, one small non-gated, one medium of each.
    "smollm2-135m", "pythia-160m", "olmo-1b", "phi-1.5",
]


def _all_zoo_keys() -> list[str]:
    """Flatten the zoo tiers into a deduplicated ordered list."""
    seen: set[str] = set()
    keys: list[str] = []
    for tier in ZOO_TIERS.values():
        for kind in ("gated", "non-gated"):
            for key in tier.get(kind, []):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def _zoo_keys_for_args(args: ExpArgs) -> list[str]:
    """Determine which models to sweep based on args."""
    if args.extra.get("quick_check"):
        return QUICK_CHECK_MODELS
    if args.scale_ladder:
        return list(args.scale_ladder)
    return _all_zoo_keys()


# =============================================================================
# Entry point
# =============================================================================


def run(args: ExpArgs) -> Verdict:
    with Run(NAME, "multi-family", args.as_config(), runs_dir=args.runs_dir) as run_:
        run_.announce_prediction(PREDICTION)
        run_.attach("swx.core.detection")

        keys = _zoo_keys_for_args(args)
        run_.section(
            "1. Zoo",
            {
                "models": keys,
                "n_models": len(keys),
                "quick_check": bool(args.extra.get("quick_check")),
                "note": (
                    "Running detection, causal confirmation, and control ablation "
                    "on every model in the zoo, grouped by MLP form (gated vs "
                    "non-gated) and scale tier."
                ),
            },
        )

        n_probe = min(64, args.n_prompts)
        prompts = load_prompts(args.prompt_source, n=n_probe, seed=args.seed)
        run_.log.info("Probe prompts: %s (%d)", prompts.name, len(prompts))

        def body(loaded: LoadedModel, item: SweepItem) -> None:
            _check_one(loaded, item, args, prompts.prompts, run_)

        items = sweep_models(args, keys, body, run_)

        run_.results["per_model"] = [item.as_dict() for item in items]
        _log_table(run_, items)
        _log_tier_summary(run_, items)
        _log_contingency(run_, items)

        confirmed = [i for i in items if i.ok and i.values.get("confirmed_super_weight")]
        detected_only = [
            i for i in items
            if i.ok and i.values.get("detected") and not i.values.get("confirmed_super_weight")
        ]

        summary = {
            "n_models_swept": len(items),
            "n_loaded_ok": sum(1 for i in items if i.ok),
            "n_detected": sum(1 for i in items if i.ok and i.values.get("detected")),
            "n_confirmed": len(confirmed),
            "confirmed_models": [i.key for i in confirmed],
            "detected_but_not_confirmed": [i.key for i in detected_only],
        }
        run_.section("2. Outcome", summary)

        verdict = _judge(items, confirmed, detected_only)
        run_.finish(verdict)
        return verdict


# =============================================================================
# One model -- same pipeline as t0._check_one
# =============================================================================


def _check_one(
    loaded: LoadedModel, item: SweepItem, args: ExpArgs, prompts: list[str], run_
) -> None:
    """Detect, confirm causally, and control -- for one model."""
    result = detect(
        loaded,
        prompt=args.extra.get("detection_prompt") or DEFAULT_DETECTION_PROMPT,
        min_spike_ratio=args.extra.get("min_spike_ratio", 3.0),
        min_weight_ratio=args.extra.get("min_weight_ratio", 20.0),
        min_dominance=args.extra.get("min_dominance", 0.5),
    )

    item.values["n_layers"] = loaded.n_layers
    item.values["d_model"] = loaded.spec.d_model
    item.values["d_ff"] = loaded.spec.d_ff
    item.values["mlp_kind"] = loaded.spec.mlp_kind
    item.values["detected"] = bool(result.accepted)
    item.values["n_detected"] = len(result.accepted)
    item.values["detection_stopped_because"] = result.stopped_because

    item.values["rejected"] = [
        {
            "coordinate": str(c.super_weight),
            "dominance": c.dominance,
            "ratio_to_mean": c.weight_stats.ratio_to_mean,
            "output_spike_ratio": c.output_spike_ratio,
            "why": c.rejected_because,
        }
        for c in result.rejected
    ]

    if not result.accepted:
        best_rejected = max(result.rejected, key=lambda c: c.dominance, default=None)
        item.values["confirmed_super_weight"] = False
        item.values["diagnosis"] = (
            (
                f"No coordinate passed. Best near-miss {best_rejected.super_weight} "
                f"explains only {best_rejected.dominance:.0%} of its layer's output "
                f"spike -- the spike is real but it is a SUM OF MANY TERMS, so there "
                "is no single weight to call the cause."
            )
            if best_rejected
            else "No layer even had an aligned input+output activation spike."
        )
        run_.log.info("    no super weight: %s", item.values["diagnosis"])
        return

    best = result.accepted[0]
    sw = best.super_weight
    item.values["coordinate"] = str(sw)
    item.values["layer"] = sw.layer
    item.values["layer_fraction"] = sw.layer / max(loaded.n_layers - 1, 1)
    item.values["residual_channel"] = sw.residual_channel
    item.values["mlp_channel"] = sw.mlp_channel
    item.values["ratio_to_mean"] = best.weight_stats.ratio_to_mean
    item.values["rank_in_tensor"] = best.weight_stats.rank
    item.values["dominance"] = best.dominance
    item.values["output_spike_ratio"] = best.output_spike_ratio

    # -- causal confirmation ---------------------------------------------------
    intact = _logits(loaded, prompts, args, scale=None)
    ablated = _logits(loaded, prompts, args, scale=0.0, super_weights=[sw])
    damage = _damage(intact, ablated)
    item.values.update({f"sw_{k}": v for k, v in damage.items()})

    # -- control: a magnitude-matched ordinary weight ---------------------------
    control_sw = _matched_control(loaded, sw, seed=args.seed)
    control_damage: dict[str, float] = {}
    if control_sw is not None:
        control_logits = _logits(loaded, prompts, args, scale=0.0, super_weights=[control_sw])
        control_damage = _damage(intact, control_logits)
        item.values["control_coordinate"] = str(control_sw)
        item.values["control_ratio_to_mean"] = weight_outlier_stats(
            loaded.down_proj(control_sw.layer), control_sw
        ).ratio_to_mean
        item.values.update({f"control_{k}": v for k, v in control_damage.items()})

    control_kl = control_damage.get("kl_from_intact", 0.0)
    ratio = damage["kl_from_intact"] / control_kl if control_kl > 1e-9 else float("inf")
    item.values["damage_ratio_vs_control"] = ratio

    catastrophic = damage["kl_from_intact"] >= MIN_KL_FOR_CATASTROPHE
    beats_control = ratio >= MIN_DAMAGE_RATIO_VS_CONTROL
    item.values["confirmed_super_weight"] = bool(catastrophic and beats_control)

    run_.log.info(
        "    %s  %.0fx mean|W|  dominance %.2f  |  ablation: KL %.3f nats, "
        "top1 %.3f -> %.3f, H %.3f -> %.3f nats",
        sw, best.weight_stats.ratio_to_mean, best.dominance,
        damage["kl_from_intact"], damage["top1_before"], damage["top1_after"],
        damage["entropy_before"], damage["entropy_after"],
    )
    if control_sw is not None:
        run_.log.info(
            "    control %s (%.0fx mean|W|): KL %.3f nats  ->  super weight does "
            "%.1fx more damage",
            control_sw, item.values["control_ratio_to_mean"], control_kl, ratio,
        )
    if not catastrophic:
        run_.log.warning(
            "    Detected but NOT causally confirmed: ablation moved the output "
            "by only %.3f nats of KL (need %.1f). A coordinate with a large "
            "magnitude that does not matter is not a super weight.",
            damage["kl_from_intact"], MIN_KL_FOR_CATASTROPHE,
        )
    elif not beats_control:
        run_.log.warning(
            "    Detected and damaging, but a magnitude-matched ORDINARY weight "
            "does comparable damage (%.1fx, need %.1fx). At this scale 'ablating "
            "a big weight hurts' may be all that is happening.",
            ratio, MIN_DAMAGE_RATIO_VS_CONTROL,
        )


@torch.no_grad()
def _logits(
    loaded: LoadedModel,
    prompts: list[str],
    args: ExpArgs,
    scale: float | None,
    super_weights: list[SuperWeight] | None = None,
) -> torch.Tensor:
    """Next-token logits over the probe prompts, optionally with a weight zeroed."""
    from ..core.data import tokenize_batch

    def forward() -> torch.Tensor:
        batches = []
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start : start + args.batch_size]
            input_ids, mask = tokenize_batch(
                loaded.tokenizer, chunk, loaded.device, max_length=args.max_length
            )
            batches.append(final_logits(loaded, input_ids, mask).cpu())
        return torch.cat(batches, dim=0)

    if scale is None or not super_weights:
        return forward()
    with scale_super_weights(loaded, super_weights, scale):
        return forward()


def _damage(intact: torch.Tensor, ablated: torch.Tensor) -> dict[str, float]:
    """How much did the output distribution move? All arithmetic in float64."""
    p, q = to_probs(intact), to_probs(ablated)
    return {
        "kl_from_intact": float(kl_divergence(p, q).mean()),
        "entropy_before": float(entropy(p).mean()),
        "entropy_after": float(entropy(q).mean()),
        "entropy_rise_nats": float((entropy(q) - entropy(p)).mean()),
        "top1_before": float(top1_prob(p).mean()),
        "top1_after": float(top1_prob(q).mean()),
    }


def _matched_control(
    loaded: LoadedModel, sw: SuperWeight, seed: int
) -> SuperWeight | None:
    """Pick the runner-up weight from the same tensor as the strongest control."""
    module = loaded.down_proj(sw.layer)
    weight, is_transposed = down_proj_weight_view(module)
    magnitudes = weight.detach().float().abs()

    i, j = (sw.mlp_channel, sw.residual_channel) if is_transposed else (
        sw.residual_channel, sw.mlp_channel
    )
    if magnitudes.numel() < 2:
        return None

    masked = magnitudes.clone()
    masked[i, j] = -1.0
    flat = int(masked.argmax())
    row, col = divmod(flat, masked.shape[1])
    return (
        SuperWeight(layer=sw.layer, residual_channel=col, mlp_channel=row)
        if is_transposed
        else SuperWeight(layer=sw.layer, residual_channel=row, mlp_channel=col)
    )


# =============================================================================
# Reporting
# =============================================================================


def _log_table(run_, items: list[SweepItem]) -> None:
    run_.log.info("")
    run_.log.info("### Multi-family zoo results")
    run_.log.info(
        "    %-22s %8s %-10s %-20s %9s %10s %10s %10s %10s",
        "model", "layers", "mlp", "coordinate", "x mean|W|", "dominance", "KL nats",
        "vs control", "confirmed",
    )
    for item in items:
        if not item.ok:
            run_.log.info("    %-22s %s", item.key, f"LOAD FAILED: {item.error}")
            continue
        v = item.values
        if not v.get("detected"):
            run_.log.info(
                "    %-22s %8d %-10s %-20s %9s %10s %10s %10s %10s",
                item.key, v.get("n_layers", 0), v.get("mlp_kind", "?"),
                "(none found)", "-", "-", "-", "-", "no",
            )
            continue
        run_.log.info(
            "    %-22s %8d %-10s %-20s %9.1f %10.2f %10.3f %10s %10s",
            item.key, v["n_layers"], v.get("mlp_kind", "?"), v["coordinate"],
            v["ratio_to_mean"], v["dominance"], v.get("sw_kl_from_intact", float("nan")),
            f"{v.get('damage_ratio_vs_control', float('nan')):.1f}x",
            "YES" if v.get("confirmed_super_weight") else "no",
        )


def _by_arch(items: list[SweepItem]) -> dict[str, dict[str, list[str]]]:
    """Group the loaded models by MLP form into confirmed / not-confirmed."""
    kinds: dict[str, dict[str, list[str]]] = {}
    for i in items:
        if not i.ok:
            continue
        bucket = kinds.setdefault(
            i.values.get("mlp_kind", "?"), {"confirmed": [], "not_confirmed": []}
        )
        key = "confirmed" if i.values.get("confirmed_super_weight") else "not_confirmed"
        bucket[key].append(i.key)
    return kinds


def _tier_of(key: str) -> str:
    """Which scale tier a model key belongs to."""
    for tier_name, tier_groups in ZOO_TIERS.items():
        for kind_models in tier_groups.values():
            if key in kind_models:
                return tier_name
    return "unknown"


def _log_tier_summary(run_, items: list[SweepItem]) -> None:
    """Print a per-tier breakdown of confirmation rates."""
    run_.log.info("")
    run_.log.info("### Per-tier confirmation rates")
    run_.log.info("    %-10s %-10s %10s %10s %10s", "tier", "mlp_kind", "swept", "confirmed", "rate")

    for tier_name in ("small", "medium", "large"):
        for kind in ("gated", "non-gated"):
            tier_items = [
                i for i in items if i.ok
                and i.values.get("mlp_kind") == kind
                and _tier_of(i.key) == tier_name
            ]
            n_confirmed = sum(1 for i in tier_items if i.values.get("confirmed_super_weight"))
            rate = f"{n_confirmed}/{len(tier_items)}" if tier_items else "-"
            run_.log.info(
                "    %-10s %-10s %10d %10d %10s",
                tier_name, kind, len(tier_items), n_confirmed, rate,
            )


def _log_contingency(run_, items: list[SweepItem]) -> None:
    """Print the 2x2 contingency table (gated/non-gated x confirmed/not)."""
    kinds = _by_arch(items)
    if len(kinds) < 2:
        return
    run_.log.info("")
    run_.log.info("### 2x2 Contingency: MLP form x super weight confirmation")
    run_.log.info("    %-15s %15s %15s", "", "CONFIRMED", "NOT CONFIRMED")
    for kind in sorted(kinds):
        b = kinds[kind]
        run_.log.info(
            "    %-15s %15d %15d",
            kind, len(b["confirmed"]), len(b["not_confirmed"]),
        )
    # The chi-squared test, if counts are large enough.
    gated = kinds.get("gated", {"confirmed": [], "not_confirmed": []})
    nongated = kinds.get("non-gated", {"confirmed": [], "not_confirmed": []})
    a, b = len(gated["confirmed"]), len(gated["not_confirmed"])
    c, d = len(nongated["confirmed"]), len(nongated["not_confirmed"])
    n = a + b + c + d
    if n > 0 and (a + c) > 0 and (b + d) > 0 and (a + b) > 0 and (c + d) > 0:
        # Fisher's exact test would be better for small n, but scipy may not
        # be installed. Report the raw counts and let the analyst decide.
        run_.log.info(
            "    Counts: gated[%d/%d], non-gated[%d/%d] (total %d models)",
            a, a + b, c, c + d, n,
        )


def _judge(
    items: list[SweepItem], confirmed: list[SweepItem], detected_only: list[SweepItem]
) -> Verdict:
    loaded_ok = [i for i in items if i.ok]
    kinds = _by_arch(items)
    gated = kinds.get("gated", {"confirmed": [], "not_confirmed": []})
    nongated = kinds.get("non-gated", {"confirmed": [], "not_confirmed": []})
    both_archs = bool(gated["confirmed"] or gated["not_confirmed"]) and bool(
        nongated["confirmed"] or nongated["not_confirmed"]
    )
    evidence = {
        "models_swept": [i.key for i in items],
        "confirmed": [i.key for i in confirmed],
        "detected_but_not_confirmed": [i.key for i in detected_only],
        "gated_confirmed": gated["confirmed"],
        "gated_not_confirmed": gated["not_confirmed"],
        "non_gated_confirmed": nongated["confirmed"],
        "non_gated_not_confirmed": nongated["not_confirmed"],
        "n_gated_total": len(gated["confirmed"]) + len(gated["not_confirmed"]),
        "n_nongated_total": len(nongated["confirmed"]) + len(nongated["not_confirmed"]),
        "partition_clean": (
            bool(both_archs)
            and bool(gated["confirmed"])
            and not nongated["confirmed"]
            and not gated["not_confirmed"]
        ),
    }

    if not loaded_ok:
        return Verdict(
            grade="INCONCLUSIVE",
            summary=(
                "No model in the zoo loaded. Check network access and the "
                "per-model errors in results.json."
            ),
            evidence=evidence,
        )

    # -- perfect partition: every gated confirms, no non-gated does -----------
    if (
        both_archs
        and gated["confirmed"]
        and not gated["not_confirmed"]
        and not nongated["confirmed"]
    ):
        return Verdict(
            grade="SUPPORTS_HYPOTHESIS",
            summary=(
                "CLEAN PARTITION across the full zoo: every gated model confirms "
                f"a super weight ({', '.join(gated['confirmed'])}), and NO non-gated "
                f"model does ({', '.join(nongated['not_confirmed'])}). Super weight "
                "localisation is driven by the multiplicative gating mechanism, "
                "not by scale, training data, position embeddings, or "
                "normalisation type."
            ),
            evidence=evidence,
        )

    # -- gated all confirm but some non-gated do too --------------------------
    if both_archs and gated["confirmed"] and not gated["not_confirmed"] and nongated["confirmed"]:
        return Verdict(
            grade="PARTIAL",
            summary=(
                "All gated models confirm super weights, but so do some non-gated "
                f"models ({', '.join(nongated['confirmed'])}). Gating is SUFFICIENT "
                "but not NECESSARY. Investigate which non-gated models break the "
                "partition and whether they share a specific property "
                "(e.g. quadratic activation, very large scale)."
            ),
            evidence=evidence,
        )

    # -- gated confirm, non-gated do not, but some gated ALSO do not ----------
    if both_archs and gated["confirmed"] and gated["not_confirmed"] and not nongated["confirmed"]:
        return Verdict(
            grade="PARTIAL",
            summary=(
                "No non-gated model confirms a super weight, but some gated models "
                f"also fail to confirm ({', '.join(gated['not_confirmed'])}). Gating "
                "is NECESSARY but not SUFFICIENT -- some additional condition "
                "(scale, training duration, data) is required within gated models."
            ),
            evidence=evidence,
        )

    # -- no confirmation at all -----------------------------------------------
    if not confirmed:
        return Verdict(
            grade="CONTRADICTS_HYPOTHESIS",
            summary=(
                "NO model in the zoo confirmed a super weight. This is likely a "
                "threshold or methodology problem rather than a genuine negative -- "
                "recheck detection thresholds and the probe prompt."
            ),
            evidence=evidence,
        )

    # -- fallback: mixed results with no clean pattern ------------------------
    return Verdict(
        grade="PARTIAL",
        summary=(
            f"Mixed results: {len(confirmed)} models confirmed super weights "
            f"out of {len(loaded_ok)} swept. Gated confirmed: "
            f"{', '.join(gated['confirmed']) or '(none)'}. Non-gated confirmed: "
            f"{', '.join(nongated['confirmed']) or '(none)'}. See the per-tier "
            "breakdown for detail."
        ),
        evidence=evidence,
    )
