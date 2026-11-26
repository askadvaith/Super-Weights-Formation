"""
Shared model-sweep and checkpoint-walking helpers.

Provides an iterator that loads, yields, and frees one model at a time, plus
helpers that walk checkpoint revisions while recording partial failures.

THE PARTIAL-FAILURE RULE
------------------------------------------------------------------------------
A multi-model or multi-checkpoint sweep records per-item failures and CONTINUES,
reporting how many items succeeded alongside its result.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import torch

from ..core.args import ExpArgs
from ..core.checkpoints import (
    CheckpointRecord,
    CoordinateStats,
    coordinate_stats,
    estimate_download_gb,
    read_down_proj,
    resolve_steps,
    revision_for_step,
)
from ..core.detection import DEFAULT_DETECTION_PROMPT, DetectionResult, detect
from ..core.modeling import LoadedModel, free, load_model
from ..core.sw_directory import (
    DEFAULT_FORMATION_MODEL,
    SCALE_LADDER,
    SEED_FAMILIES,
    SEED_FAMILY_MEANING,
    ModelSpec,
    SuperWeight,
    family_of,
    get_spec,
)

log = logging.getLogger(__name__)


# =============================================================================
# Sweeping over models
# =============================================================================


@dataclass
class SweepItem:
    """One model in a sweep, and whatever we learned about it."""

    key: str
    """Model key, e.g. `pythia-160m-data-seed2`."""
    spec: ModelSpec
    label: str
    """How to print it in a table -- key, or `key@step3000` in a checkpoint sweep."""
    revision: str | None = None
    family: str | None = None
    """Seed family, for s1/s2. None for a base model or a checkpoint sweep."""

    ok: bool = False
    error: str = ""
    values: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.key,
            "hf_id": self.spec.hf_id,
            "label": self.label,
            "ok": self.ok,
        }
        if self.revision:
            payload["revision"] = self.revision
        if self.family:
            payload["seed_family"] = self.family
        if self.error:
            payload["error"] = self.error
        payload.update(self.values)
        return payload


@contextmanager
def loaded_model(args: ExpArgs, key: str, revision: str | None = None) -> Iterator[LoadedModel]:
    """
    Load one model for the duration of a `with` block, then free it.

    The `finally` is the point. A sweep over 13 seed variants or 40 checkpoints
    leaks a model per iteration without it, and on an 8 GB laptop that means an
    OOM somewhere in the middle with no partial results.
    """
    spec = get_spec(key)
    handle: LoadedModel | None = None
    try:
        handle = load_model(
            spec,
            device=args.device,
            dtype=args.dtype,
            allow_fp16=args.allow_fp16,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            revision=revision,
        )
        yield handle
    finally:
        free(handle)


def sweep_models(
    args: ExpArgs,
    keys: Sequence[str],
    body: Callable[[LoadedModel, SweepItem], None],
    run,
    revisions: Sequence[str | None] | None = None,
) -> list[SweepItem]:
    """
    Run `body` against each model in turn, tolerating individual failures.

    `body(loaded, item)` writes its results into `item.values`; anything it
    raises is caught, recorded on the item and logged, and the sweep moves on.

    Args:
        keys: model keys.
        revisions: one revision per key, or None for all-default. Same length
            as `keys` when given.
    """
    revs: Sequence[str | None] = revisions if revisions is not None else [None] * len(keys)
    if len(revs) != len(keys):
        raise ValueError("`revisions` must be the same length as `keys`.")

    items: list[SweepItem] = []
    for index, (key, revision) in enumerate(zip(keys, revs), start=1):
        spec = get_spec(key)
        label = f"{key}@{revision}" if revision else key
        item = SweepItem(
            key=key, spec=spec, label=label, revision=revision, family=family_of(key)
        )
        run.log.info("")
        run.log.info("[%d/%d] %s  (%s)", index, len(keys), label, spec.hf_id)
        try:
            with loaded_model(args, key, revision) as loaded:
                body(loaded, item)
            item.ok = True
        except Exception as exc:  # noqa: BLE001 -- one bad model must not end the sweep
            item.error = f"{type(exc).__name__}: {exc}"
            run.log.warning("    FAILED: %s", item.error)
        items.append(item)

    ok = sum(1 for i in items if i.ok)
    run.log.info("")
    run.log.info("Sweep complete: %d/%d models succeeded.", ok, len(items))
    if ok < len(items):
        run.log.warning(
            "%d model(s) failed. Every number below is computed from the %d that "
            "succeeded -- check the per-model errors before reading the verdict.",
            len(items) - ok, ok,
        )
    return items


# =============================================================================
# Sweeping over checkpoints (weights only, no model)
# =============================================================================


def sweep_checkpoint_weights(
    args: ExpArgs,
    key: str,
    sw: SuperWeight,
    run,
    steps: Sequence[int] | None = None,
) -> list[CheckpointRecord]:
    """
    Read one coordinate's weight statistics at every requested checkpoint.

    This never constructs a model. It pulls a single [d_model, d_ff] tensor per
    checkpoint -- by HTTP range request when possible -- and computes the
    coordinate's magnitude, its rank within the tensor, and its rank within its
    own row and column.

    The row rank is the interesting one. The detection identity
    `Y_ij = sum_k X_ik W_jk` sums over exactly the row `sw.residual_channel`,
    so a coordinate that dominates its row is one that CAN dominate the sum.
    That is the structural precondition for being a super weight, and it is
    visible from the weights alone with no forward pass -- which is what makes
    a 154-checkpoint sweep affordable.
    """
    spec = get_spec(key)
    chosen = tuple(steps) if steps is not None else resolve_steps(
        args.n_checkpoints, args.checkpoint_steps
    )

    estimated = estimate_download_gb(spec, len(chosen), args.remote_slice)
    run.log.info(
        "Checkpoint sweep: %d steps of %s, estimated download %.2f GB (%s)",
        len(chosen), key, estimated,
        "remote slicing" if args.remote_slice else "FULL DOWNLOADS",
    )
    if estimated > args.max_download_gb:
        raise RuntimeError(
            f"This sweep would download about {estimated:.1f} GB, over the "
            f"--max-download-gb limit of {args.max_download_gb:.1f} GB. Either "
            f"raise the limit, lower --n-checkpoints (currently {len(chosen)}), "
            + (
                "or leave --remote-slice on -- turning it off is what makes this "
                "expensive."
                if not args.remote_slice
                else "or pick a smaller model."
            )
        )
    if not args.remote_slice:
        run.log.warning(
            "--no-remote-slice: downloading whole checkpoints. That is ~%.0fx "
            "more data than reading just the down_proj tensor, and the results "
            "are identical.",
            max(estimated / max(estimate_download_gb(spec, len(chosen), True), 1e-9), 1.0),
        )

    records: list[CheckpointRecord] = []
    transferred = 0.0
    for index, step in enumerate(chosen, start=1):
        revision = revision_for_step(step)
        record = CheckpointRecord(step=step, revision=revision)
        try:
            read = read_down_proj(
                spec.hf_id, sw.layer, revision=revision, remote_slice=args.remote_slice
            )
            record.method = read.method
            record.megabytes = read.megabytes
            transferred += read.megabytes
            record.stats = coordinate_stats(
                read.tensor, sw.residual_channel, sw.mlp_channel
            )
            del read
        except Exception as exc:  # noqa: BLE001 -- one bad revision must not end the sweep
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            run.log.warning("  step %-7d FAILED: %s", step, record.error)
        records.append(record)

        if record.ok and record.stats is not None:
            run.log.debug(
                "  [%3d/%3d] step %-7d |W|=%.6f  %7.1fx mean  rank %d/%d  "
                "row-rank %d/%d",
                index, len(chosen), step, record.stats.abs_value,
                record.stats.ratio_to_mean, record.stats.rank_in_tensor,
                record.stats.n_elements, record.stats.rank_in_row,
                record.stats.row_size,
            )

    ok = sum(1 for r in records if r.ok)
    run.log.info(
        "Read %d/%d checkpoints, %.1f MB transferred (%s).",
        ok, len(records), transferred,
        "remote slicing" if args.remote_slice else "full downloads",
    )
    return records


# =============================================================================
# Resolving which coordinate to track
# =============================================================================


@dataclass
class ResolvedCoordinate:
    """A super-weight coordinate plus the evidence that it is one."""

    super_weight: SuperWeight
    provenance: str
    detection: DetectionResult | None = None
    ratio_to_mean: float = float("nan")
    dominance: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate": str(self.super_weight),
            "layer": self.super_weight.layer,
            "residual_channel": self.super_weight.residual_channel,
            "mlp_channel": self.super_weight.mlp_channel,
            "provenance": self.provenance,
            "ratio_to_mean": self.ratio_to_mean,
            "dominance": self.dominance,
        }


def resolve_coordinate(args: ExpArgs, key: str, run, revision: str | None = None) -> ResolvedCoordinate:
    """
    Decide which coordinate a §4.6 experiment should track.

    Precedence: `--super-weight` override, then live detection. Unlike §4.4/§4.5
    there is no directory step, because no Pythia model has published
    coordinates -- that is the whole reason `t0` exists.

    Detection needs a real forward pass, so this is the one part of a
    weights-only experiment that loads a model. It happens exactly once, at the
    FINAL checkpoint, and the coordinate it finds is then tracked backwards
    through training by `sweep_checkpoint_weights` with no further model loads.
    """
    if args.super_weights:
        sw = args.super_weights[0]
        run.log.info("Tracking the coordinate given on the command line: %s", sw)
        if len(args.super_weights) > 1:
            run.log.warning(
                "%d coordinates given; §4.6 tracks one trajectory at a time and "
                "will use the first. Re-run with each in turn to compare.",
                len(args.super_weights),
            )
        return ResolvedCoordinate(sw, "override")

    run.log.info(
        "Detecting the super weight in %s%s (one forward pass) ...",
        key, f"@{revision}" if revision else "",
    )
    with loaded_model(args, key, revision) as loaded:
        result = detect(
            loaded,
            prompt=args.extra.get("detection_prompt") or DEFAULT_DETECTION_PROMPT,
            min_spike_ratio=args.extra.get("min_spike_ratio", 3.0),
            min_weight_ratio=args.extra.get("min_weight_ratio", 20.0),
            min_dominance=args.extra.get("min_dominance", 0.5),
        )

    if not result.accepted:
        raise RuntimeError(
            f"No super weight found in {key}: this scale has no localisable "
            "super weight under current detection thresholds."
        )

    best = result.accepted[0]
    run.log.info(
        "Found %s -- %.0fx mean |W|, rank %d/%d, dominance %.2f",
        best.super_weight, best.weight_stats.ratio_to_mean, best.weight_stats.rank,
        best.weight_stats.n_elements, best.dominance,
    )
    return ResolvedCoordinate(
        super_weight=best.super_weight,
        provenance=f"detected@{revision or 'final'}(prompt={result.prompt!r})",
        detection=result,
        ratio_to_mean=best.weight_stats.ratio_to_mean,
        dominance=best.dominance,
    )


# =============================================================================
# Which models to sweep
# =============================================================================


def seed_variant_keys(
    families: Sequence[str], base_size: str = "160m", include_base: bool = True
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Model keys for the requested seed families.

    Returns (all_keys, {family: keys_in_that_family}). The base model is
    included in `all_keys` but in no family, because it is the reference point
    rather than a member of any comparison group.
    """
    base = f"pythia-{base_size}"
    keys: list[str] = [base] if include_base else []
    by_family: dict[str, list[str]] = {}

    for family in families:
        if family not in SEED_FAMILIES:
            raise KeyError(
                f"Unknown seed family {family!r}. Available: "
                f"{', '.join(sorted(SEED_FAMILIES))}."
            )
        members = [f"{base}-{suffix}" for suffix in SEED_FAMILIES[family]]
        by_family[family] = members
        keys.extend(members)
    return keys, by_family


def scale_ladder_keys(args: ExpArgs) -> list[str]:
    """Models for the t0 de-risking sweep."""
    return list(args.scale_ladder) if args.scale_ladder else list(SCALE_LADDER)


def formation_model(args: ExpArgs) -> str:
    """
    The model a §4.6 experiment should use.

    Defaults to pythia-160m rather than to `args.model`'s own default of
    olmo-1b, because OLMo publishes no intermediate checkpoints and no seed
    variants -- running §4.6 on it is not a smaller version of the experiment,
    it is impossible. If the user explicitly named a non-Pythia model we honour
    it and warn, since the checkpoint machinery will then fail informatively
    rather than silently produce a one-point "trajectory".
    """
    if args.model.startswith("pythia"):
        return args.model
    if args.model != "olmo-1b":  # explicitly chosen, just not a Pythia
        log.warning(
            "%s has no published training checkpoints or seed variants, so the "
            "§4.6 experiments cannot sweep it. Using it anyway because you asked; "
            "expect the checkpoint reads to fail.",
            args.model,
        )
        return args.model
    return DEFAULT_FORMATION_MODEL


# =============================================================================
# Small shared statistics
# =============================================================================
# Deliberately kept here rather than in `core.metrics`: metrics.py is about
# probability distributions over a vocabulary, and these are about agreement
# and dispersion across a sweep. Mixing them would make both harder to read.


def coefficient_of_variation(values: Sequence[float]) -> float:
    """
    std / |mean|, the scale-free measure of spread that s2 keys off.

    Scale-free is what matters: the point of s2 is to compare the spread of
    *entropy* against the spread of *weight magnitude* against the spread of
    *KL*, which have no common unit. A raw standard deviation cannot do that.
    """
    clean = [v for v in values if v == v]
    if len(clean) < 2:
        return float("nan")
    tensor = torch.tensor(clean, dtype=torch.float64)
    mean = float(tensor.mean())
    if mean == 0:
        return float("inf")
    return float(tensor.std(unbiased=True)) / abs(mean)


def agreement_fraction(values: Sequence[Any]) -> float:
    """
    Fraction of unordered pairs that are equal. 1.0 = everything agrees.

    Used instead of "are they all the same?" because with three seeds the
    interesting middle case -- two agree, one differs -- is exactly what a
    boolean throws away.
    """
    items = list(values)
    if len(items) < 2:
        return float("nan")
    pairs = agree = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs += 1
            agree += int(items[i] == items[j])
    return agree / pairs


def chance_agreement(n_options: int) -> float:
    """
    Pairwise agreement expected if the choice were uniformly random.

    Reported next to every observed agreement number. Without it "2 of 3 seeds
    agree" is unreadable: for a residual channel out of d_model = 768 the
    chance level is 0.13%, so even weak agreement is overwhelming evidence,
    while for a layer index out of 12 it is 8.3% and the same raw number means
    much less.
    """
    return 1.0 / n_options if n_options > 0 else float("nan")


def describe_agreement(observed: float, n_options: int, what: str) -> str:
    """One readable line comparing observed agreement against chance."""
    chance = chance_agreement(n_options)
    if observed != observed:
        return f"{what}: too few models to compare."
    if observed >= 0.999:
        return (
            f"{what}: IDENTICAL across every model "
            f"(chance level {chance:.2%} across {n_options} options)."
        )
    if observed <= 1e-9:
        return (
            f"{what}: no two models agree "
            f"(chance level {chance:.2%} across {n_options} options)."
        )
    return (
        f"{what}: {observed:.1%} of pairs agree "
        f"(chance level {chance:.2%} across {n_options} options)."
    )


def summarise_stats_series(records: Sequence[CheckpointRecord]) -> dict[str, list[float]]:
    """Pull the successful records into parallel series, ready for plotting or fitting."""
    ok = [r for r in records if r.ok and r.stats is not None]
    return {
        "step": [float(r.step) for r in ok],
        "tokens_seen": [float(r.tokens_seen) for r in ok],
        "value": [r.stats.value for r in ok],                 # type: ignore[union-attr]
        "abs_value": [r.stats.abs_value for r in ok],         # type: ignore[union-attr]
        "ratio_to_mean": [r.stats.ratio_to_mean for r in ok], # type: ignore[union-attr]
        "percentile": [r.stats.percentile for r in ok],       # type: ignore[union-attr]
        "rank_in_tensor": [float(r.stats.rank_in_tensor) for r in ok],  # type: ignore[union-attr]
        "rank_in_row": [float(r.stats.rank_in_row) for r in ok],        # type: ignore[union-attr]
        "share_of_row": [r.stats.share_of_row for r in ok],   # type: ignore[union-attr]
        "mean_abs": [r.stats.mean_abs for r in ok],           # type: ignore[union-attr]
    }


def log_trajectory_table(run, records: Sequence[CheckpointRecord]) -> None:
    """Print the per-checkpoint weight trajectory table."""
    run.log.info("")
    run.log.info("### Weight trajectory (one row per checkpoint)")
    run.log.info(
        "    %-8s %-13s %12s %10s %11s %16s %12s %10s",
        "step", "tokens", "W", "|W|", "x mean|W|", "rank in tensor", "rank in row",
        "%ile",
    )
    for record in records:
        if not record.ok or record.stats is None:
            run.log.info("    %-8d %-13s %s", record.step, "-", f"FAILED: {record.error}")
            continue
        s = record.stats
        run.log.info(
            "    %-8d %-13s %+12.6f %10.6f %11.1f %16s %12s %10.4f",
            record.step, f"{record.tokens_seen / 1e9:.2f}B", s.value, s.abs_value,
            s.ratio_to_mean, f"{s.rank_in_tensor}/{s.n_elements}",
            f"{s.rank_in_row}/{s.row_size}", s.percentile,
        )
