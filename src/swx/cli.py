"""
Command-line interface for the paper experiments and replication package.

Supported workflows:
    # 1. Detection algorithm reproduction & verification:
    uv run swx detect --model olmo-1b
    uv run swx verify-directory --model olmo-1b
    uv run swx chat --model olmo-1b --compare

    # 2. Gated vs non-gated model zoo sweep:
    uv run swx g1 --quick-check
    uv run swx g1

    # 3. Controlled training twins (6 activation functions & causal interventions):
    uv run swx train-twins --variant swiglu --max-steps 100 --log-every 10
    uv run swx train-twins --variant swiglu --condition-manifest conditions/A1_hard_clamp.json
    uv run swx plot-trajectories --run-dir runs/training_twins
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .core.args import ExpArgs
from .registry import (
    DIRECTION_G,
    EXPERIMENTS,
    INFO,
)
from .core.sw_directory import SPECS, SuperWeight, get_spec, residual_channels


# =============================================================================
# Argument parsing
# =============================================================================


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options shared across experiment and detection commands."""
    model = parser.add_argument_group("model")
    model.add_argument(
        "--model", default="olmo-1b",
        help="Model key from `swx list-models` (default: olmo-1b).",
    )
    model.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    model.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"],
        help="bfloat16 by default. Quantised dtypes are deliberately unsupported.",
    )
    model.add_argument(
        "--allow-fp16", action="store_true",
        help="Permit float16 for older hardware with no bf16.",
    )
    model.add_argument(
        "--device-map", default=None,
        help="Pass 'auto' for CPU offload when a model does not fit in VRAM.",
    )
    model.add_argument("--trust-remote-code", action="store_true")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--prompt-source", default="lambada", choices=["lambada", "wikitext", "builtin"],
        help="lambada matches the anchor paper's setup (500 prompts from validation split).",
    )
    data.add_argument("--n-prompts", type=int, default=500)
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--max-length", type=int, default=256)
    data.add_argument("--seed", type=int, default=0)

    sw = parser.add_argument_group("super weights")
    sw.add_argument(
        "--super-weight", action="append", default=None, metavar="LAYER,ROW,COL",
        help="Override the coordinate. Repeatable. ROW is the residual channel "
             "(index into d_model), COL the MLP channel (index into d_ff).",
    )
    sw.add_argument(
        "--detect", action="store_true",
        help="Locate super weights live using the activation-spike method.",
    )

    fm = parser.add_argument_group("g1 sweep options")
    fm.add_argument(
        "--quick-check", action="store_true",
        help="g1: use a minimal 4-model subset (1 gated + 1 non-gated at two scales).",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--runs-dir", default=None,
        help="Where to write run directories (default: ./runs).",
    )


def _add_detection_options(parser: argparse.ArgumentParser) -> None:
    """Thresholds shared by `detect`, `verify-directory`, and `g1`."""
    from .core.detection import (
        DEFAULT_MAX_SUPER_WEIGHTS,
        DEFAULT_MIN_DOMINANCE,
        DEFAULT_MIN_SPIKE_RATIO,
        DEFAULT_MIN_WEIGHT_RATIO,
    )

    det = parser.add_argument_group("detection")
    det.add_argument(
        "--max-super-weights", type=int, default=DEFAULT_MAX_SUPER_WEIGHTS,
        help="Cap on the iterative search (default: %(default)s).",
    )
    det.add_argument(
        "--detection-prompt", default=None,
        help="Single prompt used for the activation profile.",
    )
    det.add_argument(
        "--min-spike-ratio", type=float, default=DEFAULT_MIN_SPIKE_RATIO,
        help="A layer qualifies only if BOTH its down_proj input and output "
             "exceed this multiple of the median layer (default: %(default)s).",
    )
    det.add_argument(
        "--min-weight-ratio", type=float, default=DEFAULT_MIN_WEIGHT_RATIO,
        help="A candidate is accepted only if its weight is at least this many "
             "times the mean |W| of its own down_proj tensor (default: %(default)s).",
    )
    det.add_argument(
        "--min-dominance", type=float, default=DEFAULT_MIN_DOMINANCE,
        help="Fraction of the output spike that the single weight must explain (default: %(default)s).",
    )


def _to_exp_args(ns: argparse.Namespace) -> ExpArgs:
    """Convert a parsed namespace into the dataclass the experiments take."""
    super_weights = None
    if getattr(ns, "super_weight", None):
        parsed = []
        for raw in ns.super_weight:
            try:
                layer, row, col = (int(part) for part in raw.split(","))
            except ValueError as exc:
                raise SystemExit(
                    f"--super-weight expects LAYER,ROW,COL (got {raw!r})"
                ) from exc
            parsed.append(SuperWeight(layer=layer, residual_channel=row, mlp_channel=col))
        super_weights = tuple(parsed)

    extra: dict[str, object] = {}
    for flag in (
        "quick_check",
        "detection_prompt",
        "max_super_weights",
        "min_spike_ratio",
        "min_weight_ratio",
        "min_dominance",
    ):
        val = getattr(ns, flag, None)
        if val is not None:
            extra[flag] = val

    return ExpArgs(
        model=getattr(ns, "model", "olmo-1b"),
        device=getattr(ns, "device", "auto"),
        dtype=getattr(ns, "dtype", "bfloat16"),
        allow_fp16=getattr(ns, "allow_fp16", False),
        device_map=getattr(ns, "device_map", None),
        trust_remote_code=getattr(ns, "trust_remote_code", False),
        prompt_source=getattr(ns, "prompt_source", "lambada"),
        n_prompts=getattr(ns, "n_prompts", 500),
        batch_size=getattr(ns, "batch_size", 8),
        max_length=getattr(ns, "max_length", 256),
        seed=getattr(ns, "seed", 0),
        super_weights=super_weights,
        detect=getattr(ns, "detect", False),
        extra=extra,
        runs_dir=Path(ns.runs_dir) if getattr(ns, "runs_dir", None) else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swx",
        description="Super-weight experiments: Detection reproduction, Gated-vs-Non-Gated Model Sweep, and Training Twins.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for key, info in INFO.items():
        blurb = f"[{info.section}] {info.blurb}"
        p = sub.add_parser(
            key, help=blurb,
            description=f"{blurb}\n\nCost: {info.cost}\nModule: {info.module}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_common(p)
        _add_detection_options(p)

    for name, members, blurb in (
        ("direction-g", DIRECTION_G, "Gated vs non-gated cross-family zoo evaluation (g1)"),
    ):
        p = sub.add_parser(name, help=blurb, description=blurb)
        _add_common(p)
        _add_detection_options(p)
        p.add_argument(
            "--continue-on-error", action="store_true",
            help="Keep going if one experiment fails.",
        )

    p = sub.add_parser(
        "verify-directory",
        help="Re-derive super-weight coordinates and check them against the transcribed table",
    )
    _add_common(p)
    _add_detection_options(p)

    p = sub.add_parser("detect", help="Locate super weights using the paper's activation-spike method")
    _add_common(p)
    _add_detection_options(p)

    p = sub.add_parser(
        "chat",
        help="Interactive REPL: generate with and without the super weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(p)
    _add_detection_options(p)
    p.add_argument(
        "--select", default=None, metavar="WHICH",
        help="Which pool entries start selected.",
    )
    gen = p.add_argument_group("generation")
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--greedy", action="store_true")
    gen.add_argument("--ablated", action="store_true")
    gen.add_argument("--compare", action="store_true")
    gen.add_argument("--no-reseed", action="store_true")

    p = sub.add_parser("list-models", help="Show the model registry and published super weights")
    p = sub.add_parser("summarise", help="One table of every run's verdict")
    p.add_argument("--runs-dir", default=None)

    # -- controlled pretraining (train-twins) ---------------------------------
    p = sub.add_parser(
        "train-twins",
        help="Train 124M Transformers with 6 MLP activation variants & causal interventions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant", default=None,
        help="Which MLP variant to train: swiglu, geglu, reglu, gelu_std, silu_std, relu2_std.",
    )
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for model initialisation (default: 42).")
    p.add_argument("--data-seed", type=int, default=42,
                   help="Random seed for dataset shuffling (default: 42).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    p.add_argument("--output-dir", default="", help="Override output directory.")
    p.add_argument("--resume", default=None, help="Resume from a checkpoint directory.")
    p.add_argument(
        "--condition-manifest", default=None, metavar="PATH",
        help="JSON manifest defining one named causal condition.",
    )
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset or alias (e.g. fineweb-edu, c4, slimpajama).")
    p.add_argument("--dataset-config", default=None,
                   help="Dataset configuration/subset name (e.g. sample-10BT or en).")
    p.add_argument("--dataset-split", default="train",
                   help="Dataset split to stream (default: train).")
    p.add_argument("--dataset-text-column", default="text",
                   help="Column name containing text (default: text).")
    p.add_argument("--token-cache-dir", default="",
                   help="Directory for pre-tokenised data cache (default: runs/training_twins/data).")

    # -- trajectory plotting ---------------------------------------------------
    p = sub.add_parser(
        "plot-trajectories",
        help="Analyse and plot telemetry from train-twins runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir", required=True,
        help="Path to a single variant's run dir or a parent dir containing multiple.",
    )
    p.add_argument(
        "--baseline-dir", default=None,
        help="Optional path to baseline runs directory (e.g. runs/training_twins/fineweb-edu).",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Optional dataset display name override (e.g. 'FineWeb-Edu', 'C4', 'SlimPajama').",
    )

    # -- lag dynamics and composite grid plotting ------------------------------
    p = sub.add_parser(
        "plot-lag",
        help="Generate consolidated multi-seed lag dynamics and composite 6x3 grid plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--runs-dir", default=None,
        help="Base directory containing training_twins runs (default: runs/training_twins).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory for generated plots.",
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="Resolution for generated figures (default: 200).",
    )

    return parser


# =============================================================================
# Commands
# =============================================================================


def cmd_experiment(key: str, ns: argparse.Namespace) -> int:
    module = importlib.import_module(EXPERIMENTS[key])
    verdict = module.run(_to_exp_args(ns))
    return 0 if verdict.grade != "INCONCLUSIVE" else 2


def cmd_suite(keys: tuple[str, ...], ns: argparse.Namespace) -> int:
    args = _to_exp_args(ns)
    results: list[tuple[str, str, str]] = []
    exit_code = 0

    for key in keys:
        print(f"\n{'=' * 78}\n  {key}\n{'=' * 78}", file=sys.stderr)
        try:
            module = importlib.import_module(EXPERIMENTS[key])
            verdict = module.run(args)
            results.append((key, verdict.grade, verdict.summary))
        except Exception as exc:
            results.append((key, "ERROR", str(exc)))
            exit_code = 1
            if not getattr(ns, "continue_on_error", False):
                print(f"\n{key} failed: {exc}", file=sys.stderr)
                print("Pass --continue-on-error to keep going.", file=sys.stderr)
                break

    print(f"\n{'=' * 78}\n  SUITE SUMMARY\n{'=' * 78}", file=sys.stderr)
    for key, grade, summary in results:
        print(f"  {key:>4}  {grade:<24}  {summary[:100]}", file=sys.stderr)
    return exit_code


def _detection_config(ns: argparse.Namespace, args: ExpArgs) -> dict:
    payload = args.as_config()
    payload.update(args.extra)
    return payload


def _log_weight_table(run_, title: str, rows: list[tuple[str, "object"]]) -> None:
    run_.log.info("")
    run_.log.info("### %s", title)
    if not rows:
        run_.log.info("    (none)")
        return
    run_.log.info(
        "    %-22s %11s %12s %11s %9s %20s %12s",
        "coordinate", "value", "|W|", "mean|W|", "x mean", "rank in layer", "outlier?",
    )
    for label, st in rows:
        run_.log.info(
            "    %-22s %+11.5f %12.5f %11.3e %9.1f %20s %12s",
            label, st.value, st.abs_value, st.mean_abs, st.ratio_to_mean,
            f"{st.rank}/{st.n_elements}", "yes" if st.is_outlier else "NO",
        )


def cmd_verify_directory(ns: argparse.Namespace) -> int:
    from .core.detection import DEFAULT_DETECTION_PROMPT, verify
    from .core.logging_utils import Run, Verdict
    from .core.modeling import load_model

    args = _to_exp_args(ns)
    spec = get_spec(args.model)
    prompt = getattr(ns, "detection_prompt", None) or DEFAULT_DETECTION_PROMPT

    with Run(
        "verify_directory", args.model, _detection_config(ns, args), runs_dir=args.runs_dir
    ) as run_:
        run_.attach("swx.detection")
        run_.log.info("Loading %s ...", spec.hf_id)
        loaded = load_model(
            spec,
            device=args.device,
            dtype=args.dtype,
            allow_fp16=args.allow_fp16,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

        result = verify(
            loaded,
            prompt=prompt,
            max_super_weights=ns.max_super_weights,
            min_spike_ratio=ns.min_spike_ratio,
            min_weight_ratio=ns.min_weight_ratio,
            min_dominance=ns.min_dominance,
        )

        _log_weight_table(
            run_,
            "Tabulated coordinates (Table 2) -- weight magnitude vs its tensor",
            [(str(sw), result.tabulated_stats[str(sw)])
             for sw in result.tabulated if str(sw) in result.tabulated_stats],
        )
        _log_weight_table(
            run_,
            "Independently detected coordinates -- weight magnitude vs its tensor",
            [(str(c.super_weight), c.weight_stats) for c in result.detected_candidates],
        )

        run_.section(
            "Tabulated (from Table 2)",
            {
                str(sw): result.tabulated_stats[str(sw)].as_dict()
                for sw in result.tabulated
                if str(sw) in result.tabulated_stats
            }
            or {"(none)": "model not in the paper's directory"},
        )
        run_.section(
            "Independently detected",
            {str(c.super_weight): c.as_dict() for c in result.detected_candidates}
            or {"(none)": "no aligned spike found"},
        )
        run_.section(
            "Agreement",
            {
                "matched": [str(sw) for sw in result.exact_matches],
                "tabulated_but_not_detected": [str(sw) for sw in result.tabulated_only],
                "tabulated_coordinate_is_not_an_outlier": [
                    str(sw) for sw in result.bad_tabulated
                ],
                "same_channels_different_layer": [
                    f"table says {tab} -> model says {det}"
                    for tab, det in result.layer_mismatches
                ],
                "additional_beyond_the_table": [str(sw) for sw in result.additional],
                "agrees": result.agrees,
            },
        )

        if not result.tabulated:
            verdict = Verdict(
                grade="INCONCLUSIVE",
                summary=(
                    f"{spec.key} is not in the paper's directory. Detection found "
                    f"{len(result.detected)} candidate(s): "
                    + ", ".join(str(sw) for sw in result.detected)
                ),
            )
        elif result.agrees:
            extra = ""
            if result.additional:
                extra = (
                    " Detection additionally found "
                    + ", ".join(str(sw) for sw in result.additional)
                    + ", which Table 2 does not list."
                )
            verdict = Verdict(
                grade="SUPPORTS_HYPOTHESIS",
                summary=(
                    f"Directory VERIFIED for {spec.key}: all "
                    f"{len(result.exact_matches)} tabulated coordinate(s) were "
                    f"independently reproduced and every one is a genuine outlier "
                    f"in its own tensor.{extra}"
                ),
                evidence={
                    str(sw): result.tabulated_stats[str(sw)].describe()
                    for sw in result.tabulated
                    if str(sw) in result.tabulated_stats
                },
            )
        else:
            verdict = Verdict(
                grade="CONTRADICTS_HYPOTHESIS",
                summary=f"MISMATCH for {spec.key}. {result.summary}",
                evidence={
                    str(sw): result.tabulated_stats[str(sw)].describe()
                    for sw in result.tabulated
                    if str(sw) in result.tabulated_stats
                },
            )
        run_.finish(verdict)
        return 0 if result.agrees or not result.tabulated else 1


def cmd_detect(ns: argparse.Namespace) -> int:
    from .core.detection import DEFAULT_DETECTION_PROMPT, detect
    from .core.logging_utils import Run, Verdict
    from .core.modeling import load_model

    args = _to_exp_args(ns)
    spec = get_spec(args.model)
    prompt = ns.detection_prompt or DEFAULT_DETECTION_PROMPT

    with Run(
        "detect", args.model, _detection_config(ns, args), runs_dir=args.runs_dir
    ) as run_:
        run_.attach("swx.detection")
        loaded = load_model(
            spec,
            device=args.device,
            dtype=args.dtype,
            allow_fp16=args.allow_fp16,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

        result = detect(
            loaded,
            prompt=prompt,
            max_super_weights=ns.max_super_weights,
            min_spike_ratio=ns.min_spike_ratio,
            min_weight_ratio=ns.min_weight_ratio,
            min_dominance=ns.min_dominance,
        )

        _log_weight_table(
            run_,
            "Accepted super weights -- weight magnitude vs its tensor",
            [(str(c.super_weight), c.weight_stats) for c in result.accepted],
        )

        run_.section(
            "Detected super weights",
            {
                "prompt": result.prompt,
                "min_spike_ratio": result.min_spike_ratio,
                "min_weight_ratio": result.min_weight_ratio,
                "min_dominance": result.min_dominance,
                "max_super_weights": ns.max_super_weights,
                "stopped_because": result.stopped_because,
                "candidates": [str(sw) for sw in result.candidates],
                **{f"evidence: {c.super_weight}": c.as_dict() for c in result.accepted},
            },
        )

        verdict = Verdict(
            grade="SUPPORTS_HYPOTHESIS" if result.candidates else "INCONCLUSIVE",
            summary=(
                f"Found {len(result.candidates)} super weight(s) in {spec.key}: "
                + ", ".join(
                    f"{c.super_weight} ({c.weight_stats.ratio_to_mean:.0f}x mean |W|, "
                    f"rank {c.weight_stats.rank})"
                    for c in result.accepted
                )
                if result.candidates
                else f"No super weight found in {spec.key} meeting criteria."
            ),
            evidence={
                str(c.super_weight): c.weight_stats.describe() for c in result.accepted
            },
        )
        run_.finish(verdict)
        return 0


def cmd_chat(ns: argparse.Namespace) -> int:
    from .chat import DetectionSettings, GenConfig, Target, parse_selection, repl
    from .core.detection import (
        DEFAULT_DETECTION_PROMPT,
        detect,
        resolve_super_weights,
    )
    from .core.modeling import check_fits, load_model

    args = _to_exp_args(ns)
    spec = get_spec(args.model)

    fits, message = check_fits(spec)
    print(f"VRAM pre-flight: {message}", file=sys.stderr)

    print(f"Loading {spec.hf_id} ({args.dtype}) ...", file=sys.stderr)
    loaded = load_model(
        spec,
        device=args.device,
        dtype=args.dtype,
        allow_fp16=args.allow_fp16,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    detection = DetectionSettings(
        prompt=ns.detection_prompt or DEFAULT_DETECTION_PROMPT,
        max_super_weights=ns.max_super_weights,
        min_spike_ratio=ns.min_spike_ratio,
        min_weight_ratio=ns.min_weight_ratio,
        min_dominance=ns.min_dominance,
    )

    try:
        super_weights, provenance = resolve_super_weights(
            loaded,
            override=args.super_weights,
            auto_detect=True,
        )
        targets = [Target(sw, provenance) for sw in super_weights]

        if ns.detect:
            result = detect(
                loaded,
                prompt=detection.prompt,
                max_super_weights=detection.max_super_weights,
                min_spike_ratio=detection.min_spike_ratio,
                min_weight_ratio=detection.min_weight_ratio,
                min_dominance=detection.min_dominance,
            )
            known = {sw.as_tuple() for sw in super_weights}
            targets += [
                Target(sw, "detected") for sw in result.candidates
                if sw.as_tuple() not in known
            ]
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    start_selection = [
        i for i, t in enumerate(targets) if t.provenance == provenance
    ] or list(range(len(targets)))
    if ns.select is not None:
        chosen = parse_selection(ns.select, targets)
        if chosen is None:
            return 1
        start_selection = chosen

    cfg = GenConfig(
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        top_p=ns.top_p,
        do_sample=not ns.greedy,
        seed=args.seed,
        reseed_each_turn=not ns.no_reseed,
    )
    return repl(
        loaded,
        targets,
        provenance,
        cfg,
        compare_on_start=ns.compare,
        ablated_on_start=ns.ablated,
        start_selection=start_selection,
        detection_settings=detection,
    )


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def cmd_list_models() -> int:
    print(f"{'key':<16} {'params':<8} {'layers':<7} {'d_model':<8} {'bf16 GB':<9} super weights")
    print("-" * 100)
    for key, spec in SPECS.items():
        sws = ", ".join(str(sw) for sw in spec.super_weights) or "(not in directory -- use `swx detect`)"
        print(
            f"{key:<16} {'':<8} {spec.n_layers:<7} {spec.d_model:<8} "
            f"{spec.approx_bf16_gb:<9.1f} {sws}"
        )
        if spec.notes:
            for line in _wrap(spec.notes, 92):
                print(f"{'':<16} {line}")
        print()
    print("Residual channels written to (where super activations appear):")
    for key, spec in SPECS.items():
        if spec.super_weights:
            print(f"  {key:<16} {residual_channels(spec)}")
    return 0


def cmd_summarise(ns: argparse.Namespace) -> int:
    from .core.logging_utils import summarise_runs

    print(summarise_runs(Path(ns.runs_dir) if ns.runs_dir else None))
    return 0


# =============================================================================
# Controlled pretraining (train-twins) and plotting
# =============================================================================


def cmd_train_twins(ns: argparse.Namespace) -> int:
    from .s46_formation.train_activation_twins import (
        TrainConfig,
        VARIANT_ALIASES,
        load_condition_manifest,
        train,
    )

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    condition_name = ""
    intervention = None
    if ns.condition_manifest:
        if not ns.variant:
            raise ValueError("--condition-manifest requires an explicit --variant.")
        condition_name, intervention = load_condition_manifest(ns.condition_manifest)

    variants_to_train: list[str]
    if ns.variant:
        variants_to_train = [ns.variant]
    else:
        variants_to_train = list(VARIANT_ALIASES.keys())

    import gc
    import torch

    for variant in variants_to_train:
        cfg = TrainConfig(
            variant=variant,
            max_steps=ns.max_steps,
            batch_size=ns.batch_size,
            micro_batch_size=ns.micro_batch_size,
            lr=ns.lr,
            log_every=ns.log_every,
            save_every=ns.save_every,
            seed=ns.seed,
            data_seed=getattr(ns, "data_seed", 42),
            device=ns.device,
            dtype=ns.dtype,
            output_dir=ns.output_dir,
            resume=ns.resume,
            condition_name=condition_name,
            intervention=intervention,
            dataset=getattr(ns, "dataset", "HuggingFaceFW/fineweb-edu"),
            dataset_config=getattr(ns, "dataset_config", None),
            dataset_split=getattr(ns, "dataset_split", "train"),
            dataset_text_column=getattr(ns, "dataset_text_column", "text"),
            token_cache_dir=getattr(ns, "token_cache_dir", ""),
        )
        train(cfg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


def cmd_plot_trajectories(ns: argparse.Namespace) -> int:
    from .s46_formation.plot_trajectories import run_analysis

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_analysis(
        ns.run_dir,
        baseline_dir=getattr(ns, "baseline_dir", None),
        dataset=getattr(ns, "dataset", None),
    )
    return 0


def cmd_plot_lag(ns: argparse.Namespace) -> int:
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        from plot_multi_seed_lag_dynamics import main as run_lag_plot
        lag_args = []
        if ns.runs_dir:
            lag_args.extend(["--runs-dir", str(ns.runs_dir)])
        if ns.out_dir:
            lag_args.extend(["--out-dir", str(ns.out_dir)])
        if ns.dpi:
            lag_args.extend(["--dpi", str(ns.dpi)])
        return run_lag_plot(lag_args)
    except Exception as exc:
        import subprocess
        script_path = scripts_dir / "plot_multi_seed_lag_dynamics.py"
        if not script_path.exists():
            print(f"Error: plot script not found at {script_path}", file=sys.stderr)
            return 1
        cmd = [sys.executable, str(script_path)]
        if ns.runs_dir:
            cmd.extend(["--runs-dir", str(ns.runs_dir)])
        if ns.out_dir:
            cmd.extend(["--out-dir", str(ns.out_dir)])
        if ns.dpi:
            cmd.extend(["--dpi", str(ns.dpi)])
        return subprocess.run(cmd).returncode


# =============================================================================
# Entry point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    command = ns.command

    if command in EXPERIMENTS:
        return cmd_experiment(command, ns)

    suites = {
        "direction-g": DIRECTION_G,
    }
    if command in suites:
        return cmd_suite(suites[command], ns)

    if command == "verify-directory":
        return cmd_verify_directory(ns)
    if command == "detect":
        return cmd_detect(ns)
    if command == "chat":
        return cmd_chat(ns)
    if command == "list-models":
        return cmd_list_models()
    if command == "summarise":
        return cmd_summarise(ns)

    if command == "train-twins":
        return cmd_train_twins(ns)
    if command == "plot-trajectories":
        return cmd_plot_trajectories(ns)
    if command == "plot-lag":
        return cmd_plot_lag(ns)

    raise SystemExit(f"Unknown command {command!r}")


if __name__ == "__main__":
    sys.exit(main())
