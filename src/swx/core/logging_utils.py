"""
Run logging and result reporting.

Design goal: after any experiment finishes, a reader who has not seen the code
should be able to open the run directory and understand (a) exactly what was
run, (b) what the numbers are, and (c) what they mean.  To that end every run
produces four files:

    runs/<experiment>/<model>_<timestamp>/
        config.json     what was run: model, prompts, seeds, every parameter
        run.log         full human-readable trace, including the pre-registered
                        prediction printed BEFORE any results
        results.json    all metrics, machine-readable, nested by section
        report.md       a rendered summary with tables and the verdict

The `report.md` is the important one.  Each experiment declares a
`Prediction` (what the §4.5 / §4.4 hypothesis says should happen) and a
`verdict()` function that grades the observed numbers against it.  The report
prints the prediction, the observation, and the grade side by side, so the
result is legible without re-deriving the argument.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# `rich` gives colourised console output and progress bars, but nothing here
# depends on it -- we degrade to the stdlib logger so the package works in a
# bare environment (e.g. a Kaggle kernel with a minimal install).
try:  # pragma: no cover - trivial import guard
    from rich.console import Console
    from rich.logging import RichHandler

    _HAVE_RICH = True
except ImportError:  # pragma: no cover
    _HAVE_RICH = False


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


# =============================================================================
# Verdict machinery
# =============================================================================


@dataclass
class Prediction:
    """
    A pre-registered prediction, printed before the experiment runs.

    Stating this up front is not decoration.  These experiments exist to test
    a specific hypothesis against a published claim, and it is very easy to
    read a post-hoc story into entropy numbers.  Committing to the expected
    direction first is what makes the verdict meaningful.
    """

    hypothesis: str
    """The claim under test, in one sentence."""

    if_true: str
    """What the numbers should look like if the hypothesis holds."""

    if_false: str
    """What they should look like if the paper's original framing is right."""

    key_metric: str
    """The single number the verdict keys off, named as it appears in results.json."""


@dataclass
class Verdict:
    """The graded outcome of one experiment."""

    grade: str
    """One of: SUPPORTS_HYPOTHESIS | PARTIAL | CONTRADICTS_HYPOTHESIS | INCONCLUSIVE."""

    summary: str
    """One or two sentences a reader can quote."""

    evidence: dict[str, Any] = field(default_factory=dict)
    """The specific numbers the grade was based on."""

    GRADES = (
        "SUPPORTS_HYPOTHESIS",
        "PARTIAL",
        "CONTRADICTS_HYPOTHESIS",
        "INCONCLUSIVE",
    )

    def __post_init__(self) -> None:
        if self.grade not in self.GRADES:
            raise ValueError(f"grade must be one of {self.GRADES}, got {self.grade!r}")

    @property
    def emoji(self) -> str:
        return {
            "SUPPORTS_HYPOTHESIS": "[+]",
            "PARTIAL": "[~]",
            "CONTRADICTS_HYPOTHESIS": "[-]",
            "INCONCLUSIVE": "[?]",
        }[self.grade]


# =============================================================================
# Run directory
# =============================================================================


def _json_default(obj: Any) -> Any:
    """Make dataclasses, paths, sets and numpy/torch scalars JSON-serialisable."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    # numpy / torch scalars and arrays, without importing either eagerly
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


class Run:
    """
    One experiment invocation, with its own directory and log file.

    Usage:

        with Run("e1_temperature_control", model_key="olmo-1b", config=cfg) as run:
            run.log.info("...")
            run.section("Ablated distribution", {...})
            run.finish(prediction, verdict, results)
    """

    def __init__(
        self,
        experiment: str,
        model_key: str,
        config: dict[str, Any] | None = None,
        runs_dir: Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.model_key = model_key
        self.config = config or {}

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = runs_dir or DEFAULT_RUNS_DIR
        self.dir = base / experiment / f"{model_key}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.results: dict[str, Any] = {}
        self._sections: list[tuple[str, dict[str, Any]]] = []
        self._attached: list[logging.Logger] = []

        self.log = self._make_logger()

    # -- lifecycle ---------------------------------------------------------

    def attach(self, *logger_names: str) -> None:
        """
        Route another module's logger into this run's log file and console.

        Without this, `swx.detection`'s per-candidate evidence -- spike ratios,
        weight magnitudes, why a candidate was rejected -- goes nowhere, because
        the run logger is a sibling branch with `propagate = False`. Losing
        exactly the numbers that justify a coordinate is how a run ends up
        unfalsifiable after the fact.
        """
        for name in logger_names:
            other = logging.getLogger(name)
            other.setLevel(logging.DEBUG)
            for handler in self.log.handlers:
                if handler not in other.handlers:
                    other.addHandler(handler)
            self._attached.append(other)

    def __enter__(self) -> "Run":
        self.log.info("=" * 78)
        self.log.info("EXPERIMENT  %s", self.experiment)
        self.log.info("MODEL       %s", self.model_key)
        self.log.info("RUN DIR     %s", self.dir)
        self.log.info("STARTED     %s", datetime.now(timezone.utc).isoformat())
        self.log.info("=" * 78)
        self._write_config()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.log.exception("Run failed: %s", exc)
        for handler in list(self.log.handlers):
            for other in self._attached:
                other.removeHandler(handler)
            handler.flush()
            handler.close()
            self.log.removeHandler(handler)
        self._attached.clear()

    # -- logging -----------------------------------------------------------

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"swx.{self.experiment}.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # File handler: everything, with timestamps.
        fh = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
        )
        logger.addHandler(fh)

        # Console handler: INFO and up, prettier if rich is available.
        if _HAVE_RICH:
            ch: logging.Handler = RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                show_path=False,
                markup=False,
            )
            ch.setFormatter(logging.Formatter("%(message)s"))
        else:
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        return logger

    def _write_config(self) -> None:
        payload = {
            "experiment": self.experiment,
            "model": self.model_key,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "environment": _environment_info(),
        }
        (self.dir / "config.json").write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )

    # -- structured output -------------------------------------------------

    def announce_prediction(self, prediction: Prediction) -> None:
        """Print the pre-registered prediction BEFORE any numbers exist."""
        self.log.info("")
        self.log.info("-" * 78)
        self.log.info("PRE-REGISTERED PREDICTION")
        self.log.info("-" * 78)
        self.log.info("Hypothesis  : %s", prediction.hypothesis)
        self.log.info("If TRUE     : %s", prediction.if_true)
        self.log.info("If FALSE    : %s", prediction.if_false)
        self.log.info("Key metric  : %s", prediction.key_metric)
        self.log.info("-" * 78)
        self.log.info("")
        self.results["prediction"] = asdict(prediction)

    def section(self, title: str, values: dict[str, Any]) -> None:
        """
        Record and print a named block of metrics.

        Sections appear in results.json under their title and as a table in
        report.md, in the order they were added.
        """
        self._sections.append((title, values))
        self.results.setdefault("sections", {})[title] = values

        self.log.info("")
        self.log.info("### %s", title)
        width = max((len(str(k)) for k in values), default=0)
        for key, value in values.items():
            self.log.info("    %-*s  %s", width, key, _fmt(value))

    def finish(self, verdict: Verdict) -> None:
        """Write results.json and report.md, and print the verdict."""
        self.results["verdict"] = asdict(verdict)
        self.results["finished_utc"] = datetime.now(timezone.utc).isoformat()

        (self.dir / "results.json").write_text(
            json.dumps(self.results, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (self.dir / "report.md").write_text(self._render_report(verdict), encoding="utf-8")

        self.log.info("")
        self.log.info("=" * 78)
        self.log.info("VERDICT  %s  %s", verdict.emoji, verdict.grade)
        self.log.info("=" * 78)
        for line in _wrap(verdict.summary, 74):
            self.log.info("  %s", line)
        if verdict.evidence:
            self.log.info("")
            self.log.info("  Evidence:")
            for key, value in verdict.evidence.items():
                self.log.info("    %-38s %s", key, _fmt(value))
        self.log.info("=" * 78)
        self.log.info("Wrote %s", self.dir / "report.md")

    # -- report ------------------------------------------------------------

    def _render_report(self, verdict: Verdict) -> str:
        pred = self.results.get("prediction")
        lines: list[str] = [
            f"# {self.experiment} -- {self.model_key}",
            "",
            f"Run directory: `{self.dir}`  ",
            f"Finished: {self.results['finished_utc']}",
            "",
            "## Verdict",
            "",
            f"**{verdict.emoji} {verdict.grade}**",
            "",
            verdict.summary,
            "",
        ]

        if verdict.evidence:
            lines += ["| Evidence | Value |", "|---|---|"]
            lines += [f"| {k} | {_fmt(v)} |" for k, v in verdict.evidence.items()]
            lines.append("")

        if pred:
            lines += [
                "## Pre-registered prediction",
                "",
                f"**Hypothesis.** {pred['hypothesis']}",
                "",
                f"- *If the hypothesis is true:* {pred['if_true']}",
                f"- *If it is false (paper's framing correct):* {pred['if_false']}",
                f"- *Key metric:* `{pred['key_metric']}`",
                "",
            ]

        lines += ["## Results", ""]
        for title, values in self._sections:
            lines += [f"### {title}", "", "| Metric | Value |", "|---|---|"]
            lines += [f"| `{k}` | {_fmt(v)} |" for k, v in values.items()]
            lines.append("")

        lines += [
            "## Configuration",
            "",
            "```json",
            json.dumps(self.config, indent=2, default=_json_default),
            "```",
            "",
        ]
        return "\n".join(lines)


# =============================================================================
# Helpers
# =============================================================================


def _fmt(value: Any) -> str:
    """Format a metric for display: floats to 6sf, lists truncated."""
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value == 0 or 1e-4 <= abs(value) < 1e6:
            return f"{value:.6g}"
        return f"{value:.4e}"
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            head = ", ".join(_fmt(v) for v in value[:8])
            return f"[{head}, ... ({len(value)} items)]"
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, default=_json_default)[:200]
    return str(value)


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
    return lines or [""]


def _environment_info() -> dict[str, Any]:
    """Capture enough environment detail to reproduce or debug a run."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - not a git repo, fine
        pass
    return info


def summarise_runs(runs_dir: Path | None = None) -> str:
    """
    Build a single markdown table across every run found on disk.

    Used by `swx summarise`, so you can see all verdicts at a glance after a
    batch instead of opening nine report.md files.
    """
    base = runs_dir or DEFAULT_RUNS_DIR
    rows: list[tuple[str, str, str, str, str]] = []
    for results_path in sorted(base.rglob("results.json")):
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        verdict = data.get("verdict", {})
        cfg_path = results_path.parent / "config.json"
        model = "?"
        if cfg_path.exists():
            try:
                model = json.loads(cfg_path.read_text(encoding="utf-8")).get("model", "?")
            except Exception:
                pass
        rows.append(
            (
                results_path.parent.parent.name,
                model,
                verdict.get("grade", "-"),
                verdict.get("summary", "")[:120],
                str(results_path.parent.relative_to(base)),
            )
        )

    if not rows:
        return f"No runs found under {base}."

    out = [
        f"# swx run summary ({len(rows)} runs under `{base}`)",
        "",
        "| Experiment | Model | Verdict | Summary | Run |",
        "|---|---|---|---|---|",
    ]
    out += [f"| {a} | {b} | **{c}** | {d} | `{e}` |" for a, b, c, d, e in rows]
    return "\n".join(out)
