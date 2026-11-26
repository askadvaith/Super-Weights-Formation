"""
The experiment registry -- maps experiment CLI names to modules.

Each entry's module exposes:
    run(args: ExpArgs) -> Verdict
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentInfo:
    """One registered experiment."""

    key: str
    module: str
    section: str
    """Section / group."""
    direction: str
    """Compute-plan direction or label."""
    blurb: str
    """One line, shown by `swx --help`."""
    cost: str
    """Rough wall-clock and download cost."""


_ENTRIES: tuple[ExperimentInfo, ...] = (
    ExperimentInfo(
        "g1",
        "swx.s46_formation.g1_activation_family",
        "4.6",
        "gated_vs_nongated",
        "Cross-family zoo -- does super weight localisation track the gated MLP form across model families?",
        "~4-12 h depending on zoo size; downloads each model once",
    ),
)

#: key -> module path. The CLI imports these lazily.
EXPERIMENTS: dict[str, str] = {e.key: e.module for e in _ENTRIES}

#: key -> full metadata.
INFO: dict[str, ExperimentInfo] = {e.key: e for e in _ENTRIES}

DIRECTION_G = ("g1",)
ALL = ("g1",)
