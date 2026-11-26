"""
`ExpArgs` -- the shared configuration object for CLI and experiment runs.

Keeping this dataclass independent of model-loading or dataset imports avoids
pulling heavy runtime dependencies into lightweight inspection or tensor-only
code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .sw_directory import SuperWeight


@dataclass
class ExpArgs:
    """Everything an experiment can be configured with. Populated by the CLI."""

    # -- model ---------------------------------------------------------------
    model: str = "olmo-1b"
    device: str = "auto"
    dtype: str = "bfloat16"
    allow_fp16: bool = False
    device_map: str | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    """HuggingFace revision (branch/tag/commit). For Pythia this selects a
    training checkpoint: `step0`, `step1000`, ..., `step143000`. None means the
    repo's default branch, i.e. the fully-trained model."""

    # -- data ----------------------------------------------------------------
    prompt_source: str = "lambada"
    n_prompts: int = 500
    batch_size: int = 8
    max_length: int = 256
    seed: int = 0

    # -- which scalar(s) to intervene on -------------------------------------
    super_weights: tuple[SuperWeight, ...] | None = None
    """Explicit override; None means directory-then-detect."""
    detect: bool = False
    """Force live detection even when the directory has an entry."""

    # -- §4.5 tuning ---------------------------------------------------------
    n_freq_bins: int = 20                                     # e2
    scaling_factors: tuple[float, ...] = (                    # e4
        0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
    )
    nullspace_fraction: float = 0.25                          # e5, j3
    n_random_controls: int = 32                               # e5, e6, j1

    # -- §4.4 tuning ---------------------------------------------------------
    jlens_fit_prompts: int = 256                              # j1-j3
    jlens_repo: str | None = None                             # j1-j3
    jlens_filename: str | None = None                         # j1-j3
    jlens_layer: int | None = None                            # j1-j3
    force_refit: bool = False                                 # j1-j3

    # -- §4.6 tuning ---------------------------------------------------------
    # See sw_directory.SCALE_LADDER and SEED_FAMILIES for the vocabularies.
    scale_ladder: tuple[str, ...] = ()                        # t0
    """Model keys to sweep for the de-risking check. Empty means
    sw_directory.SCALE_LADDER."""

    seed_families: tuple[str, ...] = ("data-seed", "weight-seed")   # s1, s2
    """Which Pythia seed families to compare. `data-seed` holds initialisation
    fixed and varies data order; `weight-seed` does the reverse. Those two are
    what make the experiment decisive -- see s1's docstring. `seed` (everything
    varied) is available but only bounds total variability."""

    n_checkpoints: int = 40                                   # t1, t3, t4, s3
    """How many training checkpoints to sample, log-spaced over the 154
    available. 40 resolves the emergence curve well; use 154 for a final run
    and 12 for a smoke test."""

    checkpoint_steps: tuple[int, ...] | None = None           # t1, t3, t4, s3
    """Explicit step list, overriding `n_checkpoints`."""

    remote_slice: bool = True
    """Read single tensors out of remote safetensors files with HTTP range
    requests instead of downloading whole checkpoints. Turns a ~200 GB
    multi-checkpoint sweep into ~1 GB. Falls back to a full download automatically."""

    max_download_gb: float = 60.0                             # t1, t3, t4
    """Refuse to start a sweep whose estimated download exceeds this. A
    154-checkpoint full-download sweep of pythia-410m is ~140 GB; this stops
    that happening by accident on a metered connection."""

    from_run: str | None = None                               # t2
    """Path to an existing run directory whose `results.json` supplies the
    trajectory, so curve fitting can be re-run without re-downloading."""

    activation_dataset: str | None = "Aimpoint-Digital/pythia-massive-activations"
    """HF dataset of published massive-activation statistics (arXiv:2508.03616).
    t3 joins against it so our weight-side numbers sit next to their
    activation-side numbers rather than duplicating them. None to skip."""

    # -- output --------------------------------------------------------------
    runs_dir: Any = None

    # -- extras --------------------------------------------------------------
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form passthrough for command-specific flags that do not deserve a
    field of their own (detection thresholds, generation settings). Recorded in
    config.json, so a run stays reproducible from its own record."""

    def as_config(self) -> dict[str, Any]:
        """JSON-friendly snapshot for config.json."""
        payload = {
            key: value for key, value in self.__dict__.items() if key != "runs_dir"
        }
        if self.super_weights:
            payload["super_weights"] = [str(sw) for sw in self.super_weights]
        return payload
