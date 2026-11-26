"""
Training checkpoints: which ones exist, and how to read one weight out of one
without downloading the whole model.

------------------------------------------------------------------------------
WHY THIS MODULE EXISTS
------------------------------------------------------------------------------
Tracking a single weight trajectory across training checkpoints (e.g. Pythia's
154 revisions) requires reading one scalar -- plus the tensor around it for
context -- at each checkpoint.

The obvious implementation downloads 154 checkpoints and throws away 99.99% of
each (~200 GB dl/disk), making bandwidth, not GPU, the bottleneck.

It does not have to be.  A safetensors file is a JSON header of byte offsets
followed by a flat data buffer, and HTTP supports range requests, so a single
tensor can be pulled out of a remote checkpoint without fetching the rest.  For
pythia-160m one `down_proj` tensor is 768 x 3072 x 2 bytes = 4.7 MB against a
320 MB checkpoint, which turns the full 154-step sweep from ~50 GB into
~700 MB and from an overnight download into a coffee break.

`read_tensor` does that, and falls back to a normal download when it cannot --
old revisions that only shipped `pytorch_model.bin` cannot be sliced, because
pickle has no offset table.  Which path was taken is always recorded on the
result, because "this ran fast" and "this quietly downloaded 50 GB" should not
be indistinguishable afterwards.

------------------------------------------------------------------------------
THE SAFETENSORS LAYOUT (what the slicing relies on)
------------------------------------------------------------------------------
    bytes  0 ..  8      u64 little-endian: N, the header length
    bytes  8 .. 8+N     UTF-8 JSON:
                          {"tensor.name": {"dtype": "F16",
                                           "shape": [768, 3072],
                                           "data_offsets": [begin, end]},
                           "__metadata__": {...}}
    bytes  8+N ..       the data buffer; `data_offsets` are relative to HERE

So: read 8 bytes, read N bytes, seek to 8 + N + begin, read (end - begin).
Three ranged reads totalling a few MB, whatever the checkpoint's size.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .sw_directory import ModelSpec

log = logging.getLogger(__name__)


# =============================================================================
# Which checkpoints exist
# =============================================================================
# Pythia's schedule, from the model card: 143,000 total steps at batch size
# 1024 x 2048 tokens = ~300 B tokens.  Checkpoints are LOG-SPACED over the first
# 1000 steps and then every 1000 steps, which is exactly the right shape for
# this question -- if the super weight is selected early, it happens in a region
# the schedule samples densely.


def _pythia_steps() -> tuple[int, ...]:
    """The canonical 154 Pythia checkpoint steps."""
    log_spaced = [0] + [2**i for i in range(10)]        # 0, 1, 2, 4, ..., 512
    linear = list(range(1000, 143_000 + 1, 1000))       # 1000 ... 143000
    return tuple(log_spaced + linear)


PYTHIA_STEPS: tuple[int, ...] = _pythia_steps()
assert len(PYTHIA_STEPS) == 154, f"expected 154 Pythia checkpoints, got {len(PYTHIA_STEPS)}"

PYTHIA_TOKENS_PER_STEP: int = 1024 * 2048
"""Batch size x sequence length. Lets a step index be reported in tokens, which
is the unit that makes runs of different sizes comparable."""


def revision_for_step(step: int) -> str:
    """HuggingFace revision name for a Pythia training step."""
    return f"step{step}"


def step_from_revision(revision: str | None) -> int | None:
    """Inverse of `revision_for_step`; None for a non-checkpoint revision."""
    if revision and revision.startswith("step"):
        try:
            return int(revision[4:])
        except ValueError:
            return None
    return None


def log_spaced_steps(n: int, available: Sequence[int] = PYTHIA_STEPS) -> tuple[int, ...]:
    """
    Pick `n` checkpoints spread evenly in LOG step, not linear step.

    Linear spacing would put 39 of 40 samples after step 4000 and miss the
    entire early period -- which is where the interesting thing happens, since
    Kovaleva et al. (2021) find weight outliers emerging early in pre-training
    and Gu et al. (ICLR 2025) find attention sinks emerging within the first few
    thousand steps.  Log spacing samples both regimes.

    Always includes the first and last available step.  De-duplicates, so the
    result may be shorter than `n` when the early steps crowd together.
    """
    steps = sorted(set(available))
    if n >= len(steps):
        return tuple(steps)
    if n <= 2:
        return (steps[0], steps[-1])

    # log1p keeps step 0 finite and compresses the early steps sensibly.
    targets = torch.linspace(
        0.0, float(torch.log1p(torch.tensor(float(steps[-1])))), n, dtype=torch.float64
    )
    as_steps = (torch.expm1(targets)).tolist()

    chosen: list[int] = []
    for target in as_steps:
        nearest = min(steps, key=lambda s: abs(s - target))
        if nearest not in chosen:
            chosen.append(nearest)
    for endpoint in (steps[0], steps[-1]):
        if endpoint not in chosen:
            chosen.append(endpoint)
    return tuple(sorted(chosen))


def resolve_steps(
    n_checkpoints: int, explicit: Sequence[int] | None = None
) -> tuple[int, ...]:
    """
    Turn CLI arguments into a concrete step list.

    An explicit list is validated against `PYTHIA_STEPS` and any step that does
    not exist is dropped with a warning, because a typo'd `--checkpoint-steps`
    otherwise surfaces 40 minutes later as a 404 in the middle of a sweep.
    """
    if explicit:
        valid = [s for s in explicit if s in set(PYTHIA_STEPS)]
        dropped = sorted(set(explicit) - set(valid))
        if dropped:
            log.warning(
                "Dropping %d step(s) with no Pythia checkpoint: %s. Valid steps are "
                "0, 1, 2, 4, ..., 512, then every 1000 up to 143000.",
                len(dropped), dropped,
            )
        if not valid:
            raise ValueError(
                f"None of --checkpoint-steps {list(explicit)} is a real Pythia "
                "checkpoint."
            )
        return tuple(sorted(valid))
    return log_spaced_steps(n_checkpoints)


# =============================================================================
# Locating the down_proj tensor inside a checkpoint file
# =============================================================================
# Built from the same two path lists `modeling.py` resolves live modules with,
# so there is one source of truth for "what is this architecture's down_proj"
# and the two cannot drift apart.


def down_proj_key_candidates(layer: int) -> tuple[str, ...]:
    """
    Plausible state-dict names for one layer's MLP output projection.

    We never have to guess which is right: `read_tensor` reads the safetensors
    header first and picks whichever candidate is actually present.
    """
    from .modeling import _DECODER_LAYER_PATHS, _DOWN_PROJ_PATHS

    return tuple(
        f"{prefix}.{layer}.{suffix}.weight"
        for prefix in _DECODER_LAYER_PATHS
        for suffix in _DOWN_PROJ_PATHS
    )


# =============================================================================
# Remote single-tensor reads
# =============================================================================

_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}

SINGLE_FILE = "model.safetensors"
SHARD_INDEX = "model.safetensors.index.json"


class CheckpointReadError(RuntimeError):
    """A checkpoint could not be read by any available route."""


@dataclass
class TensorRead:
    """One tensor, plus how it was obtained."""

    tensor: torch.Tensor
    key: str
    """The state-dict name that was actually found."""
    method: str
    """"remote-slice" | "full-download". Recorded so a sweep's cost is auditable
    from its own log rather than from the shape of your disk usage graph."""
    bytes_transferred: int
    """Approximate; for remote-slice this is the tensor plus its header."""

    @property
    def megabytes(self) -> float:
        return self.bytes_transferred / 1024**2


def _parse_safetensors_header(handle: Any) -> tuple[dict[str, Any], int]:
    """
    Read a safetensors header from a seekable file object.

    Returns (header_dict, data_start_offset).
    """
    handle.seek(0)
    raw_length = handle.read(8)
    if len(raw_length) != 8:
        raise CheckpointReadError("File is shorter than a safetensors header.")
    (header_length,) = struct.unpack("<Q", raw_length)
    # A sane header is kilobytes. A nonsense value means this is not
    # safetensors at all, and we would otherwise try to allocate it.
    if not 0 < header_length < 200_000_000:
        raise CheckpointReadError(
            f"Implausible safetensors header length {header_length}; this is "
            "probably not a safetensors file."
        )
    header = json.loads(handle.read(header_length).decode("utf-8"))
    return header, 8 + header_length


def _tensor_from_header(
    handle: Any, header: dict[str, Any], data_start: int, key: str
) -> torch.Tensor:
    """Range-read one tensor described by an already-parsed header."""
    entry = header[key]
    dtype_name = entry["dtype"]
    if dtype_name not in _SAFETENSORS_DTYPES:
        raise CheckpointReadError(
            f"Unsupported safetensors dtype {dtype_name!r} for {key!r}."
        )
    begin, end = entry["data_offsets"]
    handle.seek(data_start + begin)
    payload = handle.read(end - begin)
    if len(payload) != end - begin:
        raise CheckpointReadError(
            f"Short read for {key!r}: wanted {end - begin} bytes, got {len(payload)}."
        )
    # frombuffer needs a writable buffer to avoid a UserWarning and to let the
    # caller mutate the result; bytearray gives us that with one extra copy of
    # a few MB, which is not worth optimising away.
    flat = torch.frombuffer(bytearray(payload), dtype=_SAFETENSORS_DTYPES[dtype_name])
    return flat.reshape(tuple(entry["shape"]))


def _open_remote(repo_id: str, revision: str | None, filename: str):
    """Open a file in an HF repo as a seekable, range-reading file object."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"{repo_id}@{revision}/{filename}" if revision else f"{repo_id}/{filename}"
    return fs.open(path, "rb")


def _read_remote(
    repo_id: str, revision: str | None, candidates: Sequence[str]
) -> TensorRead:
    """
    Pull one tensor out of a remote safetensors checkpoint with range requests.

    Handles both the single-file layout and the sharded layout (an index JSON
    naming which shard holds each key).
    """
    try:
        handle = _open_remote(repo_id, revision, SINGLE_FILE)
    except Exception:
        # Sharded: the index tells us which shard to open, and it is small.
        with _open_remote(repo_id, revision, SHARD_INDEX) as index_handle:
            index = json.loads(index_handle.read().decode("utf-8"))
        weight_map = index.get("weight_map", {})
        present = [k for k in candidates if k in weight_map]
        if not present:
            raise CheckpointReadError(
                f"None of the candidate down_proj keys is in {repo_id}"
                f"@{revision}'s shard index."
            )
        key = present[0]
        with _open_remote(repo_id, revision, weight_map[key]) as shard:
            header, data_start = _parse_safetensors_header(shard)
            tensor = _tensor_from_header(shard, header, data_start, key)
        begin, end = header[key]["data_offsets"]
        return TensorRead(tensor, key, "remote-slice", end - begin + data_start)

    with handle:
        header, data_start = _parse_safetensors_header(handle)
        present = [k for k in candidates if k in header]
        if not present:
            sample = [k for k in header if k != "__metadata__"][:6]
            raise CheckpointReadError(
                f"None of the candidate down_proj keys is in {repo_id}@{revision}. "
                f"Header contains e.g. {sample}. Add this architecture's path to "
                "modeling._DOWN_PROJ_PATHS."
            )
        key = present[0]
        tensor = _tensor_from_header(handle, header, data_start, key)
        begin, end = header[key]["data_offsets"]
        return TensorRead(tensor, key, "remote-slice", end - begin + data_start)


def _read_downloaded(
    repo_id: str, revision: str | None, candidates: Sequence[str]
) -> TensorRead:
    """
    Fallback: download the checkpoint normally and read the tensor locally.

    Needed for revisions that predate safetensors and only ship
    `pytorch_model.bin`, which is a pickle and has no offset table to seek into.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(repo_id, SINGLE_FILE, revision=revision)
        from safetensors.torch import safe_open

        with safe_open(path, framework="pt") as f:
            present = [k for k in candidates if k in f.keys()]
            if not present:
                raise CheckpointReadError(
                    f"No candidate down_proj key found in {repo_id}@{revision}."
                )
            tensor = f.get_tensor(present[0])
            key = present[0]
    except (EntryNotFoundError, OSError):
        # Pre-safetensors checkpoint.
        path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)
        state = torch.load(path, map_location="cpu", weights_only=True)
        present = [k for k in candidates if k in state]
        if not present:
            raise CheckpointReadError(
                f"No candidate down_proj key found in {repo_id}@{revision}'s "
                "pytorch_model.bin."
            )
        key = present[0]
        tensor = state[key]

    import os

    size = os.path.getsize(path)
    return TensorRead(tensor.clone(), key, "full-download", size)


def read_down_proj(
    repo_id: str,
    layer: int,
    revision: str | None = None,
    remote_slice: bool = True,
) -> TensorRead:
    """
    Read one layer's `down_proj` weight matrix out of a checkpoint.

    This primitive makes tracking one scalar across 154 checkpoints cost
    megabytes instead of tens of gigabytes.

    Args:
        repo_id: HuggingFace repo, e.g. "EleutherAI/pythia-160m".
        layer: decoder layer index.
        revision: checkpoint, e.g. "step3000". None means the final model.
        remote_slice: try the ranged read first. Set False to force a normal
            download (useful if you want the checkpoint cached anyway, or if a
            proxy mangles range requests).

    Returns a `TensorRead` in the paper's [d_model, d_ff] convention -- see the
    Conv1D note in `modeling.down_proj_weight_view`; GPT-2-style checkpoints
    store it transposed and are transposed back here so every caller sees one
    convention.
    """
    candidates = down_proj_key_candidates(layer)

    read: TensorRead | None = None
    if remote_slice:
        try:
            read = _read_remote(repo_id, revision, candidates)
        except Exception as exc:  # noqa: BLE001 -- any failure means fall back
            log.debug(
                "Remote slice failed for %s@%s layer %d (%s); downloading instead.",
                repo_id, revision, layer, exc,
            )
    if read is None:
        read = _read_downloaded(repo_id, revision, candidates)

    # GPT-2's Conv1D stores [in, out]; everything else stores [out, in].
    # Normalising here means `residual_channel` always indexes dim 0 downstream.
    if read.key.endswith("c_proj.weight"):
        read.tensor = read.tensor.t().contiguous()

    return read


# =============================================================================
# Cost estimation and availability
# =============================================================================


def estimate_download_gb(
    spec: ModelSpec, n_checkpoints: int, remote_slice: bool = True
) -> float:
    """
    Predict a sweep's download volume, so `--max-download-gb` can refuse it
    BEFORE the first request rather than 30 GB in.
    """
    if remote_slice:
        # One down_proj tensor at 2 bytes/param, plus a ~100 KB header.
        per_checkpoint = (spec.d_model * spec.d_ff * 2 + 100_000) / 1024**3
    else:
        per_checkpoint = spec.approx_bf16_gb
    return per_checkpoint * n_checkpoints


@dataclass
class RepoProbe:
    """Whether one repo (optionally at one revision) actually exists."""

    repo_id: str
    revision: str | None = None
    exists: bool = False
    error: str = ""
    has_safetensors: bool = False
    """False means remote slicing is unavailable and the sweep will download
    whole checkpoints -- worth knowing before, not during."""
    files: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.repo_id}@{self.revision}" if self.revision else self.repo_id


def probe_repo(repo_id: str, revision: str | None = None) -> RepoProbe:
    """
    Metadata-only existence check. No weights are downloaded.

    Worth doing first for checkpoint/seed sweeps: repo names are transcribed
    from model cards, and discovering a typo after a 4 GB download is costly.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, revision=revision, files_metadata=False)
        files = tuple(s.rfilename for s in (info.siblings or []))
        return RepoProbe(
            repo_id=repo_id,
            revision=revision,
            exists=True,
            has_safetensors=any(f.endswith(".safetensors") for f in files),
            files=files,
        )
    except Exception as exc:  # noqa: BLE001 -- 404, auth, offline all report the same way
        return RepoProbe(
            repo_id=repo_id, revision=revision, exists=False, error=f"{type(exc).__name__}: {exc}"
        )


def probe_all(
    repo_ids: Iterable[str], revision: str | None = None
) -> list[RepoProbe]:
    """Probe a set of repos, preserving order."""
    return [probe_repo(repo_id, revision) for repo_id in repo_ids]


# =============================================================================
# Weight statistics from a raw tensor
# =============================================================================
# `modeling.weight_outlier_stats` needs a live nn.Module. Checkpoint sweeps have
# only a bare tensor and never build a model at all, so the same statistics are
# computed here directly. Both produce the same numbers for the same weights;
# this one additionally reports rank WITHIN the row and column, which is what
# answers "when was this coordinate selected?".


@dataclass
class CoordinateStats:
    """How one entry of a weight matrix stands out, at one checkpoint."""

    value: float
    abs_value: float
    mean_abs: float
    std_abs: float
    max_abs: float
    rank_in_tensor: int
    """1 = the largest |W| in the whole tensor."""
    n_elements: int
    rank_in_row: int
    """Rank among the d_ff weights writing into the SAME residual channel."""
    row_size: int
    rank_in_column: int
    """Rank among the d_model weights read from the SAME mlp channel."""
    column_size: int

    row_mean_abs: float
    row_max_abs: float

    @property
    def ratio_to_mean(self) -> float:
        return self.abs_value / self.mean_abs if self.mean_abs else float("inf")

    @property
    def z_score(self) -> float:
        return (self.abs_value - self.mean_abs) / self.std_abs if self.std_abs else float("inf")

    @property
    def percentile(self) -> float:
        """Percentile of |W| within the tensor; 100 = the largest weight in it.

        This, not the raw value, is the quantity that answers "was this
        coordinate already special at initialisation?" -- raw magnitudes shrink
        as a tensor's overall scale changes over training, and a percentile does
        not."""
        return 100.0 * (1.0 - (self.rank_in_tensor - 1) / max(self.n_elements, 1))

    @property
    def share_of_row(self) -> float:
        """|W| as a fraction of the row's total absolute mass.

        The detection identity Y_ij = sum_k X_ik W_jk sums over exactly this
        row, so a coordinate that owns most of its row's mass is one that CAN
        dominate the sum -- which is the structural precondition for being a
        super weight, visible from the weights alone with no forward pass."""
        total = self.row_mean_abs * self.row_size
        return self.abs_value / total if total else float("nan")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "value": self.value,
            "abs_value": self.abs_value,
            "mean_abs": self.mean_abs,
            "std_abs": self.std_abs,
            "max_abs_in_tensor": self.max_abs,
            "ratio_to_mean": self.ratio_to_mean,
            "z_score": self.z_score,
            "rank_in_tensor": self.rank_in_tensor,
            "n_elements": self.n_elements,
            "percentile": self.percentile,
            "rank_in_row": self.rank_in_row,
            "row_size": self.row_size,
            "rank_in_column": self.rank_in_column,
            "column_size": self.column_size,
            "share_of_row": self.share_of_row,
            "row_max_abs": self.row_max_abs,
        }


def coordinate_stats(
    weight: torch.Tensor, residual_channel: int, mlp_channel: int
) -> CoordinateStats:
    """
    Summarise one coordinate of a [d_model, d_ff] weight matrix.

    Computed in float32 (promoted from the checkpoint's fp16) because the
    tensor-wide mean of 2.4M values in fp16 accumulates visible error, and every
    downstream ratio divides by it.
    """
    w = weight.detach().float()
    if not (0 <= residual_channel < w.shape[0] and 0 <= mlp_channel < w.shape[1]):
        raise IndexError(
            f"Coordinate [{residual_channel}, {mlp_channel}] is outside a "
            f"{tuple(w.shape)} weight matrix."
        )

    value = float(w[residual_channel, mlp_channel])
    magnitude = abs(value)
    magnitudes = w.abs()
    row = magnitudes[residual_channel, :]
    column = magnitudes[:, mlp_channel]

    return CoordinateStats(
        value=value,
        abs_value=magnitude,
        mean_abs=float(magnitudes.mean()),
        std_abs=float(magnitudes.std()),
        max_abs=float(magnitudes.max()),
        # Counting how many are strictly larger is O(n) against a sort's
        # O(n log n), and on a 2.4M-element tensor read 154 times that matters.
        rank_in_tensor=int((magnitudes > magnitude).sum()) + 1,
        n_elements=magnitudes.numel(),
        rank_in_row=int((row > magnitude).sum()) + 1,
        row_size=row.numel(),
        rank_in_column=int((column > magnitude).sum()) + 1,
        column_size=column.numel(),
        row_mean_abs=float(row.mean()),
        row_max_abs=float(row.max()),
    )


def argmax_coordinate(weight: torch.Tensor) -> tuple[int, int]:
    """The (residual_channel, mlp_channel) of the largest |W| in the matrix."""
    flat = int(weight.detach().float().abs().argmax())
    return divmod(flat, weight.shape[1])


@dataclass
class CheckpointRecord:
    """One checkpoint's worth of results in a trajectory sweep."""

    step: int
    revision: str
    ok: bool = True
    error: str = ""
    method: str = ""
    megabytes: float = 0.0
    stats: CoordinateStats | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_seen(self) -> int:
        return self.step * PYTHIA_TOKENS_PER_STEP

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": self.step,
            "revision": self.revision,
            "tokens_seen": self.tokens_seen,
            "ok": self.ok,
            "read_method": self.method,
            "megabytes_transferred": self.megabytes,
        }
        if self.error:
            payload["error"] = self.error
        if self.stats is not None:
            payload.update(self.stats.as_dict())
        payload.update(self.extra)
        return payload
