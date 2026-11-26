"""
Model loading, architecture-agnostic module resolution, and interventions.

------------------------------------------------------------------------------
THE PRECISION RULE
------------------------------------------------------------------------------
`load_model` REFUSES to load a quantised model, and this is deliberate.

Quantisation is precisely the process that damages super outliers -- it is the
subject of the anchor paper's own Section 4.  Studying super weights through a
4-bit or 8-bit model measures the artifact, not the phenomenon.  bf16 is the
floor; fp32 is preferred for the logit/entropy comparisons in §4.5, where we
are chasing probability shifts of a few percent and fp16 accumulation error is
of comparable size.

Turing-era cards (Kaggle T4, P100) have no bf16.  fp16 has ample dynamic range
for activations of magnitude ~1e3 (fp16 max is 65504, and the largest super
activation reported in the paper is ~1500), so it will not overflow -- but all
downstream entropy and KL arithmetic is done in float64 regardless, in
metrics.py.  See the `allow_fp16` escape hatch below.
"""

from __future__ import annotations

import contextlib
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sw_directory import ModelSpec, SuperWeight


def _get_hf_token() -> str | None:
    """
    Robustly locate Hugging Face auth token across all common locations.
    Checks environment variables (HF_TOKEN, HUGGING_FACE_HUB_TOKEN),
    the huggingface_hub helper, and filesystem token files ($HF_HOME, ~/.cache, ~/.config).
    Returns None if no token is discovered so huggingface_hub uses default resolution.
    """
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok and tok.strip():
            return tok.strip()
    except Exception:
        pass

    candidates = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".config" / "huggingface" / "token",
    ]
    if "HF_HOME" in os.environ:
        candidates.insert(0, Path(os.environ["HF_HOME"]) / "token")

    for p in candidates:
        try:
            if p.is_file():
                tok = p.read_text(encoding="utf-8").strip()
                if tok:
                    return tok
        except Exception:
            continue

    return None

# -----------------------------------------------------------------------------
# Architecture adapters
# -----------------------------------------------------------------------------
# Different model families name the MLP output projection differently.  We
# resolve it by trying attribute paths in order.  `down_proj` is the third
# linear layer of the MLP -- the one that writes back into the residual stream,
# and the only place the anchor paper's detection method can find anything.

_DECODER_LAYER_PATHS: tuple[str, ...] = (
    "model.layers",        # Llama, Mistral, OLMo (new hf), Qwen2, Phi-3, Gemma
    "gpt_neox.layers",     # Pythia / GPT-NeoX
    "transformer.h",       # GPT-2, GPT-J, Falcon
    "model.decoder.layers",  # OPT
)

_DOWN_PROJ_PATHS: tuple[str, ...] = (
    "mlp.down_proj",       # Llama family and descendants
    "mlp.dense_4h_to_h",   # GPT-NeoX / Pythia
    "mlp.c_proj",          # GPT-2 (nn.Conv1D -- transposed weight, handled below)
    "mlp.fc2",             # some Phi variants
    "fc2",                 # OPT (block.fc2 directly, without mlp submodule)
    "mlp.w2",              # a few Mistral-derived forks
)


def _resolve(root: Any, dotted: str) -> Any | None:
    """Walk a dotted attribute path, returning None if any hop is missing."""
    node = root
    for part in dotted.split("."):
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the model's list of decoder blocks, whatever the family calls it."""
    for path in _DECODER_LAYER_PATHS:
        layers = _resolve(model, path)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise RuntimeError(
        "Could not locate the decoder layer list. Tried: "
        + ", ".join(_DECODER_LAYER_PATHS)
        + ". Add this architecture's path to _DECODER_LAYER_PATHS."
    )


def get_down_proj(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the MLP output projection ('down_proj') of one decoder layer."""
    block = get_decoder_layers(model)[layer_idx]
    for path in _DOWN_PROJ_PATHS:
        module = _resolve(block, path)
        if module is not None:
            return module
    raise RuntimeError(
        f"Could not locate down_proj in layer {layer_idx}. Tried: "
        + ", ".join(_DOWN_PROJ_PATHS)
    )


def down_proj_weight_view(module: nn.Module) -> tuple[torch.Tensor, bool]:
    """
    Return (weight, is_transposed) for a down-projection module.

    nn.Linear stores weight as [out_features, in_features] = [d_model, d_ff],
    which matches the paper's [row, col] convention directly.

    GPT-2's Conv1D stores it TRANSPOSED, as [in_features, out_features].
    Getting this backwards would silently ablate the wrong scalar, so we detect
    it explicitly rather than assuming.
    """
    weight = module.weight
    is_transposed = module.__class__.__name__ == "Conv1D"
    return weight, is_transposed


def sw_index(module: nn.Module, sw: SuperWeight) -> tuple[int, int]:
    """
    Map a paper-convention (residual_channel, mlp_channel) to actual tensor
    indices for this module, accounting for Conv1D transposition.
    """
    _, is_transposed = down_proj_weight_view(module)
    if is_transposed:
        return (sw.mlp_channel, sw.residual_channel)
    return (sw.residual_channel, sw.mlp_channel)


def down_proj_row(module: nn.Module, residual_channel: int) -> torch.Tensor:
    """
    Row `residual_channel` of down_proj in the paper's convention: a [d_ff]
    vector of the weights that write into one residual-stream channel.

    This is the vector the detection identity Y_ij = sum_k X_ik W_jk sums over,
    so it is what you need to find WHICH mlp channel dominates the sum.
    """
    weight, is_transposed = down_proj_weight_view(module)
    return weight[:, residual_channel] if is_transposed else weight[residual_channel, :]


# -----------------------------------------------------------------------------
# Weight outlier statistics
# -----------------------------------------------------------------------------
# A super weight is defined by being a conspicuous OUTLIER in its own tensor,
# not merely by sitting at a coordinate someone wrote down.  Every coordinate
# this codebase reports -- tabulated or detected -- is therefore accompanied by
# these numbers, so "is this actually a super weight?" is answerable from the
# log rather than by trusting a table.


@dataclass(frozen=True)
class WeightStats:
    """How far one scalar stands out from the tensor it lives in."""

    value: float
    """The signed scalar itself."""

    mean_abs: float
    """mean |W| over the whole down_proj tensor -- the 'typical' magnitude."""

    std_abs: float
    max_abs: float
    """The largest |W| anywhere in the tensor, for scale."""

    rank: int
    """1 = the largest |W| in THIS LAYER'S down_proj tensor. The single most
    telling number -- but note the ranking is per-layer, so several super
    weights in one model can each be rank 1 in their own layer, and normally
    are. Compare `mean_abs`, which differs between layers, to see at a glance
    that two coordinates were ranked against different tensors."""

    n_elements: int

    @property
    def abs_value(self) -> float:
        return abs(self.value)

    @property
    def ratio_to_mean(self) -> float:
        """|W| / mean|W|. Genuine super weights run 30-140x in the models here."""
        return self.abs_value / self.mean_abs if self.mean_abs else float("inf")

    @property
    def z_score(self) -> float:
        """(|W| - mean|W|) / std|W|. Redundant with ratio_to_mean but familiar."""
        return (self.abs_value - self.mean_abs) / self.std_abs if self.std_abs else float("inf")

    @property
    def top_percent(self) -> float:
        """What fraction of the tensor (as a %) is at least this large."""
        return 100.0 * self.rank / self.n_elements

    @property
    def is_outlier(self) -> bool:
        """The default gate used by detection: >=20x the mean magnitude."""
        return self.ratio_to_mean >= 20.0

    def describe(self) -> str:
        """One line for the log: magnitude, divergence from the mean, and rank."""
        return (
            f"value={self.value:+.5f}  |W|={self.abs_value:.5f}  "
            f"mean|W|={self.mean_abs:.3e}  {self.ratio_to_mean:7.1f}x mean  "
            f"z={self.z_score:+7.1f}  rank {self.rank}/{self.n_elements} "
            f"of its own layer's down_proj (top {self.top_percent:.6f}%)"
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "value": self.value,
            "abs_value": self.abs_value,
            "mean_abs": self.mean_abs,
            "std_abs": self.std_abs,
            "max_abs_in_tensor": self.max_abs,
            "ratio_to_mean": self.ratio_to_mean,
            "z_score": self.z_score,
            "rank": self.rank,
            "n_elements": self.n_elements,
            "top_percent": self.top_percent,
        }


@torch.no_grad()
def weight_outlier_stats(module: nn.Module, sw: SuperWeight) -> WeightStats:
    """
    Summarise how much one down_proj entry diverges from the rest of its tensor.

    `rank` is computed by an exact count rather than a sort: on a 17M-element
    tensor a full sort costs far more than a comparison, and we only need the
    position, not the ordering.
    """
    weight, _ = down_proj_weight_view(module)
    i, j = sw_index(module, sw)
    value = float(weight[i, j].item())

    magnitudes = weight.detach().float().abs()
    return WeightStats(
        value=value,
        mean_abs=float(magnitudes.mean()),
        std_abs=float(magnitudes.std()),
        max_abs=float(magnitudes.max()),
        rank=int((magnitudes > abs(value)).sum()) + 1,
        n_elements=magnitudes.numel(),
    )


def get_unembedding(model: nn.Module) -> torch.Tensor:
    """
    Return the unembedding matrix W_U with shape [vocab, d_model].

    Used by the null-space analysis (§4.5 e5) and the J-space cross-check
    (§4.4 j3).  Note that with tied embeddings this is the same tensor as the
    input embedding, which is worth knowing when interpreting the SVD.
    """
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if head is None:
        raise RuntimeError("Could not find the output head (lm_head / embed_out).")
    return head.weight


def get_final_norm(model: nn.Module) -> nn.Module | None:
    """
    Return the final normalisation layer before the unembedding, if present.

    This matters for e5/e6: the super weight's effect on logits is mediated by
    this norm, so 'the SW changes the residual norm, which rescales every
    logit' is a mechanism that only exists because this layer is here.
    """
    for path in ("model.norm", "gpt_neox.final_layer_norm", "transformer.ln_f",
                 "model.decoder.final_layer_norm"):
        module = _resolve(model, path)
        if module is not None:
            return module
    return None


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A loaded model plus the handles the experiments need."""

    model: nn.Module
    tokenizer: Any
    spec: ModelSpec
    device: torch.device
    dtype: torch.dtype

    revision: str | None = None
    """Which HuggingFace revision this is. For Pythia, the training checkpoint
    ("step3000"). Carried on the object so that a §4.6 sweep over checkpoints
    can never mislabel which one produced a number."""

    @property
    def label(self) -> str:
        """`key` or `key@step3000` -- what to print in a sweep table."""
        return f"{self.spec.key}@{self.revision}" if self.revision else self.spec.key

    @property
    def n_layers(self) -> int:
        return len(get_decoder_layers(self.model))

    def down_proj(self, layer_idx: int) -> nn.Module:
        return get_down_proj(self.model, layer_idx)

    def read_super_weight(self, sw: SuperWeight) -> float:
        """The current scalar value of one super weight."""
        module = self.down_proj(sw.layer)
        weight, _ = down_proj_weight_view(module)
        i, j = sw_index(module, sw)
        return float(weight[i, j].item())


def load_model(
    spec: ModelSpec,
    device: str = "auto",
    dtype: str = "bfloat16",
    allow_fp16: bool = False,
    device_map: str | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> LoadedModel:
    """
    Load a model at full precision, with guards.

    Args:
        spec: which model.
        device: "auto" | "cuda" | "cpu".
        dtype: "bfloat16" | "float32" | "float16" (float16 requires allow_fp16).
        allow_fp16: escape hatch for Turing cards with no bf16. Sets fp16 and
            logs a prominent warning; all metric arithmetic is float64 anyway.
        device_map: pass "auto" to shard across GPU+CPU for models that do not
            fit (e.g. 13B on a 24 GB card). Slow but workable for the ~500-prompt
            studies here.
        trust_remote_code: needed by a few HF repos (Phi-3 historically).
        revision: HuggingFace branch/tag/commit. For Pythia this selects a
            TRAINING CHECKPOINT ("step0" ... "step143000"), which is the whole
            basis of §4.6 -- see `core.checkpoints`. Note the tokenizer is
            deliberately loaded from the DEFAULT revision: Pythia's tokenizer is
            fixed for the entire run, and old checkpoint branches do not all
            carry a complete tokenizer directory.

    Raises:
        ValueError: on any request that would quantise the model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    if dtype not in dtype_map:
        raise ValueError(
            f"dtype must be one of {sorted(dtype_map)}. "
            "Quantised dtypes (int8, int4, nf4) are deliberately unsupported -- "
            "see the module docstring."
        )
    if dtype == "float16" and not allow_fp16:
        raise ValueError(
            "float16 requires allow_fp16=True. Prefer bfloat16 on Ampere or "
            "newer. Only use fp16 on Turing cards (Kaggle T4/P100) that lack "
            "bf16 support."
        )

    torch_dtype = dtype_map[dtype]

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    token = _get_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=trust_remote_code, token=token
    )
    if tokenizer.pad_token is None:
        # Most causal LMs here have no pad token; reusing EOS is standard and
        # harmless because we always pass an explicit attention mask.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=token,
        device_map=device_map,          # None => load to one device below
        low_cpu_mem_usage=True,
        revision=revision,              # None => default branch
    )
    if device_map is None:
        model = model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)  # experiments that need grads re-enable locally

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        device=resolved_device,
        dtype=torch_dtype,
        revision=revision,
    )


def free(loaded: LoadedModel | None) -> None:
    """
    Release a model and empty the CUDA cache.

    §4.6 loads up to 154 checkpoints in one process, so leaking a single model
    per iteration would exhaust VRAM well before the sweep finished. Every
    formation experiment calls this in a `finally`.

    Sets the attribute to None rather than deleting it: `del loaded.model`
    leaves the dataclass in a state where any later attribute access raises
    `AttributeError` from somewhere unrelated, which is a miserable thing to
    debug 90 checkpoints into a sweep. Accepts None and is idempotent, so it is
    safe in a `finally` that may run before the load succeeded.
    """
    if loaded is None:
        return
    loaded.model = None  # type: ignore[assignment]
    loaded.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_fits(spec: ModelSpec, headroom_gb: float = 3.0) -> tuple[bool, str]:
    """
    Cheap pre-flight VRAM check, so a 20-minute download does not end in OOM.

    Returns (fits, message). Weights-only estimate plus a fixed headroom for
    activations; short sequences and small batches keep activation memory small
    relative to weights at these scales.
    """
    if not torch.cuda.is_available():
        return True, "No CUDA device; will run on CPU (slow but unbounded by VRAM)."
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    needed = spec.approx_bf16_gb + headroom_gb
    if needed <= total:
        return True, f"{spec.approx_bf16_gb:.1f} GB weights + {headroom_gb:.1f} GB headroom <= {total:.1f} GB VRAM."
    return False, (
        f"{spec.hf_id} needs ~{needed:.1f} GB but the GPU has {total:.1f} GB. "
        "Pass --device-map auto for CPU offload, or pick a smaller model."
    )


# -----------------------------------------------------------------------------
# Interventions
# -----------------------------------------------------------------------------
# All interventions are context managers that restore exact original state on
# exit.  Every experiment compares an intervened forward pass against a clean
# one, so leaking state between them would silently corrupt every result.


@contextlib.contextmanager
def scale_super_weights(
    loaded: LoadedModel,
    super_weights: Sequence[SuperWeight],
    factor: float,
) -> Iterator[dict[str, float]]:
    """
    Multiply each super weight by `factor`, restoring originals on exit.

    factor=0.0 is the paper's "Prune SW" ablation.
    factor=1.0 is a no-op, useful as a control path through identical code.
    factor>1 is the paper's amplification experiment (Figure 6).

    Yields a dict of original scalar values, for the log.
    """
    originals: list[tuple[nn.Module, int, int, torch.Tensor]] = []
    recorded: dict[str, float] = {}

    try:
        for sw in super_weights:
            module = loaded.down_proj(sw.layer)
            weight, _ = down_proj_weight_view(module)
            i, j = sw_index(module, sw)
            original = weight[i, j].detach().clone()
            originals.append((module, i, j, original))
            recorded[str(sw)] = float(original.item())
            with torch.no_grad():
                weight[i, j] = original * factor
        yield recorded
    finally:
        with torch.no_grad():
            for module, i, j, original in originals:
                weight, _ = down_proj_weight_view(module)
                weight[i, j] = original


@contextlib.contextmanager
def ablate_super_weights(
    loaded: LoadedModel, super_weights: Sequence[SuperWeight]
) -> Iterator[dict[str, float]]:
    """Zero the super weights (the paper's 'Prune SW'). Sugar for factor=0."""
    with scale_super_weights(loaded, super_weights, 0.0) as originals:
        yield originals


@contextlib.contextmanager
def inject_residual(
    loaded: LoadedModel,
    layer_idx: int,
    channel: int,
    value: float,
    positions: Sequence[int] | None = None,
) -> Iterator[None]:
    """
    Force one residual-stream channel to a fixed value at the output of a layer.

    This is how we restore a super activation after ablating the super weight
    (the paper's 'Prune SW, +SA' condition, Table 1), and how e6 separates the
    'model needs a big constant' hypothesis from 'model needs THIS constant'.

    Args:
        layer_idx: decoder layer whose OUTPUT is patched.
        channel: index into d_model.
        value: the scalar to write.
        positions: token positions to patch; None means position 0 only, which
            is where the paper observes the super activation. Pass an empty
            tuple to patch every position.
    """
    block = get_decoder_layers(loaded.model)[layer_idx]
    target_positions = (0,) if positions is None else tuple(positions)

    def hook(_module, _inputs, output):
        # Decoder blocks return either a Tensor or a tuple whose first element
        # is the hidden state; handle both without assuming.
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        if target_positions:
            for pos in target_positions:
                hidden[:, pos, channel] = value
        else:
            hidden[:, :, channel] = value
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def inject_residual_vector(
    loaded: LoadedModel,
    layer_idx: int,
    vector: torch.Tensor,
    positions: Sequence[int] | None = None,
) -> Iterator[None]:
    """
    Add an arbitrary vector to the residual stream at a layer's output.

    Used by e6 to inject a matched-norm RANDOM direction, which isolates
    "the model needs a large-norm residual component" from "the model needs
    this specific channel".
    """
    block = get_decoder_layers(loaded.model)[layer_idx]
    target_positions = (0,) if positions is None else tuple(positions)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        delta = vector.to(device=hidden.device, dtype=hidden.dtype)
        if target_positions:
            for pos in target_positions:
                hidden[:, pos, :] = hidden[:, pos, :] + delta
        else:
            hidden = hidden + delta
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# -----------------------------------------------------------------------------
# Forward helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def final_logits(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Next-token logits at the last real (non-pad) position of each sequence.

    Returns [batch, vocab] in float32. Left as float32 rather than float64 to
    keep GPU memory sane; metrics.py promotes to float64 for entropy and KL.

    Note on padding: we index the last position where attention_mask == 1 for
    each row individually, so right-padded batches give the same answer as
    unbatched forward passes.
    """
    out = loaded.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [batch, seq, vocab]
    last_idx = attention_mask.sum(dim=1) - 1  # [batch]
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, last_idx, :].float()


@torch.no_grad()
def hidden_states(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    All residual-stream states: a tuple of (n_layers + 1) tensors [batch, seq, d].

    Index 0 is the embedding output; index i+1 is the output of decoder layer i.
    So the super activation created by a super weight in layer L first appears
    at `hidden_states[L + 1]`.
    """
    out = loaded.model(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    return out.hidden_states


@torch.no_grad()
def read_super_activation(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    channel: int,
    position: int = 0,
) -> float:
    """
    The super activation value: residual channel `channel` at `position`,
    immediately after the layer containing the super weight.

    The paper's claim is that this is constant across prompts. e6 and the
    detection code both check that claim rather than assuming it.
    """
    states = hidden_states(loaded, input_ids, attention_mask)
    return float(states[layer_idx + 1][:, position, channel].mean().item())
