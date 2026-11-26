"""
train_activation_twins -- Controlled pretraining of 124M-parameter Transformers
with 6 different MLP activation blocks, all parameter-matched.

------------------------------------------------------------------------------
PURPOSE
------------------------------------------------------------------------------
This is Pillar 2 of the gated-vs-non-gated experiment: a FROM-SCRATCH training
grid on a single RTX 3080 (10-12 GB) that directly tests whether the
multiplicative gating mechanism is what causes super weight formation, by
training architecturally identical models that differ ONLY in their MLP block.

The 6 variants are:

    GATED (d_ff = 2048, 3 weight matrices, 4,718,592 MLP params/layer):
        m1_swiglu   SwiGLU  (sigma = SiLU)
        m2_geglu    GeGLU   (sigma = GELU)
        m3_reglu    ReGLU   (sigma = ReLU)

    NON-GATED (d_ff = 3072, 2 weight matrices, 4,718,592 MLP params/layer):
        m4_gelu_std Standard GELU
        m5_silu_std Standard SiLU (non-gated, to separate SiLU from gating)
        m6_relu2_std ReLU^2 (the decisive test: polynomial O(z^2) without gating)

All models share: 12 layers, d_model=768, 12 heads, head_dim=64, ctx=1024,
Pre-RMSNorm, RoPE, tied embeddings, AdamW (lr=6e-4, cosine to 6e-5).

The VRAM budget is ~4.5-5.5 GB at bf16 with batch size 8, well within RTX 3080.

------------------------------------------------------------------------------
TELEMETRY
------------------------------------------------------------------------------
At configurable checkpoint intervals, the training loop records:

  1. Top-K weight magnitudes & outlier ratio (|W_max| / mean|W|)
  2. Dominance score D(t) for the top candidate
  3. Activation spike vs weight magnitude (lag dynamics)
  4. Adam optimizer moments (m_t, v_t) for the top candidate
  5. Kurtosis of weight and activation tensors

These are saved to a per-variant JSONL file and can be analysed/plotted
with `plot_trajectories.py`.

------------------------------------------------------------------------------
RUNNING
------------------------------------------------------------------------------
    # Single variant, 100-step sanity check:
    swx train-twins --variant swiglu --max-steps 100 --batch-size 8 --log-every 10

    # Full 6-variant grid:
    swx train-twins --max-steps 50000 --batch-size 8 --log-every 100

    # Resume from checkpoint:
    swx train-twins --variant swiglu --resume runs/training_twins/swiglu_*/
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

log = logging.getLogger(__name__)

from ..core.detection import DEFAULT_DETECTION_PROMPT, TRAINING_TWINS_PROMPT


# =============================================================================
# Opt-in causal interventions
# =============================================================================

Coordinate = tuple[int, int, int]


@dataclass(frozen=True)
class InterventionConfig:
    """A deliberately narrow, opt-in training intervention.

    ``None`` remains the default at the :class:`TrainConfig` level.  This is
    important: ordinary training must not acquire intervention fields, extra
    state, or altered numerical behaviour merely because this framework exists.
    """

    kind: Literal[
        "activation_bound", "inoculation", "weight_projection", "adam_update_cap"
    ]
    scope_layers: tuple[int, ...]
    start_step: int = 0
    end_step: int | None = None
    target_coordinate: Coordinate | None = None
    tracked_coordinates: tuple[Coordinate, ...] = ()

    # Activation-bound condition.
    activation_mode: Literal["hard", "soft"] | None = None
    activation_threshold: float | None = None

    # Inoculation condition.  The change is additive and normalised by the
    # target tensor's pre-event mean absolute weight.
    inoculation_delta_scale: float | None = None
    inoculation_sign: Literal[-1, 1] = 1
    state_policy: Literal["preserve", "reset"] = "preserve"

    # Fixed absolute caps.  They must be chosen from a pre-registered pilot or
    # baseline calibration rather than being a moving function of the tensor.
    weight_cap: float | None = None
    update_cap: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterventionConfig":
        """Parse a JSON condition manifest without accepting ambiguous shapes."""
        data = dict(payload)
        for key in ("scope_layers",):
            if key in data:
                data[key] = tuple(int(v) for v in data[key])
        for key in ("target_coordinate",):
            if data.get(key) is not None:
                data[key] = tuple(int(v) for v in data[key])
        if "tracked_coordinates" in data:
            data["tracked_coordinates"] = tuple(
                tuple(int(v) for v in coord) for coord in data["tracked_coordinates"]
            )
        return cls(**data)


@dataclass
class _InterventionMetrics:
    """Per-forward or per-step statistics, kept off normal telemetry."""

    calls: int = 0
    values_seen: int = 0
    values_modified: int = 0
    pre_max_abs: float = 0.0
    post_max_abs: float = 0.0
    parameters_modified: int = 0
    raw_update_l2: float = 0.0
    effective_update_l2: float = 0.0
    cap: float | None = None

    def as_dict(self, kind: str) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": kind, "active": True}
        if self.values_seen:
            result.update({
                "calls": self.calls,
                "values_seen": self.values_seen,
                "values_modified": self.values_modified,
                "fraction_modified": self.values_modified / self.values_seen,
                "pre_max_abs": self.pre_max_abs,
                "post_max_abs": self.post_max_abs,
            })
        if kind in {"weight_projection", "adam_update_cap"}:
            result.update({
                "parameters_modified": self.parameters_modified,
                "raw_update_l2": self.raw_update_l2,
                "effective_update_l2": self.effective_update_l2,
                "pre_max_abs": self.pre_max_abs,
                "post_max_abs": self.post_max_abs,
            })
        if self.cap is not None:
            result["cap"] = self.cap
        return result


class TrainingInterventionController:
    """Owns intervention timing, state, and conditional telemetry.

    The controller never changes a run unless a concrete
    :class:`InterventionConfig` is supplied.  It is intentionally restricted to
    down-projection inputs and weights so condition scope is auditable.
    """

    def __init__(self, config: InterventionConfig):
        self.config = config
        self.current_step = -1
        self.inoculation_applied = False
        self._metrics = _InterventionMetrics()
        self._pending_events: list[dict[str, Any]] = []
        self._pre_step_weights: dict[int, Tensor] = {}

    def validate(self, model: "TwinTransformer") -> None:
        cfg = self.config
        if not cfg.scope_layers:
            raise ValueError("Intervention scope_layers must be explicit and non-empty.")
        if any(layer < 0 or layer >= model.n_layers for layer in cfg.scope_layers):
            raise ValueError(f"Intervention layers {cfg.scope_layers} exceed 0..{model.n_layers - 1}.")
        if cfg.end_step is not None and cfg.end_step < cfg.start_step:
            raise ValueError("end_step must be >= start_step.")
        if cfg.inoculation_sign not in {-1, 1}:
            raise ValueError("inoculation_sign must be either -1 or 1.")
        if cfg.kind == "activation_bound":
            if cfg.activation_mode not in {"hard", "soft"} or not (cfg.activation_threshold and cfg.activation_threshold > 0):
                raise ValueError("activation_bound needs mode hard/soft and a positive threshold.")
            if any(not isinstance(model.layers[layer].mlp, GatedMLP) for layer in cfg.scope_layers):
                raise ValueError("activation_bound is currently defined only for gated MLP variants.")
        if cfg.kind == "inoculation":
            if cfg.target_coordinate is None or not (cfg.inoculation_delta_scale and cfg.inoculation_delta_scale > 0):
                raise ValueError("inoculation needs target_coordinate and positive inoculation_delta_scale.")
        if cfg.kind == "weight_projection" and not (cfg.weight_cap and cfg.weight_cap > 0):
            raise ValueError("weight_projection needs a positive fixed weight_cap.")
        if cfg.kind == "adam_update_cap" and not (cfg.update_cap and cfg.update_cap > 0):
            raise ValueError("adam_update_cap needs a positive absolute update_cap.")
        for coordinate in self.fixed_coordinates():
            if len(coordinate) != 3:
                raise ValueError(f"Tracked coordinate must have exactly three indices, got {coordinate}.")
            layer, j, k = coordinate
            if layer < 0 or layer >= model.n_layers:
                raise ValueError(f"Tracked coordinate {coordinate} has invalid layer.")
            weight = model.layers[layer].mlp.down_proj.weight
            if not (0 <= j < weight.shape[0] and 0 <= k < weight.shape[1]):
                raise ValueError(f"Tracked coordinate {coordinate} is outside down_proj shape {tuple(weight.shape)}.")
        if cfg.target_coordinate is not None and cfg.target_coordinate[0] not in cfg.scope_layers:
            raise ValueError("The intervention target layer must be included in scope_layers.")

    def fixed_coordinates(self) -> tuple[Coordinate, ...]:
        coordinates = list(self.config.tracked_coordinates)
        if self.config.target_coordinate is not None:
            coordinates.append(self.config.target_coordinate)
        return tuple(dict.fromkeys(coordinates))

    def set_step(self, step: int) -> None:
        self.current_step = step
        self._metrics = _InterventionMetrics()

    def is_active(self, step: int | None = None) -> bool:
        candidate = self.current_step if step is None else step
        if candidate < self.config.start_step:
            return False
        return self.config.end_step is None or candidate <= self.config.end_step

    def apply_down_proj_input(self, layer: int, x: Tensor) -> Tensor:
        """Bound a GatedMLP hidden input while capturing real binding statistics."""
        cfg = self.config
        if cfg.kind != "activation_bound" or layer not in cfg.scope_layers or not self.is_active():
            return x
        assert cfg.activation_threshold is not None
        threshold = cfg.activation_threshold
        if cfg.activation_mode == "hard":
            bounded = x.clamp(-threshold, threshold)
        else:
            bounded = threshold * torch.tanh(x / threshold)
        with torch.no_grad():
            self._metrics.calls += 1
            self._metrics.values_seen += x.numel()
            self._metrics.values_modified += int((bounded != x).sum().item())
            self._metrics.pre_max_abs = max(self._metrics.pre_max_abs, float(x.detach().abs().max()))
            self._metrics.post_max_abs = max(self._metrics.post_max_abs, float(bounded.detach().abs().max()))
        return bounded

    def before_optimizer_step(self, model: "TwinTransformer") -> None:
        """Snapshot only scoped parameters when an actual-delta cap is requested."""
        self._pre_step_weights = {}
        if self.config.kind != "adam_update_cap" or not self.is_active():
            return
        for layer in self.config.scope_layers:
            self._pre_step_weights[layer] = model.layers[layer].mlp.down_proj.weight.detach().clone()

    @torch.no_grad()
    def after_optimizer_step(self, model: "TwinTransformer", optimizer: torch.optim.Optimizer, step: int) -> None:
        """Apply all post-step interventions at one documented boundary."""
        self.current_step = step
        cfg = self.config
        if cfg.kind == "inoculation" and step == cfg.start_step and not self.inoculation_applied:
            assert cfg.target_coordinate is not None and cfg.inoculation_delta_scale is not None
            layer, j, k = cfg.target_coordinate
            weight = model.layers[layer].mlp.down_proj.weight
            tensor_scale = float(weight.detach().abs().mean())
            old_value = float(weight[j, k])
            delta = cfg.inoculation_sign * cfg.inoculation_delta_scale * tensor_scale
            weight[j, k].add_(delta)
            if cfg.state_policy == "reset":
                state = optimizer.state.get(weight, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key][j, k] = 0
            self.inoculation_applied = True
            self._pending_events.append({
                "step": step,
                "kind": "inoculation",
                "layer": layer,
                "j": j,
                "k": k,
                "tensor_mean_abs_before": tensor_scale,
                "old_value": old_value,
                "delta": delta,
                "new_value": float(weight[j, k]),
                "state_policy": cfg.state_policy,
            })
        elif cfg.kind == "weight_projection" and self.is_active():
            assert cfg.weight_cap is not None
            self._metrics.cap = cfg.weight_cap
            for layer in cfg.scope_layers:
                weight = model.layers[layer].mlp.down_proj.weight
                before = weight.detach().clone()
                weight.clamp_(-cfg.weight_cap, cfg.weight_cap)
                changed = before != weight
                self._metrics.parameters_modified += int(changed.sum().item())
                self._metrics.pre_max_abs = max(self._metrics.pre_max_abs, float(before.abs().max()))
                self._metrics.post_max_abs = max(self._metrics.post_max_abs, float(weight.detach().abs().max()))
        elif cfg.kind == "adam_update_cap" and self.is_active():
            assert cfg.update_cap is not None
            self._metrics.cap = cfg.update_cap
            for layer, before in self._pre_step_weights.items():
                weight = model.layers[layer].mlp.down_proj.weight
                raw_delta = weight - before
                capped_delta = raw_delta.clamp(-cfg.update_cap, cfg.update_cap)
                changed = capped_delta != raw_delta
                weight.copy_(before + capped_delta)
                self._metrics.parameters_modified += int(changed.sum().item())
                self._metrics.raw_update_l2 += float(raw_delta.float().pow(2).sum())
                self._metrics.effective_update_l2 += float(capped_delta.float().pow(2).sum())
                self._metrics.pre_max_abs = max(self._metrics.pre_max_abs, float(raw_delta.abs().max()))
                self._metrics.post_max_abs = max(self._metrics.post_max_abs, float(capped_delta.abs().max()))
            self._metrics.raw_update_l2 = math.sqrt(self._metrics.raw_update_l2)
            self._metrics.effective_update_l2 = math.sqrt(self._metrics.effective_update_l2)
        self._pre_step_weights = {}

    def snapshot_metrics(self) -> dict[str, Any] | None:
        """Return an optional telemetry object only while a continuous condition is active."""
        if self.config.kind in {"activation_bound", "weight_projection", "adam_update_cap"} and self.is_active():
            return self._metrics.as_dict(self.config.kind)
        return None

    def pop_events(self) -> list[dict[str, Any]]:
        events, self._pending_events = self._pending_events, []
        return events

    def state_dict(self) -> dict[str, Any]:
        return {"inoculation_applied": self.inoculation_applied, "current_step": self.current_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.inoculation_applied = bool(state.get("inoculation_applied", False))
        self.current_step = int(state.get("current_step", -1))


# =============================================================================
# MLP blocks -- the independent variable
# =============================================================================


class RMSNorm(nn.Module):
    """Pre-RMSNorm, shared across all variants."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # Multiply weight in float32 for numerical accuracy, then cast back
        # to input dtype so downstream tensors stay in bf16 under autocast.
        # This prevents float32 leakage that would disable FlashAttention.
        return ((x.float() * norm) * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE positional embedding."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Cast to input dtype so q/k stay in bf16 and FlashAttention can engage
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + _rotate_half(x) * sin


# -- Gated MLP blocks --------------------------------------------------------

class GatedMLP(nn.Module):
    """Three-projection gated MLP: down(sigma(gate(x)) * up(x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",
        layer_idx: int = -1,
        intervention: TrainingInterventionController | None = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self._act = {
            "silu": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
        }[activation]
        self.layer_idx = layer_idx
        self.intervention = intervention

    def forward(self, x: Tensor) -> Tensor:
        hidden = self._act(self.gate_proj(x)) * self.up_proj(x)
        if self.intervention is not None:
            hidden = self.intervention.apply_down_proj_input(self.layer_idx, hidden)
        return self.down_proj(hidden)


# -- Non-gated MLP blocks ----------------------------------------------------

class StandardMLP(nn.Module):
    """Two-projection non-gated MLP: down(sigma(up(x)))."""

    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self._act_name = activation

    def forward(self, x: Tensor) -> Tensor:
        h = self.up_proj(x)
        if self._act_name == "gelu":
            h = F.gelu(h)
        elif self._act_name == "silu":
            h = F.silu(h)
        elif self._act_name == "relu2":
            h = F.relu(h).pow(2)
        else:
            raise ValueError(f"Unknown activation: {self._act_name}")
        return self.down_proj(h)


# =============================================================================
# Transformer block and model
# =============================================================================


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_factory,
        layer_idx: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope, dropout)
        self.mlp = mlp_factory(layer_idx)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope
        self.dropout = dropout

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Use scaled_dot_product_attention for efficiency (FlashAttention when available)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class TwinTransformer(nn.Module):
    """A minimal GPT-style decoder for the activation-twin experiment."""

    def __init__(
        self,
        variant: str,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        intervention: TrainingInterventionController | None = None,
    ):
        super().__init__()
        self.variant = variant
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len)

        mlp_factory = _make_mlp_factory(variant, d_model, intervention)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_factory, layer_idx, self.rope, dropout)
            for layer_idx in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        # Tie embeddings: the lm_head shares weight with tok_emb
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Standard GPT-2 / nanoGPT initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Variant configuration
# =============================================================================


VARIANTS: dict[str, dict[str, Any]] = {
    "m1_swiglu": {"mlp_type": "gated", "activation": "silu", "d_ff": 2048},
    "m2_geglu": {"mlp_type": "gated", "activation": "gelu", "d_ff": 2048},
    "m3_reglu": {"mlp_type": "gated", "activation": "relu", "d_ff": 2048},
    "m4_gelu_std": {"mlp_type": "standard", "activation": "gelu", "d_ff": 3072},
    "m5_silu_std": {"mlp_type": "standard", "activation": "silu", "d_ff": 3072},
    "m6_relu2_std": {"mlp_type": "standard", "activation": "relu2", "d_ff": 3072},
}

# Shorthand aliases for CLI convenience
VARIANT_ALIASES: dict[str, str] = {
    "swiglu": "m1_swiglu",
    "geglu": "m2_geglu",
    "reglu": "m3_reglu",
    "gelu_std": "m4_gelu_std",
    "silu_std": "m5_silu_std",
    "relu2_std": "m6_relu2_std",
}

# Standard aliases for common pretraining datasets and default configurations
DATASET_ALIASES: dict[str, tuple[str, str]] = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "HuggingFaceFW/fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "c4": ("allenai/c4", "en"),
    "allenai/c4": ("allenai/c4", "en"),
    "slimpajama": ("DKYoon/SlimPajama-6B", None),
    "slimpajama-6b": ("DKYoon/SlimPajama-6B", None),
    "slimpajama_6b": ("DKYoon/SlimPajama-6B", None),
    "SlimPajama-6B": ("DKYoon/SlimPajama-6B", None),
    "DKYoon/SlimPajama-6B": ("DKYoon/SlimPajama-6B", None),
    "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "Salesforce/wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
}


def resolve_dataset(dataset: str, dataset_config: str | None = None) -> tuple[str, str | None]:
    """Resolve user-friendly dataset aliases to canonical HF repo IDs and configs."""
    if dataset in DATASET_ALIASES:
        canonical_name, default_cfg = DATASET_ALIASES[dataset]
        return canonical_name, dataset_config or default_cfg
    return dataset, dataset_config


def resolve_variant(name: str) -> str:
    """Resolve a variant name or alias to its canonical key."""
    if name in VARIANTS:
        return name
    if name in VARIANT_ALIASES:
        return VARIANT_ALIASES[name]
    raise ValueError(
        f"Unknown variant {name!r}. Available: "
        f"{', '.join(sorted(VARIANTS))} (or aliases: {', '.join(sorted(VARIANT_ALIASES))})"
    )


def _make_mlp_factory(
    variant: str,
    d_model: int,
    intervention: TrainingInterventionController | None = None,
):
    """Return a zero-argument callable that creates the MLP for this variant."""
    key = resolve_variant(variant)
    cfg = VARIANTS[key]
    d_ff = cfg["d_ff"]
    if cfg["mlp_type"] == "gated":
        return lambda layer_idx: GatedMLP(
            d_model, d_ff, cfg["activation"], layer_idx=layer_idx, intervention=intervention,
        )
    else:
        return lambda _layer_idx: StandardMLP(d_model, d_ff, cfg["activation"])


# =============================================================================
# Telemetry
# =============================================================================


@dataclass
class TelemetrySnapshot:
    """One checkpoint's worth of telemetry."""
    step: int
    tokens_seen: int
    loss: float
    lr: float
    wall_seconds: float

    # Per-layer down_proj telemetry
    per_layer: list[dict[str, Any]] = field(default_factory=list)

    # Model-wide weight-side summary (argmax|W| across all layers)
    top_weight_ratio: float = 0.0
    top_weight_layer: int = -1
    top_weight_value: float = 0.0
    weight_kurtosis_max: float = 0.0

    # Model-wide activation-side summary (detection-faithful, from output spike)
    top_dominance: float = 0.0           # max D(t) across layers (scalar ratio)
    top_dominance_layer: int = -1
    top_dominance_j: int = -1            # residual channel at the dominant layer
    top_dominance_k: int = -1            # mlp channel at the dominant layer
    top_input_spike: float = 0.0         # mean |X_ik| at dominant k, over T
    top_peak_output: float = 0.0         # |Y_ij| = |sum_k X_ik W_jk| at spike position
    top_dominant_weight: float = 0.0     # W[j*,k*] at the dominant coordinate
    activation_kurtosis_max: float = 0.0

    # Top-K coordinate watchlist (persistent tracking across steps)
    watched_coords: list[dict[str, Any]] = field(default_factory=list)

    # These fields are deliberately optional.  Baseline telemetry remains
    # schema-compatible and carries neither empty intervention fields nor
    # coordinate traces that were not requested.
    fixed_coord_traces: list[dict[str, Any]] = field(default_factory=list)
    intervention: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loss": self.loss,
            "lr": self.lr,
            "wall_seconds": self.wall_seconds,
            "top_weight_ratio": self.top_weight_ratio,
            "top_weight_layer": self.top_weight_layer,
            "top_weight_value": self.top_weight_value,
            "weight_kurtosis_max": self.weight_kurtosis_max,
            "top_dominance": self.top_dominance,
            "top_dominance_layer": self.top_dominance_layer,
            "top_dominance_j": self.top_dominance_j,
            "top_dominance_k": self.top_dominance_k,
            "top_input_spike": self.top_input_spike,
            "top_peak_output": self.top_peak_output,
            "top_dominant_weight": self.top_dominant_weight,
            "activation_kurtosis_max": self.activation_kurtosis_max,
            "per_layer": self.per_layer,
            "watched_coords": self.watched_coords,
        }
        if self.fixed_coord_traces:
            payload["fixed_coord_traces"] = self.fixed_coord_traces
        if self.intervention is not None:
            payload["intervention"] = self.intervention
        return payload


#: Default number of top candidates to track per layer for trajectory continuity.
TOP_K_WATCH = 5


@torch.no_grad()
def collect_telemetry(
    model: TwinTransformer,
    step: int,
    tokens_seen: int,
    loss: float,
    lr: float,
    wall_seconds: float,
    optimizer: torch.optim.Optimizer | None = None,
    probe_input: Tensor | None = None,
    watchlist: dict[int, set[tuple[int, int]]] | None = None,
    intervention: TrainingInterventionController | None = None,
    fixed_coordinates: Sequence[Coordinate] = (),
) -> TelemetrySnapshot:
    """
    Collect telemetry metrics across all layers' down_proj.

    Weight-side metrics (outlier ratio, kurtosis, Adam moments) are always
    collected from the static weight tensors.

    Activation-side metrics (dominance D(t), spike magnitudes, activation
    kurtosis) require a forward pass with hooks and are collected only when
    ``probe_input`` is provided.  The probe MUST be a fixed, deterministic
    tensor (e.g. the tokenised detection prompt) -- not a random training batch
    -- so that activation statistics are comparable across training steps.

    The dominance formula faithfully mirrors ``core.detection``:
        D(t) = |X_ik* · W_j*k*| / |Σ_k X_ik · W_j*k|
    where (j*, k*) is found by following the output spike peak, not argmax|W|.

    ``watchlist`` is an optional per-layer set of (j, k) coordinates whose
    weight values and Adam moments are recorded at every snapshot to ensure
    continuous trajectories.  New Top-K candidates discovered at each snapshot
    are added to the watchlist automatically.
    """
    snap = TelemetrySnapshot(
        step=step, tokens_seen=tokens_seen, loss=loss, lr=lr,
        wall_seconds=wall_seconds,
    )
    if watchlist is None:
        watchlist = {i: set() for i in range(model.n_layers)}
    for layer, j, k in fixed_coordinates:
        watchlist.setdefault(layer, set()).add((j, k))

    # -------------------------------------------------------------------------
    # Weight-side metrics: walk the static weight tensors.
    # -------------------------------------------------------------------------
    best_ratio = 0.0
    best_layer = -1
    best_value = 0.0
    max_weight_kurtosis = 0.0

    for layer_idx, block in enumerate(model.layers):
        dp = block.mlp.down_proj
        w = dp.weight.detach().float()

        # Top-K weight magnitudes & outlier ratio
        abs_w = w.abs()
        mean_abs = float(abs_w.mean())
        max_abs = float(abs_w.max())
        ratio = max_abs / mean_abs if mean_abs > 1e-12 else 0.0
        flat_idx = int(abs_w.argmax())
        max_row, max_col = divmod(flat_idx, w.shape[1])

        # Weight kurtosis
        w_flat = w.reshape(-1)
        mu = w_flat.mean()
        sigma = w_flat.std(unbiased=False)
        weight_kurtosis = float(((w_flat - mu) / sigma).pow(4).mean()) if sigma > 1e-12 else 0.0

        layer_data: dict[str, Any] = {
            "layer": layer_idx,
            "max_weight_abs": max_abs,
            "mean_weight_abs": mean_abs,
            "weight_ratio": ratio,
            "max_weight_coord": [int(max_row), int(max_col)],
            "max_weight_value": float(w[max_row, max_col]),
            "weight_kurtosis": weight_kurtosis,
        }

        # Adam moments for the argmax|W| coordinate
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p is dp.weight:
                        state = optimizer.state.get(p, {})
                        if "exp_avg" in state:
                            m = state["exp_avg"].float()
                            v = state["exp_avg_sq"].float()
                            layer_data["adam_m1_at_max"] = float(m[max_row, max_col])
                            layer_data["adam_v_at_max"] = float(v[max_row, max_col])
                        break

        # --- Top-K watchlist: register top-K weight outliers ----------------
        # Find Top-K by |W| and add them to the persistent watchlist.
        topk_vals, topk_flat = abs_w.reshape(-1).topk(TOP_K_WATCH)
        topk_coords = []
        for fi in topk_flat.tolist():
            r, c = divmod(fi, w.shape[1])
            watchlist.setdefault(layer_idx, set()).add((r, c))
            topk_coords.append([r, c])
        layer_data["topk_weight_coords"] = topk_coords
        layer_data["topk_weight_values"] = [float(w[r, c]) for r, c in topk_coords]

        snap.per_layer.append(layer_data)

        if ratio > best_ratio:
            best_ratio = ratio
            best_layer = layer_idx
            best_value = float(w[max_row, max_col])
        max_weight_kurtosis = max(max_weight_kurtosis, weight_kurtosis)

    snap.top_weight_ratio = best_ratio
    snap.top_weight_layer = best_layer
    snap.top_weight_value = best_value
    snap.weight_kurtosis_max = max_weight_kurtosis

    # -------------------------------------------------------------------------
    # Activation-side metrics: detection-faithful forward pass.
    #
    # Mirrors core.detection.profile_down_proj + _dominant_mlp_channel:
    #   1. Find argmax_{pos, j} |Y[pos, j]|  →  the output spike
    #   2. Take x_row = X[pos*, :]           →  input row at that position
    #   3. terms = x_row * W[j*, :]          →  per-channel contributions
    #   4. k* = argmax_k |terms[k]|          →  dominant mlp channel
    #   5. D(t) = |terms[k*]| / |sum(terms)| →  scalar dominance ratio
    # -------------------------------------------------------------------------
    if probe_input is not None:
        _act_cache: dict[int, tuple[Tensor, Tensor]] = {}

        handles = []
        for layer_idx, block in enumerate(model.layers):
            def _make_hook(idx: int):
                def _hook(module, inp, out):
                    _act_cache[idx] = (
                        inp[0].detach().float(),
                        out.detach().float(),
                    )
                return _hook
            handles.append(block.mlp.down_proj.register_forward_hook(_make_hook(layer_idx)))

        was_training = model.training
        model.eval()
        try:
            probe = probe_input[:1]  # [1, T]
            _ = model(probe)
        finally:
            for h in handles:
                h.remove()
            if was_training:
                model.train()

        max_activation_kurtosis = 0.0
        top_dominance = 0.0
        top_dominance_layer = -1
        top_dominance_j = -1
        top_dominance_k = -1
        top_input_spike = 0.0
        top_peak_output = 0.0
        top_dominant_weight = 0.0

        for layer_idx, block in enumerate(model.layers):
            if layer_idx not in _act_cache:
                continue
            x_act, y_act = _act_cache[layer_idx]  # [1, T, d_ff], [1, T, d_model]

            # Step 1: find the peak output position and residual channel
            out_abs = y_act[0].abs()              # [T, d_model]
            out_peak_flat = int(out_abs.argmax())
            out_pos, out_j = divmod(out_peak_flat, y_act.shape[2])

            # Step 2: input row at that position
            x_row = x_act[0, out_pos, :]          # [d_ff]

            # Step 3: find the dominant mlp channel via X * W product
            w_row = block.mlp.down_proj.weight[out_j].detach().float()  # [d_ff]
            terms = x_row * w_row                  # [d_ff]
            dom_k = int(terms.abs().argmax())
            contribution = float(terms[dom_k])
            total_output = float(terms.sum())      # = Y[out_pos, out_j]

            # Step 4: scalar dominance ratio (no hard-clamp)
            dominance = abs(contribution) / abs(total_output) if abs(total_output) > 1e-12 else 0.0

            # Activation kurtosis of the full input to down_proj
            x_flat = x_act.reshape(-1)
            mu_x = x_flat.mean()
            sigma_x = x_flat.std(unbiased=False)
            act_kurtosis = float(((x_flat - mu_x) / sigma_x).pow(4).mean()) if sigma_x > 1e-12 else 0.0

            # Spike values at the correct coordinate
            mean_input_spike = float(x_act[0, :, dom_k].abs().mean())
            dominant_weight_val = float(w_row[dom_k])

            # Register the dynamic dominant coordinate into the watchlist too
            watchlist.setdefault(layer_idx, set()).add((out_j, dom_k))

            snap.per_layer[layer_idx].update({
                "dominant_j": int(out_j),
                "dominant_k": int(dom_k),
                "dominant_weight_value": dominant_weight_val,
                "mean_input_spike": mean_input_spike,
                "peak_output": float(abs(total_output)),
                "dominance": dominance,
                "activation_kurtosis": act_kurtosis,
            })

            max_activation_kurtosis = max(max_activation_kurtosis, act_kurtosis)
            if dominance > top_dominance:
                top_dominance = dominance
                top_dominance_layer = layer_idx
                top_dominance_j = int(out_j)
                top_dominance_k = int(dom_k)
                top_input_spike = mean_input_spike
                top_peak_output = float(abs(total_output))
                top_dominant_weight = dominant_weight_val

        snap.top_dominance = top_dominance
        snap.top_dominance_layer = top_dominance_layer
        snap.top_dominance_j = top_dominance_j
        snap.top_dominance_k = top_dominance_k
        snap.top_input_spike = top_input_spike
        snap.top_peak_output = top_peak_output
        snap.top_dominant_weight = top_dominant_weight
        snap.activation_kurtosis_max = max_activation_kurtosis

        # Fixed-coordinate traces are intentionally sparse and opt-in.  The
        # instantaneous dominant coordinate above may change between snapshots;
        # these records instead follow one immutable (layer, j, k) triple.
        for layer_idx, j, k in fixed_coordinates:
            if layer_idx not in _act_cache:
                continue
            x_act, _y_act = _act_cache[layer_idx]
            weight = model.layers[layer_idx].mlp.down_proj.weight.detach().float()
            channel = x_act[0, :, k]
            contribution = channel * weight[j, k]
            trace: dict[str, Any] = {
                "layer": layer_idx,
                "j": j,
                "k": k,
                "weight_value": float(weight[j, k]),
                "mean_abs_x": float(channel.abs().mean()),
                "max_abs_x": float(channel.abs().max()),
                "max_abs_contribution": float(contribution.abs().max()),
            }
            if optimizer is not None:
                state = optimizer.state.get(model.layers[layer_idx].mlp.down_proj.weight, {})
                if "exp_avg" in state:
                    trace["adam_m1"] = float(state["exp_avg"].float()[j, k])
                    trace["adam_v"] = float(state["exp_avg_sq"].float()[j, k])
            snap.fixed_coord_traces.append(trace)

    # -------------------------------------------------------------------------
    # Watchlist snapshot: record weight values (and Adam moments) for every
    # coordinate ever registered, ensuring continuous trajectories.
    # -------------------------------------------------------------------------
    watched_entries: list[dict[str, Any]] = []
    for layer_idx, block in enumerate(model.layers):
        coords = watchlist.get(layer_idx, set())
        if not coords:
            continue
        dp = block.mlp.down_proj
        w = dp.weight.detach().float()
        # Find Adam state for this layer's down_proj (if available)
        adam_m = adam_v = None
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p is dp.weight:
                        state = optimizer.state.get(p, {})
                        if "exp_avg" in state:
                            adam_m = state["exp_avg"].float()
                            adam_v = state["exp_avg_sq"].float()
                        break
        for j, k in sorted(coords):
            entry: dict[str, Any] = {
                "layer": layer_idx,
                "j": j,
                "k": k,
                "weight_value": float(w[j, k]),
            }
            if adam_m is not None:
                entry["adam_m1"] = float(adam_m[j, k])
                entry["adam_v"] = float(adam_v[j, k])
            watched_entries.append(entry)
    snap.watched_coords = watched_entries
    if intervention is not None:
        snap.intervention = intervention.snapshot_metrics()

    return snap


# =============================================================================
# Data loading -- deterministic, pre-tokenised pipeline
# =============================================================================


def _get_tokenizer():
    """Get a GPT-2 tokenizer for training."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_training_tokens(
    output_path: Path,
    num_tokens: int = 460_000_000,
    seed: int = 42,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str | None = None,
    split: str = "train",
    text_column: str = "text",
) -> Path:
    """
    Pre-tokenise a fixed slice of a dataset (default: FineWeb-Edu) into a
    contiguous uint16 binary file for 100% deterministic training.

    The default ``num_tokens`` (460M) is enough for a full 50,000-step run with
    ``batch_size=8`` and ``seq_len=1024`` plus 10% headroom::

        50_000 * 8 * 1025 * 1.1 ≈ 451.0M  -->  460M default

    At 460M tokens in uint16, the file is ~920 MB on disk.  Every training run
    that uses the same binary file sees the exact same sequence of tokens --
    across variants, across resumes, across machines.

    The binary layout is a flat array of uint16 token IDs.  The training loop
    indexes directly into it: step *s* with batch size *B* and sequence length
    *T* reads tokens ``[s*B*T : (s+1)*B*T]``.
    """
    from datasets import load_dataset

    output_path = Path(output_path)
    if output_path.exists():
        existing_tokens = output_path.stat().st_size // 2
        if existing_tokens >= num_tokens:
            log.info(
                "Token cache already exists: %s (%d tokens, need %d). Reusing.",
                output_path, existing_tokens, num_tokens,
            )
            return output_path
        log.info(
            "Token cache %s has only %d tokens (need %d). Regenerating.",
            output_path, existing_tokens, num_tokens,
        )
        try:
            output_path.unlink()
        except Exception:
            pass

    dataset_name, dataset_config = resolve_dataset(dataset_name, dataset_config)

    log.info(
        "Pre-tokenising %d tokens from %s (%s, split=%s, seed=%d) -> %s ...",
        num_tokens, dataset_name, dataset_config or "default", split, seed, output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = _get_tokenizer()
    load_kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
    }
    if dataset_config is not None:
        load_kwargs["name"] = dataset_config

    ds = load_dataset(dataset_name, **load_kwargs)
    try:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception as exc:
        log.warning("Could not shuffle streaming dataset %s: %s", dataset_name, exc)

    # Tokenise in chunks directly into the memory-mapped file (prevents 16+ GB RAM spike)
    mm = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(num_tokens,))
    offset = 0
    chunk_buffer: list[int] = []
    chunk_flush_size = 1_000_000  # Flush to disk every 1M tokens (~2MB RAM)

    epoch = 0
    while offset < num_tokens:
        epoch += 1
        epoch_tokens_before = offset
        for example in ds:
            text = example.get(text_column, "")
            if not text and text_column != "content":
                text = example.get("content", "")
            if not text:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue
            chunk_buffer.extend(tokens)

            while len(chunk_buffer) >= chunk_flush_size and offset < num_tokens:
                write_count = min(chunk_flush_size, num_tokens - offset)
                mm[offset : offset + write_count] = np.array(chunk_buffer[:write_count], dtype=np.uint16)
                offset += write_count
                chunk_buffer = chunk_buffer[write_count:]
                if offset % 10_000_000 == 0:
                    log.info("  ... %dM tokens written to disk so far (epoch %d)", offset // 1_000_000, epoch)

            if offset >= num_tokens:
                break

        # Flush any remaining tokens
        if chunk_buffer and offset < num_tokens:
            write_count = min(len(chunk_buffer), num_tokens - offset)
            mm[offset : offset + write_count] = np.array(chunk_buffer[:write_count], dtype=np.uint16)
            offset += write_count
            chunk_buffer = chunk_buffer[write_count:]

        if offset == epoch_tokens_before:
            log.warning(
                "Dataset %s exhausted after %d tokens (target was %d).",
                dataset_name, offset, num_tokens,
            )
            break
        elif offset < num_tokens:
            log.info(
                "Dataset %s reached end of split at %d tokens. Looping for epoch %d...",
                dataset_name, offset, epoch + 1,
            )
            ds = load_dataset(dataset_name, **load_kwargs)
            try:
                ds = ds.shuffle(seed=seed + epoch, buffer_size=10_000)
            except Exception:
                pass

    mm.flush()
    del mm

    # Truncate the file if fewer than num_tokens were written, avoiding silent zero-padding
    if offset < num_tokens:
        with open(output_path, "r+b") as f:
            f.truncate(offset * 2)
        log.warning(
            "Truncated %s to actual token count %d (%.1f MB) to prevent zero-padding.",
            output_path, offset, output_path.stat().st_size / 1e6,
        )
    else:
        log.info("Wrote %d tokens to %s (%.1f MB)", offset, output_path,
                 output_path.stat().st_size / 1e6)
    return output_path


class DeterministicTokenDataset:
    """
    Zero-overhead memory-mapped dataset for deterministic training.

    Every call to ``get_batch(step, batch_size)`` returns the exact same tokens
    regardless of when it is called, making training fully deterministic and
    resume-safe.  No iterator state, no shuffling, no network I/O.
    """

    def __init__(self, bin_path: Path, seq_len: int = 1024):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.total_tokens = len(self.data)

    def get_batch(
        self, step: int, batch_size: int, device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Fetch the exact deterministic batch for training step ``step``.

        Returns (input_ids, labels) each of shape [batch_size, seq_len].
        """
        tokens_per_sample = self.seq_len + 1
        tokens_per_step = batch_size * tokens_per_sample
        start = step * tokens_per_step
        end = start + tokens_per_step
        if end > self.total_tokens:
            raise RuntimeError(
                f"Step {step} requires tokens [{start}:{end}] but the token "
                f"cache only has {self.total_tokens}. Regenerate with more tokens."
            )
        chunk = torch.from_numpy(
            self.data[start:end].astype(np.int64)
        ).reshape(batch_size, tokens_per_sample)
        return chunk[:, :-1].to(device), chunk[:, 1:].to(device)


# Keep legacy loader for backward compat (not used in train() anymore)
def _load_dataset(tokenizer, seq_len: int = 1024, seed: int = 42):
    """Load FineWeb-Edu sample and tokenize into fixed-length chunks (LEGACY)."""
    from datasets import load_dataset as _ld

    log.info("Loading FineWeb-Edu sample (streaming)...")
    ds = _ld(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    return _TokenizedIterableDataset(ds, tokenizer, seq_len)


class _TokenizedIterableDataset(torch.utils.data.IterableDataset):
    """Streams text, tokenizes, and yields fixed-length chunks (LEGACY)."""

    def __init__(self, hf_dataset, tokenizer, seq_len: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        buffer: list[int] = []
        for example in self.hf_dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t[:-1], "labels": t[1:]}


# =============================================================================
# Training loop
# =============================================================================


@dataclass
class TrainConfig:
    """Training configuration."""
    variant: str = "m1_swiglu"
    max_steps: int = 50000
    batch_size: int = 8
    micro_batch_size: int = 2  # DataLoader batch; gradient accumulation = batch_size // micro_batch_size
    seq_len: int = 1024
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 100
    # Telemetry checkpoint schedule:
    # steps 0-2000: every telemetry_early_interval steps
    # steps 2000+: every telemetry_late_interval steps
    telemetry_early_interval: int = 100
    telemetry_late_interval: int = 500
    telemetry_early_cutoff: int = 2000
    save_every: int = 5000
    seed: int = 42
    data_seed: int = 42
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257
    device: str = "auto"
    dtype: str = "bfloat16"
    output_dir: str = ""
    resume: str | None = None
    token_cache_dir: str = ""  # directory for pre-tokenised data; defaults to runs/training_twins/data
    condition_name: str = ""
    intervention: InterventionConfig | None = None
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = None
    dataset_split: str = "train"
    dataset_text_column: str = "text"


def load_condition_manifest(path: str | Path) -> tuple[str, InterventionConfig | None]:
    """Load a small, explicit JSON condition manifest for ``train-twins``.

    Example::

        {"name": "A1_act_hard", "intervention": {"kind": "activation_bound", ...}}

    A manifest with no ``intervention`` is a named no-op control.  This keeps
    the ordinary CLI useful while giving causal conditions an auditable,
    versionable definition.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Condition manifest must be a JSON object.")
    name = str(payload.get("name") or source.stem)
    raw = payload.get("intervention")
    if raw is None:
        return name, None
    if not isinstance(raw, dict):
        raise ValueError("Condition manifest intervention must be an object.")
    return name, InterventionConfig.from_dict(raw)


def _cosine_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _should_telemetry(step: int, cfg: TrainConfig) -> bool:
    """Whether to collect telemetry at this step."""
    if step == 0:
        return True
    if step <= cfg.telemetry_early_cutoff:
        return step % cfg.telemetry_early_interval == 0
    return step % cfg.telemetry_late_interval == 0


def train(cfg: TrainConfig) -> Path:
    """Train one variant and return the output directory path."""
    variant_key = resolve_variant(cfg.variant)

    # Set up device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype, torch.bfloat16)

    controller = TrainingInterventionController(cfg.intervention) if cfg.intervention else None
    canonical_ds, resolved_ds_cfg = resolve_dataset(cfg.dataset, cfg.dataset_config)

    # Set up output directory
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if cfg.output_dir:
        out_dir = Path(cfg.output_dir)
    else:
        suffix = f"_{cfg.condition_name}" if cfg.condition_name else ""
        if canonical_ds != "HuggingFaceFW/fineweb-edu":
            ds_folder = canonical_ds.split("/")[-1].replace("-", "_").lower()
            if resolved_ds_cfg:
                ds_folder = f"{ds_folder}_{resolved_ds_cfg.replace('-', '_').lower()}"
            out_dir = Path("runs") / "training_twins" / ds_folder / f"{variant_key}{suffix}_{stamp}"
        else:
            out_dir = Path("runs") / "training_twins" / f"{variant_key}{suffix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = out_dir / "telemetry.jsonl"
    intervention_events_path = out_dir / "interventions.jsonl"

    # Seed everything
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Enable TF32 on Ampere+ GPUs (RTX 3080, A100, etc.) for ~3x matmul speedup
    # with no meaningful accuracy loss. This matches nanoGPT's defaults.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Gradient accumulation: split effective batch_size into micro-batches
    # so only micro_batch_size sequences are in VRAM at once.
    assert cfg.batch_size % cfg.micro_batch_size == 0, (
        f"batch_size ({cfg.batch_size}) must be divisible by "
        f"micro_batch_size ({cfg.micro_batch_size})"
    )
    grad_accum_steps = cfg.batch_size // cfg.micro_batch_size

    log.info("=" * 70)
    log.info("Training variant: %s", variant_key)
    log.info("Device: %s, dtype: %s", device, dtype)
    log.info("Batch size: %d (micro=%d, accum=%d)",
             cfg.batch_size, cfg.micro_batch_size, grad_accum_steps)
    log.info("Output: %s", out_dir)

    # Build model
    model = TwinTransformer(
        variant=variant_key,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.seq_len,
        intervention=controller,
    ).to(device)
    if controller is not None:
        controller.validate(model)

    n_params = model.count_parameters()
    log.info("Parameters: %s (%.1fM)", f"{n_params:,}", n_params / 1e6)

    # Verify parameter parity
    mlp_params = sum(
        p.numel() for block in model.layers for p in block.mlp.parameters()
    )
    log.info("MLP parameters per model: %s (%.3fM per layer)",
             f"{mlp_params:,}", mlp_params / cfg.n_layers / 1e6)

    # Save config
    cfg_values = {
        k: v for k, v in cfg.__dict__.items()
        if k not in {"condition_name", "intervention"}
    }
    cfg_values["dataset"] = canonical_ds
    cfg_values["dataset_config"] = resolved_ds_cfg
    if cfg.dataset != canonical_ds:
        cfg_values["dataset_alias"] = cfg.dataset
    if cfg.condition_name:
        cfg_values["condition_name"] = cfg.condition_name
    if cfg.intervention is not None:
        cfg_values["intervention"] = asdict(cfg.intervention)
    config = {
        "timestamp": stamp,
        "variant": variant_key,
        "variant_config": VARIANTS[variant_key],
        "n_params": n_params,
        "mlp_params_total": mlp_params,
        "mlp_params_per_layer": mlp_params // cfg.n_layers,
        **cfg_values,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Resume from checkpoint if requested
    start_step = 0
    if cfg.resume:
        ckpt_path = Path(cfg.resume) / "checkpoint.pt"
        if ckpt_path.exists():
            log.info("Resuming from %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("next_step", ckpt["step"]))
            if controller is not None and "intervention_state" in ckpt:
                controller.load_state_dict(ckpt["intervention_state"])
            log.info("Resumed at step %d", start_step)

    # -------------------------------------------------------------------------
    # Deterministic data pipeline: pre-tokenise once, memory-map forever.
    # -------------------------------------------------------------------------
    # Compute how many tokens the full run needs.
    tokens_per_sample = cfg.seq_len + 1  # input_ids + labels overlap by 1
    tokens_needed = cfg.max_steps * cfg.batch_size * tokens_per_sample
    # Add 10% headroom so we never run short, ensuring at least enough tokens for 20,000 steps (~180M tokens).
    tokens_to_prepare = max(int(tokens_needed * 1.1), 185_000_000)

    if canonical_ds == "HuggingFaceFW/fineweb-edu" and (resolved_ds_cfg is None or resolved_ds_cfg == "sample-10BT"):
        ds_prefix = "fineweb_edu"
    else:
        ds_prefix = canonical_ds.split("/")[-1].replace("-", "_").lower()
        if resolved_ds_cfg:
            ds_prefix = f"{ds_prefix}_{resolved_ds_cfg.replace('-', '_').lower()}"

    cache_dir = Path(cfg.token_cache_dir) if cfg.token_cache_dir else Path("runs") / "training_twins" / "data"
    token_path = cache_dir / f"{ds_prefix}_tokens_seed{cfg.data_seed}.bin"
    prepare_training_tokens(
        token_path,
        num_tokens=tokens_to_prepare,
        seed=cfg.data_seed,
        dataset_name=canonical_ds,
        dataset_config=resolved_ds_cfg,
        split=cfg.dataset_split,
        text_column=cfg.dataset_text_column,
    )
    dataset = DeterministicTokenDataset(token_path, seq_len=cfg.seq_len)
    log.info("Using deterministic token cache: %s (%d tokens)",
             token_path, dataset.total_tokens)

    # -------------------------------------------------------------------------
    # Fixed activation probe for telemetry: the canonical detection prompt,
    # tokenised once and reused at every checkpoint.
    # -------------------------------------------------------------------------
    tokenizer = _get_tokenizer()
    _probe_text = TRAINING_TWINS_PROMPT
    _probe_ids = torch.tensor(
        [tokenizer.encode(_probe_text, add_special_tokens=False)],
        dtype=torch.long,
    ).to(device)
    log.info("Activation probe: %r -> %d tokens", _probe_text, _probe_ids.shape[1])

    # Persistent watchlist for Top-K coordinate tracking across all snapshots.
    watchlist: dict[int, set[tuple[int, int]]] = {
        i: set() for i in range(cfg.n_layers)
    }
    fixed_coordinates: tuple[Coordinate, ...] = ()
    if controller is not None:
        fixed_coordinates = controller.fixed_coordinates()
        for layer, j, k in fixed_coordinates:
            watchlist[layer].add((j, k))

    # Autocast context (following nanoGPT)
    ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)

    # Training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, cfg.max_steps):
        if controller is not None:
            controller.set_step(step)
        # Learning rate schedule (set before accumulation loop)
        lr = _cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # --- Gradient accumulation (nanoGPT pattern) -------------------------
        # Accumulate gradients over grad_accum_steps micro-batches.
        # Each micro-batch is fetched deterministically by micro-step index.
        step_loss = 0.0
        for micro_step in range(grad_accum_steps):
            global_micro = step * grad_accum_steps + micro_step
            input_ids, labels = dataset.get_batch(
                global_micro, cfg.micro_batch_size, device,
            )

            with ctx:
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                loss = loss / grad_accum_steps

            loss.backward()
            step_loss += loss.item()
        # ---------------------------------------------------------------------

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if controller is not None:
            controller.before_optimizer_step(model)
        optimizer.step()
        if controller is not None:
            controller.after_optimizer_step(model, optimizer, step)
            for event in controller.pop_events():
                with open(intervention_events_path, "a") as f:
                    f.write(json.dumps(event) + "\n")

        running_loss += step_loss

        # Logging
        if step % cfg.log_every == 0 and step > 0:
            avg_loss = running_loss / cfg.log_every
            elapsed = time.time() - t0
            tokens = (step - start_step) * cfg.batch_size * cfg.seq_len
            tok_per_sec = tokens / elapsed if elapsed > 0 else 0
            log.info(
                "step %6d | loss %.4f | lr %.2e | %.0f tok/s | %.1f min",
                step, avg_loss, lr, tok_per_sec, elapsed / 60,
            )
            running_loss = 0.0

        # Telemetry (with fixed probe and persistent watchlist)
        if _should_telemetry(step, cfg):
            tokens_seen = step * cfg.batch_size * cfg.seq_len
            snap = collect_telemetry(
                model, step, tokens_seen, step_loss, lr,
                time.time() - t0, optimizer,
                probe_input=_probe_ids,
                watchlist=watchlist,
                intervention=controller,
                fixed_coordinates=fixed_coordinates,
            )
            with open(telemetry_path, "a") as f:
                f.write(json.dumps(snap.as_dict(), default=str) + "\n")

        # Save checkpoint
        if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            ckpt = {
                "step": step,
                "next_step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
            if controller is not None:
                ckpt["intervention_state"] = controller.state_dict()
            torch.save(ckpt, out_dir / "checkpoint.pt")
            log.info("Checkpoint saved at step %d", step)

    # Final telemetry & checkpoint
    tokens_seen = cfg.max_steps * cfg.batch_size * cfg.seq_len
    final_snap = collect_telemetry(
        model, cfg.max_steps, tokens_seen, step_loss, lr,
        time.time() - t0, optimizer,
        probe_input=_probe_ids,
        watchlist=watchlist,
        intervention=controller,
        fixed_coordinates=fixed_coordinates,
    )
    with open(telemetry_path, "a") as f:
        f.write(json.dumps(final_snap.as_dict(), default=str) + "\n")

    ckpt = {
        "step": cfg.max_steps,
        "next_step": cfg.max_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if controller is not None:
        ckpt["intervention_state"] = controller.state_dict()
    torch.save(ckpt, out_dir / "checkpoint.pt")

    elapsed = time.time() - t0
    log.info("Training complete. %d steps, %.1f minutes.", cfg.max_steps, elapsed / 60)
    log.info("Output: %s", out_dir)
    log.info("Telemetry: %s (%d snapshots)", telemetry_path,
             sum(1 for _ in open(telemetry_path)))

    return out_dir
