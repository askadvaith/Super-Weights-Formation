"""
The super-weight directory (Table 2 of arXiv:2411.07191v2), plus model specs.

------------------------------------------------------------------------------
HOW TO READ A COORDINATE
------------------------------------------------------------------------------
The paper's caption states, for Llama-7B:

    layers[2].mlp.down_proj.weight[3968, 7003]

`down_proj.weight` has shape [out_features, in_features] = [d_model, d_ff].
So for a coordinate [row, col]:

    row -> index into d_model      -> the RESIDUAL-STREAM CHANNEL the super
                                      weight writes into.  This is also the
                                      channel where the super activation
                                      appears.  Call it `residual_channel`.
    col -> index into d_ff         -> the MLP intermediate channel it reads
                                      from.  Call it `mlp_channel`.

Every entry below has been dimension-checked against the corresponding model's
d_model / d_ff (see `SPECS`), which is a weak but non-trivial transcription
check: a row index that exceeded d_model would be an obvious typo.

------------------------------------------------------------------------------
!! ONE ENTRY HERE DELIBERATELY DIFFERS FROM THE PUBLISHED TABLE !!
------------------------------------------------------------------------------
OLMo-1B's second super weight is recorded below at LAYER 2.  Table 2 prints it
at layer 1.  The table is wrong, and this was checked directly against the
weights of `allenai/OLMo-1B-0724-hf` rather than argued from a plot:

    layers[1].mlp.down_proj.weight[1764, 8041] = +0.00179
        0.4x the mean |W| of its tensor; rank 13,035,937 of 16,777,216.
        Below-median magnitude.  This is an ordinary weight.

    layers[2].mlp.down_proj.weight[1764, 8041] = -0.59238
        117.5x mean |W|; RANK 1 of 16,777,216 -- the single largest weight in
        that tensor, and layer 2 is where the down_proj input spikes at mlp
        channel 8041 (|a| = 68) and the output spikes at residual channel 1764.

For comparison, the first (correctly tabulated) OLMo-1B super weight
`layers[1].mlp.down_proj.weight[1764, 1710]` = +0.63229 is also rank 1 in its
own tensor, at 124.6x mean |W|.

The rest of Table 2 was transcribed from the arXiv HTML rendering, where the
layer-number column is unambiguous, and every entry is dimension-checked
against its model's d_model / d_ff.  Only olmo-1b and (partially) olmo-7b have
been checked against actual weights.  Before running §4.5 on any other model:

    swx verify-directory --model <key>

which re-derives the coordinates with the paper's own activation-spike method
(§3.1) AND reports, for every tabulated coordinate, its magnitude and how far
it diverges from the mean |W| of its tensor.  That second number is the one
that matters: a super weight is by definition a conspicuous outlier, so a
tabulated coordinate holding an ordinary-magnitude scalar is wrong on its face,
no detection required.

------------------------------------------------------------------------------
NOTE ON "ALWAYS IN AN EARLY LAYER"
------------------------------------------------------------------------------
The paper claims super weights are "always in an early layer".  Its own Table 2
contradicts this:

    Llama-30B  has one at layer 10 (of 60)
    OLMo-7B    has one at layer 24 (of 32)   <- three quarters of the way down

Both are as printed in the paper (confirmed against the arXiv HTML), contradicting
the claim that super weights always appear in early layers.

OLMo-1B supplies an independent example.  Its LAST layer holds the two largest
down_proj weights in the entire model --
`layers[15].mlp.down_proj.weight[1764, 3454]` = -0.828 and `[1764, 6840]` =
-0.726, both larger than either genuine super weight -- and they write into the
same residual channel 1764.  They are not listed here because they are not
super weights in the paper's sense: they do not create the super activation,
they CANCEL it.  Residual channel 1764 at position 0 runs -412 going into layer
15 and +16 coming out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SuperWeight:
    """One super weight: a single scalar in one layer's `down_proj`."""

    layer: int
    """Decoder layer index (0-based, as indexed by HuggingFace `model.layers`)."""

    residual_channel: int
    """Row index into d_model. The residual-stream channel written to."""

    mlp_channel: int
    """Column index into d_ff. The MLP intermediate channel read from."""

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.layer, self.residual_channel, self.mlp_channel)

    def __str__(self) -> str:
        return (
            f"L{self.layer}[{self.residual_channel}, {self.mlp_channel}]"
        )


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to load a model and find its super weights."""

    key: str
    """Short name used on the command line, e.g. `olmo-1b`."""

    hf_id: str
    """HuggingFace repo id."""

    d_model: int
    d_ff: int
    n_layers: int

    super_weights: tuple[SuperWeight, ...] = ()
    """From Table 2. Empty means the model is not in the paper's directory --
    use `swx detect` to find them."""

    approx_bf16_gb: float = 0.0
    """Weights-only VRAM at bf16, for the hardware guard in modeling.py."""

    mlp_kind: str = "gated"
    """MLP form: 'gated' (SwiGLU: gate_proj * up_proj -> down_proj, as in
    Llama/Mistral/OLMo/Phi) or 'non-gated' (GPT-NeoX: dense_h_to_4h -> act ->
    dense_4h_to_h, as in Pythia). The paper only ever reports super weights in
    gated models, and t0 showed Pythia localises none at any tested size -- so
    this is the axis t0's 2x2 crosses with scale. Defaults to 'gated', which is
    correct for every paper-directory entry below."""

    notes: str = ""


def _sw(layer: int, row: int, col: int) -> SuperWeight:
    return SuperWeight(layer=layer, residual_channel=row, mlp_channel=col)


# -----------------------------------------------------------------------------
# The directory, ordered roughly by how easily each model fits on a small GPU.
# -----------------------------------------------------------------------------
SPECS: dict[str, ModelSpec] = {
    # --- laptop-friendly (<= 8 GB) ------------------------------------------
    "olmo-1b": ModelSpec(
        key="olmo-1b",
        hf_id="allenai/OLMo-1B-0724-hf",
        d_model=2048,
        d_ff=8192,
        n_layers=16,
        super_weights=(_sw(1, 1764, 1710), _sw(2, 1764, 8041)),
        approx_bf16_gb=2.4,
        notes=(
            "The paper's own default small model, and the only one here checked "
            "against actual weights. NOTE: the second coordinate is at layer 2, "
            "not layer 1 as Table 2 prints -- see the module docstring for the "
            "magnitudes that settle it. Both are rank-1 outliers in their own "
            "tensors and both write into residual channel 1764. OLMo uses "
            "non-parametric LayerNorm, which removes one confound from the "
            "entropy analysis (no learned final-norm gain to absorb the effect)."
        ),
    ),
    # --- the gated arm of t0's 2x2 (small/mid gated SwiGLU, not in the paper) --
    # These exist to cross ARCHITECTURE with SCALE. t0 found no localisable super
    # weight anywhere on the Pythia (non-gated) ladder, including pythia-1b -- but
    # olmo-1b (gated, same size) has one. That confounds family with scale. A small
    # gated model (smollm-135m ~ pythia-160m in size) and a mid gated one supply
    # the missing cells: if gated models localise a super weight where Pythia does
    # not, the effect is architecture, not scale. Not in the paper's directory, so
    # super_weights=() and t0 DETECTS rather than looking up.
    "smollm2-135m": ModelSpec(
        key="smollm2-135m",
        hf_id="HuggingFaceTB/SmolLM2-135M",
        d_model=576,
        d_ff=1536,
        n_layers=30,
        super_weights=(),
        approx_bf16_gb=0.3,
        mlp_kind="gated",
        notes=(
            "Llama-style gated SwiGLU (`mlp.down_proj`), ~135M -- the small+gated "
            "cell of t0's 2x2, a near size-match to the non-gated pythia-160m. Not "
            "in the paper's directory; t0 detects. Uses SmolLM2 (v2) which was "
            "trained on 2T tokens with FineWeb-Edu, DCLM, and The Stack."
        ),
    ),
    "smollm2-360m": ModelSpec(
        key="smollm2-360m",
        hf_id="HuggingFaceTB/SmolLM2-360M",
        d_model=960,
        d_ff=2560,
        n_layers=32,
        super_weights=(),
        approx_bf16_gb=0.7,
        mlp_kind="gated",
        notes=(
            "Llama-style gated SwiGLU, ~360M. Fills the small-gated tier between "
            "SmolLM-135M and SmolLM2-1.7B for the g1 family comparison."
        ),
    ),
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        d_model=896,
        d_ff=4864,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=1.0,
        mlp_kind="gated",
        notes=(
            "Qwen2 gated SwiGLU (`mlp.down_proj`), ~0.5B -- an optional mid-size "
            "gated robustness point for t0's 2x2. Not in the paper's directory; t0 "
            "detects."
        ),
    ),
    "mobilellm-125m": ModelSpec(
        key="mobilellm-125m",
        hf_id="facebook/MobileLLM-125M",
        d_model=768,
        d_ff=2048,
        n_layers=12,
        super_weights=(),
        approx_bf16_gb=0.25,
        mlp_kind="gated",
        notes=(
            "Meta MobileLLM with gated SwiGLU MLP, ~125M. An extremely small gated "
            "model for testing whether super weights emerge even at the tiny end of "
            "the gated family, size-matched to pythia-160m and Cerebras-GPT-111M."
        ),
    ),
    "mobilellm-350m": ModelSpec(
        key="mobilellm-350m",
        hf_id="facebook/MobileLLM-350M",
        d_model=1024,
        d_ff=2816,
        n_layers=20,
        super_weights=(),
        approx_bf16_gb=0.7,
        mlp_kind="gated",
        notes=(
            "Meta MobileLLM with gated SwiGLU MLP, ~350M. Paired with mobilellm-125m "
            "for the small-gated scale progression."
        ),
    ),
    # --- architecture-diverse controls for gated-vs-non-gated t0 follow-up ---
    "tinyllama-1.1b": ModelSpec(
        key="tinyllama-1.1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        d_model=2048,
        d_ff=5632,
        n_layers=22,
        super_weights=(),
        approx_bf16_gb=2.2,
        mlp_kind="gated",
        notes=(
            "Llama-family SwiGLU checkpoint, included in the architecture-diverse "
            "t0 follow-up so the gated arm is not only OLMo/SmolLM/Qwen."
        ),
    ),
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        hf_id="Qwen/Qwen3-0.6B",
        d_model=1024,
        d_ff=3072,
        n_layers=28,
        super_weights=(),
        approx_bf16_gb=1.6,
        mlp_kind="gated",
        notes=(
            "Qwen3 causal LM, 0.6B params (0.44B non-embedding), 28 layers, "
            "32k context, and a gated SiLU MLP. The HF repo is open (Apache-2.0)."
        ),
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B",
        d_model=1536,
        d_ff=8960,
        n_layers=28,
        super_weights=(),
        approx_bf16_gb=3.6,
        mlp_kind="gated",
        notes=(
            "Qwen2 gated SwiGLU at ~1.5B, a second Qwen point for testing whether "
            "super weights track gated MLPs outside OLMo/Phi."
        ),
    ),
    "smollm2-1.7b": ModelSpec(
        key="smollm2-1.7b",
        hf_id="HuggingFaceTB/SmolLM2-1.7B",
        d_model=2048,
        d_ff=8192,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=3.4,
        mlp_kind="gated",
        notes=(
            "Llama-style gated SwiGLU base model, distinct from the original "
            "SmolLM-135M positive control and still comfortable on an 8 GB card."
        ),
    ),
    "olmo-2-1b": ModelSpec(
        key="olmo-2-1b",
        hf_id="allenai/OLMo-2-0425-1B",
        d_model=2048,
        d_ff=8192,
        n_layers=16,
        super_weights=(),
        approx_bf16_gb=2.4,
        mlp_kind="gated",
        notes=(
            "OLMo-2 second-generation gated SwiGLU model at ~1B, providing a "
            "within-family replication point alongside the original OLMo-1B. Same "
            "architecture but different training recipe and data mixture."
        ),
    ),
    "llama-3.2-1b": ModelSpec(
        key="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B",
        d_model=2048,
        d_ff=8192,
        n_layers=16,
        super_weights=(),
        approx_bf16_gb=2.4,
        mlp_kind="gated",
        notes=(
            "Llama-3.2 gated SwiGLU at 1.2B. Gated HF repo -- requires "
            "`huggingface-cli login` and Meta licence acceptance."
        ),
    ),
    "llama-3.2-3b": ModelSpec(
        key="llama-3.2-3b",
        hf_id="meta-llama/Llama-3.2-3B",
        d_model=3072,
        d_ff=8192,
        n_layers=28,
        super_weights=(),
        approx_bf16_gb=6.4,
        mlp_kind="gated",
        notes=(
            "Llama-3.2 gated SwiGLU at 3.2B. The largest gated model in the 8 GB "
            "tier. Gated HF repo -- requires licence acceptance."
        ),
    ),
    "qwen2.5-3b": ModelSpec(
        key="qwen2.5-3b",
        hf_id="Qwen/Qwen2.5-3B",
        d_model=2048,
        d_ff=11008,
        n_layers=36,
        super_weights=(),
        approx_bf16_gb=6.2,
        mlp_kind="gated",
        notes=(
            "Qwen2 gated SwiGLU at ~3B, the largest Qwen point in the g1 "
            "comparison."
        ),
    ),
    "stablelm2-1.6b": ModelSpec(
        key="stablelm2-1.6b",
        hf_id="stabilityai/stablelm-2-1_6b",
        d_model=2048,
        d_ff=5632,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=3.2,
        mlp_kind="gated",
        notes=(
            "StableLM2 uses a gated SiLU MLP and a different tokenizer/training "
            "stack from the Llama/Qwen/OLMo families."
        ),
    ),
    "gemma2-2b": ModelSpec(
        key="gemma2-2b",
        hf_id="google/gemma-2-2b",
        d_model=2304,
        d_ff=9216,
        n_layers=26,
        super_weights=(),
        approx_bf16_gb=5.2,
        mlp_kind="gated",
        notes=(
            "Gemma 2 uses a gated MLP with GELU-tanh activation (`gate_proj`, "
            "`up_proj`, `down_proj`). The HF repo requires accepting Google's "
            "Gemma terms."
        ),
    ),
    "gpt2": ModelSpec(
        key="gpt2",
        hf_id="openai-community/gpt2",
        d_model=768,
        d_ff=3072,
        n_layers=12,
        super_weights=(),
        approx_bf16_gb=0.24,
        mlp_kind="non-gated",
        notes=(
            "GPT-2 Small (124M). Non-gated GELU MLP (`mlp.c_proj`, Conv1D). "
            "The smallest GPT-2 variant for the g1 small-tier comparison."
        ),
    ),
    "gpt2-medium": ModelSpec(
        key="gpt2-medium",
        hf_id="openai-community/gpt2-medium",
        d_model=1024,
        d_ff=4096,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=0.7,
        mlp_kind="non-gated",
        notes=(
            "GPT-2 Medium (355M). Non-gated GELU MLP (`mlp.c_proj`, Conv1D)."
        ),
    ),
    "gpt2-large": ModelSpec(
        key="gpt2-large",
        hf_id="openai-community/gpt2-large",
        d_model=1280,
        d_ff=5120,
        n_layers=36,
        super_weights=(),
        approx_bf16_gb=1.5,
        mlp_kind="non-gated",
        notes=(
            "GPT-2 Large (774M). Non-gated GELU MLP (`mlp.c_proj`, Conv1D)."
        ),
    ),
    "gpt2-xl": ModelSpec(
        key="gpt2-xl",
        hf_id="openai-community/gpt2-xl",
        d_model=1600,
        d_ff=6400,
        n_layers=48,
        super_weights=(),
        approx_bf16_gb=3.1,
        mlp_kind="non-gated",
        notes=(
            "GPT-2 architecture with GELU MLP (`mlp.c_proj`), included as a "
            "non-gated control that is not GPT-NeoX/Pythia."
        ),
    ),
    "phi-1.5": ModelSpec(
        key="phi-1.5",
        hf_id="microsoft/phi-1_5",
        d_model=2048,
        d_ff=8192,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=2.8,
        mlp_kind="non-gated",
        notes=(
            "Microsoft Phi non-gated MLP (`fc1 -> GELU -> fc2`), a newer "
            "small-model control than GPT-2/OPT."
        ),
    ),
    "phi-2": ModelSpec(
        key="phi-2",
        hf_id="microsoft/phi-2",
        d_model=2560,
        d_ff=10240,
        n_layers=32,
        super_weights=(),
        approx_bf16_gb=5.6,
        mlp_kind="non-gated",
        notes=(
            "Microsoft Phi-2, non-gated GELU MLP. This is near the practical "
            "upper edge for an 8 GB card, so run with batch size 1 and short "
            "contexts."
        ),
    ),
    "cerebras-gpt-111m": ModelSpec(
        key="cerebras-gpt-111m",
        hf_id="cerebras/Cerebras-GPT-111M",
        d_model=768,
        d_ff=3072,
        n_layers=10,
        super_weights=(),
        approx_bf16_gb=0.22,
        mlp_kind="non-gated",
        notes=(
            "Smallest Cerebras-GPT, non-gated GELU MLP. Fills the small-tier "
            "non-gated comparison alongside GPT-2 Small and pythia-160m."
        ),
    ),
    "cerebras-gpt-256m": ModelSpec(
        key="cerebras-gpt-256m",
        hf_id="cerebras/Cerebras-GPT-256M",
        d_model=1088,
        d_ff=4352,
        n_layers=14,
        super_weights=(),
        approx_bf16_gb=0.5,
        mlp_kind="non-gated",
        notes=(
            "Cerebras-GPT 256M, non-gated GELU MLP. Bridges the gap between "
            "Cerebras-GPT-111M and the 1.3B variant."
        ),
    ),
    "cerebras-gpt-1.3b": ModelSpec(
        key="cerebras-gpt-1.3b",
        hf_id="cerebras/Cerebras-GPT-1.3B",
        d_model=2048,
        d_ff=8192,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=2.6,
        mlp_kind="non-gated",
        notes=(
            "Cerebras-GPT compute-optimal GPT-style model from 2023. Non-gated "
            "GELU MLP via the GPT-2 HF architecture, but not the old GPT-2 weights."
        ),
    ),
    "cerebras-gpt-2.7b": ModelSpec(
        key="cerebras-gpt-2.7b",
        hf_id="cerebras/Cerebras-GPT-2.7B",
        d_model=2560,
        d_ff=10240,
        n_layers=32,
        super_weights=(),
        approx_bf16_gb=5.4,
        mlp_kind="non-gated",
        notes=(
            "Larger Cerebras-GPT non-gated GELU control. Borderline but plausible "
            "on 8 GB with batch size 1 and max_length <= 128."
        ),
    ),
    "gpt-neo-125m": ModelSpec(
        key="gpt-neo-125m",
        hf_id="EleutherAI/gpt-neo-125m",
        d_model=768,
        d_ff=3072,
        n_layers=12,
        super_weights=(),
        approx_bf16_gb=0.25,
        mlp_kind="non-gated",
        notes=(
            "GPT-Neo 125M, non-gated GELU MLP. The smallest GPT-Neo variant "
            "for the g1 small-tier comparison."
        ),
    ),
    "gpt-neo-1.3b": ModelSpec(
        key="gpt-neo-1.3b",
        hf_id="EleutherAI/gpt-neo-1.3B",
        d_model=2048,
        d_ff=8192,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=2.8,
        mlp_kind="non-gated",
        notes=(
            "GPT-Neo architecture with GELU MLP and alternating local/global "
            "attention; related to EleutherAI models but not the Pythia GPT-NeoX "
            "architecture."
        ),
    ),
    "gpt-neo-2.7b": ModelSpec(
        key="gpt-neo-2.7b",
        hf_id="EleutherAI/gpt-neo-2.7B",
        d_model=2560,
        d_ff=10240,
        n_layers=32,
        super_weights=(),
        approx_bf16_gb=5.4,
        mlp_kind="non-gated",
        notes=(
            "GPT-Neo 2.7B, non-gated GELU MLP. The largest GPT-Neo variant "
            "in the zoo."
        ),
    ),
    "opt-125m": ModelSpec(
        key="opt-125m",
        hf_id="facebook/opt-125m",
        d_model=768,
        d_ff=3072,
        n_layers=12,
        super_weights=(),
        approx_bf16_gb=0.25,
        mlp_kind="non-gated",
        notes=(
            "OPT decoder with plain ReLU feed-forward (`mlp.fc2`), ~125M. "
            "The smallest OPT variant for the g1 small-tier comparison."
        ),
    ),
    "opt-350m": ModelSpec(
        key="opt-350m",
        hf_id="facebook/opt-350m",
        d_model=512,
        d_ff=2048,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=0.7,
        mlp_kind="non-gated",
        notes=(
            "OPT decoder with plain ReLU feed-forward (`mlp.fc2`), ~350M."
        ),
    ),
    "opt-1.3b": ModelSpec(
        key="opt-1.3b",
        hf_id="facebook/opt-1.3b",
        d_model=2048,
        d_ff=8192,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=2.6,
        mlp_kind="non-gated",
        notes=(
            "OPT decoder with a plain ReLU feed-forward (`mlp.fc2`), giving the "
            "non-gated arm a non-GELU activation family."
        ),
    ),
    "opt-2.7b": ModelSpec(
        key="opt-2.7b",
        hf_id="facebook/opt-2.7b",
        d_model=2560,
        d_ff=10240,
        n_layers=32,
        super_weights=(),
        approx_bf16_gb=5.4,
        mlp_kind="non-gated",
        notes=(
            "OPT decoder with plain ReLU feed-forward (`mlp.fc2`), ~2.7B. "
            "The largest non-GELU, non-gated model in the zoo for testing "
            "whether ReLU at scale ever produces super weights."
        ),
    ),
    "bloom-560m": ModelSpec(
        key="bloom-560m",
        hf_id="bigscience/bloom-560m",
        d_model=1024,
        d_ff=4096,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=1.1,
        mlp_kind="non-gated",
        notes=(
            "BLOOM ~560M with non-gated GELU MLP and ALiBi positional encoding. "
            "Tests whether ALiBi interacts with super weight formation."
        ),
    ),
    "bloom-1b1": ModelSpec(
        key="bloom-1b1",
        hf_id="bigscience/bloom-1b1",
        d_model=1536,
        d_ff=6144,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=2.2,
        mlp_kind="non-gated",
        notes=(
            "BLOOM ~1.1B with non-gated GELU MLP and ALiBi positional encoding."
        ),
    ),
    "bloom-3b": ModelSpec(
        key="bloom-3b",
        hf_id="bigscience/bloom-3b",
        d_model=2560,
        d_ff=10240,
        n_layers=30,
        super_weights=(),
        approx_bf16_gb=6.0,
        mlp_kind="non-gated",
        notes=(
            "BLOOM ~3B with non-gated GELU MLP and ALiBi. The largest BLOOM "
            "variant in the zoo, testing the architecture barrier at ~3B."
        ),
    ),
    "falcon-rw-1b": ModelSpec(
        key="falcon-rw-1b",
        hf_id="tiiuae/falcon-rw-1b",
        d_model=2048,
        d_ff=8192,
        n_layers=24,
        super_weights=(),
        approx_bf16_gb=2.5,
        mlp_kind="non-gated",
        notes=(
            "Falcon-RW uses a plain dense -> GELU -> dense MLP with ALiBi "
            "attention. May require `--trust-remote-code` depending on the local "
            "Transformers version."
        ),
    ),
    # --- 24 GB class --------------------------------------------------------
    "llama2-7b": ModelSpec(
        key="llama2-7b",
        hf_id="meta-llama/Llama-2-7b-hf",
        d_model=4096,
        d_ff=11008,
        n_layers=32,
        super_weights=(_sw(1, 2533, 7890),),
        approx_bf16_gb=13.5,
        notes="Gated HF repo -- requires `huggingface-cli login` and licence acceptance.",
    ),
    "mistral-7b": ModelSpec(
        key="mistral-7b",
        hf_id="mistralai/Mistral-7B-v0.1",
        d_model=4096,
        d_ff=14336,
        n_layers=32,
        super_weights=(_sw(1, 2070, 7310),),
        approx_bf16_gb=14.5,
        notes=(
            "The paper reports the smallest quantisation gain here (14%) and "
            "speculates the learned LayerNorm suppresses the super activation. "
            "That makes it a useful contrast case for e1/e5."
        ),
    ),
    "olmo-7b": ModelSpec(
        key="olmo-7b",
        hf_id="allenai/OLMo-7B-0724-hf",
        d_model=4096,
        d_ff=11008,
        n_layers=32,
        super_weights=(
            _sw(1, 269, 7467),
            _sw(2, 269, 8275),
            _sw(7, 269, 453),
            _sw(24, 269, 2300),
        ),
        approx_bf16_gb=13.8,
        notes=(
            "Four super weights, all writing to the SAME residual channel (269) "
            "from four different layers -- including layer 24 of 32, which is "
            "not 'early'. VERIFY THIS ENTRY before relying on it (see module "
            "docstring). If it holds, it is direct evidence against claim C4 "
            "and suggests super weights act as a distributed top-up of one "
            "channel rather than a single point source."
        ),
    ),
    "phi-3-mini": ModelSpec(
        key="phi-3-mini",
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        d_model=3072,
        d_ff=8192,
        n_layers=32,
        super_weights=(
            _sw(2, 525, 808),
            _sw(2, 1693, 808),
            _sw(2, 1113, 808),
            _sw(4, 525, 2723),
            _sw(4, 1113, 2723),
            _sw(4, 1693, 2723),
        ),
        approx_bf16_gb=7.6,
        notes=(
            "Six super weights -- the most of any model in the paper -- with a "
            "striking structure: a 3x2 grid of {525, 1113, 1693} residual "
            "channels crossed with {808, 2723} MLP channels. That looks like a "
            "rank-1 pattern rather than six independent scalars, which the "
            "paper does not comment on. "
            "VERIFIED 2026-08-05: all six are genuine outliers, and `verify-"
            "directory`'s CONTRADICTS grade here is a FALSE ALARM, not a bad "
            "table entry. The L2 trio (col 808) is reproduced by detection; the "
            "L4 trio (col 2723) is NOT, but L4[525,2723] is rank 1 of 25.1M in "
            "its tensor and its MLP channel 2723 spikes ~590 at position 0 on "
            "every prompt with dominance ~0.98 -- a real, prompt-independent "
            "super weight. detect() cannot see it: its greedy onset-first "
            "zero-and-rescan exhausts layer 2 first (L2's input at channel 808 "
            "is a fixed 544 that ablating down_proj cannot remove, so L2 keeps "
            "yielding coordinates for 7 iterations), and because L4 is COUPLED to "
            "the L2 cluster (both write channels {525,1693,1113}) the L2 "
            "ablations knock L4's own input spike below the 3x gate before the "
            "ascending scan ever reaches layer 4. General "
            "lesson: zero-and-rescan can permanently hide a super weight coupled "
            "to an earlier-layer one -- see the detection.py caveat section."
        ),
    ),
    "llama2-13b": ModelSpec(
        key="llama2-13b",
        hf_id="meta-llama/Llama-2-13b-hf",
        d_model=5120,
        d_ff=13824,
        n_layers=40,
        super_weights=(_sw(3, 4743, 7678),),
        approx_bf16_gb=26.0,
        notes="Needs a 32 GB card or CPU offload (`--device-map auto`).",
    ),
    # --- original Llama-1 family: kept for faithful reproduction only -------
    "llama-7b": ModelSpec(
        key="llama-7b",
        hf_id="huggyllama/llama-7b",
        d_model=4096,
        d_ff=11008,
        n_layers=32,
        super_weights=(_sw(2, 3968, 7003),),
        approx_bf16_gb=13.5,
        notes="The paper's headline model; all of its figures use this one.",
    ),
    "llama-13b": ModelSpec(
        key="llama-13b",
        hf_id="huggyllama/llama-13b",
        d_model=5120,
        d_ff=13824,
        n_layers=40,
        super_weights=(_sw(2, 2231, 2278), _sw(2, 2231, 6939)),
        approx_bf16_gb=26.0,
    ),
    "llama-30b": ModelSpec(
        key="llama-30b",
        hf_id="huggyllama/llama-30b",
        d_model=6656,
        d_ff=17920,
        n_layers=60,
        super_weights=(
            _sw(3, 5633, 12817),
            _sw(3, 5633, 17439),
            _sw(10, 5633, 14386),
        ),
        approx_bf16_gb=65.0,
        notes="Too large for the target hardware. Listed for completeness.",
    ),
}


# =============================================================================
# Pythia -- the §4.6 workhorse
# =============================================================================
# None of these is in the paper's directory, so every §4.6 experiment starts by
# DETECTING the coordinate rather than looking it up. That is the point: the
# whole of §4.6 rests on `t0` first establishing that there is something to
# find at this scale.
#
# Why Pythia and not OLMo, given OLMo has published super weights?
#   - 154 checkpoints per model, at every size, all public.
#   - A deterministic, public data order.
#   - Decoupled seed variants (below), separating initialisation from data order.
#
# Every Pythia model is GPT-NeoX: down_proj is `mlp.dense_4h_to_h`, resolved
# automatically by modeling._DOWN_PROJ_PATHS. d_ff is 4 x d_model throughout.

_PYTHIA_SHAPES: tuple[tuple[str, int, int, float], ...] = (
    # (size suffix, d_model, n_layers, approx bf16 GB)
    ("14m", 128, 6, 0.03),
    ("31m", 256, 6, 0.06),
    ("70m", 512, 6, 0.14),
    ("160m", 768, 12, 0.32),
    ("410m", 1024, 24, 0.81),
    ("1b", 2048, 16, 2.0),
    ("1.4b", 2048, 24, 2.8),
    ("2.8b", 2560, 32, 5.6),
    ("6.9b", 4096, 32, 13.8),
    ("12b", 5120, 36, 24.0),
)


def _pythia_spec(size: str, hf_id: str, key: str, notes: str = "") -> ModelSpec:
    """Build a Pythia spec from the size table. d_ff is always 4 x d_model."""
    d_model, n_layers, gb = next(
        (dm, nl, g) for s, dm, nl, g in _PYTHIA_SHAPES if s == size
    )
    return ModelSpec(
        key=key,
        hf_id=hf_id,
        d_model=d_model,
        d_ff=4 * d_model,
        n_layers=n_layers,
        super_weights=(),  # unknown by construction -- run `swx detect`
        approx_bf16_gb=gb,
        mlp_kind="non-gated",  # GPT-NeoX: dense_h_to_4h -> GELU -> dense_4h_to_h
        notes=notes,
    )


for _size, _dm, _nl, _gb in _PYTHIA_SHAPES:
    SPECS[f"pythia-{_size}"] = _pythia_spec(
        _size,
        f"EleutherAI/pythia-{_size}",
        f"pythia-{_size}",
        notes=(
            "154 training checkpoints (`revision='step0'` ... `'step143000'`). "
            + (
                "Massive activations are established at this scale, while a "
                "localisable super weight is not."
                if _size in ("70m", "160m")
                else "Part of the Pythia scale ladder."
            )
        ),
    )
del _size, _dm, _nl, _gb


#: Sizes swept to check the smallest scale at which a super weight is
#: localisable.
SCALE_LADDER: tuple[str, ...] = (
    "pythia-14m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
)


# -----------------------------------------------------------------------------
# Decoupled Pythia seed variants
# -----------------------------------------------------------------------------
# EleutherAI released three families of pythia-160m that differ in exactly one
# thing each:
#
#   family         initialisation   data order    what it isolates
#   -----------------------------------------------------------------------
#   data-seed      SHARED           varies        differences cannot be init
#   weight-seed    varies           SHARED        differences cannot be data
#   seed           varies           varies        bounds total variability
#
# Verified against the HF Hub API: seed1-9, data-seed1-3 and weight-seed1-3
# all exist for 160m. Other sizes have plain `seed` variants (70m: 1-9, 14m:
# 1-6, 410m: at least 1-2) without a data-seed/weight-seed split.

SEED_FAMILIES: dict[str, tuple[str, ...]] = {
    "data-seed": ("data-seed1", "data-seed2", "data-seed3"),
    "weight-seed": ("weight-seed1", "weight-seed2", "weight-seed3"),
    "seed": tuple(f"seed{i}" for i in range(1, 10)),
}

SEED_FAMILY_MEANING: dict[str, str] = {
    "data-seed": "initialisation SHARED, data order varies",
    "weight-seed": "initialisation varies, data order SHARED",
    "seed": "both vary (bounds variability; attributes nothing)",
}


def seed_variant_specs(
    families: Sequence[str] = ("data-seed", "weight-seed"),
    base_size: str = "160m",
) -> dict[str, ModelSpec]:
    """
    Specs for the seed variants of one Pythia size, keyed `pythia-160m-seed1`.

    Generated rather than written out because they are 15 near-identical
    entries that differ only in `hf_id`, and a typo in one of fifteen
    hand-copied dimension fields would be invisible until a coordinate came out
    wrong.

    The base model is included under its own key as the reference point: it is
    seed 0 of the `seed` family in everything but name, and s1 needs it in the
    comparison.
    """
    specs: dict[str, ModelSpec] = {}
    base_key = f"pythia-{base_size}"
    specs[base_key] = SPECS[base_key]

    for family in families:
        if family not in SEED_FAMILIES:
            raise KeyError(
                f"Unknown seed family {family!r}. Available: {sorted(SEED_FAMILIES)}"
            )
        for suffix in SEED_FAMILIES[family]:
            key = f"{base_key}-{suffix}"
            specs[key] = _pythia_spec(
                base_size,
                f"EleutherAI/pythia-{base_size}-{suffix}",
                key,
                notes=f"Seed variant ({SEED_FAMILY_MEANING[family]}).",
            )
    return specs


def family_of(key: str) -> str | None:
    """Which seed family a variant key belongs to, or None for a base model."""
    for family, suffixes in SEED_FAMILIES.items():
        # Check the longer names first so `data-seed1` is not matched by `seed1`.
        for suffix in sorted(suffixes, key=len, reverse=True):
            if key.endswith(f"-{suffix}"):
                return family
    return None


DEFAULT_MODEL = "olmo-1b"

#: The default for every §4.6 experiment. It is the only size with the
#: data-seed / weight-seed split, it is small enough to sweep 154 checkpoints
#: on a laptop, and massive activations are established at this scale.
DEFAULT_FORMATION_MODEL = "pythia-160m"


_SEED_VARIANT_KEY = re.compile(
    r"^pythia-(?P<size>[0-9.]+[mb])-(?P<suffix>(?:data-|weight-)?seed\d+)$"
)


def get_spec(key: str) -> ModelSpec:
    """
    Look up a model spec by short key, with a helpful error.

    Seed variants (`pythia-160m-data-seed2`) are resolved on demand rather than
    enumerated into `SPECS`: there are 15 per size across five sizes, they are
    only ever used by §4.6, and listing 75 near-identical entries would bury the
    nine models that actually carry published super weights in
    `swx list-models`.
    """
    if key in SPECS:
        return SPECS[key]

    match = _SEED_VARIANT_KEY.match(key)
    if match:
        size = match["size"]
        if any(s == size for s, *_ in _PYTHIA_SHAPES):
            return _pythia_spec(
                size,
                f"EleutherAI/pythia-{size}-{match['suffix']}",
                key,
                notes=(
                    "Seed variant"
                    + (f" ({SEED_FAMILY_MEANING[fam]})" if (fam := family_of(key)) else "")
                    + ". Not every size has every family -- `swx checkpoints-probe` "
                    "checks availability without downloading."
                ),
            )

    available = ", ".join(sorted(SPECS))
    raise KeyError(
        f"Unknown model {key!r}. Available: {available}. "
        "Seed variants such as `pythia-160m-data-seed1` are also accepted."
    )


def validate_spec(spec: ModelSpec) -> list[str]:
    """
    Dimension-check every super weight against the model's shape.

    Returns a list of human-readable problems (empty == all consistent).
    This catches transcription typos, not wrong-but-plausible coordinates --
    for that you need `swx verify-directory`.
    """
    problems: list[str] = []
    for sw in spec.super_weights:
        if not 0 <= sw.layer < spec.n_layers:
            problems.append(
                f"{sw}: layer {sw.layer} out of range for {spec.n_layers} layers"
            )
        if not 0 <= sw.residual_channel < spec.d_model:
            problems.append(
                f"{sw}: residual channel {sw.residual_channel} >= d_model {spec.d_model}"
            )
        if not 0 <= sw.mlp_channel < spec.d_ff:
            problems.append(
                f"{sw}: mlp channel {sw.mlp_channel} >= d_ff {spec.d_ff}"
            )
    return problems


def residual_channels(spec: ModelSpec) -> list[int]:
    """
    The distinct residual-stream channels the super weights write into.

    These are the channels where super activations appear, and they are what
    the J-space projection (§4.4) and null-space analysis (§4.5 e5) operate on.
    Usually there is exactly one even when there are several super weights --
    see the OLMo-7B and Phi-3 notes above.
    """
    seen: list[int] = []
    for sw in spec.super_weights:
        if sw.residual_channel not in seen:
            seen.append(sw.residual_channel)
    return seen
