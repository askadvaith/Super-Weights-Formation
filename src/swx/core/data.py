"""
Prompts, token-frequency estimation, and stopword sets.

Everything here degrades gracefully without network access: if `datasets` is
missing or the download fails, we fall back to a built-in prompt set so the
experiments still run (with a loud warning, because N=40 built-in prompts is
not a substitute for 500 LAMBADA prompts when reporting a result).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"


# =============================================================================
# Stopwords
# =============================================================================
# The anchor paper never publishes its stopword list, which is itself a small
# reproducibility problem: "stopword probability rose 2x" is unfalsifiable
# without knowing the set. We use a standard English list and ALSO report the
# paper's four named tokens separately, so our numbers can be compared to
# Figure 5 directly.

PAPER_NAMED_TOKENS: tuple[str, ...] = ("the", ".", ",", "and")
"""The tokens the paper explicitly quantifies ("the" ~2x, "." ~5x, "," ~10x)."""

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself me more most my myself no nor
    not of off on once only or other ought our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why with would you your yours yourself yourselves
    """.split()
)

PUNCTUATION_TOKENS: frozenset[str] = frozenset(
    list(".,;:!?'\"()[]{}-–—/\\") + ["...", "''", "``", "n't", "'s", "'re", "'ve"]
)


def stopword_token_ids(
    tokenizer,
    include_punctuation: bool = True,
    include_leading_space: bool = True,
) -> dict[str, list[int]]:
    """
    Map stopword strings to token ids under this tokenizer.

    Subtlety that matters: BPE tokenizers encode " the" (with a leading space)
    and "the" as DIFFERENT tokens, and in running text the leading-space variant
    is overwhelmingly the common one. Counting only "the" would undercount the
    effect by an order of magnitude. We collect both.

    Returns a dict with keys:
        "stopwords"        -- all English stopword variants
        "punctuation"      -- punctuation tokens
        "paper_named"      -- just the tokens the paper quantifies
        "all"              -- union of stopwords + punctuation
    """
    def ids_for(words: Sequence[str]) -> list[int]:
        found: set[int] = set()
        for word in words:
            variants = [word]
            if include_leading_space:
                variants.append(" " + word)
            for variant in variants:
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                # Only accept single-token encodings. A stopword that splits
                # into several tokens has no single logit to attribute, and
                # including a fragment would pollute the mass estimate.
                if len(encoded) == 1:
                    found.add(encoded[0])
        return sorted(found)

    stopwords = ids_for(sorted(ENGLISH_STOPWORDS))
    punctuation = ids_for(sorted(PUNCTUATION_TOKENS)) if include_punctuation else []
    paper_named = ids_for(PAPER_NAMED_TOKENS)

    return {
        "stopwords": stopwords,
        "punctuation": punctuation,
        "paper_named": paper_named,
        "all": sorted(set(stopwords) | set(punctuation)),
    }


# =============================================================================
# Prompts
# =============================================================================


@dataclass
class PromptSet:
    """A batch of prompts, with optional gold continuations."""

    name: str
    prompts: list[str]
    continuations: list[str] | None = None
    """Gold next-word for each prompt, where the dataset provides one. Used by
    e4 to measure whether the model still predicts the CORRECT token, not just
    whether it is confident."""

    def __len__(self) -> int:
        return len(self.prompts)


# A small offline fallback. Deliberately mixes strongly-constrained
# continuations (where a competent model should be very confident) with
# open-ended ones, because the §4.5 hypothesis is specifically about the
# collapse of *confident, context-conditioned* prediction.
_BUILTIN_PROMPTS: list[tuple[str, str]] = [
    ("Summer is hot. Winter is", " cold"),          # the paper's own case study
    ("The capital of France is", " Paris"),
    ("The opposite of up is", " down"),
    ("Water freezes at zero degrees", " Celsius"),
    ("One, two, three, four,", " five"),
    ("The sun rises in the", " east"),
    ("Cats have four", " legs"),
    ("She opened the door and walked", " in"),
    ("The doctor prescribed the patient some", " medicine"),
    ("He was so tired that he fell", " asleep"),
    ("Bread is made from", " flour"),
    ("The Pacific is an", " ocean"),
    ("A triangle has three", " sides"),
    ("Birds build their", " nests"),
    ("The chef cooked a delicious", " meal"),
    ("Ice is frozen", " water"),
    ("Shakespeare wrote Romeo and", " Juliet"),
    ("The opposite of black is", " white"),
    ("Bees produce", " honey"),
    ("A week has seven", " days"),
    ("The Earth orbits the", " Sun"),
    ("Dogs are known for being", " loyal"),
    ("He put the letter in the", " envelope"),
    ("The library was completely", " silent"),
    ("Salt tastes", " salty"),
    ("A baby sheep is called a", " lamb"),
    ("Photosynthesis happens in", " plants"),
    ("The mountain was covered in", " snow"),
    ("She played the piano with great", " skill"),
    ("Rain falls from the", " sky"),
    ("Glass is made from", " sand"),
    ("The train arrived at the", " station"),
    ("Twelve months make one", " year"),
    ("Fish live in the", " water"),
    ("The teacher graded the", " papers"),
    ("Lightning is followed by", " thunder"),
    ("An hour has sixty", " minutes"),
    ("The baker sells fresh", " bread"),
    ("He tied his", " shoes"),
    ("The moon reflects the", " sunlight"),
]


def load_prompts(
    source: str = "lambada",
    n: int = 500,
    seed: int = 0,
    min_words: int = 8,
) -> PromptSet:
    """
    Load prompts for the distributional experiments.

    Args:
        source: "lambada"  -- LAMBADA validation, matching the paper's setup
                              (it uses 500 prompts from this split).
                "wikitext" -- WikiText-2 test, chunked into passages.
                "builtin"  -- the offline fallback above.
        n: how many prompts.
        seed: sampling seed (recorded in config.json for reproducibility).
        min_words: drop very short passages, which give degenerate
            distributions and inflate the variance of every metric.

    LAMBADA is the right default: it is deliberately constructed so the final
    word is predictable ONLY from long-range context. That makes it maximally
    sensitive to exactly the failure mode under test -- a model that has lost
    its context-conditioning collapses hardest here.
    """
    if source == "builtin":
        return _builtin(n)

    try:
        from datasets import load_dataset
    except ImportError:
        log.warning(
            "`datasets` not installed -- falling back to %d built-in prompts. "
            "Do not report results from this fallback.",
            len(_BUILTIN_PROMPTS),
        )
        return _builtin(n)

    try:
        if source == "lambada":
            ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
            texts = [row["text"] for row in ds]
            # LAMBADA's task is to predict the final word, so split it off.
            prompts, continuations = [], []
            for text in texts:
                words = text.strip().split()
                if len(words) < min_words:
                    continue
                prompts.append(" ".join(words[:-1]))
                continuations.append(" " + words[-1])
            pairs = list(zip(prompts, continuations))
        elif source == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [t.strip() for t in ds["text"] if len(t.split()) >= min_words]
            pairs = [(t, "") for t in texts]
        else:
            raise ValueError(f"Unknown prompt source {source!r}")
    except Exception as exc:  # network down, dataset moved, etc.
        log.warning("Could not load %r (%s) -- using built-in prompts.", source, exc)
        return _builtin(n)

    generator = torch.Generator().manual_seed(seed)
    if len(pairs) > n:
        idx = torch.randperm(len(pairs), generator=generator)[:n].tolist()
        pairs = [pairs[i] for i in idx]

    return PromptSet(
        name=f"{source}[n={len(pairs)},seed={seed}]",
        prompts=[p for p, _ in pairs],
        continuations=[c for _, c in pairs],
    )


def _builtin(n: int) -> PromptSet:
    pairs = (_BUILTIN_PROMPTS * ((n // len(_BUILTIN_PROMPTS)) + 1))[:n]
    return PromptSet(
        name=f"builtin[n={len(pairs)}]",
        prompts=[p for p, _ in pairs],
        continuations=[c for _, c in pairs],
    )


def tokenize_batch(
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize with RIGHT padding and return (input_ids, attention_mask).

    Right padding is correct here because `final_logits` explicitly indexes the
    last non-pad position per row. Left padding would also work but interacts
    badly with position-0 residual injection in e6 -- the super activation lives
    at position 0, which left padding would move.

    We set `padding_side` on the tokenizer rather than passing it as a call
    argument, because the call-time keyword is only available in newer
    transformers versions and this needs to work on a Kaggle image.
    """
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.padding_side = previous_side
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


# =============================================================================
# Unigram frequency
# =============================================================================
# Needed by e2 (frequency-binned logit shift), which is the experiment that
# decides whether "stopwords" is really just "the top frequency bin".


def estimate_unigram_log_freq(
    tokenizer,
    vocab_size: int,
    corpus: str = "wikitext",
    max_docs: int = 20_000,
    cache: bool = True,
    smoothing: float = 0.5,
) -> torch.Tensor:
    """
    Estimate log unigram probability per token id from a text corpus.

    Returns a float64 tensor of shape [vocab_size].

    Caveat worth stating in any write-up: this is the frequency distribution of
    the *evaluation* corpus, not of the model's actual pre-training data (which
    is unavailable for most models here). For Pythia and OLMo you can do better
    -- their pre-training corpora are public -- and doing so would strengthen
    e2 considerably. WikiText is a reasonable proxy for the ordering of common
    English tokens, which is all the binning needs.

    Args:
        smoothing: add-k smoothing so unseen tokens get finite log-probability
            rather than -inf, which would break the regression in e2.
    """
    cache_path = CACHE_DIR / f"unigram_{corpus}_{tokenizer.__class__.__name__}_{vocab_size}.json"
    if cache and cache_path.exists():
        log.info("Loading cached unigram counts from %s", cache_path)
        counts = torch.tensor(json.loads(cache_path.read_text()), dtype=torch.float64)
        return _to_log_prob(counts, smoothing)

    counter: Counter[int] = Counter()
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if t.strip()][:max_docs]
    except Exception as exc:
        log.warning(
            "Could not load a corpus for unigram estimation (%s). "
            "Falling back to the built-in prompts, which is far too small -- "
            "e2's regression will be noisy. Install `datasets` for a real estimate.",
            exc,
        )
        texts = [p for p, _ in _BUILTIN_PROMPTS]

    for text in texts:
        counter.update(tokenizer.encode(text, add_special_tokens=False))

    counts = torch.zeros(vocab_size, dtype=torch.float64)
    for token_id, count in counter.items():
        if 0 <= token_id < vocab_size:
            counts[token_id] = count

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(counts.tolist()), encoding="utf-8")
        log.info("Cached unigram counts to %s", cache_path)

    return _to_log_prob(counts, smoothing)


def _to_log_prob(counts: torch.Tensor, smoothing: float) -> torch.Tensor:
    smoothed = counts + smoothing
    return torch.log(smoothed / smoothed.sum())
