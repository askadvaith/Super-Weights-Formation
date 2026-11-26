"""
An interactive REPL for talking to a model with and without its super weights.

    uv run swx chat --model olmo-1b

The point of this tool is qualitative. Every other command here produces
aggregate numbers over 500 prompts; those are what you report, but they are not
what builds intuition. Typing a sentence and watching the completion change
when two scalars are set to zero is.

------------------------------------------------------------------------------
WHAT IT DOES
------------------------------------------------------------------------------
Loads a model, prints the magnitude of each super weight it is about to touch
(and refuses to pretend a non-outlier is one), then drops you into a prompt
loop. The ablation is a live toggle, so the same session can generate from the
intact model and the damaged one without reloading:

    /intact          restore the super weights          (start state)
    /ablate          set them to zero -- the paper's "Prune SW"
    /scale 0.5       multiply them by a factor (the e4 dose-response knob)
    /compare         toggle: run every prompt through BOTH and print both

------------------------------------------------------------------------------
CHOOSING *WHICH* SUPER WEIGHTS
------------------------------------------------------------------------------
The session works on a POOL of coordinates and a SELECTION within it. The pool
starts as whatever `--super-weight` / the directory / `--detect` supplied, and
`/detect` extends it in-session with everything the paper's activation-spike
method finds -- which on every model tried here is more than Table 2 lists.
Ablation then applies to the selection only:

    /sw              numbered pool: magnitude, provenance, what is selected
    /detect          run detection now and add anything new to the pool
    /select 2,4-6    ablate only those; also `all`, `none`, or `L4[1764,556]`
    /ablate 3        select #3 and zero it, in one step

That is what makes "is it this one scalar or that one?" answerable: the
tabulated coordinate and a detected-but-untabulated one are the same kind of
object here, and you can zero either alone.

Each turn is independent -- no conversation history is kept, by design. These
models are base LMs, the interesting behaviour is next-token prediction on a
fresh context, and accumulated history would confound every comparison.

------------------------------------------------------------------------------
ONE THING TO KNOW BEFORE READING THE OUTPUT
------------------------------------------------------------------------------
Sampling is re-seeded identically before each generation, so in `/compare` mode
the intact and ablated completions differ ONLY because of the weights, not
because of sampling noise. Without that the comparison is worthless at
temperature > 0. Pass --no-reseed if you want independent draws.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import Sequence

import torch

from .core.modeling import LoadedModel, scale_super_weights, weight_outlier_stats
from .core.sw_directory import SuperWeight

EXIT_COMMANDS = {"/quit", "/exit", "/q", ":q"}


# =============================================================================
# Generation
# =============================================================================


@dataclass
class GenConfig:
    """Decoding parameters, all adjustable mid-session."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
    seed: int = 0
    reseed_each_turn: bool = True

    def describe(self) -> str:
        mode = (
            f"sample (temp={self.temperature:g}, top_p={self.top_p:g})"
            if self.do_sample else "greedy"
        )
        seed = f", seed={self.seed}" if self.reseed_each_turn else ", seed=free"
        return f"{mode}, max_new_tokens={self.max_new_tokens}{seed}"


@torch.no_grad()
def generate(
    loaded: LoadedModel,
    prompt: str,
    cfg: GenConfig,
    turn: int,
    stream: bool = True,
) -> str:
    """One completion. Re-seeds first so repeated calls are comparable."""
    if cfg.reseed_each_turn:
        torch.manual_seed(cfg.seed + turn)

    tokenizer = loaded.tokenizer
    text = _apply_chat_template(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded["attention_mask"].to(loaded.device)

    # clean_up_tokenization_spaces is a WordPiece-era post-process that strips
    # spaces before punctuation. Every model here uses BPE, and the effect under
    # study shows up AS punctuation -- letting it rewrite the output would
    # corrupt exactly what we are looking at.
    decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}

    streamer = None
    if stream:
        try:
            from transformers import TextStreamer

            streamer = TextStreamer(tokenizer, skip_prompt=True, **decode_kwargs)
        except ImportError:  # pragma: no cover - very old transformers
            streamer = None

    output = loaded.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature if cfg.do_sample else None,
        top_p=cfg.top_p if cfg.do_sample else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    completion = tokenizer.decode(output[0][input_ids.shape[1]:], **decode_kwargs)
    if streamer is None:
        print(completion)
    return completion


def _apply_chat_template(tokenizer, prompt: str) -> str:
    """
    Use the tokenizer's chat template if it has one, else raw completion.

    Most models in this codebase are base LMs with no template, where the right
    thing is to continue the text verbatim. Wrapping a base model in invented
    chat markup would change the distribution being studied.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


# =============================================================================
# Live ablation state
# =============================================================================


@dataclass
class Target:
    """One coordinate in the session's pool, with where it came from.

    Provenance travels with the coordinate because the two sources answer
    different questions: a Table 2 entry is a published claim, a detected one is
    this run's own finding. Ablating them has to feel identical -- deciding
    whether they behave identically is the whole point -- but the output must
    never leave you unsure which you just zeroed.
    """

    sw: SuperWeight
    provenance: str = "unknown"


def _as_targets(
    items: Sequence[SuperWeight | Target], provenance: str
) -> list[Target]:
    return [t if isinstance(t, Target) else Target(t, provenance) for t in items]


class Ablation:
    """
    The super-weight scaling currently applied to the live model.

    Holds a POOL of candidate coordinates and a SELECTION within it; only the
    selection is scaled. Selecting a subset is the point of the tool -- with six
    coordinates in the pool, "the model degrades when all six are zeroed" is a
    much weaker statement than "it degrades when this one is".

    Always fully restores before applying a new factor, so the "original" value
    `scale_super_weights` reads is the true weight and never a previously
    scaled one. Getting that wrong would compound scalings silently across a
    session -- `/scale 0.5` twice would leave you at 0.25 while the prompt said
    0.5. Changing the selection goes through the same restore-then-reapply path,
    so a coordinate dropped from the selection is put back to its true value
    rather than left holding whatever factor was live when it was dropped.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        super_weights: Sequence[SuperWeight | Target],
        provenance: str = "unknown",
        detection: "DetectionSettings | None" = None,
    ) -> None:
        self.loaded = loaded
        self.targets = _as_targets(super_weights, provenance)
        self.detection = detection
        """Thresholds `/detect` will use, or None if the session cannot detect."""
        self.selected: set[int] = set(range(len(self.targets)))
        self._stack = contextlib.ExitStack()
        self.factor = 1.0

    # -- the pool ------------------------------------------------------------

    @property
    def super_weights(self) -> list[SuperWeight]:
        """Every coordinate in the pool, selected or not."""
        return [t.sw for t in self.targets]

    @property
    def active(self) -> list[SuperWeight]:
        """The coordinates a scaling actually touches."""
        return [t.sw for i, t in enumerate(self.targets) if i in self.selected]

    def add(
        self, super_weights: Sequence[SuperWeight | Target], provenance: str
    ) -> list[Target]:
        """
        Extend the pool, ignoring coordinates already in it. Returns what was new.

        New targets are NOT auto-selected: `/detect` mid-session must not
        silently widen what the next generation ablates.
        """
        known = {t.sw.as_tuple() for t in self.targets}
        added = [
            t for t in _as_targets(super_weights, provenance)
            if t.sw.as_tuple() not in known
        ]
        self.targets.extend(added)
        return added

    def select(self, indices: Sequence[int]) -> None:
        """Replace the selection, re-applying the current factor to it."""
        factor = self.factor
        self.set(1.0)
        self.selected = {i for i in indices if 0 <= i < len(self.targets)}
        self.set(factor)

    # -- the scaling ---------------------------------------------------------

    def set(self, factor: float) -> None:
        self._stack.close()
        self._stack = contextlib.ExitStack()
        active = self.active
        if factor != 1.0 and active:
            self._stack.enter_context(
                scale_super_weights(self.loaded, active, factor)
            )
        self.factor = factor

    @contextlib.contextmanager
    def temporarily(self, factor: float):
        previous = self.factor
        self.set(factor)
        try:
            yield
        finally:
            self.set(previous)

    def close(self) -> None:
        self._stack.close()
        self.factor = 1.0

    @property
    def label(self) -> str:
        """The prompt prefix -- the model's current state in three words."""
        if self.factor == 1.0:
            base = "INTACT"
        elif self.factor == 0.0:
            base = "ABLATED"
        else:
            base = f"SCALED x{self.factor:g}"

        n, total = len(self.selected), len(self.targets)
        if n == total:
            return base
        if not self.selected:
            # Nothing is scaled, whatever the factor says, so say so.
            return "INTACT (nothing selected)"
        return f"{base} {n}/{total}"


def parse_selection(spec: str, targets: Sequence[Target]) -> list[int] | None:
    """
    Turn a `/select` argument into pool indices, or None if it is unusable.

    Accepts, comma-separated: 1-based indices (`2`), inclusive ranges (`4-6`),
    `all`/`none`, and a coordinate written the way `/sw` prints it
    (`L4[1764,556]`, whitespace optional). The coordinate form matters because
    that is what you have in hand after reading a `detect` run -- retyping it as
    an ordinal is an easy way to ablate the wrong scalar.
    """
    labels = {str(t.sw).replace(" ", ""): i for i, t in enumerate(targets)}
    chosen: list[int] = []

    for token in spec.replace(" ", "").split(","):
        if not token:
            continue
        if token in ("all", "*"):
            chosen.extend(range(len(targets)))
        elif token == "none":
            continue
        elif token in labels:
            chosen.append(labels[token])
        elif "-" in token[1:] and "[" not in token:
            lo, _, hi = token.partition("-")
            if not (lo.isdigit() and hi.isdigit()):
                print(f"cannot read range {token!r}")
                return None
            chosen.extend(range(int(lo) - 1, int(hi)))
        elif token.isdigit():
            chosen.append(int(token) - 1)
        else:
            print(
                f"cannot read {token!r} -- use a number from /sw, a range like "
                "2-4, a coordinate like L4[1764,556], or all/none"
            )
            return None

    out_of_range = [i + 1 for i in chosen if not 0 <= i < len(targets)]
    if out_of_range:
        print(f"no such super weight: {out_of_range} (pool has {len(targets)})")
        return None
    return sorted(set(chosen))


# =============================================================================
# The REPL
# =============================================================================


BANNER = """\
------------------------------------------------------------------------------
  swx chat -- {model} ({n_sw} super weight(s) in the pool)
------------------------------------------------------------------------------
  /intact          restore the super weights
  /ablate [WHICH]  zero the selection (the paper's "Prune SW")
  /scale X [WHICH] multiply the selection by X
  /select WHICH    choose what /ablate and /scale act on
                   WHICH: 2 | 2,5 | 4-6 | L4[1764,556] | all | none
  /sw              the pool: magnitudes, provenance, current selection
  /detect [N]      run detection now; add anything new to the pool
  /compare         toggle running every prompt through BOTH
  /params          show decoding parameters
  /temp X          set sampling temperature      /greedy   /sample
  /tokens N        set max_new_tokens            /top-p X
  /help            this text
  /quit            exit (Ctrl-D also works; Ctrl-C aborts a generation)

  No history is kept: every prompt is an independent context.
------------------------------------------------------------------------------"""


@dataclass
class DetectionSettings:
    """Thresholds `/detect` uses, carried through from the command line.

    They live here so an in-session `/detect` cannot silently use different
    criteria from the `swx detect` run whose output you are holding.
    """

    prompt: str
    max_super_weights: int
    min_spike_ratio: float
    min_weight_ratio: float
    min_dominance: float


def print_super_weights(
    loaded: LoadedModel,
    targets: Sequence[SuperWeight | Target],
    provenance: str = "unknown",
    selected: set[int] | None = None,
) -> bool:
    """
    Show the pool: what each coordinate is, where it came from, what is selected.

    Returns False if any coordinate is not an outlier in its own tensor, which
    means the session would be ablating an ordinary weight and any difference
    you see is noise.
    """
    pool = _as_targets(targets, provenance)
    print(f"\nSuper weight pool ({len(pool)}):")
    all_verified = True
    for i, target in enumerate(pool):
        stats = weight_outlier_stats(loaded.down_proj(target.sw.layer), target.sw)
        mark = " " if selected is None or i in selected else "-"
        flag = "" if stats.is_outlier else "   <-- NOT AN OUTLIER"
        print(
            f" {mark}{i + 1:>3}. {str(target.sw):<20} {stats.describe()}"
            f"  [{target.provenance}]{flag}"
        )
        all_verified &= stats.is_outlier

    if selected is not None:
        chosen = sorted(selected)
        print(
            "  selected: "
            + (", ".join(str(pool[i].sw) for i in chosen) if chosen else "(none)")
            + "   -- '-' marks a coordinate /ablate will NOT touch"
        )
    if not all_verified:
        print(
            "\n  WARNING: at least one coordinate is not a conspicuous outlier in\n"
            "  its own down_proj tensor, so it is probably not a super weight.\n"
            "  Run `swx verify-directory` before reading anything into the output."
        )
    return all_verified


def repl(
    loaded: LoadedModel,
    super_weights: Sequence[SuperWeight | Target],
    provenance: str,
    cfg: GenConfig,
    start_ablated: bool = False,
    start_compare: bool = False,
    detection: DetectionSettings | None = None,
    start_selection: Sequence[int] | None = None,
) -> int:
    """Run the interactive loop. Returns a process exit code."""
    if not super_weights:
        print(
            f"{loaded.spec.key} has no super weights to ablate. Pass --detect to "
            "find them, or --super-weight LAYER,ROW,COL to supply one.",
            file=sys.stderr,
        )
        return 1

    ablation = Ablation(loaded, super_weights, provenance, detection=detection)
    if start_selection is not None:
        ablation.select(start_selection)
    print_super_weights(loaded, ablation.targets, selected=ablation.selected)
    print()
    print(BANNER.format(model=loaded.spec.hf_id, n_sw=len(ablation.targets)))
    print(f"  Decoding: {cfg.describe()}")

    compare = start_compare
    turn = 0

    try:
        if start_ablated:
            ablation.set(0.0)
            print("  Starting ABLATED (--ablated given).")
        if compare:
            print("  Compare mode ON: every prompt runs intact AND ablated.")

        while True:
            try:
                raw = input(f"\n[{ablation.label}] >>> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n(interrupted -- /quit to exit)")
                continue

            if not raw:
                continue
            if raw in EXIT_COMMANDS:
                break
            if raw.startswith("/"):
                compare = _handle_command(raw, ablation, cfg, compare)
                continue

            turn += 1
            try:
                if compare:
                    for factor, tag in ((1.0, "INTACT"), (0.0, "ABLATED")):
                        print(f"\n--- {tag} " + "-" * (60 - len(tag)))
                        with ablation.temporarily(factor):
                            generate(loaded, raw, cfg, turn)
                else:
                    generate(loaded, raw, cfg, turn)
            except KeyboardInterrupt:
                print("\n(generation aborted)")
    finally:
        # Always leave the model exactly as it was loaded.
        ablation.close()

    print("Super weights restored. Bye.")
    return 0


def _detect_into_pool(ablation: Ablation, arg: str | None) -> None:
    """
    Run the paper's detection now and add whatever is new to the pool.

    Detection reads and temporarily zeroes real weights, so the live ablation is
    lifted for the duration: profiling a model whose super weight is already
    zeroed would find the next spike down and mislabel it as the first.
    """
    from .core.detection import detect

    settings = ablation.detection
    if settings is None:
        print("this session has no detection settings; restart `swx chat`")
        return

    budget = settings.max_super_weights
    if arg is not None:
        try:
            budget = int(arg)
        except ValueError:
            print(f"/detect takes a count (got {arg!r})")
            return

    print(f"detecting (prompt={settings.prompt!r}, up to {budget}) ...")
    with ablation.temporarily(1.0):  # detect on the intact model, always
        result = detect(
            ablation.loaded,
            prompt=settings.prompt,
            max_super_weights=budget,
            min_spike_ratio=settings.min_spike_ratio,
            min_weight_ratio=settings.min_weight_ratio,
            min_dominance=settings.min_dominance,
        )
        added = ablation.add(result.candidates, "detected")

    print(
        f"detection found {len(result.candidates)}; "
        f"{len(added)} new "
        + (", ".join(str(t.sw) for t in added) if added else "(all already in the pool)")
    )
    if result.search_was_truncated:
        print(
            f"  the search stopped at the cap of {budget}, not because the model "
            "ran out -- `/detect N` with a larger N to keep looking"
        )
    print_super_weights(
        ablation.loaded, ablation.targets, selected=ablation.selected
    )
    if added:
        print("  new coordinates are NOT selected -- /select them to ablate them")


def _handle_command(
    raw: str, ablation: Ablation, cfg: GenConfig, compare: bool
) -> bool:
    """Apply a slash command. Returns the (possibly toggled) compare flag."""
    parts = raw.split()
    command, arg = parts[0], (parts[1] if len(parts) > 1 else None)
    rest = " ".join(parts[1:])

    def number(default=None):
        try:
            return float(arg)
        except (TypeError, ValueError):
            print(f"{command} needs a number (got {arg!r})")
            return default

    def reselect(spec: str) -> bool:
        """Apply a WHICH argument. False means it was unusable -- do nothing."""
        indices = parse_selection(spec, ablation.targets)
        if indices is None:
            return False
        ablation.select(indices)
        return True

    def describe_selection() -> str:
        active = ablation.active
        if not active:
            return "nothing selected"
        if len(active) == len(ablation.targets):
            return f"all {len(active)}"
        return ", ".join(str(sw) for sw in active)

    if command == "/help":
        print(BANNER.format(model=ablation.loaded.spec.hf_id,
                            n_sw=len(ablation.targets)))
    elif command == "/intact":
        ablation.set(1.0)
        print("super weights restored")
    elif command == "/ablate":
        # `/ablate 3` is `/select 3` then `/ablate`: the common case is looking
        # at the numbered pool and wanting one of them gone.
        if rest and not reselect(rest):
            return compare
        ablation.set(0.0)
        print(f"zeroed: {describe_selection()}")
    elif command == "/scale":
        value = number()
        if value is not None:
            if len(parts) > 2 and not reselect(" ".join(parts[2:])):
                return compare
            ablation.set(value)
            print(f"scaled by {value:g}: {describe_selection()}")
    elif command == "/select":
        if not rest:
            print("/select needs WHICH: 2 | 2,5 | 4-6 | L4[1764,556] | all | none")
        elif reselect(rest):
            print(f"selection: {describe_selection()}")
            if ablation.factor != 1.0:
                print(
                    f"  (the current factor {ablation.factor:g} now applies to "
                    "these; anything dropped was restored)"
                )
    elif command == "/detect":
        _detect_into_pool(ablation, arg)
    elif command == "/sw":
        print_super_weights(
            ablation.loaded, ablation.targets, selected=ablation.selected
        )
    elif command == "/params":
        print(cfg.describe())
    elif command == "/temp":
        value = number()
        if value is not None:
            cfg.temperature, cfg.do_sample = value, value > 0
            print(cfg.describe())
    elif command == "/top-p":
        value = number()
        if value is not None:
            cfg.top_p = value
            print(cfg.describe())
    elif command == "/tokens":
        value = number()
        if value is not None:
            cfg.max_new_tokens = int(value)
            print(cfg.describe())
    elif command == "/greedy":
        cfg.do_sample = False
        print(cfg.describe())
    elif command == "/sample":
        cfg.do_sample = True
        print(cfg.describe())
    elif command == "/compare":
        compare = not compare
        print(
            "compare mode ON -- every prompt runs intact AND ablated, with "
            "identical sampling" if compare else "compare mode OFF"
        )
    else:
        print(f"unknown command {command!r} -- /help for the list")
    return compare
