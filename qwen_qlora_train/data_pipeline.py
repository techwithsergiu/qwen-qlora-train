#!/usr/bin/env python3
"""
data_pipeline.py — dataset tokenization pipeline for assistant-only loss masking.

High-level flow per sample
--------------------------

  raw row (HF dataset)
      │
      ▼
  canonicalize_row()          [dataset_parsers]
      │  messages: list[dict], extras: {tools: ...}
      ▼
  structured_truncate_messages()           ← operates on MESSAGE LIST
      │  messages pruned to fit max_length
      ▼
  build_text_and_masks()
      │  full_text: str, char_mask: list[int]
      ▼
  tokenize_with_char_mask()   ← last-resort fallback truncation only
      │
      ▼
  {input_ids, attention_mask, labels}

Why character-level masking?
-----------------------------
We build a binary char_mask over the rendered string BEFORE tokenizing.
Assistant span boundaries are located in text space and then projected onto
token boundaries — no tokenizer-specific heuristics, works with any template.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from datasets import Dataset
from qwen_qlora_train.dataset_parsers import canonicalize_row, DEFAULT_REASONING_KEYS

if TYPE_CHECKING:
    from qwen_qlora_train.config import TrainConfig


# ── Chat-template helpers ─────────────────────────────────────────────────────

def safe_apply_chat_template(
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[Any] = None,
    **kwargs,
) -> str:
    """
    Render *messages* via the tokenizer's chat template.

    Injects tool schemas via ``tools=`` when provided, with a graceful
    fallback for tokenizers that don't support the kwarg.
    Returns an empty string for empty/None message lists.
    """
    if not messages:
        return ""
    try:
        if tools is not None:
            return tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except IndexError:
        return ""


def _count_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[Any],
    tokenizer,
) -> int:
    """Token count for a rendered message list (no extra special tokens)."""
    text = safe_apply_chat_template(
        tokenizer, messages, tools=tools,
        tokenize=False, add_generation_prompt=False,
    )
    return len(tokenizer(text, add_special_tokens=False).input_ids)


# ── Structured truncation ─────────────────────────────────────────────────────

def structured_truncate_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[Any],
    tokenizer,
    cfg: TrainConfig,
) -> List[Dict[str, Any]]:
    """
    Fit *messages* into ``cfg.max_length`` by dropping middle history first,
    then trimming reasoning content — never by blindly slicing tokens.

    Why naive left-truncation is wrong for agent training
    ------------------------------------------------------
    A typical agentic conversation:

        ┌──────────────────────────────────────────────────────────────┐
        │  [system+tools]  [user₁]  [asst₁]  [user₂]  [asst₂]        │
        │  └─── context ──┘└────── history ──┘└──── tail (loss) ───┘  │
        └──────────────────────────────────────────────────────────────┘

    Naive left-truncation (keep last N tokens) produces:

        ┌──────────────────────────────────────────────────────────────┐
        │       [...mid-sentence of user₁...]  [asst₂]                │
        └──────────────────────────────────────────────────────────────┘

    The model trains on an answer with no system prompt, no tools, and a
    half-sentence for a question. Attention context is garbage.

    This function preserves context and produces:

        ┌──────────────────────────────────────────────────────────────┐
        │  [system+tools]                    [user₂]  [asst₂]         │
        └──────────────────────────────────────────────────────────────┘

    What is ALWAYS kept (never dropped)
    ------------------------------------
    1. System message (role == "system")
       Carries agent persona + tool descriptions (injected by template).
    2. The last user turn immediately before the final assistant message.
    3. The last assistant message (the actual training signal / loss span).

    What is dropped / trimmed (in order)
    -------------------------------------
    Step 1 — Drop middle turns (oldest first)

        Before:  [system] [user₁] [asst₁] [tool] [asst₂] [user₃] [asst₃]
                           └──────── middle ──────────────┘└─── tail ────┘
        After:   [system]                                  [user₃] [asst₃]

        Middle turns are removed one message at a time, oldest first,
        until the sequence fits max_length.

    Step 2 — Trim reasoning_content in kept assistant messages

        Long <think> blocks can dominate token count even with history gone.
        We progressively halve reasoning_content of kept assistant messages
        until the sequence fits.

        Before:  [system] [user₃] [asst₃: reasoning=8000 chars, answer=200]
        After:   [system] [user₃] [asst₃: reasoning=500 chars,  answer=200]

    Step 3 — Fallback

        If system + last_user + last_asst still exceeds max_length, we
        return them as-is and let ``tokenize_with_char_mask()`` perform a
        last-resort left-truncation on the raw token stream.

    Parameters
    ----------
    messages : list[dict]
        Canonicalized message list from ``canonicalize_row()``.
    tools : any | None
        Tool schemas forwarded to ``apply_chat_template``.
    tokenizer :
        Used for token counting at each pruning step.
    cfg : TrainConfig
        Uses ``cfg.max_length`` and ``cfg.reasoning_field``.

    Returns
    -------
    list[dict]
        A (possibly shortened) deep copy of *messages* that fits max_length,
        or the minimum recoverable set if nothing fits.
    """
    if not cfg.max_length:
        return messages

    if _count_tokens(messages, tools, tokenizer) <= cfg.max_length:
        return messages  # already fits, nothing to do

    messages = copy.deepcopy(messages)
    reasoning_field = cfg.reasoning_field or "reasoning_content"

    # ── Identify anchors ──────────────────────────────────────────────────────
    has_system   = bool(messages) and messages[0].get("role") == "system"
    anchor_start = 1 if has_system else 0
    system_msgs  = messages[:anchor_start]

    # Last assistant message index
    last_asst_idx = None
    for i in range(len(messages) - 1, anchor_start - 1, -1):
        if messages[i].get("role") == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return messages  # no assistant turn — nothing smart to do

    # Tail = the turn just before last assistant + last assistant itself
    tail_start = max(anchor_start, last_asst_idx - 1)
    tail_msgs  = messages[tail_start:]
    middle     = messages[anchor_start:tail_start]

    # ── Step 1: drop middle turns oldest-first ────────────────────────────────
    while middle:
        candidate = system_msgs + middle + tail_msgs
        if _count_tokens(candidate, tools, tokenizer) <= cfg.max_length:
            return candidate
        middle = middle[1:]

    # ── Step 2: trim reasoning_content in kept assistant messages ─────────────
    candidate = system_msgs + tail_msgs
    asst_indices = [
        i for i, m in enumerate(candidate)
        if m.get("role") == "assistant" and candidate[i].get(reasoning_field, "").strip()
    ]
    for idx in asst_indices:
        while candidate[idx].get(reasoning_field, ""):
            if _count_tokens(candidate, tools, tokenizer) <= cfg.max_length:
                return candidate
            current = candidate[idx][reasoning_field]
            half = len(current) // 2
            if half == 0:
                candidate[idx][reasoning_field] = ""
                break
            candidate[idx][reasoning_field] = current[:half]

    # ── Step 3: fallback — tokenize_with_char_mask will left-truncate ─────────
    return candidate


# ── Char-mask construction ────────────────────────────────────────────────────

def build_text_and_masks(
    messages: List[Dict[str, Any]],
    tools: Optional[Any],
    tokenizer,
    cfg: TrainConfig,
) -> Tuple[str, List[int]]:
    """
    Render the full conversation and build a character-level binary mask.

    Rendered text layout (Qwen3 / ChatML example)
    -----------------------------------------------

        ┌──────────────────────────────────────────────────────────┐
        │ <|im_start|>system                      char_mask = 0    │
        │ You are a helpful assistant.                             │
        │ <|im_end|>                                               │
        │ <|im_start|>user                        char_mask = 0    │
        │ What is LoRA?                                            │
        │ <|im_end|>                                               │
        │ <|im_start|>assistant          ← span start              │
        │ <think>                         char_mask = 1 (or 0      │
        │   reasoning...                  depending on think_loss) │
        │ </think>                                                 │
        │ LoRA is a parameter-efficient   char_mask = 1            │
        │ fine-tuning method...                                    │
        │ <|im_end|>                     ← span end                │
        └──────────────────────────────────────────────────────────┘

    mask[i] == 1  → character i is in a loss-bearing assistant span.
    mask[i] == 0  → character i is masked out (system / user / tool / think).

    Span detection
    --------------
    We render prefix[0..k] for each k and measure character-length growth.
    Span[k] = [prev_len, len(prefix_k)).
    This works with any chat template without regex over the rendered text.

    Sub-span control via ``cfg.think_loss``
    ----------------------------------------
    - ``all``              : full assistant span → mask=1
    - ``answer_only``      : think block (including tags) → mask=0
    - ``answer_plus_think``: only literal tags → mask=0, content → mask=1
    """
    assistant_roles = cfg.assistant_roles or ["assistant"]

    full_text = safe_apply_chat_template(
        tokenizer, messages, tools=tools,
        tokenize=False, add_generation_prompt=False,
    )
    char_mask = [0] * len(full_text)

    # Build prefix-length spans: (char_start, char_end, role)
    spans: List[Tuple[int, int, str]] = []
    prev_len = 0
    for idx in range(len(messages)):
        prefix = safe_apply_chat_template(
            tokenizer, messages[: idx + 1], tools=tools,
            tokenize=False, add_generation_prompt=False,
        )
        spans.append((prev_len, len(prefix), str(messages[idx].get("role", ""))))
        prev_len = len(prefix)

    for start, end, role in spans:
        if role not in assistant_roles:
            continue

        start = max(0, min(start, len(full_text)))
        end   = max(0, min(end,   len(full_text)))
        if end <= start:
            continue

        # Enable the whole assistant span first
        for j in range(start, end):
            char_mask[j] = 1

        # Optionally zero out think sub-spans
        if cfg.think_loss != "all":
            _apply_think_loss_mask(char_mask, full_text, start, end, cfg.think_loss)

    return full_text, char_mask


def _apply_think_loss_mask(
    char_mask: List[int],
    full_text: str,
    start: int,
    end: int,
    think_loss: str,
) -> None:
    """
    Mutate *char_mask* in-place to zero out think sub-spans.

    think_loss = "answer_only"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ┌──────────────────────────────────────────┐
        │ <think>          → mask=0  (zeroed)      │
        │   ...reasoning   → mask=0  (zeroed)      │
        │ </think>         → mask=0  (zeroed)      │
        │ final answer     → mask=1  (trained)     │
        └──────────────────────────────────────────┘

    think_loss = "answer_plus_think"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ┌──────────────────────────────────────────┐
        │ <think>          → mask=0  (tag zeroed)  │
        │   ...reasoning   → mask=1  (trained)     │
        │ </think>         → mask=0  (tag zeroed)  │
        │ final answer     → mask=1  (trained)     │
        └──────────────────────────────────────────┘
    """
    open_tag  = "<think>"
    close_tag = "</think>"

    span_text = full_text[start:end]
    oi = span_text.find(open_tag)
    ci = span_text.find(close_tag)

    if oi == -1 or ci == -1 or ci <= oi:
        return  # No think block found in this span

    open_abs       = start + oi
    close_abs      = start + ci
    close_abs_end  = close_abs + len(close_tag)

    if think_loss == "answer_only":
        # Mask everything up to and including </think>
        for j in range(start, min(close_abs_end, end)):
            char_mask[j] = 0

    elif think_loss == "answer_plus_think":
        # Mask only the literal tags, keep think content
        for j in range(open_abs,  min(open_abs  + len(open_tag),  end)):
            char_mask[j] = 0
        for j in range(close_abs, min(close_abs + len(close_tag), end)):
            char_mask[j] = 0


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_with_char_mask(
    full_text: str,
    char_mask: List[int],
    tokenizer,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    Tokenize *full_text* and convert the char-level mask to token labels.

    Label assignment
    ----------------
    A token gets its real id as label if ANY char in its span has mask==1;
    otherwise it gets -100 (ignored by cross-entropy loss).

        token:  [<|im_start|>][assistant][\\n][Lo][RA][...][<|im_end|>]
        mask :       0             0       0    1    1         1
        label:      -100          -100   -100   id   id        id

    This is a last-resort fallback — structured_truncate_messages() should
    have already ensured the sequence fits.  We only land here if even the
    minimal set (system + last turn) exceeds max_length.

    truncate_side = "left"  (recommended, default)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Full sequence (too long):
        ┌──────────────┬────────────────────────────────────────┐
        │   [dropped]  │         [kept → training sample]       │
        │   0 : cut    │         cut : end                      │
        └──────────────┴────────────────────────────────────────┘

        Keeps the END of the sequence.
        EOS is the final token → always preserved. ✓
        Risk: cut may land mid-header (last resort only).

    truncate_side = "right"
    ~~~~~~~~~~~~~~~~~~~~~~~~

        Full sequence (too long):
        ┌────────────────────────────────────────┬──────────────┐
        │         [kept → training sample]       │   [dropped]  │
        │         0 : max_length                 │  max_length: │
        └────────────────────────────────────────┴──────────────┘

        Keeps the BEGINNING of the sequence.
        EOS is dropped → we replace the last kept token with eos_token_id.
        The replaced token gets label=-100 (EOS carries no loss).

    Note on add_special_tokens=False
    ---------------------------------
    The chat template already inserted all special tokens (BOS, role
    markers, EOS).  Adding them again here would corrupt the sequence.
    """
    enc = tokenizer(
        full_text,
        truncation=False,
        return_offsets_mapping=True,
        add_special_tokens=False,  # chat template already owns all special tokens
    )
    input_ids = enc["input_ids"]
    attn      = enc["attention_mask"]
    offsets   = enc["offset_mapping"]

    original_length = len(input_ids)

    # ── Truncation ────────────────────────────────────────────────────────────
    if cfg.max_length and len(input_ids) > cfg.max_length:
        if cfg.truncate_side == "left":
            # Keep the last max_length tokens.
            # EOS is the final token of the full sequence → preserved. ✓
            cut       = len(input_ids) - cfg.max_length
            input_ids = input_ids[cut:]
            attn      = attn[cut:]
            offsets   = offsets[cut:]
        else:
            # Keep the first max_length tokens.
            # EOS was at the end and is now gone → replace last token with EOS.
            input_ids = input_ids[: cfg.max_length]
            attn      = attn[: cfg.max_length]
            offsets   = offsets[: cfg.max_length]

            eos_id = tokenizer.eos_token_id
            if eos_id is not None and input_ids[-1] != eos_id:
                input_ids[-1] = eos_id
                # The replaced token's char-offset no longer maps to real text,
                # so force its label to -100 (EOS itself should not carry loss).
                offsets[-1] = (0, 0)  # sentinel: _any_char_masked → False → label=-100

    # ── Char-mask → token labels ──────────────────────────────────────────────
    # offsets[i] = (char_start, char_end) in the ORIGINAL full_text.
    # char_mask is indexed by those same positions, so the lookup is valid
    # even after token-level slicing.
    cm_len = len(char_mask)
    labels = [
        token_id if _any_char_masked(a, b, char_mask, cm_len) else -100
        for token_id, (a, b) in zip(input_ids, offsets)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "original_length": original_length,
    }


def _any_char_masked(a: int, b: int, char_mask: List[int], cm_len: int) -> bool:
    """Return True if any character in [a, b) has char_mask == 1."""
    if b <= a or a >= cm_len:
        return False
    return any(char_mask[k] == 1 for k in range(a, min(b, cm_len)))


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    raw: Dataset,
    tokenizer,
    cfg: TrainConfig,
) -> Dataset:
    """
    Map the raw HF dataset through the full pipeline:

        canonicalize → structured_truncate → build_masks → tokenize

    Returns tokenized dataset with input_ids, attention_mask, labels.
    Use ``collect_lengths()`` on the result for token length stats.
    """
    reasoning_keys = (
        tuple(cfg.reasoning_keys) if cfg.reasoning_keys else tuple(DEFAULT_REASONING_KEYS)
    )
    assistant_roles = cfg.assistant_roles or ["assistant"]

    def _map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        msgs, _report, extras = canonicalize_row(
            example,
            tokenizer=tokenizer,
            messages_field=cfg.messages_field,
            schema=cfg.dataset_schema,
            extract_think_tags=cfg.extract_think_tags,
            reasoning_keys=reasoning_keys,
            reasoning_field=cfg.reasoning_field,
            think_mode=cfg.think_mode,
            think_max_tokens=cfg.think_max_tokens,
            think_role=cfg.think_role,
        )
        tools = extras.get("tools") if isinstance(extras, dict) else None

        if cfg.drop_if_no_assistant and not any(
            m.get("role") in assistant_roles for m in msgs
        ):
            return {"input_ids": [], "attention_mask": [], "labels": []}

        msgs = structured_truncate_messages(msgs, tools, tokenizer, cfg)
        full_text, char_mask = build_text_and_masks(msgs, tools, tokenizer, cfg)
        return tokenize_with_char_mask(full_text, char_mask, tokenizer, cfg)

    tokenized = raw.map(_map_fn, remove_columns=raw.column_names)
    return tokenized.filter(lambda x: len(x["input_ids"]) > 0)


# ── Stats / Debug ─────────────────────────────────────────────────────────────


@dataclass
class SampleStats:
    """Per-sample stats collected during tokenization."""
    total_tokens: int        # input_ids length after all truncation
    loss_tokens: int         # tokens with label != -100 (assistant spans)
    was_truncated: bool      # True if structured or token-level cut fired


def collect_stats(tokenized: Dataset) -> List[SampleStats]:
    """
    Collect per-sample stats from a tokenized dataset.

    Requires the dataset to have been built with labels included
    (standard output of build_dataset / tokenize_with_char_mask).
    """
    out = []
    for ids, labels in zip(tokenized["input_ids"], tokenized["labels"]):
        out.append(SampleStats(
            total_tokens=len(ids),
            loss_tokens=sum(1 for l in labels if l != -100),
            was_truncated=False,  # patched in build_dataset if needed; safe default
        ))
    return out


def collect_lengths(tokenized: Dataset) -> List[int]:
    """Return post-truncation token lengths for every sample."""
    return [len(ids) for ids in tokenized["input_ids"]]


def collect_raw_lengths(tokenized: Dataset) -> List[int]:
    """Return pre-truncation token lengths (original dataset lengths)."""
    if "original_length" in tokenized.column_names:
        return list(tokenized["original_length"])
    return collect_lengths(tokenized)


def print_length_stats(
    lengths: List[int],
    max_length: int = 0,
    raw_lengths: Optional[List[int]] = None,
) -> None:
    """
    Print token length distribution and truncation impact.

    Parameters
    ----------
    lengths : list[int]
        Token lengths after all truncation (output of collect_lengths).
    max_length : int
        cfg.max_length — used to compute truncation stats. Pass 0 to skip.
    raw_lengths : list[int], optional
        Pre-truncation lengths (output of collect_raw_lengths).
        When provided, shows the real dataset max alongside the capped max.
    """
    if not lengths:
        print("No lengths collected.")
        return

    s = sorted(lengths)
    n = len(s)

    def pct(p: float) -> int:
        return s[int((p / 100.0) * (n - 1))]

    ml = max_length or 0
    header = f"[Token length stats]  n={n}"
    if ml:
        header += f"  max_length={ml}"
    print(header)

    # ── Distribution ──────────────────────────────────────────────────────────
    print("\n  Distribution")
    percentiles = [("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99)]
    for label, p in percentiles:
        val = pct(p)
        marker = "  ← above max_length" if ml and val >= ml else ""
        print(f"    {label:4s}: {val:6d}{marker}")

    dataset_max = max(raw_lengths) if raw_lengths else None
    if dataset_max and dataset_max > s[-1]:
        print(f"    {'max':4s}: {s[-1]:6d}  ← above max_length  (dataset max: {dataset_max})")
    else:
        print(f"    {'max':4s}: {s[-1]:6d}")

    # ── Window utilisation ────────────────────────────────────────────────────
    if ml:
        print("\n  Window utilisation (tokens / max_length)")
        buckets = [
            ("< 25% ",   0,       ml * 0.25),
            ("25-50%",   ml*0.25, ml * 0.50),
            ("50-75%",   ml*0.50, ml * 0.75),
            ("75-99%",   ml*0.75, ml * 1.00 - 1),
            ("at max",   ml,      float("inf")),
        ]
        bar_width = 20
        for label, lo, hi in buckets:
            count = sum(1 for v in s if lo <= v <= hi)
            ratio = count / n
            filled = round(ratio * bar_width)
            bar_filled = "█" * filled + "░" * (bar_width - filled)
            print(f"    {label}: {count:5d} samples  ({ratio*100:5.1f}%)   [{bar_filled}]")

    # ── Truncation summary ────────────────────────────────────────────────────
    if ml:
        truncated = sum(1 for v in s if v >= ml)
        not_trunc = n - truncated
        print("\n  Truncation")
        print(f"    truncated    : {truncated:5d} / {n}  ({truncated/n*100:.1f}%)")
        print(f"    not truncated: {not_trunc:5d} / {n}  ({not_trunc/n*100:.1f}%)")


def debug_render_samples(raw: Dataset, tokenizer, cfg: TrainConfig, n: int) -> None:
    """
    Print the first *n* samples as they will actually appear in training.

    Pipeline applied (identical to build_dataset):
        canonicalize → structured_truncate → build_masks → tokenize → decode

    The decoded text reflects the final token-level cut, so what you see
    here is exactly what the model trains on — nothing more, nothing less.

    Header line explanation
    -----------------------
        tokens: 496 → 286 → 65  (max=65)
                 │     │     │
                 │     │     └─ after tokenize_with_char_mask (final)
                 │     └─────── after structured_truncate_messages
                 └─────────────  before any truncation
    """
    reasoning_keys = (
        tuple(cfg.reasoning_keys) if cfg.reasoning_keys else tuple(DEFAULT_REASONING_KEYS)
    )
    n = min(n, len(raw))
    print(f"\n[DEBUG] showing {n} samples (what the model actually trains on):\n")

    for i in range(n):
        msgs, report, extras = canonicalize_row(
            raw[i],
            tokenizer=tokenizer,
            messages_field=cfg.messages_field,
            schema=cfg.dataset_schema,
            extract_think_tags=cfg.extract_think_tags,
            reasoning_keys=reasoning_keys,
            reasoning_field=cfg.reasoning_field,
            think_mode=cfg.think_mode,
            think_max_tokens=cfg.think_max_tokens,
            think_role=cfg.think_role,
        )
        tools = extras.get("tools") if isinstance(extras, dict) else None

        # Step 1: count before any truncation
        original_len = _count_tokens(msgs, tools, tokenizer)

        # Step 2: structured truncation (message-level)
        msgs = structured_truncate_messages(msgs, tools, tokenizer, cfg)
        structured_len = _count_tokens(msgs, tools, tokenizer)

        # Step 3: build masks + tokenize (token-level fallback cut happens here)
        full_text, char_mask = build_text_and_masks(msgs, tools, tokenizer, cfg)
        result = tokenize_with_char_mask(full_text, char_mask, tokenizer, cfg)
        final_len = len(result["input_ids"])

        # Decode back to text so we see exactly what goes into training
        final_text = tokenizer.decode(result["input_ids"], skip_special_tokens=False)

        # Show which tokens carry loss (label != -100)
        loss_token_count = sum(1 for l in result["labels"] if l != -100)

        print("=" * 80)
        print(
            f"row {i} | schema={report.schema} "
            f"| tools={report.tools_present} "
            f"| extracted_think={report.extracted_think_from_content}"
            + (f" | unknown_roles={report.unknown_roles}" if report.unknown_roles else "")
        )
        print(
            f"tokens : {original_len} → {structured_len} → {final_len}  (max={cfg.max_length})"
            f"  |  loss tokens: {loss_token_count} / {final_len}"
        )
        print("-" * 80)
        print(final_text)
        print()
