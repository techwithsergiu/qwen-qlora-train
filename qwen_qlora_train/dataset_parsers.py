#!/usr/bin/env python3
"""
dataset_parsers.py — minimal dataset parsing + policy layer (Qwen-friendly, tools-aware)
======================================================================================

We keep a Qwen-friendly "canonical" message structure, and we *delegate rendering*
(including tool schemas, tool calls, and tool results) to the model's chat template.

Key idea:
- Many HF datasets store tool schemas in a separate column like `tools`.
- Qwen/Unsloth chat templates can inject those tool schemas into the system prompt
  *if* you pass them into `tokenizer.apply_chat_template(..., tools=TOOLS)`.

So we return:
  - canonical `messages` (list[dict])
  - `extras` (dict) e.g. {"tools": ...} to be passed to apply_chat_template

We also normalize common "thinking" variants:
- reasoning fields in message dicts -> `reasoning_content`
- <think>...</think> inside assistant.content -> extracted into `reasoning_content`
- cap/drop thinking with a simple policy
"""
from __future__ import annotations

import warnings
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL)

DEFAULT_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking", "reason")

# Roles the Qwen3 chat template understands natively.
# Anything outside this set will produce either a template error or
# silently malformed text — better to warn early.
KNOWN_QWEN_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ParseReport:
    schema: str
    extracted_think_from_content: int = 0
    moved_reasoning_keys: int = 0
    dropped_think_messages: int = 0
    tools_present: bool = False
    tools_parsed_from_string: bool = False
    unknown_roles: List[str] = None

    def __post_init__(self):
        if self.unknown_roles is None:
            self.unknown_roles = []


def detect_schema(row: Dict[str, Any], messages_field: str = "messages") -> str:
    if messages_field in row and isinstance(row[messages_field], list):
        return "messages"
    if "prompt" in row and ("response" in row or "completion" in row or "answer" in row):
        return "prompt_response"
    return "unknown"


def _ensure_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _normalize_tools(tools: Any) -> Tuple[Any, bool]:
    """
    Tools can appear as:
      - list[dict]
      - dict
      - JSON string
      - None
    We keep the object as-is, but if it's a string we try json.loads.
    Returns (tools_obj, parsed_from_string).
    """
    if tools is None:
        return None, False
    if isinstance(tools, (list, dict)):
        return tools, False
    if isinstance(tools, str):
        s = tools.strip()
        if not s:
            return None, False
        try:
            return json.loads(s), True
        except Exception:
            return tools, False
    return tools, False


def parse_row_to_messages(
    row: Dict[str, Any],
    messages_field: str = "messages",
    schema: str = "auto",
) -> Tuple[List[Dict[str, Any]], ParseReport, Dict[str, Any]]:
    """
    Convert one HF row into canonical messages + a report + extras (currently tools).
    """
    if schema == "auto":
        schema = detect_schema(row, messages_field=messages_field)

    report = ParseReport(schema=schema)
    extras: Dict[str, Any] = {}

    # tools column (common names). take first match.
    tools_val = None
    for k in ("tools", "tool_schemas", "functions", "function_schemas"):
        if k in row:
            tools_val = row.get(k)
            break

    tools_obj, parsed = _normalize_tools(tools_val)
    if tools_obj is not None:
        extras["tools"] = tools_obj
        report.tools_present = True
        report.tools_parsed_from_string = parsed

    if schema == "messages":
        msgs = row.get(messages_field)
        if not isinstance(msgs, list):
            return [], report, extras
        out: List[Dict[str, Any]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            mm = dict(m)
            mm["role"]    = _ensure_str(mm.get("role", "")).strip()
            # always normalise, _ensure_str(None) == ""
            mm["content"] = _ensure_str(mm.get("content"))

            # Drop empty system messages — they waste tokens and produce
            # <|im_start|>system\n<|im_end|> with no content in the template.
            if mm["role"] == "system" and not mm["content"].strip():
                continue

            out.append(mm)
        return out, report, extras

    if schema == "prompt_response":
        prompt = _ensure_str(row.get("prompt"))
        response = _ensure_str(row.get("response") or row.get("completion") or row.get("answer"))
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        return msgs, report, extras

    return [], report, extras


def extract_think_from_assistant_content(
    messages: List[Dict[str, Any]],
    to_field: str = "reasoning_content",
    assistant_role: str = "assistant",
) -> Tuple[List[Dict[str, Any]], int]:
    extracted = 0
    out: List[Dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        role = _ensure_str(mm.get("role")).strip()
        if role == assistant_role:
            content = _ensure_str(mm.get("content"))
            mt = THINK_RE.search(content) if content else None
            if mt:
                think_text = mt.group(1)
                new_content = THINK_RE.sub("", content, count=1).lstrip()
                mm["content"] = new_content
                existing = _ensure_str(mm.get(to_field)).strip()
                if not existing:
                    mm[to_field] = think_text
                extracted += 1
        out.append(mm)
    return out, extracted


def normalize_reasoning_keys(
    messages: List[Dict[str, Any]],
    reasoning_keys: Tuple[str, ...] = DEFAULT_REASONING_KEYS,
    to_field: str = "reasoning_content",
) -> Tuple[List[Dict[str, Any]], int]:
    moved = 0
    out: List[Dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        current = _ensure_str(mm.get(to_field)).strip()
        if not current:
            for rk in reasoning_keys:
                if rk == to_field:
                    continue
                if rk in mm:
                    val = _ensure_str(mm.get(rk)).strip()
                    if val:
                        mm[to_field] = val
                        moved += 1
                        break
        out.append(mm)
    return out, moved


def cap_tokens(text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens is None or max_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids)


def apply_think_policy(
    messages: List[Dict[str, Any]],
    tokenizer,
    think_mode: str = "keep",
    think_max_tokens: int = 0,
    think_role: str = "think",
    reasoning_field: str = "reasoning_content",
) -> Tuple[List[Dict[str, Any]], int]:
    dropped = 0
    out: List[Dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        role = _ensure_str(mm.get("role")).strip()

        # Dedicated think role (rare)
        if role == think_role:
            if think_mode == "drop":
                dropped += 1
                continue
            mm["content"] = cap_tokens(
                _ensure_str(mm.get("content")),
                tokenizer,
                think_max_tokens
            )

        # Qwen reasoning field
        if reasoning_field in mm:
            if think_mode == "drop":
                mm[reasoning_field] = ""
            else:
                mm[reasoning_field] = cap_tokens(
                    _ensure_str(mm.get(reasoning_field)),
                    tokenizer,
                    think_max_tokens
                )

        out.append(mm)
    return out, dropped


def canonicalize_row(
    row: Dict[str, Any],
    tokenizer,
    messages_field: str = "messages",
    schema: str = "auto",
    extract_think_tags: bool = True,
    reasoning_keys: Tuple[str, ...] = DEFAULT_REASONING_KEYS,
    reasoning_field: str = "reasoning_content",
    think_mode: str = "keep",
    think_max_tokens: int = 0,
    think_role: str = "think",
) -> Tuple[List[Dict[str, Any]], ParseReport, Dict[str, Any]]:
    msgs, report, extras = parse_row_to_messages(row, messages_field=messages_field, schema=schema)
    if not msgs:
        return [], report, extras

    msgs, moved = normalize_reasoning_keys(
        msgs,
        reasoning_keys=reasoning_keys,
        to_field=reasoning_field
    )
    report.moved_reasoning_keys += moved

    if extract_think_tags:
        msgs, extracted = extract_think_from_assistant_content(msgs, to_field=reasoning_field)
        report.extracted_think_from_content += extracted

    msgs, dropped = apply_think_policy(
        msgs,
        tokenizer=tokenizer,
        think_mode=think_mode,
        think_max_tokens=think_max_tokens,
        think_role=think_role,
        reasoning_field=reasoning_field,
    )
    report.dropped_think_messages += dropped

    # ── Role validation ───────────────────────────────────────────────────────
    # Warn about any role the Qwen template won't recognise.
    # We don't drop the message — the template may handle it or raise a
    # clear error on its own; the warning just surfaces it early.
    for m in msgs:
        role = _ensure_str(m.get("role")).strip()
        if role and role not in KNOWN_QWEN_ROLES and role != think_role:
            if role not in report.unknown_roles:
                report.unknown_roles.append(role)
            warnings.warn(
                f"[dataset_parsers] Unknown role '{role}' in row "
                f"(schema={report.schema}). "
                f"Qwen3 template may not handle it correctly. "
                f"Known roles: {sorted(KNOWN_QWEN_ROLES)}",
                stacklevel=2,
            )

    return msgs, report, extras
