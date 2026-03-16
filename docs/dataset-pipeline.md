---
title: Dataset pipeline
---

# Dataset pipeline

## How it works

The pipeline never builds prompts manually. Instead it works in three layers:

```text
HF dataset row
      │
      ▼  dataset_parsers.py — canonicalize_row()
      │  • detect schema (messages / prompt_response / auto)
      │  • normalize reasoning fields → reasoning_content
      │  • extract <think>...</think> from assistant content
      │  • apply think_mode / think_max_tokens policy
      │  • detect tools column → extras{"tools": ...}
      │
      ▼  data_pipeline.py — structured_truncate_messages()
      │  • prune middle history turns to fit max_length
      │  • trim reasoning_content in kept turns
      │  • fallback: token-level left-truncation
      │
      ▼  data_pipeline.py — build_text_and_masks()
      │  • tokenizer.apply_chat_template(messages, tools=tools)
      │  • char-level mask: 1 = loss, 0 = masked
      │  • assistant_roles masking
      │  • think_loss sub-span masking
      │
      ▼  data_pipeline.py — tokenize_with_char_mask()
         • tokenize + project char mask → token labels
         • returns {input_ids, attention_mask, labels}
```

The chat template (Jinja2) lives in the tokenizer — we just call it.
`get_chat_template(tokenizer, chat_template="qwen3")` patches it once at load
time so the correct template is always used regardless of what the Hub model
ships with.

---

## Supported dataset schemas

### `messages` (default — OpenAI API format)

The most common format. Detected when the row has a list under `messages_field`.

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

Config:

```yaml
dataset_schema: auto      # or "messages" explicitly
messages_field: messages  # column name
```

### `prompt_response`

Detected when the row has `prompt` + one of `response` / `completion` / `answer`.
Converted to a two-message conversation internally.

```json
{"prompt": "What is 2+2?", "response": "4"}
```

Config:

```yaml
dataset_schema: auto   # auto-detected, no extra config needed
```

---

## Reasoning / thinking fields

Qwen3 stores reasoning inside assistant messages as `reasoning_content`.
Many datasets use different field names — all of these are recognized and
normalized to `reasoning_content` before the template sees them:

```yaml
reasoning_keys: ["reasoning", "reasoning_content", "thinking", "reason"]
```

Also handles `<think>...</think>` inline inside `assistant.content`:

```json
{"role": "assistant", "content": "<think>let me think...</think>\n4"}
```

→ extracted into `reasoning_content`, content becomes `"4"`.

---

## Tools support

Tool schemas are auto-detected from common column names and passed directly
to `apply_chat_template(messages, tools=tools)` — the Qwen template injects
them into the system prompt automatically.

Detected columns (first match wins): `tools`, `tool_schemas`, `functions`, `function_schemas`.

Tools can be stored as a list, dict, or JSON string — all normalized.

> ⚠️ Tool schemas can add hundreds or thousands of tokens per sample.
> Always run `--stats-only` first when training on tool datasets.

---

## Assistant-only loss masking

```yaml
assistant_roles: ["assistant"]
```

Loss is computed **only on spans produced by roles in `assistant_roles`**.
System / user / tool messages are masked with `-100`.

Masking is done in **character space** before tokenizing — the pipeline finds
assistant spans in the rendered string and projects a `1/0` char mask onto token
labels. This avoids tokenizer-specific heuristics (sentinel tokens, role tag ids)
and works correctly with any subword vocabulary.

### Known roles

The Qwen3 chat template natively understands: `system`, `user`, `assistant`, `tool`.

If a row contains a message with any other role, `canonicalize_row()` emits a
`UserWarning` identifying the role and the schema, but does **not** drop the
message — the template will either handle it or raise its own error. The warning
surfaces the problem early so you can decide whether to filter those rows.

---

## Truncation strategy

```yaml
truncate_side: left   # always recommended for chat
max_length: 4096
```

Naive left-truncation (keep last N tokens) destroys the system prompt and
tool schemas. The pipeline uses **structured truncation** instead:

1. Drop oldest middle turns first (keep system + last user + last assistant)
2. Trim `reasoning_content` in kept turns
3. Fallback: token-level left-truncation only if still over `max_length`

---

## Dataset diagnostics

```bash
qlora-train --config configs/qwen35/0.8b.yaml --stats-only
```

Output:

```text
[Token length stats]  n=247  max_length=5120

  Distribution
    p25 :    360
    p50 :    567
    p75 :   1741
    p90 :   3608
    p95 :   4404
    p99 :   5120  ← above max_length
    max :   5120  ← above max_length  (dataset max: 8327)

  Window utilisation (tokens / max_length)
    < 25% :   180 samples  ( 72.9%)   [███████████████░░░░░]
    25-50%:    19 samples  (  7.7%)   [██░░░░░░░░░░░░░░░░░░]
    50-75%:    26 samples  ( 10.5%)   [██░░░░░░░░░░░░░░░░░░]
    75-99%:    16 samples  (  6.5%)   [█░░░░░░░░░░░░░░░░░░░]
    at max:     6 samples  (  2.4%)   [░░░░░░░░░░░░░░░░░░░░]

  Truncation
    truncated    :     6 / 247  (2.4%)
    not truncated:   241 / 247  (97.6%)

(stats-only) exiting without training.
```

What to look for:

- **`at max` > 30%** — too many samples truncated, raise `max_length` or pre-filter
- **`< 25%` > 50%** — samples are very short, lower `max_length` to save VRAM

```bash
# Debug render — see exact tokens + loss mask for 2 samples
qlora-train --config configs/qwen35/0.8b.yaml --debug-samples 2
```

---

## ParseReport — canonicalization telemetry

`canonicalize_row()` returns a `ParseReport` dataclass alongside the message list.
The training pipeline aggregates these across the dataset and can surface them in
logs. Fields:

| Field | Type | What it counts |
|-------|------|----------------|
| `schema` | `str` | Detected schema: `"messages"` / `"prompt_response"` / `"unknown"` |
| `extracted_think_from_content` | `int` | Assistant messages where `<think>…</think>` was extracted out of `content` into `reasoning_content` |
| `moved_reasoning_keys` | `int` | Messages where an alternative reasoning key (`reasoning`, `thinking`, `reason`) was renamed to `reasoning_content` |
| `dropped_think_messages` | `int` | Messages with role `think_role` dropped because `think_mode: drop` |
| `tools_present` | `bool` | Whether a tools column was found in the row |
| `tools_parsed_from_string` | `bool` | Whether the tools value was a JSON string that needed parsing |
| `unknown_roles` | `list[str]` | Any role names not in `{system, user, assistant, tool}` |
