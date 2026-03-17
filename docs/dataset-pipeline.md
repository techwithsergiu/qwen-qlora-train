---
title: Dataset pipeline
---

# Dataset pipeline

## Problem

Training quality depends on how raw dataset rows are converted into tokenized samples with correct loss masks.
Naive prompt construction and naive truncation can break system context, tools context, and assistant-only loss behavior.

## Architecture / mechanics

The pipeline is layered and avoids manual prompt assembly.

```text
Input:
  HF dataset row

Stage 1 — dataset_parsers.py / canonicalize_row():
  - Detect schema (`messages` / `prompt_response` / `auto`)
  - Normalize reasoning fields -> `reasoning_content`
  - Extract inline `<think>...</think>` from assistant content
  - Apply `think_mode` / `think_max_tokens`
  - Detect tools column -> `extras{"tools": ...}`

Stage 2 — data_pipeline.py / structured_truncate_messages():
  - Prune middle history turns to fit `max_length`
  - Trim `reasoning_content` in kept turns
  - Fallback to token-level left truncation

Stage 3 — data_pipeline.py / build_text_and_masks():
  - Render with `tokenizer.apply_chat_template(messages, tools=tools)`
  - Build char-level loss mask (`1` = loss, `0` = masked)
  - Apply `assistant_roles` masking
  - Apply `think_loss` sub-span masking

Stage 4 — data_pipeline.py / tokenize_with_char_mask():
  - Tokenize rendered text
  - Project char mask -> token `labels`
  - Return `{input_ids, attention_mask, labels}`

Output:
  Trainer-ready tensors with role-accurate loss masking
```

`get_chat_template(tokenizer, chat_template="qwen3")` patches template usage at load time,
so processing stays consistent across model repos.
The pipeline always delegates final rendering to tokenizer Jinja template
(`apply_chat_template`) instead of manually concatenating prompt strings.

## Supported schemas

### `messages`

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

```yaml
dataset_schema: auto
messages_field: messages
```

### `prompt_response`

```json
{"prompt": "What is 2+2?", "response": "4"}
```

```yaml
dataset_schema: auto
```

## Reasoning and tools handling

### Reasoning normalization

Alternative fields are normalized into `reasoning_content` before templating:

```yaml
reasoning_keys: ["reasoning", "reasoning_content", "thinking", "reason"]
```

Inline `<think>...</think>` in assistant content is extracted into `reasoning_content`.

### Tools injection

Tools schemas are auto-detected from common columns and passed into chat template:
- `tools`, `tool_schemas`, `functions`, `function_schemas`.
First matching column is used.

Supported input forms: list, dict, JSON string.

> [!WARNING]
> Tool schemas can add many tokens per sample. Run `--stats-only` before training on tool-heavy datasets.

## Loss masking behavior

```yaml
assistant_roles: ["assistant"]
```

Loss is applied only on spans produced by roles in `assistant_roles`.
Masking is built in character space first, then projected to token labels,
which avoids tokenizer-specific role-token heuristics.

Known template roles: `system`, `user`, `assistant`, `tool`.
Unknown roles are warned but not silently dropped.

## Truncation strategy

```yaml
truncate_side: left
max_length: 4096
```

Structured truncation order:
1. Preserve conversation anchors:
   - keep system context and latest user/assistant turns.
2. Remove oldest middle turns first:
   - shrink history with minimal semantic damage.
3. Trim `reasoning_content` in kept turns:
   - reduce token load before cutting answer context.
4. Apply token-level left truncation only as final fallback:
   - used only when structured pruning is not enough.

Why this order:
- naive left truncation can drop system/tool context first,
- structured truncation keeps instruction and recent interaction integrity longer.

## Practical impact

- Keeps conversational structure more stable under max-length pressure.
- Improves consistency of assistant-only loss behavior.
- Reduces hidden regressions from schema mismatch and unknown roles.

## Recommended defaults

```yaml
dataset_schema: auto
messages_field: messages
assistant_roles: ["assistant"]
truncate_side: left
extract_think_tags: true
```

Diagnostics commands:

```bash
# Length distribution and truncation rates
qlora-train --config configs/qwen35/0.8b.yaml --stats-only

# Render processed samples and masks
qlora-train --config configs/qwen35/0.8b.yaml --debug-samples 2
```

Heuristics:
- `at max > 30%` -> increase `max_length` or pre-filter long samples.
- `< 25% > 50%` -> lower `max_length` to reduce VRAM usage.

## ParseReport telemetry

`canonicalize_row()` emits `ParseReport` for dataset-level telemetry.

| Field | Type | Meaning |
|-------|------|---------|
| `schema` | `str` | Detected schema: `messages` / `prompt_response` / `unknown` |
| `extracted_think_from_content` | `int` | Count of extracted inline `<think>` blocks |
| `moved_reasoning_keys` | `int` | Count of renamed reasoning keys |
| `dropped_think_messages` | `int` | Messages dropped due to `think_mode: drop` |
| `tools_present` | `bool` | Tools column found in row |
| `tools_parsed_from_string` | `bool` | Tools parsed from JSON string |
| `unknown_roles` | `list[str]` | Roles not in `{system, user, assistant, tool}` |

## Related

- [Config reference](config-reference.md)
- [Reasoning control](reasoning.md)
- [Quickstart](quickstart.md)
