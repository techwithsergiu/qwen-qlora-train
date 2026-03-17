---
title: Reasoning control
---

# Reasoning / thinking control

## Purpose

Control how reasoning (`<think>...</think>`) is kept, removed, and trained.
This page covers `think_mode`, `think_loss`, and `think_max_tokens` behavior.

## When to use

- You want answer-only behavior for agent/tool datasets.
- You want to preserve full reasoning traces for reasoning datasets.
- You need to cap reasoning token length for memory/perf constraints.

## Syntax (config fields)

```yaml
think_mode: keep                 # keep | drop
think_loss: answer_only          # all | answer_only | answer_plus_think
think_max_tokens: 0              # 0 = no cap
```

## Options

### `think_mode`

| Value | Effect |
|-------|--------|
| `keep` | Keep reasoning content (with optional cap) |
| `drop` | Remove reasoning content entirely |

### `think_loss`

| Value | Gradient scope |
|-------|----------------|
| `all` | Full assistant span: think + answer |
| `answer_only` | Tokens after `</think>` only |
| `answer_plus_think` | Think content + answer, excluding literal tags |

### `think_max_tokens`

- `0` means no cap.
- Positive values cap reasoning token count per message.

## Decision flow

```text
1. Decide content retention (`think_mode`):
   - `keep` to preserve reasoning content.
   - `drop` for answer-only datasets/behavior.
2. Decide gradient scope (`think_loss`):
   - `all`, `answer_only`, or `answer_plus_think`.
3. Decide cap (`think_max_tokens`):
   - `0` for unlimited.
   - positive value to constrain long reasoning traces.
4. Validate against eval target:
   - train/eval behavior should match (answer-only vs full reasoning).
```

## Examples

```yaml
# Agent/tool training: keep think text, train only final answer
think_mode: keep
think_loss: answer_only
```

```yaml
# Full reasoning training
think_mode: keep
think_loss: all
```

```yaml
# Pure answer-style behavior
think_mode: drop
think_loss: all
```

```yaml
# Reasoning cap example
think_max_tokens: 512
```

## Edge cases / limitations

> [!WARNING]
> Aggressive `think_max_tokens` can truncate reasoning mid-thought and teach abrupt stop patterns.
> Prefer cleaner datasets over heavy truncation when possible.

- `think_loss` has no practical effect on reasoning text when `think_mode: drop`.
- Keep reasoning settings aligned with your evaluation target (answer-only vs full reasoning).

## Related

- [Config reference](config-reference.md#reasoning--thinking)
- [Dataset pipeline](dataset-pipeline.md)
- [Quickstart](quickstart.md)
