---
title: Reasoning control
---

# Reasoning / thinking control

Qwen3 and Qwen3.5 embed reasoning inside assistant messages as `<think>…</think>`.

For the full field reference see [Config reference](config-reference.md#reasoning-thinking).

---

## think_mode — keep or drop reasoning

```yaml
think_mode: keep   # keep | drop
```

- `keep` — preserve reasoning content (apply `think_max_tokens` cap if set)
- `drop` — remove reasoning entirely; trains pure answer-style behavior

---

## think_loss — gradient scope inside assistant turns

```yaml
think_loss: all   # all | answer_only | answer_plus_think
```

| Value | What gets gradient |
|-------|--------------------|
| `all` | Full assistant span — `<think>` content + answer |
| `answer_only` | Tokens after `</think>` only — recommended for agent/tool training |
| `answer_plus_think` | Think content + answer, but not the literal `<think>`/`</think>` tags |

---

## Recommended patterns

```yaml
# Agent / tool training — stable answers, no think gradient
think_loss: answer_only
think_mode: keep

# Reasoning dataset — train full chain of thought
think_loss: all
think_mode: keep

# Pure answer-style (no reasoning at all)
think_mode: drop
think_loss: all   # irrelevant when think_mode: drop
```

---

## Token cap warning

```yaml
think_max_tokens: 512
```

> ⚠️ Capping creates truncated reasoning mid-thought. The model may learn
> to start a chain of thought and abruptly stop. Prefer shorter-think datasets
> over aggressive capping. Use only if you have a specific reason.
