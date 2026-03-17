---
title: Inference
---

# Inference

## Purpose

Run inference for:
- base model only,
- base model + LoRA adapter,
- merged model.

Primary use case: validate adapter behavior immediately after training, without CPU merge.

## When to use

- You need a fast sanity check after `qlora-train`.
- You want to compare adapter and merged output.
- You need single-prompt or interactive checks.

## Syntax

```text
qlora-infer --model <repo_or_path> [options]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF repo id or local path (**required**) |
| `--adapter` | `null` | LoRA adapter path; omit for merged models |
| `--hf-token` | `null` | HF access token |
| `--backend` | `auto` | `auto` -> `unsloth` for `bnb-4bit`, otherwise `transformers` |
| `--dtype` | `f16` | `f16` or `bf16` |
| `--chat-template` | `qwen3` | Unsloth template key (unsloth path only) |
| `--max-seq-length` | `4096` | KV cache allocation (unsloth path only) |
| `--max-new` | `1024` | Max generated tokens |
| `--temp` | `0.7` | Sampling temperature (`0` for greedy) |
| `--top-p` | `0.9` | Top-p sampling |
| `--prompt` | `null` | Single prompt mode |
| `--interactive` | `false` | Interactive chat loop |
| `--no-thinking` | `false` | Disable `<think>` output |

## Backend decision logic

```text
1. `--backend auto`:
   - if model id contains `bnb-4bit` -> use `unsloth`
   - otherwise -> use `transformers`
2. Adapter attachment:
   - if `--adapter` provided -> base + adapter path
   - if omitted -> merged/base-only inference
3. Interaction mode:
   - `--prompt` for single-shot check
   - `--interactive` for chat loop
4. Reasoning output mode:
   - default: thinking enabled
   - `--no-thinking`: answer-only output view
```

## Examples

```bash
# Smoke tests (default mode)
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity
```

```bash
# Single prompt
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity \
  --prompt  "Explain LoRA in two sentences."
```

```bash
# Interactive chat
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity \
  --interactive
```

```bash
# Thinking on/off
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity

qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity \
  --no-thinking
```

```bash
# Merged model (no adapter)
qlora-infer --model merged/qwen3-1.7b-merged-f16 --dtype f16
```

## Edge cases / limitations

> [!NOTE]
> `--backend auto` picks `unsloth` when model id contains `bnb-4bit`,
> and `transformers` otherwise.

- `--interactive` keeps conversation history until `exit` or `Ctrl-C`.
- For merged fp16 models, omit `--adapter`.
- Use `--no-thinking` for answer-only evaluation.

## Related

- [Quickstart](quickstart.md)
- [CPU merge](merge.md)
- [Troubleshooting](troubleshooting.md)
