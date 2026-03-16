---
title: Inference
---

# Inference

`qlora-infer` runs inference on a base model, a base model with a LoRA adapter, or a merged model. The primary use case is verifying an adapter immediately after training — **no CPU merge needed**.

`--backend auto` selects the backend automatically: `unsloth` (4-bit) if the model id contains `bnb-4bit`, `transformers` (fp16/bf16) otherwise.

---

## Modes

### Smoke tests

Run four predefined prompts covering short answer, code generation, stop behavior, and thinking. The fastest way to verify an adapter works correctly after training.

```bash
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity
```

### Single prompt

Run one prompt and exit. Useful for scripted checks or quick comparisons between adapters.

```bash
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity \
  --prompt  "Explain LoRA in two sentences."
```

### Interactive chat

Maintains full conversation history across turns. Useful for testing multi-turn behavior and system prompt adherence.

```bash
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity \
  --interactive
```

Type `exit` or press `Ctrl-C` to quit.

### Thinking mode

Qwen3 models generate a `<think>…</think>` block before answering. Use `--no-thinking` to suppress it — useful for testing `answer_only` trained adapters or for cleaner output during review.

```bash
# With thinking (default)
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity

# Without thinking
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/qwen3-1.7b-sanity \
  --no-thinking
```

### Merged model

Run a merged fp16 model directly — no adapter needed. Use `--dtype f16` and `--backend transformers` (or let `auto` detect it from the model path).

```bash
qlora-infer --model merged/qwen3-1.7b-merged-f16 --dtype f16
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF repo id or local path (**required**) |
| `--adapter` | `null` | LoRA adapter directory — omit for merged models |
| `--hf-token` | `null` | HF access token |
| `--backend` | `auto` | `auto` detects from model id · `unsloth` (4-bit) · `transformers` (fp16/bf16) |
| `--dtype` | `f16` | `f16` or `bf16` |
| `--chat-template` | `qwen3` | Unsloth template key (unsloth path only) |
| `--max-seq-length` | `4096` | KV cache allocation — match your training `max_seq_length` (unsloth path only) |
| `--max-new` | `1024` | Max new tokens to generate |
| `--temp` | `0.7` | Sampling temperature — set to `0` for greedy decoding |
| `--top-p` | `0.9` | Top-p sampling |
| `--prompt` | `null` | Single user prompt — skips predefined smoke tests |
| `--interactive` | `false` | Interactive chat loop with full conversation history |
| `--no-thinking` | `false` | Set `enable_thinking=False` — suppresses `<think>` block |

---

If you want a standalone model for GGUF conversion, see [CPU merge](merge.md).
