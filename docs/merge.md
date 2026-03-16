---
title: CPU merge
---

# CPU merge

Needed only if you want a standalone merged model (for GGUF conversion or publishing).
If you just want to test the adapter, use [`qlora-infer`](inference.md) — no merge needed.

```bash
# Qwen3 1.7B
qlora-merge \
  --base    unsloth/Qwen3-1.7B \
  --adapter adapters/qwen3-1.7b-sanity \
  --output  merged/qwen3-1.7b-merged-f16 \
  --dtype   f16

# Qwen3.5 0.8B
qlora-merge \
  --base    unsloth/Qwen3.5-0.8B \
  --adapter adapters/qwen35-text-0.8b-sanity \
  --output  merged/qwen35-text-0.8b-merged-f16 \
  --dtype   f16
```

Hardware: ~10–20 GB RAM depending on model size, no VRAM needed.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | — | HF repo id or local path of full-precision base model (**required**) |
| `--adapter` | — | Path to trained LoRA adapter directory (**required**) |
| `--output` | — | Directory where merged model will be saved (**required**) |
| `--dtype` | `bf16` | `bf16` (matches Qwen3.5 training dtype) or `f16` (llama.cpp compat) |
| `--loader` | `auto` | `auto` detects from name · `qwen3.5` forces `Qwen3_5ForConditionalGeneration` |
| `--hf-token` | `null` | HF access token |

## Loader auto-detection

| Model name contains | Loader used |
|---------------------|-------------|
| `qwen3.5` or `qwen3_5` | `Qwen3_5ForConditionalGeneration` |
| anything else | `AutoModelForCausalLM` |

Use `--loader qwen3.5` to override if auto-detection gets it wrong.

## Adding support for a new model family

The loader registry lives in `merge_cpu.py` as two module-level constants:

```python
# 1. Map a loader key to a transformers class name
LOADER_CLASSES: dict[str, str] = {
    "qwen3.5": "Qwen3_5ForConditionalGeneration",
    "auto":    "AutoModelForCausalLM",
}

# 2. Map a substring in the model name to a loader key (auto-detection)
AUTO_DETECT: list[tuple[str, str]] = [
    ("qwen3.5", "qwen3.5"),
    ("qwen3_5", "qwen3.5"),
]
```

To add a new family (e.g. a hypothetical `Qwen4ForCausalLM`):
1. Add an entry to `LOADER_CLASSES`: `"qwen4": "Qwen4ForCausalLM"`
2. Add a row to `AUTO_DETECT`: `("qwen4", "qwen4")`
3. The new key becomes available as `--loader qwen4` and fires automatically when the model name contains `"qwen4"`

---

After merging, see [Post-merge workflow](post-merge-workflow.md) for GGUF conversion, quantization, and Hub upload.
