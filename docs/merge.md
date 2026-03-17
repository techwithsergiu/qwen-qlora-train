---
title: CPU merge
---

# CPU merge

## Purpose

Merge a LoRA adapter into base weights on CPU to produce a standalone model.
This is usually needed for export/publishing workflows (for example GGUF conversion).

## When to use

- You need a standalone merged model directory.
- You plan GGUF conversion and quantization.
- You want to publish merged weights.

If you only want to test adapter behavior, use [Inference](inference.md) instead.

## Syntax

```text
qlora-merge --base <repo_or_path> --adapter <dir> --output <dir> [options]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | — | HF repo id or local path of full-precision base model (**required**) |
| `--adapter` | — | Path to trained LoRA adapter (**required**) |
| `--output` | — | Output directory for merged model (**required**) |
| `--dtype` | `bf16` | `bf16` or `f16` (`f16` commonly used for llama.cpp workflows) |
| `--loader` | `auto` | Loader auto-detect; `qwen3.5` forces `Qwen3_5ForConditionalGeneration` |
| `--hf-token` | `null` | HF access token |

## Examples

```bash
# Qwen3 1.7B
qlora-merge \
  --base    unsloth/Qwen3-1.7B \
  --adapter adapters/qwen3-1.7b-sanity \
  --output  merged/qwen3-1.7b-merged-f16 \
  --dtype   f16
```

```bash
# Qwen3.5 0.8B
qlora-merge \
  --base    unsloth/Qwen3.5-0.8B \
  --adapter adapters/qwen35-text-0.8b-sanity \
  --output  merged/qwen35-text-0.8b-merged-f16 \
  --dtype   f16
```

## Loader auto-detection

| Model name contains | Loader used |
|---------------------|-------------|
| `qwen3.5` or `qwen3_5` | `Qwen3_5ForConditionalGeneration` |
| anything else | `AutoModelForCausalLM` |

Use `--loader qwen3.5` to override when needed.

## Output examples

### Example output

```text
base    : unsloth/Qwen3-1.7B
adapter : adapters/qwen3-1.7b-sanity
output  : merged/qwen3-1.7b-merged-f16
dtype   : f16
loader  : AutoModelForCausalLM  (resolved from --loader=auto)

RAM : 47.5 GB free / 58.7 GB total
⏳ Loading base model (AutoModelForCausalLM) …
✅ Base model loaded  (1s)
⏳ Loading tokenizer …
✅ Tokenizer loaded
⏳ Loading LoRA adapter …
✅ Adapter loaded
⏳ Merging adapter into base weights …
✅ Merged  (3s)
💾 Saving merged model to 'merged/qwen3-1.7b-merged-f16' …
✅ Saved  (2s)
✅ Done. Merged model: merged/qwen3-1.7b-merged-f16
```

Interpretation: merge completed on CPU and produced standalone f16 weights.

Stable fields: merge stage sequence and final output marker.
Variable fields: RAM values, stage timings, path names, deprecation warnings.

## Edge cases / limitations

> [!NOTE]
> Typical merge runtime requires ~10-20 GB RAM depending on model size.
> VRAM is not required for the merge operation itself.

- Wrong loader selection can break model load; use `--loader` override.
- `--dtype f16` is usually preferred for downstream llama.cpp conversion.

## Related

- [Inference](inference.md)
- [Post-merge workflow](post-merge-workflow.md)
- [Training pipeline](training-pipeline.md)
