---
title: Post-merge workflow
---

# Post-merge workflow

After `qlora-merge`, you have a standalone fp16 model directory.
Use this workflow to produce GGUF artifacts and publish them.

## Prerequisites

- A successful `qlora-merge` output directory.
- `qwen35-toolkit` installed.
- `llama.cpp` tools available for GGUF conversion/quantization.

## Steps

### Step 1 — Convert merged fp16 model to GGUF f16
Create a GGUF base file from merged weights.

| Step | Tool | What it produces |
|------|------|-----------------|
| Convert to GGUF | `llama.cpp/convert_hf_to_gguf.py` | `model-F16.gguf` |

Success criteria: `model-F16.gguf` exists.

### Step 2 — Quantize GGUF
Generate inference-friendly quant files.

| Step | Tool | What it produces |
|------|------|-----------------|
| Quantize | `llama-quantize` | `Q4_K_M` · `Q5_K_M` · `Q6_K` · `Q8_0` |

Success criteria: at least one quantized GGUF file (commonly `Q4_K_M`) is created.

### Step 3 — Upload artifacts to Hugging Face Hub
Publish GGUF output directory.

| Step | Tool | What it produces |
|------|------|-----------------|
| Upload | `qwen35-upload` | HuggingFace Hub repo |

Success criteria: Hub repo contains expected GGUF files.

## Expected result

- Merged fp16 model is converted to GGUF f16.
- Quantized GGUF variants are generated.
- Final artifacts are published to Hub.

## Common failures

- Conversion fails on wrong input path -> verify merge output directory first.
- `llama-quantize` not found -> build llama.cpp binaries before running step 2.
- Hub push fails -> rerun `hf auth login` or pass `--hf-token`.

## GGUF conversion command reference

See **[qwen35-toolkit -> GGUF conversion](https://techwithsergiu.github.io/qwen35-toolkit/gguf.html)** for exact command lines.

## Related

- [CPU merge](merge.md)
- [Inference](inference.md)
- [Training pipeline](training-pipeline.md)
