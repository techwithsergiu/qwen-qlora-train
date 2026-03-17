---
title: Training pipeline
---

# Training pipeline

## What this page covers

This page maps the full training workflow across two repositories:
- model preparation (`qwen35-toolkit`),
- adapter training (`qwen-qlora-train`),
- export back into GGUF workflow (`qwen35-toolkit`).

## When to use

- You need a high-level map before starting end-to-end training.
- You want to understand where adapter inference and merge fit.
- You need to align training outputs with export/publishing steps.

## Input -> Output

| Input | Output |
|------|--------|
| Qwen3.5 source VLM checkpoint | LoRA adapter (training artifact) |
| LoRA adapter + base model | merged fp16/bf16 model |
| merged fp16/bf16 model | GGUF quant files for inference/distribution |

## Diagram

```mermaid
flowchart TD
    SRC["Qwen/Qwen3.5-{size}<br/>f16 · source"]

    subgraph PREP ["1) Model prep — qwen35-toolkit"]
        BNBVLM["Qwen3.5-{size}-bnb-4bit<br/>BNB NF4 · VLM"]
        TEXTBNB["Qwen3.5-text-{size}-bnb-4bit<br/>BNB NF4 · text-only"]
        V1{{"verified"}}
        V3{{"verified"}}
        BNBVLM -->|"qwen35-strip --mode bnb"| TEXTBNB
        BNBVLM -->|"qwen35-verify-qwen35"| V1
        TEXTBNB -->|"qwen35-verify"| V3
        TEXTF16["Qwen3.5-text-{size}<br/>bf16 · text-only"]
        V2{{"verified"}}
        TEXTF16 -->|"qwen35-verify"| V2
    end

    subgraph TRAIN_PHASE ["2) Training — qwen-qlora-train"]
        TRAIN["QLoRA training"]
        ADAPTER["LoRA adapter"]
        INFER2["qlora-infer"]
        MERGED["merged fp16"]
        TRAIN --> ADAPTER
        ADAPTER -->|"qlora-infer"| INFER2
        ADAPTER -->|"qlora-merge"| MERGED
    end

    subgraph EXPORT ["3) Export — qwen35-toolkit"]
        GGUFT["GGUF f16"]
        QUANT["Q4_K_M / Q5_K_M / Q6_K / Q8_0"]
        GGUFT -->|"llama-quantize"| QUANT
    end

    HUB[("HuggingFace Hub")]

    SRC -->|"qwen35-convert"| BNBVLM
    SRC -->|"qwen35-strip --mode f16"| TEXTF16
    TEXTBNB --> TRAIN
    MERGED -->|"convert_hf_to_gguf.py"| GGUFT
    TEXTF16 -->|"convert_hf_to_gguf.py"| GGUFT
    V1 -->|"qwen35-upload"| HUB
    V3 -->|"qwen35-upload"| HUB
    V2 -->|"qwen35-upload"| HUB
    QUANT -->|"qwen35-upload"| HUB
```

## Steps

1. Prepare source model into a text-only training-ready artifact.
2. Train LoRA adapter and validate with `qlora-infer`.
3. Optionally merge adapter into standalone fp16.
4. Export merged model into GGUF and quantize.
5. Upload validated artifacts to Hub.

## Merge decision point

Use adapter-only path (no merge) when:
- you only need to evaluate or iterate quickly (`qlora-infer`).

Use merge path when:
- you need a standalone model directory,
- you need GGUF export,
- you need publishable merged weights.

## Cross-repo ownership

```text
qwen35-toolkit:
  convert / strip / verify / upload / GGUF conversion + quantization

qwen-qlora-train:
  train / adapter inference / CPU merge
```

After `qlora-merge`, GGUF conversion and upload are handled by `qwen35-toolkit`.

## Phase gates

```text
Gate 1 — Prep gate:
  - Training input checkpoint is text-only and verified.

Gate 2 — Train gate:
  - Adapter artifact is produced.
  - Basic `qlora-infer` checks pass.

Gate 3 — Merge gate (optional):
  - Standalone merged fp16/bf16 directory is created.

Gate 4 — Export gate:
  - GGUF f16 exists.
  - Required quant outputs are generated.

Gate 5 — Publish gate:
  - Upload dry-run looks correct.
  - Final push/pull sync completes without unexpected changes.
```

## Phase map

| Phase | Primary tools | Result |
|------|---------------|--------|
| Model prep | `qwen35-convert`, `qwen35-strip`, `qwen35-verify` | text-only checkpoint |
| Training | `qlora-train`, `qlora-infer` | adapter + validation |
| Merge/export | `qlora-merge`, `convert_hf_to_gguf.py`, `llama-quantize` | merged fp16 + GGUF quants |

## Related

- [Quickstart](quickstart.md)
- [Inference](inference.md)
- [Post-merge workflow](post-merge-workflow.md)
