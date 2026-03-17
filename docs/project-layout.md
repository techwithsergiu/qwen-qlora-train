---
title: Project layout
---

# Project layout

## What this page covers

This page maps the repository structure and explains what each major module is responsible for.

## When to use

- You are onboarding and need quick orientation in the codebase.
- You want to find where config, data processing, training, inference, and merge logic live.
- You are preparing code changes and need file ownership context.

## Repository tree

```bash
qwen-qlora-train/
├── pyproject.toml
├── configs/
│   ├── qwen3/
│   │   ├── 1.7b.yaml          # primary validated starting config
│   │   ├── 4b.yaml
│   │   └── 8b.yaml            # reference; OOM on unsloth 2026.3.4+ (RTX 3070 8 GB)
│   └── qwen35/
│       ├── 0.8b.yaml          # default smoke-test
│       ├── 2b.yaml
│       ├── 4b.yaml
│       └── 9b.yaml            # reference only on 8 GB VRAM class
└── qwen_qlora_train/
    ├── train.py               # training entry point
    ├── infer.py               # inference entry point (base/adapter)
    ├── config.py              # TrainConfig dataclass + YAML validation
    ├── model_utils.py         # model/tokenizer loading + LoRA setup + diagnostics
    ├── data_pipeline.py       # truncation, masking, tokenization, stats
    ├── dataset_parsers.py     # row canonicalization (schema/reasoning/tools)
    └── merge_cpu.py           # CPU merge: base + adapter -> fp16/bf16
```

## Module responsibilities

| Area | Files | Responsibility |
|------|-------|----------------|
| Configs | `configs/*` | Training presets by family/size |
| Data processing | `dataset_parsers.py`, `data_pipeline.py` | Canonicalization, truncation, masks, tokenization |
| Training | `train.py`, `model_utils.py` | QLoRA setup and run orchestration |
| Inference | `infer.py` | Base + adapter / merged inference |
| Merge | `merge_cpu.py` | CPU merge for export workflows |

## Related

- [Quickstart](quickstart.md)
- [Config reference](config-reference.md)
- [Training pipeline](training-pipeline.md)
