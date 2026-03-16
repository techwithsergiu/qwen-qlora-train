---
title: Project layout
---


# Project layout

```bash
qwen-qlora-train/
├── pyproject.toml
├── configs/
│   ├── qwen3/
│   │   ├── 1.7b.yaml          # sanity check
│   │   ├── 4b.yaml
│   │   └── 8b.yaml            # legacy config (ooms on unsloth 2026.3.4+; kept for reference)
│   └── qwen35/
│       ├── 0.8b.yaml          # default smoke-test
│       ├── 2b.yaml
│       ├── 4b.yaml
│       └── 9b.yaml            # reference only — not a training target on 8 GB VRAM
└── qwen_qlora_train/
    ├── train.py               # entry point — orchestrates training, saves LoRA adapter
    ├── infer.py               # entry point — base + adapter inference, interactive chat
    ├── config.py              # TrainConfig dataclass, YAML loading, validation
    ├── model_utils.py         # model + tokenizer loading, LoRA setup, GPU diagnostics
    ├── data_pipeline.py       # structured truncation, char-mask, tokenization, stats
    ├── dataset_parsers.py     # row canonicalization (reasoning fields, tools, schemas)
    └── merge_cpu.py           # merge base + adapter on CPU into fp16/bf16 weights
```
