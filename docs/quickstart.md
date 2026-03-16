---
title: Quickstart
---

# Quickstart

## 1. Install

See [Setup](setup.md) for prerequisites and install order. Short version:

```bash
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git
pip install git+https://github.com/techwithsergiu/qwen-qlora-train.git
```

Authenticate once:

```bash
hf auth login
```

## 2. Pick a model and config

**Qwen3 (validated):**

| Config | Model | max_length |
|--------|-------|-----------|
| `configs/qwen3/1.7b.yaml` | `unsloth/Qwen3-1.7B-bnb-4bit` | 5120 |
| `configs/qwen3/4b.yaml` | `unsloth/Qwen3-4B-bnb-4bit` | 5120 |
| `configs/qwen3/8b.yaml` | `unsloth/Qwen3-8B-bnb-4bit` | 1024 |

> **8B note:** currently OOMs on RTX 3070 8 GB with unsloth 2026.3.4+.
> See [Troubleshooting](troubleshooting.md#known-issue-qwen3-8b-oom-on-unsloth-202634).

**Qwen3.5 (configs ready, end-to-end not yet validated):**

Pre-quantized models are published on HuggingFace — no local prep needed.

| Config | Model | max_length |
|--------|-------|-----------|
| `configs/qwen35/0.8b.yaml` | `techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit` | 5120 |
| `configs/qwen35/2b.yaml` | `techwithsergiu/Qwen3.5-text-2B-bnb-4bit` | 5120 |
| `configs/qwen35/4b.yaml` | `techwithsergiu/Qwen3.5-text-4B-bnb-4bit` | 5120 |
| `configs/qwen35/9b.yaml` | `techwithsergiu/Qwen3.5-text-9B-bnb-4bit` | 1024 |

> **9B note:** not a realistic training target on RTX 3070 8 GB — use 0.8B or 2B.

## 3. Set your dataset

Open the YAML config and set `dataset_id` — it is required and has no default:

```yaml
dataset_id: "your-hf-username/your-dataset"
```

See [Config reference](config-reference.md) for all available fields.

## 4. Check token length distribution

Before training, verify your dataset fits the configured `max_length`:

```bash
qlora-train --config configs/qwen3/1.7b.yaml --stats-only
```

- **`at max` > 30%** — too many samples truncated, raise `max_length` or pre-filter
- **`< 25%` > 50%** — samples are short, lower `max_length` to save VRAM

See [Dataset pipeline → diagnostics](dataset-pipeline.md#dataset-diagnostics) for full output explanation.

## 5. Inspect processed samples

See exactly what the model will train on — rendered text and loss mask:

```bash
qlora-train --config configs/qwen3/1.7b.yaml --debug-samples 2
```

## 6. Train

```bash
qlora-train --config configs/qwen3/1.7b.yaml
```

Outputs:
- LoRA adapter: `adapters/<run_name>/`
- Checkpoints: `outputs/<run_name>/`

## 7. Test the adapter

No merge needed — run inference directly on base model + adapter:

```bash
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/<run_name>
```

See [Inference](inference.md) for all modes: single prompt, interactive chat, thinking control.

## 8. Merge (optional)

Only needed for GGUF conversion or publishing:

```bash
qlora-merge \
  --base    unsloth/Qwen3-1.7B \
  --adapter adapters/<run_name> \
  --output  merged/<run_name>-f16 \
  --dtype   f16
```

See [CPU merge](merge.md) and [Post-merge workflow](post-merge-workflow.md).
