---
title: Quickstart
---

# Quickstart

> [!NOTE]
> Validated scope on RTX 3070 8 GB in this repo: Qwen3 1.7B and 4B.
> Qwen3 8B currently OOMs on unsloth 2026.3.4+, and larger models should be treated as experimental on similar hardware.

## Prerequisites

See [Setup](setup.md) for required install order.

```bash
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git
pip install git+https://github.com/techwithsergiu/qwen-qlora-train.git
hf auth login
```

## Steps

### Step 1 — Choose model and config
Start with a validated config first.

```text
Recommended first run: configs/qwen3/1.7b.yaml
```

Model/config references:

| Config | Model | max_length | Status |
|--------|-------|------------|--------|
| `configs/qwen3/1.7b.yaml` | `unsloth/Qwen3-1.7B-bnb-4bit` | 5120 | validated |
| `configs/qwen3/4b.yaml` | `unsloth/Qwen3-4B-bnb-4bit` | 5120 | validated |
| `configs/qwen3/8b.yaml` | `unsloth/Qwen3-8B-bnb-4bit` | 1024 | currently OOM |
| `configs/qwen35/0.8b.yaml` | `techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit` | 5120 | experimental |
| `configs/qwen35/2b.yaml` | `techwithsergiu/Qwen3.5-text-2B-bnb-4bit` | 5120 | experimental |
| `configs/qwen35/4b.yaml` | `techwithsergiu/Qwen3.5-text-4B-bnb-4bit` | 5120 | experimental |
| `configs/qwen35/9b.yaml` | `techwithsergiu/Qwen3.5-text-9B-bnb-4bit` | 1024 | experimental |

Success criteria: chosen config matches your hardware constraints.

### Step 2 — Set dataset id
Set required dataset source in YAML.

```yaml
dataset_id: "your-hf-username/your-dataset"
```

Success criteria: `dataset_id` is non-empty and points to accessible data.

### Step 3 — Check token-length distribution
Run stats-only pass before training.

```bash
qlora-train --config configs/qwen3/1.7b.yaml --stats-only
```

Success criteria: length distribution is acceptable for current `max_length`.

### Step 4 — Inspect processed samples
Review rendered text and loss masks.

```bash
qlora-train --config configs/qwen3/1.7b.yaml --debug-samples 2
```

Success criteria: samples look structurally correct for your target behavior.

### Step 5 — Train adapter
Run training with selected config.

```bash
qlora-train --config configs/qwen3/1.7b.yaml
```

Success criteria: adapter/checkpoints are created without fatal errors.

### Step 6 — Validate adapter with inference
Test base + adapter directly (no merge required).

```bash
qlora-infer \
  --model   unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter adapters/<run_name>
```

Success criteria: outputs are coherent and aligned with your training objective.

### Step 7 — Merge (optional)
Merge only if you need standalone fp16 for export/publishing.

```bash
qlora-merge \
  --base    unsloth/Qwen3-1.7B \
  --adapter adapters/<run_name> \
  --output  merged/<run_name>-f16 \
  --dtype   f16
```

Success criteria: merged model directory is created.

## Training command output examples

These examples correspond to this page's command sequence:
- Step 3 (`--stats-only`) for token-length diagnostics.
- Step 5 (`qlora-train`) for full-run success signal.

### Example output (`--stats-only`)

```text
[Token length stats]  n=247  max_length=5120

Distribution
  p25 :    360
  p50 :    567
  p75 :   1741
  p90 :   3608
  p95 :   4404
  p99 :   5120  ← above max_length
  max :   5120

Truncation
  truncated    :     6 / 247  (2.4%)
  not truncated:   241 / 247  (97.6%)

(stats-only) exiting without training.
```

Interpretation: dataset mostly fits `max_length`, with a small truncated subset to watch.

Stable fields: percentile/truncation keys and stats-only exit marker.
Variable fields: sample counts, percentiles, truncation rate.

### Example output (`qlora-train` full run success signal)

```text
Num examples = 247 | Num Epochs = 9 | Total steps = 500
Batch size per device = 1 | Gradient accumulation steps = 4
Total batch size (1 x 4 x 1) = 4
Trainable parameters = 17,432,576 of 1,738,007,552 (1.00% trained)
...
{'train_runtime': '1353', 'train_samples_per_second': '1.478', 'train_steps_per_second': '0.369', 'train_loss': '0.3997', 'epoch': '8.065'}
100%|...| 500/500 [22:33<00:00,  2.71s/it]

✅  LoRA adapter saved to: adapters/qwen3-1.7b-sanity
```

Interpretation: run finished and produced a valid adapter artifact.

Stable fields: run summary keys and adapter-save success marker.
Variable fields: throughput/loss/runtime metrics.

## Expected result

- LoRA adapter saved under `adapters/<run_name>/`.
- Training checkpoints under `outputs/<run_name>/`.
- Optional merged fp16 artifact under `merged/<run_name>-f16`.

## Common failures

- OOM before first step -> reduce `max_length`, lower `lora_r`, keep batch size 1.
- Many samples at max length -> adjust dataset or raise `max_length` if hardware allows.
- HF access errors -> rerun `hf auth login` or pass `--hf-token`.
- Unexpected behavior after training -> inspect `--debug-samples` output and reasoning settings.

## Related

- [Setup](setup.md)
- [Troubleshooting](troubleshooting.md)
- [Config reference](config-reference.md)
