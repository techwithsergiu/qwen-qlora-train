---
title: Config reference
---

# Config reference

## Purpose

Reference for all `TrainConfig` fields and defaults.
Any YAML field not set explicitly falls back to the default listed here.

## When to use

- While creating a new training config.
- When checking which settings are safe on limited hardware.
- When validating reasoning/truncation/loss settings.

## Config load behavior

```text
1. Token resolution order:
   - `--hf_token` CLI flag
   - `HF_TOKEN` environment variable
   - YAML `hf_token` field
2. Default resolution:
   - A default is applied only when the field is omitted in YAML.
```

## Schema (YAML shape)

```yaml
run_name: "run"
model_name: "unsloth/Qwen3-4B-bnb-4bit"
dataset_id: "your-hf-username/your-dataset"
max_length: 2048
```

## Options by section

### Identity

| Field | Default | Description |
|-------|---------|-------------|
| `run_name` | `"run"` | Subdirectory name under `output_dir/` and `adapter_base_dir/` |
| `output_dir` | `"outputs"` | Root dir for trainer checkpoints and logs |
| `adapter_base_dir` | `"adapters"` | Root dir for saved LoRA adapter |
| `hf_token` | `null` | HF access token (prefer `HF_TOKEN` env var) |

### Model

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `"unsloth/Qwen3-4B-bnb-4bit"` | HF repo id or local path |
| `chat_template` | `"qwen3"` | Unsloth template key (`qwen3` works for Qwen3 and Qwen3.5) |

Valid `chat_template` keys commonly used:

| Key | Use for |
|-----|---------|
| `"qwen3"` | Qwen3 and Qwen3.5 |
| `"qwen3-thinking"` | Qwen3 with explicit thinking template |
| `"qwen3-instruct"` | Qwen3 instruct without thinking |
| `"qwen-2.5"` | Qwen2.5 family |

### Dataset

| Field | Default | Description |
|-------|---------|-------------|
| `dataset_id` | `""` | HF dataset id or local path (**required**) |
| `dataset_split` | `"train"` | Split passed to `load_dataset` |
| `messages_field` | `"messages"` | Column with conversation list |
| `dataset_schema` | `"auto"` | `auto` / `messages` / `prompt_response` |

### Reasoning / thinking

| Field | Default | Description |
|-------|---------|-------------|
| `reasoning_field` | `"reasoning_content"` | Canonical field for `<think>` content |
| `reasoning_keys` | `null` | Alternative keys normalized into `reasoning_field` |
| `extract_think_tags` | `true` | Extract inline `<think>...</think>` from assistant content |
| `think_mode` | `"keep"` | `keep` preserve reasoning, `drop` remove reasoning |
| `think_max_tokens` | `0` | Cap tokens per reasoning block (`0` = no cap) |
| `think_role` | `"think"` | Role name for separate-think-message datasets |
| `think_loss` | `"all"` | `all` / `answer_only` / `answer_plus_think` |

`think_loss` behavior:

| Value | Gradient scope |
|-------|----------------|
| `all` | Full assistant span (`<think>` + answer) |
| `answer_only` | Tokens after `</think>` |
| `answer_plus_think` | Think content + answer (excluding literal tags) |

### Sequence / truncation

| Field | Default | Description |
|-------|---------|-------------|
| `max_seq_length` | `2048` | Positional allocation; set >= `max_length` |
| `max_length` | `2048` | Max tokens/sample; primary VRAM lever |
| `truncate_side` | `"left"` | Fallback token-level truncation side |

### Precision / hardware

| Field | Default | Description |
|-------|---------|-------------|
| `load_in_4bit` | `true` | BNB NF4 quantization |
| `attn_implementation` | `"sdpa"` | `sdpa` or `flash_attention_2` |
| `fp16` | `true` | Float16 path (commonly Qwen3) |
| `bf16` | `false` | BFloat16 path (commonly Qwen3.5) |

### LoRA

| Field | Default | Description |
|-------|---------|-------------|
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | Scaling factor (commonly `2 * lora_r`) |
| `lora_dropout` | `0.0` | LoRA dropout |
| `lora_target_modules` | `null` | `null` -> all 7 projection layers |
| `gradient_checkpointing` | `"unsloth"` | `unsloth` / `true` / `false` |

### Training

| Field | Default | Description |
|-------|---------|-------------|
| `per_device_train_batch_size` | `1` | Keep at 1 for limited VRAM setups |
| `gradient_accumulation_steps` | `8` | Effective batch = batch size x accumulation |
| `learning_rate` | `2e-4` | Peak LR for AdamW |
| `warmup_ratio` | `0.05` | LR warmup fraction |
| `max_steps` | `1000` | Total optimizer steps |
| `logging_steps` | `20` | Log interval |
| `save_steps` | `200` | Checkpoint interval |
| `seed` | `3407` | Random seed |
| `optim` | `"adamw_8bit"` | `adamw_8bit` or `adamw_torch` |

### Loss masking

| Field | Default | Description |
|-------|---------|-------------|
| `assistant_roles` | `null` | Roles that carry loss (`null` -> `["assistant"]`) |
| `drop_if_no_assistant` | `true` | Drop samples with no assistant turn |

## Validation rules

```text
1. Required dataset source:
   - `dataset_id` must be non-empty.
2. Precision flags:
   - `fp16` and `bf16` should not be enabled at the same time.
3. Sequence budget:
   - high `max_length` increases VRAM pressure and can trigger OOM on 8 GB GPUs.
4. Positional allocation:
   - `max_seq_length` should be >= `max_length`.
```

## Edge cases / limitations

> [!WARNING]
> `dataset_id` is required. Empty `dataset_id` cannot be resolved into a train dataset.

- `fp16` and `bf16` should not be enabled simultaneously.
- Aggressive `max_length` on 8 GB VRAM can cause OOM.

## Related

- [Quickstart](quickstart.md)
- [Dataset pipeline](dataset-pipeline.md)
- [Reasoning control](reasoning.md)
