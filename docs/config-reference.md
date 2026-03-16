---
title: Config reference
---

# Config reference

All `TrainConfig` fields with their defaults.
Any field omitted from YAML falls back to the default shown here.

Load order for `hf_token`: `--hf_token` flag → `HF_TOKEN` env var → YAML field.

---

## Identity

| Field | Default | Description |
|-------|---------|-------------|
| `run_name` | `"run"` | Subdirectory name under `output_dir/` and `adapter_base_dir/` |
| `output_dir` | `"outputs"` | Root dir for trainer checkpoints and logs |
| `adapter_base_dir` | `"adapters"` | Root dir for saved LoRA adapter |
| `hf_token` | `null` | HF access token — prefer `HF_TOKEN` env var |

## Model

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `"unsloth/Qwen3-4B-bnb-4bit"` | HF repo id or local path |
| `chat_template` | `"qwen3"` | Unsloth template key — `"qwen3"` works for both Qwen3 and Qwen3.5 |

> **All valid `chat_template` values** (from `unsloth/chat_templates.py`):
>
> | Key | Use for |
> |-----|---------|
> | `"qwen3"` | Qwen3 and Qwen3.5 (same `im_start/im_end` format) |
> | `"qwen3-thinking"` | Qwen3 with explicit thinking enabled in the template |
> | `"qwen3-instruct"` | Qwen3 instruct variant without thinking |
> | `"qwen-2.5"` | Qwen2.5 family |

## Dataset

| Field | Default | Description |
|-------|---------|-------------|
| `dataset_id` | `""` | HF dataset id or local path (**required**) |
| `dataset_split` | `"train"` | Split passed to `load_dataset` |
| `messages_field` | `"messages"` | Column name containing the conversation list |
| `dataset_schema` | `"auto"` | `"auto"` / `"messages"` / `"prompt_response"` |

## Reasoning / thinking

| Field | Default | Description |
|-------|---------|-------------|
| `reasoning_field` | `"reasoning_content"` | Canonical field the Qwen template reads for `<think>` content |
| `reasoning_keys` | `null` | Alternative field names to normalize into `reasoning_field`. Default: `["reasoning_content", "reasoning", "thinking", "reason"]` |
| `extract_think_tags` | `true` | Extract inline `<think>…</think>` from `assistant.content` |
| `think_mode` | `"keep"` | `"keep"` preserve reasoning · `"drop"` remove entirely |
| `think_max_tokens` | `0` | Hard token cap on reasoning per message. `0` = no cap |
| `think_role` | `"think"` | Role name for datasets that store thinking as a separate message (rare) |
| `think_loss` | `"all"` | `"all"` · `"answer_only"` · `"answer_plus_think"` — see below |

**`think_loss` values:**

| Value | What gets gradient |
|-------|--------------------|
| `all` | Full assistant span — `<think>` content + answer |
| `answer_only` | Only tokens after `</think>` — recommended for agent/tool training |
| `answer_plus_think` | Think content + answer, but not the literal `<think>`/`</think>` tags |

## Sequence / truncation

| Field | Default | Description |
|-------|---------|-------------|
| `max_seq_length` | `2048` | Model's position embedding allocation (FastLanguageModel). Set ≥ `max_length` |
| `max_length` | `2048` | Max tokens per training sample. **Primary VRAM lever on 8 GB** |
| `truncate_side` | `"left"` | Fallback token-level cut: `"left"` keeps last N tokens (recommended) |

## Precision / hardware

| Field | Default | Description |
|-------|---------|-------------|
| `load_in_4bit` | `true` | BNB NF4 quantization — required for >1.7B on 8 GB VRAM |
| `attn_implementation` | `"sdpa"` | Attention backend — `"sdpa"` or `"flash_attention_2"` |
| `fp16` | `true` | Float16 training — use for Qwen3 |
| `bf16` | `false` | BFloat16 training — use for Qwen3.5. Mutually exclusive with `fp16` |

## LoRA

| Field | Default | Description |
|-------|---------|-------------|
| `lora_r` | `16` | LoRA rank. Higher = more capacity + VRAM |
| `lora_alpha` | `32` | Scaling factor. Convention: `2 * lora_r` |
| `lora_dropout` | `0.0` | Dropout on LoRA layers. Keep `0.0` with gradient checkpointing |
| `lora_target_modules` | `null` | Layers to apply LoRA to. `null` = all 7 projection layers |
| `gradient_checkpointing` | `"unsloth"` | `"unsloth"` (recommended, ~30% less VRAM) · `true` · `false` |

## Training

| Field | Default | Description |
|-------|---------|-------------|
| `per_device_train_batch_size` | `1` | Keep at 1 for 8 GB VRAM |
| `gradient_accumulation_steps` | `8` | Effective batch = batch_size × accumulation_steps |
| `learning_rate` | `2e-4` | Peak LR for AdamW |
| `warmup_ratio` | `0.05` | Fraction of steps used for LR warmup |
| `max_steps` | `1000` | Total optimizer steps |
| `logging_steps` | `20` | Log loss every N steps |
| `save_steps` | `200` | Save checkpoint every N steps |
| `seed` | `3407` | Random seed for torch, LoRA init, and SFTConfig |
| `optim` | `"adamw_8bit"` | `"adamw_8bit"` (saves ~1 GB VRAM) or `"adamw_torch"` |

## Loss masking

| Field | Default | Description |
|-------|---------|-------------|
| `assistant_roles` | `null` | Roles that carry loss. `null` = `["assistant"]`. All others masked with `-100` |
| `drop_if_no_assistant` | `true` | Skip samples with no assistant turn — they produce zero loss and waste compute |
