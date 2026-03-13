#!/usr/bin/env python3
"""
config.py — TrainConfig dataclass + YAML loading/merging/validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TrainConfig:
    """
    All training hyperparameters and pipeline settings in one place.

    Load from YAML via ``load_config(path)``. Any field not present in the
    YAML falls back to the default defined here.

    Token-resolution priority for hf_token:
        1. --hf_token CLI flag
        2. HF_TOKEN environment variable
        3. hf_token field in YAML
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    run_name: str = "run"
    """
    Short name for this run. Used as:
      - subdirectory under output_dir:     outputs/<run_name>/
      - subdirectory under adapter_base_dir: adapters/<run_name>/
    Keep it filesystem-safe (no spaces).
    """

    output_dir: str = "outputs"
    """Root directory for SFTTrainer checkpoints and logs."""

    adapter_base_dir: str = "adapters"
    """Root directory where the final LoRA adapter is saved after training."""

    hf_token: Optional[str] = None
    """
    HuggingFace access token for gated models and private datasets.
    Prefer setting via HF_TOKEN env var rather than hardcoding in YAML.
    """

    # ── Model ─────────────────────────────────────────────────────────────────

    model_name: str = "unsloth/Qwen3-4B-bnb-4bit"
    """
    HF repo id or local path of the base model to fine-tune.
    For Qwen3.5 text-only training use techwithsergiu/Qwen3.5-text-{size}-bnb-4bit.
    Passed directly to FastLanguageModel.from_pretrained().
    """

    chat_template: str = "qwen3"
    """
    Chat template key passed to unsloth's get_chat_template().
    Patches the tokenizer's Jinja2 template once at load time.

    Valid values (from unsloth/chat_templates.py):
      "qwen3"          — Qwen3 and Qwen3.5 (same im_start/im_end format)
      "qwen3-thinking" — Qwen3 with explicit thinking enabled in the template
      "qwen3-instruct" — Qwen3 instruct variant without thinking
      "qwen-2.5"       — Qwen2.5 family
    """

    # ── Dataset ───────────────────────────────────────────────────────────────

    dataset_id: str = ""
    """
    HuggingFace dataset repo id (e.g. "TeichAI/claude-sonnet-4.5-high-reasoning-250x")
    or local path. Required — validation fails if empty.
    Passed to datasets.load_dataset(dataset_id, split=dataset_split).
    """

    dataset_split: str = "train"
    """Dataset split to load. Passed to load_dataset(..., split=dataset_split)."""

    messages_field: str = "messages"
    """
    Column name that contains the conversation turns.
    Only used when dataset_schema is "messages" or auto-detected as such.
    Expected value: list of dicts with at least {"role": str, "content": str}.
    """

    dataset_schema: str = "auto"
    """
    How to interpret each dataset row. Options:

      "auto"           — detect automatically (recommended):
                         • list under messages_field → "messages"
                         • prompt + response/completion/answer → "prompt_response"
      "messages"       — OpenAI API format: list of {role, content} dicts
                         under messages_field column.
      "prompt_response"— Two-column format: "prompt" + one of
                         "response" / "completion" / "answer".
                         Converted internally to a two-message conversation.
    """

    # ── Reasoning / Thinking ──────────────────────────────────────────────────

    reasoning_field: str = "reasoning_content"
    """
    The canonical field name for reasoning content inside message dicts.
    The Qwen3 chat template reads this field to render <think>...</think> blocks.
    All other reasoning field names (see reasoning_keys) are normalized to this.
    """

    reasoning_keys: Optional[List[str]] = None
    """
    Alternative field names to look for reasoning content in message dicts.
    The first non-empty match is moved into reasoning_field.
    Defaults to: ["reasoning_content", "reasoning", "thinking", "reason"]
    """

    extract_think_tags: bool = True
    """
    If True, extract <think>...</think> found inline inside assistant.content
    into reasoning_field. The tags and their content are removed from content.
    Set to False only if your dataset never uses inline think tags.
    """

    think_mode: str = "keep"
    """
    What to do with reasoning content during preprocessing.

      "keep" — preserve reasoning_content as-is (apply think_max_tokens cap if set)
      "drop" — clear reasoning_content entirely; trains pure answer-style behavior
    """

    think_max_tokens: int = 0
    """
    Hard token cap on reasoning_content per message. 0 or None = no cap.

    ⚠️  Capping creates truncated reasoning mid-thought. The model may learn
    to start a chain of thought and abruptly stop. Prefer shorter-think datasets
    over aggressive capping. Use only if you have a specific reason.
    """

    think_role: str = "think"
    """
    Role name used by datasets that store thinking as a separate message
    (rare — most datasets use reasoning_content inside assistant messages).
    Messages with this role are treated like reasoning_content: kept or dropped
    based on think_mode, capped by think_max_tokens.
    """

    think_loss: str = "all"
    """
    Controls which tokens inside assistant spans carry gradient.

      "all"              — loss on the full assistant span (think + answer).
                           Good for reasoning datasets.
      "answer_only"      — mask everything up to and including </think>.
                           Trains only the final answer. Recommended for
                           agent / tool-oriented training.
      "answer_plus_think"— mask only the literal <think> and </think> tags.
                           Trains on think content + answer, but not the tags.
    """

    # ── Sequence ──────────────────────────────────────────────────────────────

    max_seq_length: int = 2048
    """
    Maximum sequence length passed to FastLanguageModel.from_pretrained().
    Sets the model's internal position embedding / KV cache allocation.
    Should be >= max_length. On RTX 3070 8 GB this is the primary VRAM lever.
    """

    max_length: int = 2048
    """
    Maximum token length for training samples, used by the data pipeline.
    Samples exceeding this are truncated via structured_truncate_messages()
    before tokenization, with token-level left-truncation as a last resort.
    Also passed to SFTConfig(max_length=...).
    Set max_seq_length to the same value or higher.
    """

    truncate_side: str = "left"
    """
    Direction for the token-level fallback truncation (step 3 of structured
    truncation, applied only when the message list still exceeds max_length
    after dropping middle history and trimming reasoning).

      "left"  — keep the LAST max_length tokens (recommended for chat;
                preserves the most recent user/assistant turn)
      "right" — keep the FIRST max_length tokens + restore EOS
    """

    # ── Precision / Hardware ──────────────────────────────────────────────────

    load_in_4bit: bool = True
    """
    Load base model in BitsAndBytes NF4 4-bit quantization.
    Required for training >1.7B models on 8 GB VRAM.
    Passed to FastLanguageModel.from_pretrained(load_in_4bit=...).
    """

    attn_implementation: str = "sdpa"
    """
    Attention backend. "sdpa" (Scaled Dot-Product Attention, PyTorch native)
    is recommended. "flash_attention_2" is faster but requires a separate install.
    """

    fp16: bool = True
    """
    Train in float16. Used for Qwen3 models (trained in fp16).
    Mutually exclusive with bf16 — set exactly one to True.
    Passed to SFTConfig(fp16=...) and used to pick torch dtype for model load.
    """

    bf16: bool = False
    """
    Train in bfloat16. Preferred for Qwen3.5 models (trained in bf16).
    Mutually exclusive with fp16 — set exactly one to True.
    """

    # ── LoRA ──────────────────────────────────────────────────────────────────

    lora_r: int = 16
    """
    LoRA rank. Higher rank = more trainable parameters = more VRAM.
    Typical range: 8–64. 16 is a safe default; use 32–64 for larger capacity.
    """

    lora_alpha: int = 32
    """
    LoRA scaling factor. Effective learning rate scale = lora_alpha / lora_r.
    Convention: set to 2 * lora_r (e.g. r=16 → alpha=32).
    """

    lora_dropout: float = 0.0
    """
    Dropout probability applied to LoRA layers. 0.0 is recommended with
    gradient checkpointing — dropout + checkpointing can interact poorly.
    """

    lora_target_modules: Optional[List[str]] = None
    """
    Which linear layers to apply LoRA to. None = use default set:
      ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    Override to target a subset (less VRAM, less capacity) or superset.
    """

    gradient_checkpointing: str = "unsloth"
    """
    Gradient checkpointing mode passed to FastLanguageModel.get_peft_model().
      "unsloth" — Unsloth's optimized implementation (recommended, ~30% less VRAM)
      True       — standard HuggingFace gradient checkpointing
      False      — disabled (fastest, but highest VRAM)
    """

    # ── Training ──────────────────────────────────────────────────────────────

    per_device_train_batch_size: int = 1
    """Batch size per GPU. Keep at 1 for 8 GB VRAM."""

    gradient_accumulation_steps: int = 8
    """
    Number of steps to accumulate gradients before an optimizer update.
    Effective batch size = per_device_train_batch_size * gradient_accumulation_steps.
    Increase this (not batch size) to simulate larger batches without extra VRAM.
    """

    learning_rate: float = 2e-4
    """Peak learning rate for the AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Fraction of total steps used for linear LR warmup."""

    max_steps: int = 1000
    """Total number of optimizer steps. Overrides num_train_epochs."""

    logging_steps: int = 20
    """Log training loss every N steps."""

    save_steps: int = 200
    """Save a checkpoint every N steps under output_dir/<run_name>/."""

    seed: int = 3407
    """
    Random seed. Set for torch.manual_seed(), LoRA random_state, and SFTConfig.
    3407 is the Unsloth conventional default.
    """

    optim: str = "adamw_8bit"
    """
    Optimizer. Options:
      "adamw_8bit"  — bitsandbytes 8-bit AdamW (recommended, saves ~1 GB VRAM)
      "adamw_torch" — standard PyTorch AdamW (fallback if bitsandbytes issues)
    """

    # ── Loss masking ──────────────────────────────────────────────────────────

    assistant_roles: Optional[List[str]] = None
    """
    List of role names whose tokens carry loss. All other roles (system, user,
    tool) are masked with -100. None defaults to ["assistant"].
    Masking is performed in character space before tokenization.
    """

    drop_if_no_assistant: bool = True
    """
    If True, skip samples that contain no message from any role in
    assistant_roles after canonicalization. Such samples produce zero loss
    tokens and waste compute. Set to False to keep them (e.g. for debugging).
    """


_VALID_THINK_LOSS = ("all", "answer_only", "answer_plus_think")
_VALID_TRUNCATE_SIDE = ("left", "right")


def load_config(yaml_path: str, hf_token_override: Optional[str] = None) -> TrainConfig:
    """Load YAML, merge into TrainConfig defaults, resolve HF token."""
    raw = _read_yaml(yaml_path)
    cfg = TrainConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    cfg.hf_token = (
        hf_token_override
        or os.environ.get("HF_TOKEN")
        or cfg.hf_token
    )
    return cfg


def validate_config(cfg: TrainConfig) -> None:
    """Raise SystemExit with a clear message if the config is invalid."""
    if not cfg.dataset_id:
        raise SystemExit("Config error: dataset_id is required.")
    if cfg.think_loss not in _VALID_THINK_LOSS:
        raise SystemExit(
            f"Config error: think_loss must be one of {' | '.join(_VALID_THINK_LOSS)}"
        )
    if cfg.truncate_side not in _VALID_TRUNCATE_SIDE:
        raise SystemExit(
            f"Config error: truncate_side must be one of {' | '.join(_VALID_TRUNCATE_SIDE)}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
