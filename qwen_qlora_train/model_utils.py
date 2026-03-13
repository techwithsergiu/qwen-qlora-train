#!/usr/bin/env python3
"""
model_utils.py — model loading, LoRA setup, and GPU diagnostics.

Unsloth imports are deferred to function call time so that importing this
module (e.g. for --help) does not trigger the Unsloth banner or torchao.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch
    from .config import TrainConfig


_DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def print_gpu_state() -> None:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"[GPU]   {props.name} | VRAM {total_gb:.1f} GB")
        print(f"[Torch] {torch.__version__} | CUDA {torch.version.cuda}")
    else:
        print("[GPU] CUDA not available — training will run on CPU (very slow).")


def pick_dtype(cfg: TrainConfig) -> torch.dtype:
    import torch
    return torch.bfloat16 if cfg.bf16 else torch.float16


def load_model_and_tokenizer(cfg: TrainConfig) -> Tuple[torch.nn.Module, object]:
    """Load base model + tokenizer via Unsloth, apply chat template."""
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    dtype = pick_dtype(cfg)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        token=cfg.hf_token,
        attn_implementation=cfg.attn_implementation,
        dtype=dtype,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    return model, tokenizer


def setup_lora(model: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    """Wrap model with LoRA adapters."""
    from unsloth import FastLanguageModel

    target_modules = cfg.lora_target_modules or _DEFAULT_LORA_TARGETS
    return FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        random_state=cfg.seed,
    )
