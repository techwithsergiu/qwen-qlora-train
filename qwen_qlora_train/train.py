#!/usr/bin/env python3
"""
train.py — Agent-style SFT / QLoRA with assistant-only loss masking
(Qwen template-driven, tools-aware)
================================================================================

This training script is built for **Qwen/Qwen3-style chat models** (via Unsloth)
and **HF chat datasets** (usually a `messages` column), with optional `tools`
schemas stored per row.

Core idea
---------
We do NOT manually "compose prompts" for tools / tool calls / tool results.
Instead, we **delegate rendering** to the model's chat template:

    tokenizer.apply_chat_template(messages, tools=tools, ...)

So if your dataset provides:
- `messages`: list of {role, content, ...}
- optional `tools`: list/dict of function schemas (or JSON string)
- optional tool-call / tool-result fields inside messages (template-specific)

…the Qwen template is responsible for turning that into the final training text.

Think policy
------------
  think_mode:
    - "keep" : preserve reasoning content (cap with think_max_tokens if set)
    - "drop" : remove reasoning entirely

  think_loss controls what gets trained inside assistant spans:
    - "all"              : loss on full assistant span (think + answer)
    - "answer_only"      : mask everything up to and including </think>
    - "answer_plus_think": mask only the literal <think> / </think> tags,
                           train on think content + answer

  Qwen3 reasoning fields are normalized to `reasoning_content` so the template
  never renders empty <think></think> blocks.

Tools support (template-delegated)
-----------------------------------
  Columns detected: tools / tool_schemas / functions / function_schemas
  Passed to the template as tools=...; falls back gracefully if unsupported.

  ⚠ Tool schemas can add hundreds/thousands of tokens per sample.
  Use --stats-only to check if you're constantly truncating.

Truncation
----------
  truncate_side: "left"  → keep the LAST max_length tokens  (recommended for chat)
                 "right" → keep the FIRST max_length tokens

Assistant-only loss masking
---------------------------
  Loss is computed ONLY on spans whose role is in assistant_roles (default: ["assistant"]).
  Everything else (system / user / tool) is masked with -100.
  Benefits: role discipline, stopping behavior, stable tool-call formatting.

Dataset expectations
--------------------
  messages_field (default: "messages") — list of dicts with at least:
    {"role": "...", "content": "..."}
  Additional keys are preserved (important for reasoning/tool variants).

  Also supported: prompt/response schema (auto-detected).

Usage
-----
  qlora-train --config configs/qwen3/1.7b.yaml
  qlora-train --config configs/qwen3/1.7b.yaml --stats-only
  qlora-train --config configs/qwen3/1.7b.yaml --debug-samples 2

  qlora-train --config configs/qwen35/0.8b.yaml
  qlora-train --config configs/qwen35/0.8b.yaml --stats-only
  qlora-train --config configs/qwen35/0.8b.yaml --debug-samples 2
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

# ── Third-party: unsloth MUST come before trl / transformers / peft ──────────
import unsloth
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ── Local modules ─────────────────────────────────────────────────────────────
from qwen_qlora_train.config import load_config, validate_config
from qwen_qlora_train.data_pipeline import (
  build_dataset,
  collect_lengths,
  collect_raw_lengths,
  debug_render_samples,
  print_length_stats
)
from qwen_qlora_train.model_utils import (
  load_model_and_tokenizer,
  print_gpu_state,
  setup_lora
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--hf_token", default=None, help="HF token (overrides config / env).")
    ap.add_argument("--stats-only", action="store_true",
                    help="Print dataset token-length stats and exit.")
    ap.add_argument("--debug-samples", type=int, default=0,
                    help="Render N processed samples and exit.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config, hf_token_override=args.hf_token)
    validate_config(cfg)

    print_gpu_state()
    torch.manual_seed(cfg.seed)

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = setup_lora(model, cfg)

    raw = load_dataset(cfg.dataset_id, split=cfg.dataset_split)

    # ── Debug / stats early-exit paths ────────────────────────────────────────
    if args.debug_samples > 0:
        debug_render_samples(raw, tokenizer, cfg, n=args.debug_samples)
        print("[debug-samples] exiting without training.")
        return

    tokenized = build_dataset(raw, tokenizer, cfg)

    if args.stats_only:
        print_length_stats(
            collect_lengths(tokenized),
            max_length=cfg.max_length,
            raw_lengths=collect_raw_lengths(tokenized),
        )
        print("\n(stats-only) exiting without training.")
        return

    # ── Training ──────────────────────────────────────────────────────────────
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = SFTConfig(
        output_dir=str(out_dir / cfg.run_name),
        dataset_text_field="__unused__",
        max_length=cfg.max_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to=[],
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )

    def formatting_prompts_func(_examples):
        return {"__unused__": [""] * len(_examples[cfg.messages_field])}

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized,
        formatting_func=formatting_prompts_func,
        args=trainer_args,
    )
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_dir = Path(cfg.adapter_base_dir) / cfg.run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n✅  LoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
