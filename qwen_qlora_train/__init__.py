"""
qwen_qlora_train — Agent-style SFT / QLoRA training pipeline.

Supports Qwen3 and Qwen3.5 models via Unsloth + TRL on limited VRAM.

Module layout
-------------
  train.py           — entry point (this file)
  config.py          — TrainConfig dataclass, YAML loading, validation
  model_utils.py     — model + tokenizer loading, LoRA setup, GPU diagnostics
  data_pipeline.py   — char-mask building, tokenization, dataset mapping, stats
  dataset_parsers.py — row canonicalization (reasoning fields, tools, schemas)
"""

__version__ = "0.1.0"
