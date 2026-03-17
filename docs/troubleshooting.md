---
title: Troubleshooting
---

# Troubleshooting

## Purpose

Collect common training-time failures and fast mitigation steps.

## When to use

- You hit OOM before or during first training steps.
- You are tuning config for limited VRAM hardware.
- You need quick monitoring commands during training.

## OOM on RTX 3070 8 GB

Apply in order (largest impact first):

1. Reduce `max_length`.
2. Reduce `lora_r`.
3. Keep `per_device_train_batch_size: 1`.
4. Increase `gradient_accumulation_steps` to preserve effective batch size.

> [!WARNING]
> Tool-schema datasets can add thousands of tokens per sample.
> Run `--stats-only` before full training.

## Known issue: Qwen3-8B OOM on unsloth 2026.3.4+

`configs/qwen3/8b.yaml` (`max_length: 1024`) previously ran on RTX 3070 8 GB.
On newer unsloth builds, memory headroom changed and this run now OOMs before first step.
The config remains for reference while upstream behavior evolves.

## Monitoring commands

```bash
# GPU
nvtop
# or
watch -n 1 nvidia-smi

# RAM
htop
# or
free -h
```

## Training output references

For canonical training output examples (and command context), use Quickstart:
- [Quickstart](quickstart.md), section `Training command output examples`
  - `Example output (--stats-only)`
  - `Example output (qlora-train full run success signal)`

Why this is here: troubleshooting focuses on diagnosis and mitigation, while
command output examples are maintained next to the corresponding run steps.

## Edge cases / limitations

- OOM can happen from sequence length + dataset shape even when base model size looks acceptable.
- Effective VRAM pressure depends on model, max_length, LoRA rank, and runtime versions.

## Related

- [Quickstart](quickstart.md)
- [Config reference](config-reference.md)
- [Dataset pipeline](dataset-pipeline.md)
