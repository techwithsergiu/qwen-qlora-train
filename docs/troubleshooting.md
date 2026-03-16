---
title: Troubleshooting
---

# Troubleshooting

## OOM (RTX 3070 8 GB)

Apply in order — `max_length` is the strongest lever:

1. Reduce `max_length` — KV cache and activations scale with sequence length
2. Reduce `lora_r`
3. Keep `per_device_train_batch_size: 1`
4. Increase `gradient_accumulation_steps` — maintains effective batch size without extra VRAM

If using tool-schema datasets, schemas can add thousands of tokens per sample.
Always run `--stats-only` first.

### Known issue: Qwen3-8B OOM on unsloth 2026.3.4+

`configs/qwen3/8b.yaml` (`max_length: 1024`) was previously validated on RTX 3070 8 GB.
Newer unsloth versions allocate memory more conservatively, leaving ~500 MB headroom
that is no longer available for training — the run now OOMs before the first step.
The config is kept for when this is resolved.

---

## Monitoring during training

```bash
# GPU (VRAM, utilization)
nvtop
# or
watch -n 1 nvidia-smi

# RAM
htop
# or
free -h
```
