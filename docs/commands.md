---
title: Commands
---

# Commands

## Purpose

This page is the command index for training workflows.
Use it to select the right CLI entry point.

## When to use

- You are starting a training run and need the correct command.
- You need quick navigation to inference or merge steps.

## Command map

| CLI command | Module | Purpose | Docs |
|---|---|---|---|
| `qlora-train` | `qwen_qlora_train.train` | Run SFT / QLoRA training, save LoRA adapter | [Quickstart](quickstart.md) |
| `qlora-infer` | `qwen_qlora_train.infer` | Run inference on base model or base + adapter | [Inference](inference.md) |
| `qlora-merge` | `qwen_qlora_train.merge_cpu` | Merge adapter into base weights on CPU | [CPU merge](merge.md) |

## Notes

- Run `--help` on each command for full flag list.
- For constrained GPUs, start with the smallest validated config.

## Related

- [Quickstart](quickstart.md)
- [Troubleshooting](troubleshooting.md)
- [Config reference](config-reference.md)
