---
layout: home

hero:
  name: Qwen QLoRA train
  text: Fine-tune Qwen3 and Qwen3.5 on your GPU
  tagline: If you only have 8 GB VRAM, every training step has to count. Structured truncation, assistant-only loss masking, and full reasoning control — end-to-end from dataset to llama.cpp.
  actions:
    - theme: brand
      text: Get started
      link: /quickstart
    - theme: alt
      text: GitHub →
      link: https://github.com/techwithsergiu/qwen-qlora-train

features:
  - icon: 🎯
    title: Assistant-only loss masking
    details: Gradients only where they belong. Masking is computed in character space before tokenization — works correctly with any subword vocabulary.
    link: /dataset-pipeline#assistant-only-loss-masking
    linkText: Learn more

  - icon: ✂️
    title: Structured truncation
    details: Preserves system prompt and tool schemas under any max_length. Drops oldest middle turns first — never corrupts the conversation structure.
    link: /dataset-pipeline#truncation-strategy
    linkText: Learn more

  - icon: 🧠
    title: Full reasoning control
    details: Keep, drop, or selectively train Qwen3 think content. Control gradient scope with think_mode and think_loss per training run.
    link: /reasoning
    linkText: Learn more

  - icon: 🔧
    title: Tools-aware
    details: Auto-detects tool schema columns and passes them directly to the chat template. No manual prompt building required.
    link: /dataset-pipeline#tools-support
    linkText: Learn more

  - icon: 🚀
    title: Test before merging
    details: Run inference on the base model + LoRA adapter directly — no CPU merge needed. Smoke tests, single prompt, or interactive chat. Verify your adapter immediately after training.
    link: /inference
    linkText: Learn more

  - icon: 🔀
    title: CPU merge
    details: Merge LoRA adapter into base weights on CPU — no VRAM needed. Produces a standalone fp16 model ready for GGUF conversion.
    link: /merge
    linkText: Learn more
---

## Part of a two-repo ecosystem

| Repo | Purpose |
|------|---------|
| [qwen35-toolkit](https://github.com/techwithsergiu/qwen35-toolkit) | Model prep — BNB quantization, visual tower strip, verify, upload |
| **qwen-qlora-train** (this repo) | LoRA training, adapter inference, CPU merge |

> ⚠️ Validated training on RTX 3070 8 GB currently covers Qwen3 1.7B and 4B (see [Quickstart](quickstart)). The 8B config OOMs on unsloth 2026.3.4+ and remains in `configs/qwen3` for reference only; treat anything larger than 4B as experimental until the tooling stabilizes.
