---
layout: home

hero:
  name: Qwen QLoRA train
  text: Fine-tune Qwen3 and Qwen3.5 on limited GPU memory
  tagline: Practical QLoRA workflow for smaller setups. Focused on clear dataset handling, controlled truncation, and reproducible training/inference steps.
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
    link: /dataset-pipeline#loss-masking-behavior
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
    link: /dataset-pipeline#tools-injection
    linkText: Learn more

  - icon: 🚀
    title: Test before merging
    details: Run inference on base model + LoRA adapter directly. Check outputs before deciding whether you need a merge.
    link: /inference
    linkText: Learn more

  - icon: 🔀
    title: CPU merge
    details: Merge LoRA adapter into base weights on CPU. Useful when you need a standalone fp16 model for export.
    link: /merge
    linkText: Learn more
---

## Part of a two-repo ecosystem

| Repo | Purpose |
|------|---------|
| [qwen35-toolkit](https://techwithsergiu.github.io/qwen35-toolkit) | Model prep — BNB quantization, visual tower strip, verify, upload |
| **qwen-qlora-train** (this repo) | LoRA training, adapter inference, CPU merge |

> ⚠️ Validated training on RTX 3070 8 GB currently covers Qwen3 1.7B and 4B (see [Quickstart](quickstart)).
> Qwen3 8B OOMs on unsloth 2026.3.4+, and sizes above 4B should be treated as experimental on this hardware class.
