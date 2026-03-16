# qwen-qlora-train

Agent-style SFT / QLoRA pipeline for **Qwen3** and **Qwen3.5** on consumer hardware.

Built around one idea: if you only have 8 GB VRAM, every training step has to count.

**What it does:**
- Train Qwen3 / Qwen3.5 with QLoRA on a single consumer GPU via Unsloth
- Structured truncation — preserves system prompt and tool schemas under any `max_length`
- Assistant-only loss masking — gradients only where they belong
- Full reasoning control — keep, drop, or selectively train think content
- Tools-aware — passes tool schemas directly to the chat template
- CPU merge — produces a standalone fp16 model ready for GGUF conversion

If you have a mid-range GPU and want to fine-tune Qwen3 or Qwen3.5 on your own data,
this is the training pipeline that makes it work — end-to-end from dataset to llama.cpp.

> 💡 Validated on an RTX 3070 8 GB for Qwen3 1.7B / 4B (see `configs/qwen3`). The 8B config now OOMs on unsloth 2026.3.4+ and is kept only for reference; treat anything larger than 4B as experimental until the tooling stabilizes.

Part of a two-repo ecosystem:

| Repo | Purpose |
|------|---------|
| [qwen35-toolkit](https://github.com/techwithsergiu/qwen35-toolkit) | Model prep — BNB quantization, visual tower strip, verify, upload |
| **qwen-qlora-train** (this repo) | LoRA training, adapter inference, CPU merge |

---

## Setup

### Prerequisites

- Arch Linux (or any Linux with NVIDIA driver)
- Python 3.11
- CUDA via driver (`nvidia-smi` works → CUDA is fine)

```bash
# I'm using Arch btw
yay -S python311
python3.11 -m venv venv
source venv/bin/activate
```

### Install

```bash
# 1. Unsloth — installs torch automatically with the right CUDA wheel
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# 2. qwen35-toolkit
pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git

# 3a. Editable install (development / local clone)
pip install -e .

# 3b. Install directly from GitHub (no local clone needed)
pip install git+https://github.com/techwithsergiu/qwen-qlora-train.git
```

> **Why this order?** Unsloth hard-pins `torch` and `transformers`. Installing it
> first prevents `pip install -e .` from overriding those versions.

### Authentication

`huggingface_hub` is installed automatically with this package. Run once before using any command:

```bash
hf auth login
```

Alternatively, pass `--hf-token hf_...` per command or set `HF_TOKEN` as an environment variable.

---

## Quick usage

```bash
# Train
qlora-train --config configs/qwen3/1.7b.yaml

# Test adapter (no merge needed)
qlora-infer --model unsloth/Qwen3-1.7B-bnb-4bit --adapter adapters/qwen3-1.7b-sanity

# Merge to fp16 (for GGUF conversion)
qlora-merge --base unsloth/Qwen3-1.7B --adapter adapters/qwen3-1.7b-sanity --output merged/qwen3-1.7b-merged-f16 --dtype f16
```

> Before training: open the YAML config and set `dataset_id` — it is required and has no default.

---

## Documentation

Full docs available at **[techwithsergiu.github.io/qwen-qlora-train](https://techwithsergiu.github.io/qwen-qlora-train/)**.

| Doc | Contents |
|-----|----------|
| [docs/setup.md](docs/setup.md) | Prerequisites, install order, authentication |
| [docs/quickstart.md](docs/quickstart.md) | Step-by-step from install to trained adapter |
| [docs/commands.md](docs/commands.md) | All CLI commands with module and docs links |
| [docs/project-layout.md](docs/project-layout.md) | File and directory structure of the repo |
| [docs/training-pipeline.md](docs/training-pipeline.md) | End-to-end pipeline diagram across both repos |
| [docs/dataset-pipeline.md](docs/dataset-pipeline.md) | Pipeline internals, schemas, tools, truncation, diagnostics |
| [docs/reasoning.md](docs/reasoning.md) | Reasoning control — keep, drop, or selectively train think content |
| [docs/config-reference.md](docs/config-reference.md) | All TrainConfig fields with defaults and descriptions |
| [docs/inference.md](docs/inference.md) | Adapter inference — smoke tests, interactive chat, thinking control |
| [docs/merge.md](docs/merge.md) | CPU merge — produce a standalone fp16 model from adapter + base |
| [docs/post-merge-workflow.md](docs/post-merge-workflow.md) | GGUF conversion, quantization, and Hub upload via qwen35-toolkit |
| [docs/troubleshooting.md](docs/troubleshooting.md) | OOM fixes, known issues, monitoring commands |

---

## License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software in both open-source
and commercial applications, as long as you comply with the terms of the
Apache 2.0 License.

Full license text:  
[LICENSE](LICENSE)

---

## Third-party Licenses

This project relies on several third-party components, all using permissive
licenses compatible with Apache License 2.0.

Full list:  
[docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md)
