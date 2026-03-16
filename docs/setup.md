---
title: Setup
---

# Setup

## Prerequisites

- Arch Linux (or any Linux with NVIDIA driver)
- Python 3.11
- CUDA via driver (`nvidia-smi` works → CUDA is fine)

```bash
yay -S python311
python3.11 -m venv venv
source venv/bin/activate
```

## Install

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

## Authentication

`huggingface_hub` is installed automatically with this package. Run once before using any command:

```bash
hf auth login
```

Alternatively, pass `--hf-token hf_...` per command or set `HF_TOKEN` as an environment variable.
