---
title: Setup
---

# Setup

## Prerequisites

- Arch Linux (or any Linux with NVIDIA driver)
- Python 3.11
- CUDA via driver (`nvidia-smi` works → CUDA is fine)

## Steps

### Step 1 — Create and activate a Python environment
Create an isolated environment before installing dependencies.

```bash
yay -S python311
python3.11 -m venv venv
source venv/bin/activate
```

Success criteria: `python --version` resolves to Python 3.11 in venv.

### Step 2 — Install dependencies in the required order
Install Unsloth first, then toolkit, then train package.

```bash
# 1) Unsloth (pins compatible torch/transformers versions)
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# 2) qwen35-toolkit
pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git

# 3a) Editable install (development / local clone)
pip install -e .

# 3b) Install directly from GitHub (no local clone needed)
pip install git+https://github.com/techwithsergiu/qwen-qlora-train.git
```

Success criteria: `qlora-train --help` prints CLI usage.

> [!WARNING]
> Keep the install order above. Installing train package first can cause
> version conflicts with Unsloth-pinned dependencies.

### Step 3 — Authenticate with Hugging Face
Authenticate once before using Hub models/datasets.

```bash
hf auth login
```

Success criteria: private dataset/model access works from CLI commands.

## Expected result

- `qlora-train`, `qlora-infer`, `qlora-merge` are available.
- Hub authentication is active for training/inference workflows.

## Common failures

- Import or version errors after install -> recreate venv and reinstall in order.
- `CUDA out of memory` at startup -> start with 1.7B config and shorter `max_length`.
- HF auth errors (401/403) -> rerun `hf auth login` or pass `--hf-token`.

Alternatively, pass `--hf-token hf_...` per command or set `HF_TOKEN` as an environment variable.

## Related

- [Quickstart](quickstart.md)
- [Troubleshooting](troubleshooting.md)
- [Config reference](config-reference.md)
