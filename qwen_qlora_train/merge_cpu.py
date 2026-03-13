"""
merge_cpu.py
────────────
Merges a base model with a trained LoRA adapter on CPU, producing a
standalone full-precision model ready for GGUF conversion.

Loads the base model in bf16/f16 into CPU RAM (NOT 4-bit), merges the LoRA
adapter into the base weights, and saves a clean merged model.

Model class selection:
  The loader is chosen automatically based on the model name/path.
  Qwen3.5 models are loaded with Qwen3_5ForConditionalGeneration to ensure
  the correct forward signature and generation config (required for the
  enable_thinking flag). All other families fall back to AutoModelForCausalLM.
  Use --loader auto|qwen3.5 to override if auto-detection gets it wrong.

Hardware requirements:
  RAM  : ~20 GB  (full bf16 model + adapter must fit simultaneously)
  VRAM : none    (everything runs on CPU)

Usage:
  qlora-merge --base unsloth/Qwen3-1.7B --adapter adapters/qwen3-1.7b-sanity --output merged/qwen3-1.7b-merged-f16 --dtype f16

  qlora-merge --base unsloth/Qwen3.5-0.8B --adapter adapters/qwen35-text-0.8b-sanity --output merged/qwen35-text-0.8b-merged-f16 --dtype f16
"""

import argparse
import os
from pathlib import Path

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_DTYPE      = "bf16"   # bf16 matches Qwen3.5 training dtype; use f16 for llama.cpp compat
DEFAULT_LOADER     = "auto"   # "auto" | "qwen3.5"
RAM_WARN_THRESHOLD = 20.0     # GB — warn if less free RAM than this before loading

# Loader key → qualified class name (imported lazily to keep --help instant).
# Extend this dict to add support for new model families.
LOADER_CLASSES: dict[str, str] = {
    "qwen3.5": "Qwen3_5ForConditionalGeneration",
    "auto":    "AutoModelForCausalLM",
}

# Substrings that trigger automatic loader selection (case-insensitive).
# Checked against the model name/path before falling back to AutoModelForCausalLM.
AUTO_DETECT: list[tuple[str, str]] = [
    ("qwen3.5", "qwen3.5"),   # (substring_in_name, loader_key)
    ("qwen3_5", "qwen3.5"),
]


# ── Model loader resolver ──────────────────────────────────────────────────────

def _import_model_class(class_name: str):
    """Lazily import a model class from transformers by name."""
    import transformers
    return getattr(transformers, class_name)


def resolve_loader(model_name: str, loader_arg: str) -> tuple:
    """
    Return the model class and a human-readable label for the given model.

    Imports are deferred to keep --help instant (no torchao/transformers banner).

    Resolution order:
      1. If loader_arg is not "auto" — use it directly (explicit override).
      2. Otherwise scan AUTO_DETECT for a substring match in model_name.
      3. Fall back to AutoModelForCausalLM.

    Args:
        model_name: HF repo id or local path (used for auto-detection).
        loader_arg: Value of the --loader CLI flag.

    Returns:
        (model_class, loader_key)
    """
    if loader_arg != "auto":
        if loader_arg not in LOADER_CLASSES:
            raise ValueError(
                f"Unknown --loader value '{loader_arg}'. "
                f"Valid options: {list(LOADER_CLASSES)}"
            )
        return _import_model_class(LOADER_CLASSES[loader_arg]), loader_arg

    name_lower = model_name.lower()
    for substring, key in AUTO_DETECT:
        if substring in name_lower:
            return _import_model_class(LOADER_CLASSES[key]), key

    return _import_model_class("AutoModelForCausalLM"), "auto"


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_ram(required_gb: float = RAM_WARN_THRESHOLD) -> None:
    """Warn if available RAM looks insufficient for a full-precision load."""
    import psutil
    available = psutil.virtual_memory().available / 1024**3
    total     = psutil.virtual_memory().total / 1024**3
    print(f"   RAM : {available:.1f} GB free / {total:.1f} GB total")
    if available < required_gb:
        print(
            f"   ⚠️  Warning: a full-precision load needs ~{required_gb:.0f} GB RAM. "
            f"Only {available:.1f} GB available — may OOM."
        )


def fmt_time(seconds: float) -> str:
    """Format elapsed seconds as '1m 23s' or '45s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ── Main merge ─────────────────────────────────────────────────────────────────

def merge(
    base: str,
    adapter: str,
    output_dir: str,
    dtype: str,
    hf_token: str | None,
    loader: str = DEFAULT_LOADER,
) -> None:
    """
    Full merge pipeline: load base → attach adapter → merge → save.

    Args:
        base:       HF Hub repo id or local path of the full-precision base model.
        adapter:    Path to the trained LoRA adapter directory.
        output_dir: Where to write the merged model.
        dtype:      Weight dtype for the merge — "bf16" or "f16".
        hf_token:   HF access token; None uses cached credentials.
        loader:     Model class to use — "auto" (detect from name) or "qwen3.5".
    """
    import gc
    import time
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    torch_dtype         = torch.float16 if dtype == "f16" else torch.bfloat16
    model_class, loader_key = resolve_loader(base, loader)
    out                 = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  base    : {base}")
    print(f"  adapter : {adapter}")
    print(f"  output  : {out}")
    print(f"  dtype   : {dtype}")
    print(f"  loader  : {model_class.__name__}  (resolved from --loader={loader_key})")
    print(f"{'─' * 60}\n")

    check_ram()

    # ── 1. Load base model ─────────────────────────────────────────────────────
    print(f"⏳ Loading base model ({model_class.__name__}) …")
    t0         = time.perf_counter()
    base_model = model_class.from_pretrained(
        base,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=hf_token,
    )
    print(f"✅ Base model loaded  ({fmt_time(time.perf_counter() - t0)})\n")

    # ── 2. Load tokenizer ──────────────────────────────────────────────────────
    print("⏳ Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True,
        token=hf_token,
    )
    print("✅ Tokenizer loaded\n")

    # ── 3. Attach LoRA adapter ─────────────────────────────────────────────────
    print("⏳ Loading LoRA adapter …")
    model = PeftModel.from_pretrained(base_model, adapter, device_map="cpu")
    print("✅ Adapter loaded\n")

    # ── 4. Merge ───────────────────────────────────────────────────────────────
    print("⏳ Merging adapter into base weights …")
    t0     = time.perf_counter()
    merged = model.merge_and_unload()
    print(f"✅ Merged  ({fmt_time(time.perf_counter() - t0)})\n")

    # Free pre-merge objects from RAM before saving.
    del model, base_model
    gc.collect()

    # ── 5. Save ────────────────────────────────────────────────────────────────
    print(f"💾 Saving merged model to '{out}' …")
    t0 = time.perf_counter()
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"✅ Saved  ({fmt_time(time.perf_counter() - t0)})\n")

    del merged
    gc.collect()

    print(f"✅ Done. Merged model: {out}")
    print("   Next steps: convert to GGUF → quantize to Q4_K_M → upload_to_hf.py\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge a base model + LoRA adapter on CPU → full-precision merged model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--base", required=True,
        help="HF Hub repo id or local path of the full-precision base model.",
    )
    ap.add_argument(
        "--adapter", required=True,
        help="Path to the trained LoRA adapter directory.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Directory where the merged model will be saved.",
    )
    ap.add_argument(
        "--loader", default=DEFAULT_LOADER, choices=list(LOADER_CLASSES),
        help=(
            "Model class to use for loading. "
            "'auto' detects from the model name (qwen3.5 → Qwen3_5ForConditionalGeneration, "
            "everything else → AutoModelForCausalLM)."
        ),
    )
    ap.add_argument(
        "--dtype", default=DEFAULT_DTYPE, choices=["bf16", "f16"],
        help="Weight precision for the merge. bf16 matches Qwen3.5 training dtype.",
    )
    ap.add_argument(
        "--hf-token", default=None,
        help="HF access token. Omit to use credentials from `huggingface-cli login`.",
    )
    return ap.parse_args()


def main() -> None:
    """CLI entrypoint — registered as ``qlora-merge`` in pyproject.toml."""
    args = parse_args()
    merge(
        base=args.base,
        adapter=args.adapter,
        output_dir=args.output,
        dtype=args.dtype,
        loader=args.loader,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
