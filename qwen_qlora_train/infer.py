#!/usr/bin/env python3
"""
infer.py — run inference on a base model, base + LoRA adapter, or merged model.

Primary use case: verify a LoRA adapter immediately after training,
before committing to a CPU merge.

Usage
-----
  # Predefined smoke tests (base + adapter)
  qlora-infer --model unsloth/Qwen3-1.7B-bnb-4bit --adapter adapters/qwen3-1.7b-sanity

  # Single prompt
  qlora-infer --model unsloth/Qwen3-1.7B-bnb-4bit --adapter adapters/qwen3-1.7b-sanity --prompt "Explain LoRA in two sentences."

  # Interactive chat
  qlora-infer --model unsloth/Qwen3-1.7B-bnb-4bit --adapter adapters/qwen3-1.7b-sanity --interactive

  # Merged fp16 model (no adapter)
  qlora-infer --model merged/qwen3-1.7b-merged-f16 --dtype f16

  # Disable thinking
  qlora-infer --model ... --adapter ... --no-thinking
"""


def main() -> None:
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(
        description="Run inference on a base model or base + LoRA adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model",        required=True,  help="HF repo id or local model path.")
    ap.add_argument("--adapter",      default=None,   help="LoRA adapter directory (optional).")
    ap.add_argument("--hf-token",     default=None,   help="HF token (or set HF_TOKEN env).")
    ap.add_argument("--dtype",        default="f16",  choices=["f16", "bf16"])
    ap.add_argument("--chat-template",default="qwen3",help="Unsloth chat template key.")
    ap.add_argument("--max-seq-length",type=int, default=4096, help="Max sequence length (Unsloth path).")
    ap.add_argument("--backend",      default="auto", choices=["auto", "unsloth", "transformers"],
                    help="auto: detect from model id. unsloth: 4-bit. transformers: fp16/bf16.")
    ap.add_argument("--max-new",      type=int,   default=1024, help="Max new tokens to generate.")
    ap.add_argument("--temp",         type=float, default=0.7,  help="Sampling temperature.")
    ap.add_argument("--top-p",        type=float, default=0.9,  help="Top-p sampling.")
    ap.add_argument("--prompt",       default=None, help="Single user prompt — skips predefined tests.")
    ap.add_argument("--interactive",  action="store_true", help="Interactive chat loop.")
    ap.add_argument("--no-thinking",  action="store_true", help="Disable Qwen3 thinking mode.")
    args = ap.parse_args()

    # ── Heavy imports (deferred so --help is instant) ─────────────────────────
    import torch

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    dtype     = torch.float16 if args.dtype == "f16" else torch.bfloat16

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — inference will run on CPU (slow).")

    # ── Load ──────────────────────────────────────────────────────────────────
    model, tok, backend = _load(
        model_id   = args.model,
        adapter    = args.adapter,
        hf_token   = hf_token,
        dtype      = dtype,
        template   = args.chat_template,
        max_seq    = args.max_seq_length,
        backend    = args.backend,
    )

    print(f"\n[Loaded]  backend={backend}")
    print(f"  model   : {args.model}")
    if args.adapter:
        print(f"  adapter : {args.adapter}")
    print(f"  dtype   : {args.dtype}")
    print(f"  thinking: {'off' if args.no_thinking else 'on'}")
    print(f"[Tokens]  eos={tok.eos_token!r} ({tok.eos_token_id})  "
          f"pad={tok.pad_token!r} ({tok.pad_token_id})\n")

    gen_kwargs = dict(
        model          = model,
        tokenizer      = tok,
        max_new_tokens = args.max_new,
        temperature    = args.temp,
        top_p          = args.top_p,
        no_thinking    = args.no_thinking,
    )

    # ── Modes ─────────────────────────────────────────────────────────────────
    if args.prompt:
        _run_single(args.prompt, **gen_kwargs)
        return

    if args.interactive:
        _interactive(**gen_kwargs)
        return

    _predefined_tests(**gen_kwargs)


# ── Loader ────────────────────────────────────────────────────────────────────

def _is_4bit(model_id: str) -> bool:
    s = (model_id or "").lower()
    return "bnb-4bit" in s or "unsloth-bnb-4bit" in s


def _load(model_id, adapter, hf_token, dtype, template, max_seq, backend):
    use_unsloth = (
        backend == "unsloth"
        or (backend == "auto" and _is_4bit(model_id))
    )

    if use_unsloth:
        return _load_unsloth(model_id, adapter, hf_token, dtype, template, max_seq)
    else:
        return _load_transformers(model_id, adapter, hf_token, dtype)


def _load_unsloth(model_id, adapter, hf_token, dtype, template, max_seq):
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        raise SystemExit(
            "Unsloth not installed. Install it first or use --backend transformers."
        )
    from peft import PeftModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name     = model_id,
        max_seq_length = max_seq,
        load_in_4bit   = True,
        token          = hf_token,
        dtype          = dtype,
        attn_implementation = "sdpa",
    )
    tok = get_chat_template(tok, chat_template=template)

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    model = FastLanguageModel.for_inference(model)
    model.eval()
    return model, tok, "unsloth-4bit"


def _load_transformers(model_id, adapter, hf_token, dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, use_fast=True, fix_mistral_regex=True
        )
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token          = hf_token,
        torch_dtype    = dtype,
        device_map     = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        low_cpu_mem_usage = True,
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        try:
            model = model.merge_and_unload()
        except Exception:
            pass  # keep as PeftModel if merge fails

    model.eval()
    return model, tok, "transformers-fp"


# ── Generation ────────────────────────────────────────────────────────────────

def _build_prompt(tokenizer, messages, no_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not no_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate(model, tokenizer, messages, max_new_tokens, temperature, top_p, no_thinking) -> str:
    import torch
    import logging
    import warnings

    # transformers 5.2.0 bug: warning_once(MSG, FutureWarning) tries MSG % FutureWarning
    # which raises TypeError because MSG has no % placeholder.
    # Suppress the broken log record so it doesn't pollute output.
    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    text   = _build_prompt(tokenizer, messages, no_thinking)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0.0

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = do_sample,
            temperature    = temperature if do_sample else None,
            top_p          = top_p if do_sample else None,
            eos_token_id   = tokenizer.eos_token_id,
            pad_token_id   = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ── Run helpers ───────────────────────────────────────────────────────────────

_SYSTEM = "You are a concise helpful assistant."


def _run_single(prompt, *, model, tokenizer, max_new_tokens, temperature, top_p, no_thinking):
    msgs = [
        {"role": "system",  "content": _SYSTEM},
        {"role": "user",    "content": prompt},
    ]
    out = _generate(model, tokenizer, msgs, max_new_tokens, temperature, top_p, no_thinking)
    print(out.strip())


def _interactive(*, model, tokenizer, max_new_tokens, temperature, top_p, no_thinking):
    print("Interactive mode — type 'exit' or Ctrl-C to quit.\n")
    history = [{"role": "system", "content": _SYSTEM}]

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_text.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        out = _generate(model, tokenizer, history, max_new_tokens, temperature, top_p, no_thinking)
        reply = out.strip()
        print(f"\nAssistant> {reply}\n")
        history.append({"role": "assistant", "content": reply})


def _predefined_tests(*, model, tokenizer, max_new_tokens, temperature, top_p, no_thinking):
    tests = [
        {
            "name": "Short answer",
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": "Explain what LoRA is in 2 short sentences."},
            ],
        },
        {
            "name": "Code snippet",
            "messages": [
                {"role": "system", "content": "You are a concise assistant. Keep answers short."},
                {"role": "user",   "content": "Write a TypeScript function debounce(fn, ms) in ~15 lines."},
            ],
        },
        {
            "name": "Stop behavior",
            "messages": [
                {"role": "system", "content": "Answer with exactly one line."},
                {"role": "user",   "content": "Say hello and stop."},
            ],
        },
        {
            "name": "Think check",
            "messages": [
                {"role": "system", "content": "Write ONE short thought (max 1 line), then answer."},
                {"role": "user",   "content": "Explain what LoRA is in 2 short sentences."},
            ],
        },
    ]

    for t in tests:
        print("=" * 72)
        print(f"TEST: {t['name']}")
        print("-" * 72)
        out = _generate(model, tokenizer, t["messages"], max_new_tokens, temperature, top_p, no_thinking)
        print(out.strip())
        print()

    print("✅  Done.")


if __name__ == "__main__":
    main()
