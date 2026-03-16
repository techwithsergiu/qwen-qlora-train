---
title: Post-merge workflow
---


## Post-merge workflow

After `qlora-merge` you have a standalone fp16 model. The next steps —
GGUF conversion, quantization, and Hub upload — are handled by
**[qwen35-toolkit](https://github.com/techwithsergiu/qwen35-toolkit)**.

| Step | Tool | What it produces |
|------|------|-----------------|
| Convert to GGUF | `llama.cpp/convert_hf_to_gguf.py` | `model-F16.gguf` |
| Quantize | `llama-quantize` | `Q4_K_M` · `Q5_K_M` · `Q6_K` · `Q8_0` |
| Upload | `qwen35-upload` | HuggingFace Hub repo |

> See **[qwen35-toolkit → Usage examples](https://github.com/techwithsergiu/qwen35-toolkit#usage-examples)**
> for exact commands.
