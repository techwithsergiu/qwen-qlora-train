---
title: Training pipeline
---

# Training pipeline

The full pipeline spans three phases across two repos.

```mermaid
flowchart TD
    SRC["Qwen/Qwen3.5-{size}<br/>f16 · source"]

    subgraph PREP ["① Model Prep — qwen35-toolkit"]
        BNBVLM["Qwen3.5-{size}-bnb-4bit<br/>BNB NF4 · VLM"]
        TEXTBNB["Qwen3.5-text-{size}-bnb-4bit<br/>BNB NF4 · text-only ✅"]
        V1{{"✅ verified"}}
        V3{{"✅ verified"}}
        BNBVLM -->|"qwen35-strip --mode bnb"| TEXTBNB
        BNBVLM -->|"qwen35-verify-qwen35"| V1
        TEXTBNB -->|"qwen35-verify"| V3
        TEXTF16["Qwen3.5-text-{size}<br/>bf16 · text-only"]
        V2{{"✅ verified"}}
        TEXTF16 -->|"qwen35-verify"| V2
    end

    subgraph TRAIN_PHASE ["② Training — qwen-qlora-train"]
        TRAIN["Unsloth LoRA Training<br/>QLoRA · rank 16–64 · TRL + PEFT"]
        ADAPTER["LoRA Adapter<br/>~100–500 MB"]
        INFER2["qlora-infer<br/>base + adapter · smoke test"]
        MERGED["Merged f16<br/>full weights · CPU merge"]
        TRAIN --> ADAPTER
        ADAPTER -->|"qlora-infer"| INFER2
        ADAPTER -->|"qlora-merge"| MERGED
    end

    subgraph EXPORT ["③ Export — qwen35-toolkit"]
        GGUFT["GGUF f16"]
        QUANT["Q4_K_M ✅ · Q5_K_M<br/>Q6_K · Q8_0"]
        INFER["LM Studio / llama.cpp<br/>~35 tok/s · RTX 3070"]
        GGUFT -->|"llama-quantize"| QUANT
        QUANT --> INFER
    end

    HUB[("HuggingFace Hub")]

    SRC -->|"qwen35-convert"| BNBVLM
    SRC -->|"qwen35-strip --mode f16"| TEXTF16
    V1 -->|"qwen35-upload"| HUB
    V3 -->|"qwen35-upload"| HUB
    TEXTBNB --> TRAIN
    MERGED -->|"convert_hf_to_gguf.py"| GGUFT
    V2 -->|"qwen35-upload"| HUB
    TEXTF16 -->|"convert_hf_to_gguf.py"| GGUFT
    QUANT -->|"qwen35-upload"| HUB

    style PREP        fill:#fefce8,stroke:#ca8a04
    style TRAIN_PHASE fill:#eff6ff,stroke:#3b82f6
    style EXPORT      fill:#fdf4ff,stroke:#9333ea

    style TEXTBNB fill:#dcfce7,stroke:#16a34a
    style TRAIN   fill:#dbeafe,stroke:#3b82f6
    style ADAPTER fill:#dbeafe,stroke:#3b82f6
    style MERGED  fill:#dbeafe,stroke:#3b82f6
    style INFER2  fill:#dbeafe,stroke:#3b82f6
    style QUANT   fill:#fce7f3,stroke:#db2777
    style HUB     fill:#f3e8ff,stroke:#9333ea
```

## Phases

**① Model prep — [qwen35-toolkit](https://github.com/techwithsergiu/qwen35-toolkit)**

Convert source Qwen3.5 VLM to a text-only BNB NF4 4-bit model ready for training.
Pre-quantized models are already published on HuggingFace — no local prep needed unless you want to build your own.

**② Training — qwen-qlora-train (this repo)**

Fine-tune with QLoRA on a single consumer GPU. Produces a LoRA adapter (~100–500 MB).
Use `qlora-infer` to verify the adapter immediately after training, then `qlora-merge` to produce a standalone fp16 model.

**③ Export — [qwen35-toolkit](https://github.com/techwithsergiu/qwen35-toolkit)**

Convert the merged fp16 model to GGUF, quantize, and upload to HuggingFace Hub.
