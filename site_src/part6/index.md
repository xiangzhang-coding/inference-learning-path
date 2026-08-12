# Part 6 · Advanced Inference Topics

> The specialized serving forms you'll be asked about once the Part 5 basics are solid — where "one model per use case" and "the model just emits text" both stop being good enough.

## What this part covers

- **Multi-LoRA serving**: one base model + many low-rank adapters, swapped per request and even *mixed within one batch* — how to serve dozens of fine-tunes on a single [24 GB card](../part0/gpu-hardware.md) instead of one full copy each.
- **Guided / structured decoding**: constrain the output to valid **JSON / regex / grammar / enum** by masking impossible tokens at every decode step — turning "prompt and pray" into schema-valid-by-construction.
- **Long-context inference** *(later ticket)*: RoPE extrapolation, attention sink, KV compression, and the memory/scheduling problems of long sequences — it pushes directly on the [KV cache](../part0/kv-cache.md) growth problem.

Both topics here ride on the same [PagedAttention block pool](../part5/paged-attention.md) and [continuous-batching](../part5/continuous-batching.md) machinery from Part 5 — they're what you *layer on top* once throughput is handled.

## Lessons

- **[Multi-LoRA Serving: One Base, Many Adapters](multi-lora-serving.md)** — why serving N fine-tunes naively means N full model copies (and why that's impossible on one card), how [LoRA](../glossary.md)'s low-rank delta $\Delta W = BA$ shrinks each fine-tune to megabytes, how vLLM keeps one frozen base plus a shelf of adapters and applies a *different* adapter per row of the *same* batch via grouped GEMM kernels, and the knobs (`--max-lora-rank`, `max_loras`, dynamic loading) that decide how many you can co-serve.
- **[Guided / Structured Decoding: Make Invalid Tokens Impossible](structured-decoding.md)** — why prompting for JSON still yields broken JSON some fraction of the time, how a schema/regex/grammar compiles to a finite-state machine that produces a **token mask** at every step, how vLLM sets disallowed logits to $-\infty$ so only schema-valid tokens can be sampled (xgrammar/guidance backends), and the sharp edge every interviewer probes: it guarantees *shape*, never *truth*.

!!! note "Part 6 status"
    This ticket (#15, "Part 5A") lands the first two lessons — **[multi-LoRA serving](multi-lora-serving.md)** and **[structured decoding](structured-decoding.md)** — each with a two-way-linked interview question ([multi-LoRA](../interview/multi-lora-serving.md), [structured decoding](../interview/structured-decoding.md)). **Long-context inference** follows in the next ticket (#16). All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**, and every performance number is an **illustrative / order-of-magnitude reference**. See the **[Glossary](../glossary.md)** and the [Interview Bank](../interview/index.md).
