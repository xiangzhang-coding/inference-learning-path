# Part 6 · Advanced Inference Topics

> The specialized serving forms you'll be asked about once the Part 5 basics are solid — where "one model per use case" and "the model just emits text" both stop being good enough.

## What this part covers

- **Multi-LoRA serving**: one base model + many low-rank adapters, swapped per request and even *mixed within one batch* — how to serve dozens of fine-tunes on a single [24 GB card](../part0/gpu-hardware.md) instead of one full copy each.
- **Guided / structured decoding**: constrain the output to valid **JSON / regex / grammar / enum** by masking impossible tokens at every decode step — turning "prompt and pray" into schema-valid-by-construction.
- **Long-context inference**: RoPE extrapolation, attention sink, KV compression/quantization, and the memory/scheduling problems of long sequences — it pushes directly on the [KV cache](../part0/kv-cache.md) growth problem.

All three topics here ride on the same [PagedAttention block pool](../part5/paged-attention.md) and [continuous-batching](../part5/continuous-batching.md) machinery from Part 5 — they're what you *layer on top* once throughput is handled.

## Lessons

- **[Multi-LoRA Serving: One Base, Many Adapters](multi-lora-serving.md)** — why serving N fine-tunes naively means N full model copies (and why that's impossible on one card), how [LoRA](../glossary.md)'s low-rank delta $\Delta W = BA$ shrinks each fine-tune to megabytes, how vLLM keeps one frozen base plus a shelf of adapters and applies a *different* adapter per row of the *same* batch via grouped GEMM kernels, and the knobs (`--max-lora-rank`, `max_loras`, dynamic loading) that decide how many you can co-serve.
- **[Guided / Structured Decoding: Make Invalid Tokens Impossible](structured-decoding.md)** — why prompting for JSON still yields broken JSON some fraction of the time, how a schema/regex/grammar compiles to a finite-state machine that produces a **token mask** at every step, how vLLM sets disallowed logits to $-\infty$ so only schema-valid tokens can be sampled (xgrammar/guidance backends), and the sharp edge every interviewer probes: it guarantees *shape*, never *truth*.
- **[Long-Context Inference: RoPE Scaling, Attention Sink & the KV Wall](long-context-inference.md)** — why a model trained to 32K produces garbage at 128K (RoPE angles going out-of-distribution) and how Position Interpolation / NTK / **YaRN** rescale position back (`--hf-overrides` `rope_parameters` + `--max-model-len`), why you can't just keep the last N tokens (**attention sinks**), and why the [KV cache](../part0/kv-cache.md) — *linear* in length — is the real long-context ceiling, mitigated by fp8 KV, GQA, sliding windows, and [chunked prefill](../part5/scheduler-chunked-prefill-pd.md).

!!! note "Part 6 complete"
    All three lessons are in — **[multi-LoRA serving](multi-lora-serving.md)** & **[structured decoding](structured-decoding.md)** and **[long-context inference](long-context-inference.md)** — each with a two-way-linked interview question ([multi-LoRA](../interview/multi-lora-serving.md), [structured decoding](../interview/structured-decoding.md), [long-context](../interview/long-context-inference.md)). Together they cover the specialized serving forms layered on top of Part 5's throughput core. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**, and every performance number is an **illustrative / order-of-magnitude reference**. See the **[Glossary](../glossary.md)** and the [Interview Bank](../interview/index.md).
