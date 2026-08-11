# Part 0 · Foundations & Motivation

> **Motivation-first.** Before any optimization, we build the one mental model everything else hangs on: **why LLM inference is memory-bound**. Once that clicks, every later trick — KV cache tuning, quantization, PagedAttention, continuous batching — becomes a conclusion you can *derive*, not a fact to memorize.

## What this part covers

- Why LLM inference is **memory-bound** (the throughput story starts here)
- The two phases: **prefill** vs **decode**, and which optimizations act on which
- **[KV cache](kv-cache.md)** — what it is, why it exists, how it grows, and why it is the core tension behind the throughput ceiling
- Inference **metrics**: TTFT, TPOT/ITL, throughput, goodput — and how to measure them
- **Number formats**: FP16 / BF16 / FP8 / INT8 / INT4 — enough to enter the quantization part unblocked

## Lessons

- **[Inference Flow: Prefill & Decode](inference-flow.md)** — the two phases of autoregressive generation, and why prefill is compute-bound while decode is memory-bound.
- **[Transformer, the Infra View](transformer-infra.md)** — read a decoder block as a cost model: which parts cost weights, prefill FLOPs, and KV cache.
- **[KV Cache](kv-cache.md)** — what it is, why it grows, and why it is the core tension behind the throughput ceiling.

!!! note "Scaffolding status"
    This learning path is being built part by part. Part 0A (tickets #2, #4) is fully written — inference flow, the Transformer infra view, and KV cache — with linked interview questions. Part 0B (metrics, number formats) and Parts 1–8 land in later tickets. See the [Interview Bank](../interview/index.md) for the linked question set as it grows.
