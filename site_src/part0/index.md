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
- **[GPU Hardware Mental Model](gpu-hardware.md)** — the memory pyramid (HBM vs SRAM), the SM/warp execution model, and the roofline that proves decode is bandwidth-bound.
- **[Inference Performance Metrics](metrics.md)** — TTFT, TPOT/ITL, throughput, and goodput, plus how to measure each with `vllm bench serve` and Prometheus.
- **[Number Formats: FP16 · BF16 · FP8 · INT8 · INT4](number-formats.md)** — the range-vs-precision trade behind every dtype, paving the way into quantization.

!!! note "Scaffolding status"
    This learning path is being built part by part. **Part 0A** (tickets #2, #4) — inference flow, the Transformer infra view, KV cache — and **Part 0B** (ticket #5) — GPU hardware, metrics, number formats — are fully written, each with linked interview questions. Parts 1–8 land in later tickets. See the [Interview Bank](../interview/index.md) for the linked question set as it grows.
