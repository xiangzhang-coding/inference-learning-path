# Part 5 · Serving & Throughput (vLLM Core)

> The heart of the path. This is where "maximize concurrency and backend throughput" is actually won.

## What this part covers

- From **static → continuous batching**: the first lever on throughput
- **PagedAttention**: managing the [KV cache](../part0/kv-cache.md) like virtual memory — the root of vLLM's high throughput
- **Scheduler**: chunked prefill, PD disaggregation — tuning the TTFT/throughput balance
- **Prefix caching** & **speculative decoding**: further speedups in the right scenarios
- A vLLM **end-to-end architecture map**: engine / scheduler / block manager / worker
- The core **tuning knobs** and how each moves the throughput/latency curve

## Lessons

- **[From Static to Continuous Batching](continuous-batching.md)** — why static batching leaves the GPU idle (bubbles, head-of-line blocking), how Orca's iteration-level scheduling (evict-finished, admit-waiting every step) keeps the batch full, and why that is the first lever on throughput because decode is memory-bound.
- **[PagedAttention: KV Cache as Virtual Memory](paged-attention.md)** — how the block manager kills the internal fragmentation that caps concurrency: fixed-size blocks in a pool sized by profiling (`num_gpu_blocks`), a per-sequence block table, grow-on-demand and free-on-finish, block sharing for prefixes — and how that reclaimed VRAM turns into a bigger continuous batch. (The [kernel that reads](../part3/paged-attention-kernel.md) these blocks is Part 3.)

!!! note "Scaffolding status"
    Batching + PagedAttention (ticket #12) are in, each with a two-way-linked interview question: [continuous batching](../interview/continuous-batching.md) and the [block manager](../interview/kv-cache-block-manager.md). Still to land: the **scheduler** (chunked prefill, PD disaggregation), **prefix caching** & **speculative decoding**, the vLLM **architecture map**, and the **tuning knobs** (later tickets). All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**. See the **[Glossary](../glossary.md)** and the [Interview Bank](../interview/index.md).
