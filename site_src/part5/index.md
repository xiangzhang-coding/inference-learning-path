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
- **[The Scheduler: Chunked Prefill & PD Disaggregation](scheduler-chunked-prefill-pd.md)** — why a long prefill freezes ongoing decodes, how chunked prefill slices it to share each step's `max_num_batched_tokens` budget (the TTFT↔ITL dial), and how PD disaggregation splits prefill and decode across GPU pools at scale.
- **[Prefix Caching: Reuse Shared-Prefix KV](prefix-caching.md)** — how content-hashed blocks (token + parent hash) let requests sharing a system prompt / few-shot / chat history skip the shared prefill entirely, with byte-identical outputs; when it helps and what silently kills the hit rate.
- **[Speculative Decoding: Guess Many, Verify Once](speculative-decoding.md)** — a cheap draft proposes K tokens, the target verifies K+1 in one pass; why it's nearly free only because decode is memory-bound, what the acceptance rate sets, and when it backfires at large batch.

!!! note "Scaffolding status"
    Batching + PagedAttention (ticket #12) and the scheduler + prefix caching + speculative decoding (ticket #13) are in, each with a two-way-linked interview question: [continuous batching](../interview/continuous-batching.md), [block manager](../interview/kv-cache-block-manager.md), [chunked prefill & PD](../interview/chunked-prefill-pd.md), [prefix caching](../interview/prefix-caching.md), [speculative decoding](../interview/speculative-decoding.md). Still to land: the vLLM **end-to-end architecture map** and the core **tuning knobs** (later tickets). All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**. See the **[Glossary](../glossary.md)** and the [Interview Bank](../interview/index.md).
