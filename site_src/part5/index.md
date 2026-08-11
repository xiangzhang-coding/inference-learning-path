# Part 5 · Serving & Throughput (vLLM Core)

> The heart of the path. This is where "maximize concurrency and backend throughput" is actually won.

## What this part covers

- From **static → continuous batching**: the first lever on throughput
- **PagedAttention**: managing the [KV cache](../part0/kv-cache.md) like virtual memory — the root of vLLM's high throughput
- **Scheduler**: chunked prefill, PD disaggregation — tuning the TTFT/throughput balance
- **Prefix caching** & **speculative decoding**: further speedups in the right scenarios
- A vLLM **end-to-end architecture map**: engine / scheduler / block manager / worker
- The core **tuning knobs** and how each moves the throughput/latency curve

!!! note "Scaffolding status"
    This part's lessons land in later tickets. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**. See the **[Glossary](../glossary.md)**.
