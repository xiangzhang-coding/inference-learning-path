# Interview Bank

> A growing bank of **high-frequency** interview questions, organized by module (Part 0–8). Each question follows one schema: **direct answer → deep dive → code (if applicable) → interviewer follow-up → linked concept**.

Every question links back to the lesson it tests, and every lesson's "Interview links" section links here — a closed learn-and-practice loop.

## By module

- **Part 0 · Foundations**
    - [Prefill vs decode](prefill-vs-decode.md) — which phase is compute- vs memory-bound, and why.
    - [Attention variants: MHA/MQA/GQA](attention-variants.md) — how KV heads set the KV cache and the throughput ceiling.
    - [KV cache & throughput ceiling](kv-cache.md) — why the KV cache, not compute, is usually the bottleneck.
    - [GPU memory hierarchy & roofline](gpu-memory-hierarchy.md) — walk the memory tiers and use the roofline to explain why decode is memory-bound.
    - [Latency vs throughput metrics](latency-throughput-metrics.md) — TTFT/TPOT/ITL/throughput/goodput, how to measure, and the batch-size trade.
    - [Number formats & precision](number-formats.md) — FP16/BF16/FP8/INT8/INT4, range vs precision, and why low-bit speeds up decode.
- **Part 2 · Single-GPU Inference Performance**
    - [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md) — derive an operator's intensity from its shapes, why decode attention is context-independent, and the batch that crosses the ridge.
    - [VRAM budget & max concurrency](vram-capacity-planning.md) — walk the full VRAM budget and size max concurrency; the knobs that hit a concurrency target.
    - [FlashAttention & IO-aware attention](flash-attention.md) — why it's faster at the same FLOPs, online softmax, and where it does/doesn't help.
    - [CUDA graphs & kernel fusion](cuda-graphs-fusion.md) — decode launch overhead, why it hits decode not prefill, and what `enforce_eager` trades.
- **Parts 1, 3–8** — questions land alongside their lessons in later tickets.

!!! note "Scaffolding status"
    Part 0 (tickets #2, #4, #5) and Part 2 (tickets #6, #7) questions are in, each two-way-linked to the lesson it tests. The full ~100-question bank grows as parts land. Difficulty tiers / frequency tags / weighting are intentionally out of scope for now.
