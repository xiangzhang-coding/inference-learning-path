# Part 2 · Single-GPU Inference Performance

> You have the hardware mental model from [Part 0](../part0/index.md); this part makes it *quantitative* — derive any operator's regime from its shapes, size a deployment's memory budget, and apply the two kernel-level wins (FlashAttention, CUDA graphs) that squeeze the most out of one GPU.

## What this part covers

- **[Operator roofline](roofline-analysis.md)**: derive arithmetic intensity for GEMMs and attention — why decode is memory-bound at $I\approx1$, prefill compute-bound, and the batch size that crosses the ridge
- **[KV cache memory math](kv-cache-math.md)**: the full VRAM budget (weights + KV + activations + overhead) and how to solve it for max concurrency
- **[FlashAttention](flash-attention.md)**: the IO-aware idea (tiling + online softmax) — same FLOPs, $O(S^2)\to O(S)$ memory, and why long-context prefill is feasible
- **[Kernel fusion & CUDA graphs](kernel-fusion-cuda-graphs.md)**: why decode-stage launch overhead is deadly, and how fusion and graph replay reclaim it

The hardware primer this builds on — the [memory hierarchy & roofline](../part0/gpu-hardware.md) and the [latency/throughput metrics](../part0/metrics.md) — lives in **[Part 0](../part0/index.md)**. See the **[Glossary](../glossary.md)** for the performance vocabulary.

## Lessons

- **[Operator Roofline: Arithmetic Intensity of GEMM & Attention](roofline-analysis.md)** — decompose a decoder layer into its matmuls and attention op, and compute each operator's intensity and roofline regime from its shapes alone.
- **[KV Cache Memory Math: Sizing a Deployment](kv-cache-math.md)** — assemble the full VRAM budget and solve for max concurrent sequences (and the inverse: max context at a target concurrency).
- **[FlashAttention: the IO-aware Attention Kernel](flash-attention.md)** — tile Q/K/V and use online softmax to keep the $S\times S$ scores in SRAM, turning $O(S^2)$ HBM traffic into $O(S)$ while computing the exact same output.
- **[Kernel Fusion & CUDA Graphs: Killing Decode Launch Overhead](kernel-fusion-cuda-graphs.md)** — why hundreds of tiny decode kernels make the step launch-bound, and how fusion and CUDA-graph replay collapse the overhead.

!!! note "Scaffolding status"
    Part 2 is complete: all four lessons (tickets #6, #7) are written, each with a two-way-linked interview question. Next up is **Part 3 · GPU Programming (Triton)**. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**. See the [Interview Bank](../interview/index.md) for the linked question set.
