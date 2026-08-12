# Part 2 · Single-GPU Inference Performance

> You have the hardware mental model from [Part 0](../part0/index.md); now learn to *quantify* a single GPU's limits — derive any operator's regime from its shapes, and size a deployment's memory budget before you rent the card.

## What this part covers

- **[Operator roofline](roofline-analysis.md)**: derive arithmetic intensity for GEMMs and attention — why decode is memory-bound at $I\approx1$, prefill compute-bound, and the batch size that crosses the ridge
- **[KV cache memory math](kv-cache-math.md)**: the full VRAM budget (weights + KV + activations + overhead) and how to solve it for max concurrency
- **FlashAttention** and **kernel fusion / CUDA graphs**: the IO-aware and launch-overhead wins (ticket #7)

The hardware primer this builds on — the [memory hierarchy & roofline](../part0/gpu-hardware.md) and the [latency/throughput metrics](../part0/metrics.md) — lives in **[Part 0](../part0/index.md)**. See the **[Glossary](../glossary.md)** for the performance vocabulary.

## Lessons

- **[Operator Roofline: Arithmetic Intensity of GEMM & Attention](roofline-analysis.md)** — decompose a decoder layer into its matmuls and attention op, and compute each operator's intensity and roofline regime from its shapes alone.
- **[KV Cache Memory Math: Sizing a Deployment](kv-cache-math.md)** — assemble the full VRAM budget and solve for max concurrent sequences (and the inverse: max context at a target concurrency).

!!! note "Scaffolding status"
    **Roofline analysis** and **KV cache memory math** (ticket #6) are written, each with a linked interview question. **FlashAttention** and **kernel fusion / CUDA graphs** (ticket #7) land next. See the [Interview Bank](../interview/index.md) for the linked question set.
