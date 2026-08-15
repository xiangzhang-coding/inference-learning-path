# Part 4 · Quantization

> Why quantization raises throughput, what precision it costs, and how to choose a method in the real world.

## What this part covers

- **[Why quantization helps throughput](quantization-basics.md)**, and the **precision trade-offs** — the affine map and its error bound
- **[Weight-only vs weight+activation](quantization-schemes.md)**, granularity (per-tensor/channel/group), symmetric/asymmetric, and PTQ vs QAT
- **[Method families](quantization-methods.md)**: **GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** and **KV-cache quantization**
- **[A complete runnable path](quantization-lab.md)** to quantize `Qwen2.5-7B-Instruct` and serve INT4 in vLLM, comparing quality and throughput

KV-cache quantization connects directly back to **[KV cache](../part0/kv-cache.md)**.

## Lessons

- **[Why Quantization Speeds Up Inference: the Affine Map & the Precision Trade-off](quantization-basics.md)** — why fewer weight bits mean faster memory-bound decode (bandwidth, not FLOPs), the affine quantization map ($\hat{x}=\text{scale}\cdot(q-z)$), and the error bound ($\le \text{scale}/2$) that outliers inflate.
- **[Quantization Choices: Granularity, Symmetry, What to Quantize, and PTQ vs QAT](quantization-schemes.md)** — the four engineering choices that keep error small at low bits: per-tensor/channel/group, symmetric/asymmetric, weight-only (`W4A16`) vs weight+activation (`W8A8`), and why inference uses PTQ.
- **[Quantization Method Families: GPTQ, AWQ, SmoothQuant, FP8, LLM.int8(), KV-cache](quantization-methods.md)** — each method as a point in the design space plus one anti-outlier trick, and how to pick one for a given bottleneck.
- **[Hands-On: Quantize Qwen2.5-7B to INT4, Serve in vLLM, Compare Quality & Throughput](quantization-lab.md)** — the complete runnable path: quantize with llm-compressor (or a prebuilt AWQ checkpoint), serve (auto-detected), A/B quality on the small eval set and measure throughput.

!!! note "Scaffolding status"
    Part 4 is complete: all four lessons (tickets #10, #11) are written, each with a two-way-linked interview question — the principles ([basics](quantization-basics.md), [schemes](quantization-schemes.md)) and the applied half ([method families](quantization-methods.md), the [hands-on INT4 lab](quantization-lab.md)). Hand-written CUDA kernels stay out of scope (ADR-0002); spending the freed VRAM on concurrency is **Part 5**. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**. See the **[Glossary](../glossary.md)** for the quantization vocabulary and the [Interview Bank](../interview/index.md) for the linked questions.
