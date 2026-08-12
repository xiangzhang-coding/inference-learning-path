# Part 4 · Quantization

> Why quantization raises throughput, what precision it costs, and how to choose a method in the real world.

## What this part covers

- **[Why quantization helps throughput](quantization-basics.md)**, and the **precision trade-offs** — the affine map and its error bound
- **[Weight-only vs weight+activation](quantization-schemes.md)**, granularity (per-tensor/channel/group), symmetric/asymmetric, and PTQ vs QAT
- Method families: **GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** and **KV-cache quantization** *(next ticket)*
- A complete runnable path to quantize `Qwen2.5-7B-Instruct` and serve INT4 in vLLM *(next ticket)*

KV-cache quantization connects directly back to **[KV cache](../part0/kv-cache.md)**.

## Lessons

- **[Why Quantization Speeds Up Inference: the Affine Map & the Precision Trade-off](quantization-basics.md)** — why fewer weight bits mean faster memory-bound decode (bandwidth, not FLOPs), the affine quantization map ($\hat{x}=\text{scale}\cdot(q-z)$), and the error bound ($\le \text{scale}/2$) that outliers inflate.
- **[Quantization Choices: Granularity, Symmetry, What to Quantize, and PTQ vs QAT](quantization-schemes.md)** — the four engineering choices that keep error small at low bits: per-tensor/channel/group, symmetric/asymmetric, weight-only (`W4A16`) vs weight+activation (`W8A8`), and why inference uses PTQ.

!!! note "Scaffolding status"
    The principles half of Part 4 is in (ticket #10): [quantization basics](quantization-basics.md) and [the four scheme choices](quantization-schemes.md), each with a two-way-linked interview question. The concrete method families (GPTQ/AWQ/SmoothQuant/FP8/LLM.int8(), KV-cache quant) and the hands-on `Qwen2.5-7B` → INT4 run in vLLM land next (#11). See the **[Glossary](../glossary.md)** for the quantization vocabulary and the [Interview Bank](../interview/index.md) for the linked questions.
