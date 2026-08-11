# Part 4 · Quantization

> Why quantization raises throughput, what precision it costs, and how to choose a method in the real world.

## What this part covers

- Why quantization helps throughput, and the **precision trade-offs**
- **Weight-only vs weight+activation**, granularity (per-tensor/channel/group), symmetric/asymmetric
- Method families: **GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** and **KV-cache quantization**
- A complete runnable path to quantize `Qwen2.5-7B-Instruct` and serve INT4 in vLLM

KV-cache quantization connects directly back to **[KV cache](../part0/kv-cache.md)**.

!!! note "Scaffolding status"
    This part's lessons land in a later ticket. See the **[Glossary](../glossary.md)** for the quantization vocabulary.
