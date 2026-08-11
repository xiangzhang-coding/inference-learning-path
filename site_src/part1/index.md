# Part 1 · Transformer (Infra View)

> The same Transformer you know from training, re-read through an **inference-cost lens**: every architectural choice maps directly to memory and throughput.

## What this part covers

- **Q / K / V → KV cache**: where the cache comes from and why attention needs history
- **MHA / MQA / GQA**: fewer KV heads → smaller [KV cache](../part0/kv-cache.md) → higher throughput ceiling
- **FFN / MLP**: where most FLOPs and weight memory live
- **RoPE**: rotary position embedding and why its extrapolation matters for long context

See the **[Glossary](../glossary.md)** for the architecture vocabulary.

!!! note "Scaffolding status"
    This part's lessons land in a later ticket. The foundations they build on are in **[Part 0](../part0/index.md)**.
