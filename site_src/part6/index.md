# Part 6 · Advanced Inference Topics

> The specialized serving forms you'll be asked about once the basics are solid.

## What this part covers

- **Multi-LoRA serving**: one base model + many adapters swapped dynamically
- **Guided / structured decoding**: constrained JSON / regex / grammar output
- **Long-context inference**: RoPE extrapolation, attention sink, KV compression, and the memory/scheduling problems of long sequences

Long-context inference pushes directly on the [KV cache](../part0/kv-cache.md) growth problem.

!!! note "Scaffolding status"
    This part's lessons land in later tickets. See the **[Glossary](../glossary.md)**.
