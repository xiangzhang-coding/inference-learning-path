# Part 2 · Single-GPU Inference Performance

> Build the hardware mental model, then learn to *quantify* any optimization before trusting it.

## What this part covers

- **GPU hardware model**: SM / warp, **HBM vs SRAM**, bandwidth vs compute
- **Metrics & measurement**: TTFT, TPOT/ITL, throughput, goodput
- **Roofline & arithmetic intensity**: is attention / GEMM compute- or bandwidth-limited?
- **FlashAttention**: the IO-aware idea (tiling, online softmax) and why it's faster
- **Kernel fusion & CUDA graphs**: why decode-stage launch overhead is deadly

See the **[Glossary](../glossary.md)** for the performance vocabulary.

!!! note "Scaffolding status"
    This part's lessons land in a later ticket, building on **[Part 0](../part0/index.md)**.
