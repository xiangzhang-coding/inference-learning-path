# Capstone

> The finale: on a **single RTX 4090** within a **¥500 AutoDL budget**, push `Qwen2.5-7B-Instruct` throughput as far as it goes — and write the **"before → optimization → after" report**.

## The goal

Take everything from Parts 0–8 and turn knobs that actually move the number:

- Quantization (Part 4) to fit more KV cache in 24 GB
- Continuous batching + PagedAttention tuning (Part 5)
- Prefix caching where the workload has shared prefixes
- The right `gpu-memory-utilization`, `max-model-len`, and batching settings for your SLO

You produce a report that measures throughput and latency **before and after**, on your own AutoDL box.

!!! gpu "Capstone Lab"
    - **Min VRAM:** 24 GB
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** a few hours of GPU time, well within ¥500 (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** compatibility differences noted per technique where relevant

!!! warning "On the numbers"
    All targets and figures here are **illustrative / order-of-magnitude references** (ADR-0004). The real "before → after" numbers are the ones *you* measure.

!!! note "Scaffolding status"
    The full Capstone brief lands after the parts it depends on. It builds directly on **[Part 5](../part5/index.md)** and **[Part 8](../part8/index.md)**.
