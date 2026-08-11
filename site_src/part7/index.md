# Part 7 · Multi-GPU & Distributed

> When one 4090 isn't enough: how to split a model across GPUs and read/configure the result.

## What this part covers

- **Tensor / Pipeline / Data / Expert parallelism** and **NCCL** collective communication
- How to enable **TP / PP in vLLM** and how to choose the **TP degree**
- **Load testing** and finding the concurrency **knee** — the real throughput ceiling of a service

!!! gpu "Multi-GPU note"
    Per ADR-0001, the main line runs on a single RTX 4090. The 1–2 topics that truly need multiple GPUs (e.g. a TP/PP demo) use an A100 on a "power-on-then-off" basis. Everything else stays single-card.

!!! note "Scaffolding status"
    This part's lessons land in later tickets. See the **[Glossary](../glossary.md)**.
