# Part 7 · Multi-GPU & Distributed

> When one 4090 isn't enough: how to split a model across GPUs and read/configure the result.

## What this part covers

- **Tensor / Pipeline / Data / Expert parallelism** and **NCCL** collective communication
- How to enable **TP / PP in vLLM** and how to choose the **TP degree**
- **Load testing** and finding the concurrency **knee** — the real throughput ceiling of a service

!!! gpu "Multi-GPU note"
    Per ADR-0001, the main line runs on a single RTX 4090. The 1–2 topics that truly need multiple GPUs (e.g. a TP/PP demo) use an A100 on a "power-on-then-off" basis. Everything else stays single-card.

## Lessons

- **[Why Parallelize, and How: Tensor / Pipeline / Data / Expert Parallelism](parallelism-strategies.md)** — the two reasons to leave one GPU (**it won't fit** → split the model; **it's too slow** → replicate it), and what each of the four cuts splits and costs to communicate: **TP** shards each layer's matrices with two all-reduces per layer (bandwidth-hungry → NVLink → *within a node*), **PP** splits the layers into stages with a cheap point-to-point handoff (→ *across nodes*, at the cost of the pipeline bubble), **DP** replicates the whole model for throughput (needs it to fit), and **EP** splits MoE experts with an all-to-all — plus the decision tree (`tensor_parallel_size` / `pipeline_parallel_size` / `--data-parallel-size` / `--enable-expert-parallel`, verified on vLLM 0.26.0).

!!! note "Scaffolding status"
    The **[parallelism strategies](parallelism-strategies.md)** lesson (ticket #17) is in, two-way-linked to its interview question ([Parallelism: TP/PP/DP/EP](../interview/parallelism-strategies.md)). Still landing in later tickets: **NCCL collective communication + actually launching TP/PP on multiple GPUs** (the hands-on companion to this conceptual lesson), then load testing and the concurrency **knee**. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**, and every performance number is an **illustrative / order-of-magnitude reference**. See the **[Glossary](../glossary.md)** and the [Interview Bank](../interview/index.md).
