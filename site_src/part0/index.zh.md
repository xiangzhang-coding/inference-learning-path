# Part 0 · 基础与动机

> **动机先行。** 在任何优化之前，先建立支撑后续一切的那个心智模型：**为什么 LLM 推理是 memory-bound（带宽受限）**。一旦想通这一点，后面所有招式——KV cache 调优、量化、PagedAttention、continuous batching——都会变成你能**推导**出的结论，而非要死记的事实。

## 本 Part 覆盖

- 为什么 LLM 推理是 **memory-bound**（吞吐的故事从这里开始）
- 两个阶段：**prefill** vs **decode**，以及每种优化作用在哪个阶段
- **[KV 缓存](kv-cache.md)** — 是什么、为何存在、如何增长、为何是吞吐上限的核心矛盾
- 推理**度量**：TTFT、TPOT/ITL、throughput、goodput——以及如何测量
- **数值格式**：FP16 / BF16 / FP8 / INT8 / INT4——够顺畅进入量化 Part

## 课程

- **[推理流程：Prefill 与 Decode](inference-flow.md)** — 自回归生成的两个阶段，以及为何 prefill 是 compute-bound 而 decode 是 memory-bound。
- **[Transformer 的 Infra 视角](transformer-infra.md)** — 把 decoder 层读成成本模型：哪些部件花权重、prefill FLOPs、KV 缓存。
- **[KV 缓存](kv-cache.md)** — 是什么、为何增长、为何是吞吐上限的核心矛盾。
- **[GPU 硬件心智模型](gpu-hardware.md)** — 内存金字塔（HBM vs SRAM）、SM/warp 执行模型，以及证明 decode 带宽受限的 roofline。
- **[推理性能度量](metrics.md)** — TTFT、TPOT/ITL、throughput、goodput，以及如何用 `vllm bench serve` 与 Prometheus 测每一个。
- **[数值格式：FP16 · BF16 · FP8 · INT8 · INT4](number-formats.md)** — 每种 dtype 背后的范围-vs-精度权衡，为进入量化铺路。

!!! note "Part 0 完成"
    **Part 0A**——推理流程、Transformer infra 视角、KV 缓存——与 **Part 0B**——GPU 硬件、度量、数值格式——均已成文，各附带一道双向链接的面试题。所有 vLLM flag/API 经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**。链接题集见 [面试题库](../interview/index.md)。
