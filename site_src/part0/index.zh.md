# Part 0 · 基础与动机

> **动机先行。** 在任何优化之前，先建立支撑后续一切的那个心智模型：**为什么 LLM 推理是 memory-bound（带宽受限）**。一旦想通这一点，后面所有招式——KV cache 调优、量化、PagedAttention、continuous batching——都会变成你能**推导**出的结论，而非要死记的事实。

## 本 Part 覆盖

- 为什么 LLM 推理是 **memory-bound**（吞吐的故事从这里开始）
- 两个阶段：**prefill** vs **decode**，以及每种优化作用在哪个阶段
- **[KV 缓存](kv-cache.md)** — 是什么、为何存在、如何增长、为何是吞吐上限的核心矛盾
- 推理**度量**：TTFT、TPOT/ITL、throughput、goodput——以及如何测量
- **数值格式**：FP16 / BF16 / FP8 / INT8 / INT4——够顺畅进入量化 Part

## 课程

- **[KV 缓存](kv-cache.md)** — 完整演示 9 段骨架的样板课。

!!! note "脚手架状态"
    本学习路径按 Part 逐步搭建。**KV 缓存** 是第一节完整成文的课（票 #2）；Part 0 其余课程与 Part 1–8 会在后续票落地。题库随进度增长，见 [面试题库](../interview/index.md)。
