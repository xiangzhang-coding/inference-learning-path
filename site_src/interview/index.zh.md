# 面试题库

> 一个持续增长的**高频**面试题库，按模块（Part 0–8）归类。每题遵循同一 schema：**直接答案 → 深入原理 → 代码（若适用）→ 面试官追问 → 关联知识点**。

每题都反链回它考察的课程，每节课的「面试连线」也链到这里——学练闭环。

## 按模块

- **Part 0 · 基础**
    - [Prefill vs decode](prefill-vs-decode.md) — 哪个阶段 compute- vs memory-bound，为什么。
    - [注意力变体：MHA/MQA/GQA](attention-variants.md) — KV 头如何决定 KV 缓存与吞吐上限。
    - [KV 缓存与吞吐上限](kv-cache.md) — 为什么瓶颈通常是 KV cache 而非算力。
    - [GPU 内存层级与 roofline](gpu-memory-hierarchy.md) — 走一遍内存层级，用 roofline 解释为何 decode 是 memory-bound。
    - [延迟与吞吐度量](latency-throughput-metrics.md) — TTFT/TPOT/ITL/throughput/goodput，如何测，以及 batch size 的权衡。
    - [数值格式与精度](number-formats.md) — FP16/BF16/FP8/INT8/INT4，范围 vs 精度，以及低比特为何加速 decode。
- **Part 2 · 单卡推理性能**
    - [GEMM 与 attention 的算术强度](arithmetic-intensity.md) — 仅凭形状推导一个算子的强度、decode attention 为何与上下文无关、以及越过拐点的 batch。
    - [显存预算与最大并发](vram-capacity-planning.md) — 走一遍完整显存预算、估最大并发；达到并发目标的旋钮。
    - [FlashAttention 与 IO-aware attention](flash-attention.md) — 同 FLOPs 为何更快、online softmax、以及它在哪帮得上/帮不上。
    - [CUDA graphs 与 kernel fusion](cuda-graphs-fusion.md) — decode 启动开销、为何伤 decode 而非 prefill、`enforce_eager` 权衡什么。
- **Part 3 · GPU 编程（Triton）**
    - [CUDA 执行模型：warp、SIMT 与 occupancy](cuda-execution-model.md) — 什么是 warp、SIMT divergence 的代价，以及为何拉满 occupancy 未必更快。
    - [Memory coalescing、shared memory 与 bank conflict](memory-coalescing.md) — 什么让一次访问 coalesced、uncoalesced 的代价，以及 shared memory 与 bank conflict 是什么。
    - [Triton 编程模型](triton-programming.md) — 一个 Triton program 映射到什么、`program_id`/offset/mask、FP32 累加，以及何时选 Triton。
    - [PagedAttention kernel 与 block table](paged-attention-kernel.md) — 为何 KV 存成块、block table 做什么、kernel 如何 gather KV，以及它为何等于稠密 attention。
- **Part 1、4–8** — 各题随对应课程在后续票落地。

!!! note "脚手架状态"
    Part 0（票 #2、#4、#5）、Part 2（票 #6、#7）与 Part 3（票 #8、#9）题目已入库，每题与它考察的课程双向链接。完整 ~100 道题库随各 Part 落地增长。难度档 / 频率标签 / 权重暂不在范围。
