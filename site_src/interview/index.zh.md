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
- **Part 1–8** — 各题随对应课程在后续票落地。

!!! note "脚手架状态"
    Part 0 题目（票 #2、#4、#5）已入库，每题与它考察的课程双向链接。完整 ~100 道题库随各 Part 落地增长。难度档 / 频率标签 / 权重暂不在范围。
