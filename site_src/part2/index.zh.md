# Part 2 · 单卡推理性能

> 你已从 [Part 0](../part0/index.md) 拿到硬件心智模型；现在学会*量化*单张 GPU 的极限——仅凭形状推导任何算子的状态，并在租卡之前算好部署的显存预算。

## 本 Part 覆盖

- **[算子 roofline](roofline-analysis.md)**：为 GEMM 与 attention 推导算术强度——为什么 decode 在 $I\approx1$ memory-bound、prefill compute-bound，以及越过拐点的 batch 大小
- **[KV 缓存显存数学](kv-cache-math.md)**：完整显存预算（权重 + KV + 激活 + 开销）与如何解出最大并发
- **FlashAttention** 与 **kernel fusion / CUDA graphs**：IO-aware 与 launch-overhead 的收益（票 #7）

本 Part 建立于的硬件入门——[内存层级与 roofline](../part0/gpu-hardware.md) 与 [延迟/吞吐度量](../part0/metrics.md)——在 **[Part 0](../part0/index.md)**。性能术语见 **[术语表](../glossary.md)**。

## 课程

- **[算子 Roofline：GEMM 与 Attention 的算术强度](roofline-analysis.md)** —— 把一个 decoder 层拆成它的 matmul 与 attention 算子，仅凭形状算出每个算子的强度与 roofline 状态。
- **[KV 缓存显存数学：为部署做容量规划](kv-cache-math.md)** —— 拼出完整显存预算，解出最大并发序列（及逆：目标并发下的最大上下文）。

!!! note "脚手架状态"
    **Roofline 分析**与 **KV 缓存显存数学**（票 #6）已写，各带一道链接的面试题。**FlashAttention** 与 **kernel fusion / CUDA graphs**（票 #7）接下来落地。链接题集见 [面试题库](../interview/index.md)。
