# Part 2 · 单卡推理性能

> 你已从 [Part 0](../part0/index.md) 拿到硬件心智模型；这个 Part 让它*可量化*——仅凭形状推导任何算子的状态、算好部署的显存预算，并用两个 kernel 级收益（FlashAttention、CUDA graphs）把单张 GPU 榨到极致。

## 本 Part 覆盖

- **[算子 roofline](roofline-analysis.md)**：为 GEMM 与 attention 推导算术强度——为什么 decode 在 $I\approx1$ memory-bound、prefill compute-bound，以及越过拐点的 batch 大小
- **[KV 缓存显存数学](kv-cache-math.md)**：完整显存预算（权重 + KV + 激活 + 开销）与如何解出最大并发
- **[FlashAttention](flash-attention.md)**：IO-aware 思想（tiling + online softmax）——同 FLOPs、$O(S^2)\to O(S)$ 内存，以及长上下文 prefill 为何可行
- **[Kernel fusion 与 CUDA graphs](kernel-fusion-cuda-graphs.md)**：为什么 decode 阶段的 launch overhead 致命，以及 fusion 与图重放如何收回它

本 Part 建立于的硬件入门——[内存层级与 roofline](../part0/gpu-hardware.md) 与 [延迟/吞吐度量](../part0/metrics.md)——在 **[Part 0](../part0/index.md)**。性能术语见 **[术语表](../glossary.md)**。

## 课程

- **[算子 Roofline：GEMM 与 Attention 的算术强度](roofline-analysis.md)** —— 把一个 decoder 层拆成它的 matmul 与 attention 算子，仅凭形状算出每个算子的强度与 roofline 状态。
- **[KV 缓存显存数学：为部署做容量规划](kv-cache-math.md)** —— 拼出完整显存预算，解出最大并发序列（及逆：目标并发下的最大上下文）。
- **[FlashAttention：IO-aware 的注意力 kernel](flash-attention.md)** —— 分块 Q/K/V、用 online softmax 把 $S\times S$ scores 留在 SRAM，把 $O(S^2)$ 的 HBM 流量变成 $O(S)$，同时算出完全相同的输出。
- **[Kernel Fusion 与 CUDA Graphs：干掉 decode 的 launch overhead](kernel-fusion-cuda-graphs.md)** —— 为什么几百个小 decode kernel 让一步变 launch-bound，以及 fusion 与 CUDA-graph 重放如何塌缩这份开销。

!!! note "Part 2 完成"
    四节课全部写就，各带一道双向链接的面试题。所有 vLLM flag/API 经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**。链接题集见 [面试题库](../interview/index.md)。
