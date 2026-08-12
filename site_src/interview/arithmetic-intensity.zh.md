# GEMM 与 attention 的算术强度

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 2 · 单卡推理性能   ·   **考察课程：** [算子 Roofline：GEMM 与 Attention 的算术强度](../part2/roofline-analysis.md)

---

## Q：推导（a）单 token decode 里一个权重 matmul 与（b）decode 里 attention 算子的算术强度。解释为什么 decode attention 的强度与上下文长度无关、GQA 对它做了什么。然后：一个 projection GEMM 在 4090 上在多大 batch 变 compute-bound？

### 直接答案

**GEMM。** 对 $Y=XW$，$X\in\mathbb{R}^{M\times K}$、$W\in\mathbb{R}^{K\times N}$、每元素 $b$ 字节：FLOPs $=2MKN$，字节 $=(MK+KN+MN)b$，故 $I=\frac{2MKN}{(MK+KN+MN)b}$。**$M=1$** 时权重读 $KN$ 主导分母，得 $I\approx\frac{2}{b}=1$ FLOP/字节（BF16）——**memory-bound**，与矩阵大小无关。这就是单条流 decode memory-bound 的原因：一个 token 每个权重字节的活儿不够。

**Attention（decode）。** 一个查询 token 关注 $S$ 个缓存 token：FLOPs $\approx 4n_qSd$（$QK^\top$ + scores·$V$），字节 $\approx 2n_{\text{kv}}Sd\,b$（读 K,V）。故 $I=\frac{4n_qSd}{2n_{\text{kv}}Sd\,b}=\frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}$。**$S$ 抵消**——FLOPs 与字节都随上下文缩放——所以强度是个由 GQA 比例决定的常数。对 Qwen2.5-7B（$n_q/n_{\text{kv}}=28/4=7$，BF16）：$I=7$。GQA 把 attention 强度*抬* $7\times$（每个 K/V 被 7 个查询头复用）但 $7\ll165$，所以 decode attention 仍 memory-bound；GQA 更大的战绩是 KV 字节小 $7\times$。

**越过拐点的 batch。** 大 $M$ 极限（$M\ll N$）下 $I\approx\frac{2M}{b}$，故在 $M^{*}\approx\frac{I^{*}b}{2}\approx165$ 个 token（4090，BF16）命中拐点 $I^{*}\approx165$——计入激活流量后略高。

### 深入原理

- **为什么整模型平均仍 ≈1。** 按字节加权的均值被每步读的 ~14 GiB 权重主导；不到 1 GiB 的 KV 读（attention，$I\approx7$）几乎不动它。Part 0 的「decode $I\approx1$」就是这个平均——算子视角细化它。
- **权重每步读一次，不是每 token 读一次。** GEMM 分母是 $KN$、不是 $MKN$；把 $M$ 个 token 塞进一次权重读，是 continuous batching 的全部机械基础。数错这个，batching 收益就没了。
- **瘦矩阵越线晚。** GQA `k_proj`/`v_proj`（$N=512$）在 $M=256$ 只到 $I\approx163$——仍在拐点下——而肥 FFN GEMM（$N=18944$）早已越过。$M^{*}$ 是按算子的。
- **prefill attention 翻转。** $S$ 个查询复用 KV 时，$I\approx\frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}\cdot S=7S$，$S\approx24$ 后 compute-bound——*除非*朴素 attention 把 $S\times S$ scores 物化到 HBM，加 $\sim S^2b$ 字节把它拽回（FlashAttention 的动机）。

### 代码

形状 → 强度，无 GPU：

```python
def gemm_I(M, K, N, b=2):
    return 2*M*K*N / ((M*K + K*N + M*N) * b)

def attn_decode_I(n_q, n_kv, b=2):
    return 2 * n_q / (n_kv * b)          # S 抵消

print(round(gemm_I(1,    3584, 3584), 2))   # 1.0    -> memory-bound
print(round(gemm_I(256,  3584, 3584), 1))   # 224.0  -> compute-bound（> 165 拐点）
print(round(attn_decode_I(28, 4), 1))       # 7.0    -> memory-bound（GQA 28/4）
print(round(attn_decode_I(28, 28), 1))      # 1.0    -> 假想 MHA
```

### 面试官追问

- *"为什么更大的 GPU（更多 TFLOPS）不加速单条流 decode？"* → decode 在 $I\approx1\ll I^{*}$，被钉在带宽屋顶 $I\cdot B$；更多 FLOPs 抬的是 $P$，不成约束。你要更多带宽、更少字节（量化）、或更高 $I$（批处理）。
- *"FlashAttention 在这张 roofline 的哪？"* → prefill attention *仅当* $S\times S$ scores 留在片上才 compute-bound；FlashAttention 用 tiling + online softmax 把它们留在 SRAM，避开否则会拉低强度的 HBM 往返。
- *"GQA 改变 decode 状态吗？"* → 不——它把 attention $I$ 抬 $n_q/n_{\text{kv}}$（Qwen 到 7）并同倍缩小 KV 字节，但 7 仍远低于拐点。状态由 $I$ vs $I^{*}$ 定，不由 GQA 定。
- *"你 batch 256 了但 KV projection 仍 memory-bound——为什么？"* → 它们瘦（$N=512$），故 $I(256)\approx163<165$。窄 GEMM 要更大 $M$（或与邻居融合）才越线。

### 关联知识点

- 课程：[算子 Roofline：GEMM 与 Attention 的算术强度](../part2/roofline-analysis.md)
- 相关：[GPU 内存层级与 roofline](gpu-memory-hierarchy.md)（本题所建立于的 roofline 与拐点）、[Prefill vs decode](prefill-vs-decode.md)
- 术语表：[Roofline / 算术强度、Memory-bound / Compute-bound、GQA、FlashAttention](../glossary.md)
