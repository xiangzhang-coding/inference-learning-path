# FlashAttention 与 IO-aware attention

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 2 · 单卡推理性能   ·   **考察课程：** [FlashAttention：IO-aware 的注意力 kernel](../part2/flash-attention.md)

---

## Q：FlashAttention 与朴素 attention 做同样的 FLOPs，却更快、用内存更少。怎么做到的？「online softmax」算什么、为何要运行最大值，以及 FlashAttention 在哪里*帮不上*？

### 直接答案

FlashAttention 是 **IO-aware** 的：它用相同 FLOPs 算出完全相同的输出，但过 HBM 的字节少得多。朴素 attention 在 HBM 里物化 $S\times S$ score 矩阵、读回做 softmax、再读一次乘 $V$——对一个 $O(S^2)$ 中间量三趟往返。FlashAttention 把 Q、K、V **分块（tiling）**，用 **online softmax** 让每个 score 分块在 SRAM 里算出并消费、从不写 HBM。结果：HBM 流量从 $O(S^2)$ 降到 $O(S\cdot d)$、内存从 $O(S^2)$ 降到 $O(S)$。在 roofline 上它靠缩字节*分母*（而非 FLOP 分子）抬高算术强度，把 attention 推回算力屋顶。

**Online softmax** 把精确 softmax 算成流式归约：维护运行最大值 $m$、归一化因子 $\ell$、输出累加器 $O$；当新块揭示更大最大值时，在加入该块前把已累加的 $\ell$ 与 $O$ 乘 $e^{\,m-m^{\text{new}}}$ 重标定。**运行最大值必不可少**有两个原因——它让 $e^{(\cdot)}$ 不溢出（稳定），且重标定让流式结果*精确*等于一次性 softmax，而非近似。

**哪里帮不上：** 单条流 **decode**。一个 decode 步只有一个查询，故 scores 是 $1\times S$ 向量——没有 $O(S^2)$ 矩阵要避开。decode 因 **KV-cache 读**而 memory-bound，FlashAttention 不改变它。

### 深入原理

- **它精确、非近似。** 分块/流式计算与一次性 softmax 是同一个函数（至多浮点重排序）。这是跳过建 $S\times S$ 矩阵的许可证。
- **收益随 $S$ 放大。** 避开的项是 $S^2$，故上下文越长，内存与流量节省越大——这正是 32k 上下文 prefill 得以可行的原因。
- **fusion 是机制。** FlashAttention 是融合的 attention kernel：$QK^\top$、softmax、$\cdot\,V$ 三步在一个 kernel 里完成，中间量从不离开 SRAM。这也意味着更少的 kernel 启动（接 [CUDA graphs / fusion](cuda-graphs-fusion.md)）。
- **变体。** FlashAttention-2 改进 GPU 工作划分；FlashAttention-3 加 FP8/Hopper 调度；FlashDecoding 针对 decode，沿 KV 长度切分给多 SM 提占用率。

### 代码

online softmax 等于完整 softmax——tiling 精确的证明（纯 CPU）：

```python
import math
def online(q, K, V, block=2):
    d=len(q); m,l,acc=-math.inf,0.0,[0.0]*d
    for i in range(0,len(K),block):
        for k,v in zip(K[i:i+block],V[i:i+block]):
            s=sum(a*b for a,b in zip(q,k))/math.sqrt(d)
            mn=max(m,s); c=math.exp(m-mn) if m!=-math.inf else 0.0; p=math.exp(s-mn)
            l=l*c+p; acc=[acc[j]*c+p*v[j] for j in range(d)]; m=mn
    return [a/l for a in acc]     # == softmax(QKᵀ/√d)·V，到机器精度
```

### 面试官追问

- *"FLOPs 相同——那提速究竟从哪来？"* → 更少 HBM 字节。attention 曾在 $S\times S$ 流量上 memory/IO-bound；砍字节把强度抬向算力屋顶。是带宽收益，不是 FLOP 收益。
- *"为什么不能不要最大值、直接流式累加 exp 和？"* → `exp(大 score)` 会溢出成 inf。运行最大值把每个指数归一，$e^{m-m^{\text{new}}}$ 重标定在最大值增长时保持部分和一致——这正是它既稳定又精确的原因。
- *"它加速 decode 吗？"* → 根本上不：decode 的 scores 是 $1\times S$、本就 $O(S)$。decode 因 KV 读而 bandwidth-bound（强度 ≈ 7）。FlashDecoding 用另一机制（KV 长度并行提占用率）帮 decode。
- *"为什么你的『flash』attention 可能没更快？"* → 它可能因不支持的 head_dim/dtype/布局回退到物化路径。确认 flash 后端真的派发了。

### 关联知识点

- 课程：[FlashAttention：IO-aware 的注意力 kernel](../part2/flash-attention.md)
- 相关：[GEMM 与 attention 的算术强度](arithmetic-intensity.md)（$S\times S$ 字节项与 prefill-attention 强度）、[CUDA graphs 与 kernel fusion](cuda-graphs-fusion.md)
- 术语表：[FlashAttention、HBM / SRAM、Kernel fusion、Roofline](../glossary.md)
