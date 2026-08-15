# Speculative decoding：猜测-校验

!!! info "基线：**vLLM 0.26.0** · 配置经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [Speculative Decoding：猜多个，验一次](../part5/speculative-decoding.md)

---

## Q：解释 speculative decoding。为何它只因 decode memory-bound 才是「免费午餐」？什么设定加速？它改变输出吗？何时反噬？

### 直接答案

便宜的 **draft** 提议接下来 K 个 token；大 **target** 对全部 K+1 个位置跑**一次** forward 校验，接受最长正确前缀、在第一个不匹配处吐出它自己的 token。于是每次昂贵的 target 权重读取拿到多个 token——且**输出逐位相同**于 vanilla target decoding（校验使之精确）。

它近乎免费**只因 [decode 是 memory-bound](../part0/inference-flow.md)**：target 对 K+1 个 token 的 forward 只读权重**一次**（与一个 token 相同），用 GPU 本来空闲的计算处理额外位置。把空闲 FLOPs → 更少 HBM 往返。

**加速 = 期望接受串长** $\sum_{i=0}^{K}\alpha^i=(1-\alpha^{K+1})/(1-\alpha)$，$\alpha$ 是每 token 的 draft/target 一致率。$\alpha=0.7$、K=4 时 ≈ 2.77× 更少 target pass。

**它反噬**在批变 compute-bound 时：那「空闲」计算现被服务批用掉，校验额外 token 不再免费，draft 开销可能让你*更慢*。它是低批量**延迟**工具，不是吞吐工具。

### 深入原理

- **draft 来源**（`method`）：`ngram`（prompt-lookup，无 draft 模型——输出回响输入时极好：摘要/RAG/编辑）、`eagle`/`eagle3`（训练过的小头，高接受率、小 VRAM——现代默认）、`draft_model`（独立小模型，更通用但自身 forward 更贵）。`num_speculative_tokens` = K。
- **接受率就是一切。** 便宜但少一致（低 $\alpha$）的 draft 收益寥寥；设计张力是一个*既*便宜*又*常一致的 draft（EAGLE 存在的原因）。
- **K 的递减。** $\sum\alpha^i$ 饱和；过了某点，额外 draft token 很少存活却总耗 draft 计算。按 $\alpha$ 调 K。
- **VRAM。** `draft_model`/EAGLE 检查点与 target 同占 GPU 显存，缩小 [KV-cache 预算](../part5/paged-attention.md)；`ngram` 是零 VRAM 选项。

### 代码

加速作为接受率的确定性函数：

```python
def tokens_per_target_pass(alpha, k):
    return sum(alpha**i for i in range(k+1))    # 1 + a + ... + a^k == (1 - a^(k+1))/(1 - a)
for a in (0.5, 0.7, 0.9):
    print(a, round(tokens_per_target_pass(a, 4), 2))   # 0.5→1.94  0.7→2.77  0.9→4.10
# vanilla = 1.00 token/pass，所以这个数字就是 target forward 上的加速（未减 draft 代价）
```

### 面试官追问

- *「它伤质量吗？」* → 不——校验使输出与 target 单独相同。纯延迟。
- *「为何在大批时消退？」* → decode 转 compute-bound；让校验免费的空闲计算没了，draft 开销收不回。
- *「K 越大越好？」* → 不——$\sum\alpha^i$ 递减；低 $\alpha$ 上大 K 浪费 draft 计算。按 $\alpha$ 匹配 K。
- *「摘要用哪种方法？」* → `ngram`——输出回响输入，prompt-lookup 以零 draft 模型代价/无额外 VRAM 常命中。
- *「最大的单一杠杆？」* → 接受率 $\alpha$（draft/target 对齐）——它主导整个 $\sum\alpha^i$。

### 关联概念

- 课程：[Speculative Decoding](../part5/speculative-decoding.md)
- 相关：[Prefill vs decode](prefill-vs-decode.md)（为何 decode memory-bound——前提）、[算术强度](arithmetic-intensity.md)（它消退的 compute ridge）、[Static vs continuous batching](continuous-batching.md)（它*不*触及的吞吐轴）
- 术语：[Speculative decoding、Decode](../glossary.md)
