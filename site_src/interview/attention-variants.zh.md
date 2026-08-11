# 注意力变体：MHA / MQA / GQA

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 0 · 基础   ·   **考察课程：** [Transformer 的 Infra 视角](../part0/transformer-infra.md)

---

## 问：MHA、MQA、GQA 有何区别？各自如何改动 KV 缓存与吞吐上限，质量上如何权衡？

### 直接答案

它们只在**KV 头数**上不同，即 query 头共享多少份不同的 Key/Value 投影：

- **MHA**（Multi-Head Attention）：每个 query 头一份 K/V——$n_{\text{kv}} = h$。KV 缓存最大。
- **MQA**（Multi-Query Attention）：所有 query 头共享一份 K/V——$n_{\text{kv}} = 1$。KV 缓存最小（缩小 $h\times$），但最激进、质量风险最高。
- **GQA**（Grouped-Query Attention）：query 头按 $g$ 组共享 K/V——$1 < n_{\text{kv}} = g < h$。实用折中；MHA 与 MQA 是它的两个极端。

KV 缓存为 $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$ 字节/token——**与 $n_{\text{kv}}$ 成线性**。缩小 KV 头就成比例缩小缓存，让更多序列塞进剩余 VRAM，从而**抬高吞吐上限**（decode 是 memory-bound，KV 容量越多 ≈ 并发越多）。关键是它几乎不动 **FLOP/参数** 预算——K/V 投影只是层内一小片，而大头 FFN 完全没变。质量：MQA 可能明显掉点；GQA 用适中的组数（如 4–8）就能恢复接近 MHA 的质量，这正是几乎所有现代模型都采用 GQA 的原因。

### 深入原理

- **Qwen2.5-7B 算例。** $h = 28$ 个 query 头、$n_{\text{kv}} = 4$ 个 KV 头 → KV 缓存比等价 MHA 小 $28/4 = 7\times$：**56 KiB/token 而非 392 KiB/token**。在 24 GB 卡上，这 7× 就是「几条」与「几十条」并发序列的差别。
- **为什么几乎不改 FLOPs。** 只有 K、V 投影矩阵缩小（从 $d\times d$ 到 $d\times n_{\text{kv}}d_h$）。Q、O 及整个 FFN 不变——而光 FFN 就占 ~75% 参数。所以 GQA 是**显存**优化，不是算力优化。→ [Transformer 的 Infra 视角](../part0/transformer-infra.md)。
- **为什么质量能保住。** query 头仍各自独立注意力；只是查阅共享的 K/V 表示。组数够多时，模型保留了 MHA 的大部分表达力。MQA 的单组最常掉质量，尤其在长上下文或检索密集任务上。
- **与下游一切的联动。** 更少 KV 头会与 `kv_cache_dtype=fp8` 和 PagedAttention 叠加——它们都在抢同一块决定并发的剩余 VRAM 预算。

### 代码

整个故事就是一个比值——KV 头 → KV 字节：

```python
L, head_dim, b = 28, 128, 2          # Qwen2.5-7B 层数、head_dim、BF16 字节

def kib_per_token(n_kv): return 2 * L * n_kv * head_dim * b / 1024

for name, n_kv in [("MHA", 28), ("GQA-4 (Qwen)", 4), ("MQA", 1)]:
    print(f"{name:<14} n_kv={n_kv:>2} -> {kib_per_token(n_kv):6.0f} KiB/token")
# MHA            n_kv=28 ->    392 KiB/token
# GQA-4 (Qwen)   n_kv= 4 ->     56 KiB/token   （小 7 倍）
# MQA            n_kv= 1 ->     14 KiB/token   （小 28 倍）
```

### 面试官追问

- *「GQA 能加速 decode 的算术吗？」* → 不明显；它缩的是 KV 缓存（搬运字节），而这*正是* decode 瓶颈，所以它经由**带宽/容量**帮 decode，而非 FLOPs。
- *「什么时候你还会选 MHA？」* → 极小模型或 KV 缓存不构成约束的研究场景，或当「每参数最高质量」比服务并发更重要时。
- *「组数怎么定？」* → 靠实测——够多以在你的 eval 上恢复 MHA 级质量，够少以最小化 KV。4–8 是常见甜点。
- *「MLA（DeepSeek）在哪一环？」* → 同一目标（缩 KV），但用*低秩潜在* KV 表示而非更少头——同一条「压缩 KV 列」轴上更进一步的点。

### 关联知识点

- 课程：[Transformer 的 Infra 视角](../part0/transformer-infra.md)
- 相关题：[KV 缓存与吞吐上限](kv-cache.md)（为什么更小 KV → 更多并发）
- 术语表：[MHA / MQA / GQA、KV cache](../glossary.md)
