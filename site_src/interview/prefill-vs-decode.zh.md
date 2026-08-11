# Prefill vs decode

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 0 · 基础   ·   **考察课程：** [推理流程：Prefill 与 Decode](../part0/inference-flow.md)

---

## 问：解释 LLM 推理的 prefill 与 decode 两阶段。哪个 compute-bound、哪个 memory-bound，为什么？各自对延迟（TTFT/TPOT）与批处理意味着什么？

### 直接答案

推理分两个阶段，是因为生成是**自回归**的。**Prefill** 是消化整段 prompt 的那一次前向——所有 prompt token **并行**处理，产出第一个输出 token 及 prompt 的 KV 缓存。**Decode** 是随后的循环：每个输出 token 一次前向，每次**串行**（token *t+1* 需要 token *t* 的取值），每次复用此前一切的 KV。

Prefill 是 **compute-bound**：它在很多 token 上各做 ~$2N$ FLOPs，而每个权重只从 HBM 读一次，于是算术强度高、GPU 算术单元打满。Decode 是 **memory-bound**：每步重读*全部*权重和*整个* KV 缓存，只为吐一个 token，于是算术强度 ≈ 1 FLOP/字节，GPU 卡在 HBM 带宽上。

后果：**TTFT**（首 token 延迟）由 prefill 主导；**TPOT**（每输出 token 时间）由 decode 主导。批处理对 decode 帮助很大（把很多序列的步一起跑，权重跨整批复用、抬高强度），但对单条流几乎无用。

### 深入原理

- **为什么必须分裂。** token 50 还没出现你算不了 token 51——decode 天生串行。但 prompt 事先完全已知，所以 prefill 能是一次宽的并行矩阵乘。同样的数学，相反的并行画像。
- **算术强度，精确地。** 设 $N$ 参数、$b$ 字节/权重、$\kappa$ KV 字节/token：prefill 强度 $\approx \frac{2NS}{Nb + \kappa S} \approx \frac{2S}{b}$（当 $\kappa S \ll Nb$，故在任何现实上下文内都随 prompt 长度 $S$ 近似线性增长）；decode 强度 $\approx \frac{2N}{Nb + \kappa S} \le \frac{2}{b} = 1$（BF16）。GPU 要几百 FLOP/字节 才不空转——prefill 越线，decode 永远够不着。
- **墙钟去哪了。** 请求延迟 ≈ TTFT（对 prompt 的一次 prefill）+（输出 token 数 × TPOT）。长 prompt → prefill 主导 TTFT；长答案 → decode 主导总时间，大致线性。
- **各阶段优化针对什么。** Prefill：chunked prefill、PD 分离（把 TTFT 压住）。Decode：continuous batching、PagedAttention、speculative decoding（对付内存墙与串行依赖）。

### 代码

无需 GPU，两种状态从 FLOP/字节 估算里自然浮现：

```python
N = 7_615_000_000        # ~7.6B 参数（Qwen2.5-7B）
b = 2                    # BF16 字节/权重
kappa = 57344            # KV 字节/token：2*28*4*128*2

def prefill_I(S): return (2*N*S) / (N*b + kappa*S)   # 随 prompt 长度增长
def decode_I(S):  return (2*N)   / (N*b + kappa*S)   # 钉在 1 附近

print(round(prefill_I(1024), 1), round(decode_I(1024), 2))   # ~1020.1  ~1.00
```

### 面试官追问

- *「为什么不能直接并行 decode？」* → 串行数据依赖：token *t+1* 以 token *t* 采样出的取值为条件。你只能*跨*请求并行（批处理）或*提前猜*（speculative decoding），绝不能在单条流自己的未来里并行。
- *「用户抱怨首 token 慢、但流式很快，诊断一下。」* → 长 prompt → prefill 重、TTFT 高；流式快说明 decode 没问题。修法：裁剪 prompt、prefix caching、chunked prefill。
- *「更大的 batch 会降低 TPOT 吗？」* → 单请求 TPOT 可能略升，但*聚合* token 吞吐大涨，因为 decode 的权重读被整批摊薄——这正是 continuous batching 的意义。
- *「prefill 还是 decode 更受益于更高 HBM *带宽* vs 更多 *FLOPs*？」* → decode 要带宽（memory-bound）；prefill 要 FLOPs（compute-bound）。

### 关联知识点

- 课程：[推理流程：Prefill 与 Decode](../part0/inference-flow.md)
- 相关课程：[Transformer 的 Infra 视角](../part0/transformer-infra.md)（$2N$ FLOPs 从哪来）
- 术语表：[Prefill、Decode、Memory-bound / Compute-bound、TTFT、TPOT、Roofline](../glossary.md)
