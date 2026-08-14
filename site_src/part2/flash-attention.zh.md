# FlashAttention：IO-aware 的注意力 kernel

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    vLLM 注意力后端 flag（`--attention-backend`）与 PyTorch SDPA API 经 Context7 对照 vLLM 0.26.0 / PyTorch 核实（ADR-0004）。内存/延迟数字是**示例 / 量级参考**。§4 的 online-softmax 等价性是*精确*算术（它把完整 softmax 复现到机器精度）。

---

## 1 · 直觉 & 为什么重要

这是 FlashAttention 解决的谜题。在 [算子 Roofline](roofline-analysis.md) 那节课我们发现 prefill attention *本应* compute-bound——它的强度像 $7S$ 一样增长。可教科书式的 attention 实现常常**memory-bound、且在长序列上爆显存**。差在哪？因为教科书版在 HBM 里建出完整的 $S\times S$ score 矩阵，再读回来做 softmax，再读一次乘 $V$——对一个本不必存在的 $O(S^2)$ 中间量做了三趟往返。

FlashAttention 是解药，而且不是近似：它用**完全相同的 FLOPs** 算出**完全相同的输出**，但它是 *IO-aware* 的。它把 Q、K、V **分块（tiling）**，用 **online softmax** 在单趟流式过程里产出结果，把工作集留在 [SRAM](../part0/gpu-hardware.md)，**从不在 HBM 里物化那个 $S\times S$ 矩阵**。回报：内存从 $O(S^2)$ 降到 $O(S)$（于是 32k 上下文的 prefill 才塞得下）、HBM 流量骤减、attention 终于够到 roofline 承诺的 compute-bound 潜力。这就是为什么每个正经引擎——包括 vLLM——默认都用 FlashAttention 家族的 kernel。→ 见[术语表](../glossary.md)的 *FlashAttention*、*HBM / SRAM*、*Kernel fusion*。

## 2 · 心智模型

算同一个 attention 的两种方式，用「什么越过 HBM 线」来对比：

```text
NAIVE attention — three HBM round-trips over an S×S matrix
  Q,K ─► [ S = QKᵀ ] ──write──► HBM   (S×S, e.g. 4096² = 16.8M floats PER HEAD)
                                 │
         [ P = softmax(S) ] ◄─read──┘ ──write──► HBM   (another S×S)
                                                  │
         [ O = P·V ] ◄────────────────────read───┘     -> O(S²) memory, 3× the traffic

FLASH attention — one streaming pass, S×S is born and dies in SRAM
  for each Q tile (rows):
      init running (m=-inf, l=0, O=0)         # in SRAM/registers
      for each K,V tile (cols):               # stream over the sequence
          S_ij = Q_i · K_jᵀ                    # small tile, stays on-chip
          update m,l,O with ONLINE SOFTMAX     # rescale, accumulate
      write O_i once                          # -> O(S) memory, read Q,K,V once
```

要握住的两个形状：

- **$S\times S$ 矩阵是实现产物，不是需求。** *输出*只有 $S\times d$。attention 要求每个查询看到每个 key，但并不要求所有 $S^2$ 个 score 同时驻留——你可以在算出每个 score 的瞬间就消费它。FlashAttention 恰恰利用了这点。
- **同样的数学，不同的内存调度。** FlashAttention 改的是字节*何时、何地*存在，而非*算什么*。FLOPs 完全一样；HBM 字节从 $O(S^2)$ 降到 $O(S\cdot d)$。这是 [roofline](roofline-analysis.md) 上的一步纯粹操作：靠砍分母（字节）抬强度，不碰分子（FLOPs）。

## 3 · 原理与数学

### 3.1 $S\times S$ 中间量的问题

标准 attention 对一个查询块算 $O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V$。物化出来就是三个张量：scores $S=QK^\top\in\mathbb{R}^{S\times S}$、概率 $P=\operatorname{softmax}(S)$、输出 $O=PV$。$S\times S$ 张量随序列长度二次增长——$S=4096$ 时一个头的 score 矩阵是 $4096^2 = 16.8$M 个元素（BF16 下 ~34 MB），*每头、每层*。写它、再读它主宰了内存占用与 HBM 流量。这就是 [算子 Roofline](roofline-analysis.md) 课里点出的、把 prefill attention 拽下算力屋顶的那个 $\sim S^2 b$ 字节项。

### 3.2 Online softmax —— 关键技巧

softmax 需要沿行的两个归约：**最大值**（数值稳定）与**指数和**（归一化因子）。朴素地看两者都要整行同时在手。但两者都是*运行中*的归约，可在流式过程里增量维护，只要在运行最大值移动时重标定部分结果。把 score 按块处理，维护运行最大值 $m$、运行归一化因子 $\ell$、运行输出累加器 $O$：

$$
m^{\text{new}} = \max(m,\ \tilde m), \qquad
\ell^{\text{new}} = e^{\,m - m^{\text{new}}}\,\ell + \sum_{j\in\text{块}} e^{\,s_j - m^{\text{new}}}
$$

$$
O^{\text{new}} = e^{\,m - m^{\text{new}}}\,O + \sum_{j\in\text{块}} e^{\,s_j - m^{\text{new}}}\,v_j
$$

其中 $\tilde m$ 是该块的局部最大值。因子 $e^{\,m-m^{\text{new}}}$ **重标定**此前累加的一切到新的参考最大值——正是这个修正让流式结果*精确*等于一次性 softmax，而非近似。最后一块之后，$O \mathbin{/}\ell$ 就是最终 attention 输出。

### 3.3 分块 → $O(S)$ 内存与更高强度

FlashAttention 把那个递推包在两层循环里：外层遍历 **Q 分块**（输出的行），内层遍历 **K,V 分块**（被流式的序列）。每个分块小到能住进 SRAM，于是一个 score 分块被算出、被 online-softmax 更新消费、再丢弃——它从不碰 HBM。HBM 流量变成：$Q,K,V$ 各读一次、$O$ 写一次，即 $O(S\cdot d)$ 而非 $O(S^2)$。

回到 roofline：FLOPs 不变（prefill 约 $4n_qS^2d$），但字节从 $\sim S^2b$ 降到 $\sim S\,d\,b$，故算术强度抬高约 $\sim S/d$ 倍——把 attention 推回它本该在的算力屋顶上。（FlashAttention-2 进一步改进 GPU 工作划分；FlashAttention-3 加入 FP8 与 Hopper 专用调度。IO-aware 内核一致。）

!!! note "哪里帮得上——哪里帮不上"
    $O(S^2)\to O(S)$ 的收益是 **prefill / 长上下文**的故事：prefill 有 $S$ 个查询行，故 score 矩阵确实是 $S\times S$。在 **decode**，一步只有一个查询（$S_q=1$），故 scores 只是 $1\times S$ 向量——本就 $O(S)$。decode 的 memory-bound 墙是重读 **KV cache**（强度 $\approx 7$，见 [算子 Roofline](roofline-analysis.md)），FlashAttention 不改变它；decode 专用变体（FlashDecoding）转而沿 KV 长度*并行*以抬占用率。

## 4 · 完整可跑代码 + 逐行讲解

这段证明 online softmax（flash 的内层循环）等于一次性 softmax——**纯 CPU、可离线运行**，无 GPU、无张量。如果流式结果与完整结果匹配到机器精度，那整个 tiling 方案就是精确的。

```python title="online_softmax.py"
"""Online（分块）softmax attention == 完整 softmax attention（纯 CPU，离线）。"""
import math


def full_attention(q, K, V):
    """教科书版：物化全部 scores，softmax，再加权 V。"""
    d = len(q)
    scores = [sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d) for k in K]
    m = max(scores)                                   # 全 S 宽的最大值（要整行）
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps)
    p = [e / Z for e in exps]                         # 完整的 S×1 概率行
    return [sum(p[i] * V[i][j] for i in range(len(V))) for j in range(d)]


def online_attention(q, K, V, block=2):
    """FlashAttention 的内层循环：分块流式 K,V，随最大值移动而重标定。"""
    d = len(q)
    m, l, acc = -math.inf, 0.0, [0.0] * d             # 运行最大值、归一化因子、输出
    for start in range(0, len(K), block):             # <- 沿序列分块
        for k, v in zip(K[start:start + block], V[start:start + block]):
            s = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)
            m_new = max(m, s)
            corr = math.exp(m - m_new) if m != -math.inf else 0.0   # 重标定因子
            p = math.exp(s - m_new)
            l = l * corr + p                          # 重标定旧归一化因子，加新的
            acc = [acc[j] * corr + p * v[j] for j in range(d)]      # 输出同理
            m = m_new
    return [a / l for a in acc]                       # 末尾归一化一次


if __name__ == "__main__":
    q = [0.5, -0.3, 0.8, 0.1]
    K = [[0.2, 0.1, -0.4, 0.6], [0.9, -0.2, 0.3, 0.0], [-0.5, 0.4, 0.7, -0.1],
         [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, -0.8, 0.2], [-0.3, 0.5, 0.2, 0.9]]
    V = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]

    full = full_attention(q, K, V)
    online = online_attention(q, K, V, block=2)       # 分块大小不改变结果
    diff = max(abs(a - b) for a, b in zip(full, online))
    print("完整   :", [round(x, 6) for x in full])
    print("在线   :", [round(x, 6) for x in online])
    print(f"max abs diff = {diff:.2e}   （分块即等于完整 softmax）")
```

**逐行讲解：**

- `full_attention` —— 朴素路径：它需要先对**整**行求 `m = max(scores)` 才能安全取指数，这正是它想让所有 $S$ 个 score 驻留的原因。
- `online_attention` —— flash 递推。`corr = exp(m - m_new)` 是 §3.2 的重标定因子；当后来的块揭示更大的最大值时，它回溯修正归一化因子 `l` 与输出累加器 `acc`。任何时刻都不存在超过一个 `block` 宽的 score。
- `block` 参数是分块大小——改它（1、2、6）答案不变；tiling 是内存调度，不是近似。
- `__main__` —— 在同一组固定输入上跑两者，报告逐元素最大差。

预期输出（精确算术，非跑分）：

```text
完整   : [0.363083, 0.449897, 0.392487, 0.386499]
在线   : [0.363083, 0.449897, 0.392487, 0.386499]
max abs diff = 1.11e-16   （分块即等于完整 softmax）
```

差异是机器 epsilon——浮点噪声，而非算法误差。流式、分块的计算与一次性 softmax 是*同一个函数*。这个等价性就是 FlashAttention 从不建 $S\times S$ 矩阵的全部许可证。

### 在 vLLM 源码里读它（v0.26.0）

你刚推理过的 flash kernel，被 vLLM 的 V1 attention backend 包裹起来。两个文件锚定这段 read-along：

- 注册表 [`vllm/v1/attention/backends/registry.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/backends/registry.py) 把 `AttentionBackendEnum.FLASH_ATTN → "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"`。这正是 `--attention-backend FLASH_ATTN`（或 `AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN)`）所选中的。
- 实现在 [`vllm/v1/attention/backends/flash_attn.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/backends/flash_attn.py)：`FlashAttentionBackend` 声明元数据/形状契约，而 `FlashAttentionImpl.forward(...)` 就是那次把 Q、K、V——外加分页 KV cache 的 **block tables**——交给融合 flash kernel 的调用。那个 kernel *就是* §3 的 online-softmax tiling，用 CUDA 写成。

你不会去重写这个 kernel（ADR-0002——读，不手写），但你现在能打开 `FlashAttentionImpl.forward` 并认出每一个参数：query、分页 K/V、softmax scale、causal 标志。顺着 block-tables 参数再走一跳，你就进入了 [Part 5](../part5/paged-attention.md) 要打开的 PagedAttention 机制。

## 5 · Lab —— 看着 $S^2$ 内存消失

!!! gpu "GPU Lab"
    - **最低显存：** 8 GB（只分配 attention 张量；不加载模型）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）——对齐基线
    - **预估耗时 / 花费：** ~5 分钟 · ~¥0.5 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** `scaled_dot_product_attention` 在 ROCm/CPU 也能跑，但它派发到哪个融合后端（以及是否有 flash）因平台而异——查你平台上的 `torch.backends.cuda`。

PyTorch 的 `scaled_dot_product_attention`（SDPA）在合适 GPU 上自动派发到 FlashAttention kernel。拿它的峰值内存对比朴素物化 attention，看 $O(S^2)$ 项只在朴素路径出现：

```python title="flash_vs_naive_memory.py"
import torch
import torch.nn.functional as F

assert torch.cuda.is_available()
dev, dt = "cuda", torch.bfloat16
B, H, D = 1, 28, 128                                   # Qwen2.5-7B attention 形状

def peak_mb(fn, S):
    q = torch.randn(B, H, S, D, device=dev, dtype=dt)
    k, v = torch.randn_like(q), torch.randn_like(q)
    torch.cuda.reset_peak_memory_stats()
    fn(q, k, v)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def naive(q, k, v):                                    # 在 HBM 里物化 S×S scores
    scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)    # [B,H,S,S]  <- O(S²) 张量
    return torch.softmax(scores, dim=-1) @ v

def flash(q, k, v):                                    # 从不物化 S×S
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

for S in (1024, 2048, 4096):
    print(f"S={S:>5}: naive {peak_mb(naive, S):8.1f} MB   flash {peak_mb(flash, S):7.1f} MB")
```

**观察什么：** 朴素峰值内存**二次**增长——$S$ 每翻倍它约翻四倍（$B\times H\times S\times S$ 的 scores）——远在 32k 上下文之前就 OOM。SDPA/flash **线性**增长且很小，因为 scores 从不离开 SRAM。那条平坦曲线正是长上下文推理得以可能的原因。vLLM 里这个 kernel 是 CUDA 上默认的 attention 后端（`FLASH_ATTN`）；你可以用 `vllm serve … --attention-backend FLASH_ATTN` 显式指定（0.26.0 已核实）。

## 6 · 常见坑 / 反直觉点

- **「FlashAttention 近似了 attention。」** 并没有——它逐比特是同一个函数（至多有浮点重排序），FLOPs 相同。§4 展示了等价性。它是 *IO* 优化，不是*数学*优化。
- **指望 FLOP 提速。** FLOPs 不变；收益是更少 HBM 字节（与 $O(S)$ 内存）。在 roofline 上它靠缩分母*抬强度*——一个带宽/内存收益，所以它在 attention 曾 memory-bound 或爆显存处帮得最多。
- **以为它拯救 decode。** 它头条的 $O(S^2)\to O(S)$ 收益是 **prefill / 长上下文**效应。decode 的单查询步没有 $S\times S$ 矩阵；decode 因 KV-cache 读而 memory-bound（见 [算子 Roofline](roofline-analysis.md)）。不同问题、不同解（FlashDecoding）。
- **忘了运行最大值。** 流式累加指数和却*不*追踪运行最大值，会在真实 score 上让 `exp()` 溢出。`exp(m - m_new)` 重标定不是可选的记账——它正是让流式 softmax 既稳定又精确的东西。
- **head-dim / 布局约束。** flash kernel 支持特定 `head_dim`、dtype 与连续布局；不支持的形状会静默回退到更慢的（物化）路径。如果你的「flash」attention 没更快，检查它是否真派发到了 flash 后端。
- **只针对 attention 算子。** FlashAttention 融合的是 attention 计算；projection 与 FFN GEMM 是分开的 kernel（那是 [kernel fusion / CUDA graphs](kernel-fusion-cuda-graphs.md) 课的地盘）。

## 7 · 面试连线

- [FlashAttention 与 IO-aware attention](../interview/flash-attention.md) —— 本课为你准备的高频题：*FlashAttention 若 FLOPs 相同为何更快；online softmax 算什么、为何要运行最大值；以及它把 attention 在 roofline 上挪到哪。*

## 8 · 小结 & 延伸阅读

**一句话：** FlashAttention 用相同 FLOPs 算精确 attention，但分块 Q/K/V、用 online softmax 把 $S\times S$ scores 留在 SRAM——把 $O(S^2)$ 的 HBM 流量与内存变成 $O(S)$，抬高算术强度，让长上下文 prefill 既快又可行。

延伸阅读：

- Dao 等 —— *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* —— 源头；其中 §3 就是 online-softmax 推导。
- Dao —— *FlashAttention-2*（更好的工作划分）与 *FlashAttention-3*（FP8、Hopper）—— 同一 IO-aware 内核，调优版。
- *FlashDecoding* —— decode 期变体，沿 KV 长度切分以提占用率。
- [算子 Roofline](roofline-analysis.md) 那节课 —— $S\times S$ 字节项与 prefill-attention 强度从哪来。

## 9 · 自测小问

??? question "FlashAttention 与朴素 attention 做同样的 FLOPs。为什么它更快？"
    因为它**过 HBM 的字节少得多**。朴素 attention 把 $S\times S$ score 矩阵写到 HBM、读回做 softmax、再读一次做 $\cdot V$——对一个 $O(S^2)$ 张量三趟往返。FlashAttention 分块 Q/K/V、用 online softmax，让每个 score 分块在 SRAM 里算出并消费、从不写 HBM：流量降到 $O(S\cdot d)$、内存降到 $O(S)$。在 roofline 上它靠缩字节分母抬强度，而非砍 FLOPs。

??? question "「online softmax」算什么，为什么运行最大值必不可少？"
    它把精确 softmax 算成一个**流式归约**：维护运行最大值 $m$、归一化因子 $\ell$、输出累加器 $O$，当新块揭示更大最大值时，在加入该块贡献前把已累加的 $\ell$ 与 $O$ 乘以 $e^{\,m-m^{\text{new}}}$ 重标定。运行最大值必不可少有两个原因：它让 $e^{(\cdot)}$ 不溢出（数值稳定），且重标定让流式结果*精确*等于一次性 softmax 而非近似。

??? question "FlashAttention 让单条流 *decode* 更快吗？为什么？"
    根本上不会。decode 每步处理一个查询 token，所以 attention scores 是 $1\times S$ 向量——没有 $O(S^2)$ 矩阵要避开。decode 之所以 memory-bound，是每步重读整个 **KV cache**（强度 ≈ 7），这点 FlashAttention 不改变。它的大收益在 prefill / 长上下文——那里 score 矩阵确实是 $S\times S$。decode 专用 kernel（FlashDecoding）转而把 KV 长度切给多个 SM 以抬占用率。
