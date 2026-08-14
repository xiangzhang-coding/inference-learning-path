# 算子 Roofline：GEMM 与 Attention 的算术强度

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    硬件数字（峰值 FLOP/s、HBM 带宽）是消费级 RTX 4090 的**示例 / 量级参考**。下面的强度算术（FLOPs ÷ 字节）是*精确*的——它只取决于张量形状与 dtype，不依赖任何跑分。

---

## 1 · 直觉 & 为什么重要

[Part 0 的 roofline 课](../part0/gpu-hardware.md) 交给你模型 $\min(P, I\cdot B)$ 及其拐点 $I^{*}=P/B$，而 [推理流程](../part0/inference-flow.md) *断言*了结论：prefill $I\approx\text{几千}$、decode $I\approx1$。那些是**整模型平均**——把模型当成一个 $2N$-FLOP 的整块。

打开 profiler，那个整块就碎成一个个具名 kernel：`q_proj`、`k_proj`、attention 算子、`gate_proj`、`down_proj`。每个都有**自己的**算术强度、**自己的** roofline 位置。本课要建立的技能，是**仅凭形状**推导一个算子的强度——于是你看着一条 trace 就能说「这个 matmul 在 batch 1 时 memory-bound，但 batch 256 时 compute-bound」，或「decode attention 为一个 token 重读整个 KV cache，所以再多 FLOPs 也救不了它」。Part 2 及之后的每个优化——[FlashAttention](../part0/gpu-hardware.md)、kernel fusion、[continuous batching](../glossary.md)——都是*这张按算子铺开的 roofline 上的一步棋*，而你算不出来的一步棋，也就评估不了。→ 见[术语表](../glossary.md)的 *Roofline / 算术强度*、*Memory-bound / Compute-bound*。

## 2 · 心智模型

一个 decoder 层就是两类算子，它们因相反的理由落在 roofline 上：

```text
ONE DECODER LAYER, decomposed into operators
                                             intensity set by...
  x ──► [ q_proj ]  [ k_proj ]  [ v_proj ]   <- GEMM: weight W[K×N] reused
             │           │           │           across M tokens. I rises with M.
             └─────► [  ATTENTION  ] ◄──┘      <- NOT a weight matmul: Q·Kᵀ, ·V.
                          │                       "weights" = KV cache, which GROWS
                          ▼                       with context S. I set by GQA ratio.
                     [  o_proj  ]              <- GEMM
                          │
                     [ gate ][ up ]  ─► SwiGLU ─► [ down ]   <- GEMMs (the FLOP bulk)

  GEMM  archetype:  fixed weight bytes, FLOPs ∝ M (tokens in flight)  → batch to cross the ridge
  ATTN  archetype:  bytes ∝ KV size (∝ S), FLOPs ∝ S too             → the S cancels; regime is fixed
```

再把这些算子实际画到 roofline 上——斜的**内存屋顶**（$I\cdot B$）、平的**算力屋顶**（$P$），以及分隔二者的拐点 $I^{*}$：

<svg viewBox="0 0 760 430" role="img" xmlns="http://www.w3.org/2000/svg" aria-label="Roofline plot (log-log): a sloped memory roof I·B meets a flat compute roof P at the ridge point I*≈165 FLOP/byte. Decode GEMM sits at I≈1 and decode attention at I≈7 — both on the sloped memory-bound roof, far left of the ridge; prefill sits on the flat compute-bound roof past the ridge." style="max-width:100%;height:auto;font-family:inherit">
  <title>Per-operator roofline (RTX 4090, illustrative)</title>
  <g stroke="currentColor" stroke-opacity="0.12">
    <line x1="220" y1="45" x2="220" y2="360"/><line x1="370" y1="45" x2="370" y2="360"/>
    <line x1="520" y1="45" x2="520" y2="360"/><line x1="670" y1="45" x2="670" y2="360"/>
    <line x1="70" y1="260" x2="700" y2="260"/><line x1="70" y1="160" x2="700" y2="160"/>
    <line x1="70" y1="60" x2="700" y2="60"/>
  </g>
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="70" y1="360" x2="700" y2="360"/><line x1="70" y1="360" x2="70" y2="45"/>
  </g>
  <g stroke="currentColor" stroke-width="2.5" fill="none">
    <line x1="70" y1="360" x2="403" y2="138"/><line x1="403" y1="138" x2="700" y2="138"/>
  </g>
  <line x1="403" y1="138" x2="403" y2="360" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" stroke-opacity="0.6"/>
  <g fill="currentColor">
    <circle cx="72" cy="358" r="4"/><circle cx="197" cy="275" r="4"/>
    <circle cx="403" cy="138" r="4.5"/><circle cx="530" cy="138" r="4"/>
  </g>
  <g fill="currentColor" font-size="12.5">
    <text x="98" y="352">GEMM decode · I≈1</text>
    <text x="210" y="272">attn decode · I≈7</text>
    <text x="300" y="120" text-anchor="end">ridge  I*=P/B ≈ 165</text>
    <text x="524" y="128" text-anchor="end">prefill · compute-bound</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-style="italic" opacity="0.7">
    <text x="120" y="205">memory-bound</text>
    <text x="545" y="175">compute-bound</text>
  </g>
  <g fill="currentColor" font-size="11" opacity="0.75" text-anchor="middle">
    <text x="70" y="378">1</text><text x="220" y="378">10</text><text x="370" y="378">100</text>
    <text x="520" y="378">1k</text><text x="670" y="378">10k</text>
  </g>
  <g fill="currentColor" font-size="11" opacity="0.75" text-anchor="end">
    <text x="62" y="364">1</text><text x="62" y="264">10</text><text x="62" y="164">100</text><text x="62" y="64">1k</text>
  </g>
  <text x="385" y="404" fill="currentColor" font-size="12.5" text-anchor="middle">arithmetic intensity  I  (FLOP/byte, log)</text>
  <text x="24" y="200" fill="currentColor" font-size="12.5" text-anchor="middle" transform="rotate(-90 24 200)">attainable  (TFLOP/s, log)</text>
</svg>

要握住的两个形状：

- **GEMM 的强度是一个你用 batch/token 数 $M$ 去拧的旋钮。** 权重矩阵每步只从 HBM 读*一次*，无论多少 token 搭车；把更多 token（$M$）塞进这一次读，每个权重字节就做更多 FLOPs。这就是 batching 抬吞吐的全部机械原因——而且是个你能算出来的数。
- **Attention 的强度（几乎）由架构而非 batch 钉死。** decode 时你把整个 KV cache 拖过 HBM，只为服务**一个**查询 token；FLOPs 与字节都随上下文 $S$ 缩放，于是 $S$ *抵消*，强度落在一个由 [GQA](../glossary.md) 比例决定的常数上。单条流的 attention 无法靠 batch 逃出 memory-bound——你只能缩小字节、或把它们留在 SRAM。

## 3 · 原理与数学

### 3.1 GEMM 的 roofline

一个线性层计算 $Y = XW$，激活 $X\in\mathbb{R}^{M\times K}$、权重 $W\in\mathbb{R}^{K\times N}$，其中 $M$ 是一起处理的 token 数。把一次乘加记作 2 FLOPs、每元素 $b$ 字节：

$$
\text{FLOPs} = 2MKN, \qquad
\text{bytes} = \underbrace{(MK}_{\text{读 }X} + \underbrace{KN}_{\text{读 }W} + \underbrace{MN)}_{\text{写 }Y}\, b
$$

$$
I_{\text{gemm}}(M) = \frac{2MKN}{(MK + KN + MN)\,b}
$$

读它的两个极限：

**单 token decode，$M=1$**（且 $1 \ll K,N$，故 $KN$ 权重项主导分母）：

$$
I_{\text{gemm}}(1) \approx \frac{2KN}{KN\cdot b} = \frac{2}{b} = 1 \ \text{FLOP/字节（BF16）}
$$

你读 $KN$ 个权重，只在一个 token 上做 $2KN$ FLOPs——强度 $\approx 2/b$，不管矩阵多大都被钉在**带宽屋顶**上。这就是*为什么*单条流 decode 是 memory-bound：不是模型大小的锅，而是 $M=1$ 的 matmul 每个权重字节的活儿远远不够。

**批处理 / prefill，$M$ 大**（仍 $M \ll N$，故固定权重读 $KN$ 主导）：

$$
I_{\text{gemm}}(M) \approx \frac{2MKN}{KN\cdot b} = \frac{2M}{b}
$$

强度**随 $M$ 线性增长**。它在此处越过拐点、变 compute-bound：

$$
M^{*} \approx \frac{I^{*} b}{2} = \frac{P b}{2B} \approx \frac{165 \times 2}{2} = 165 \ \text{个 token（4090，BF16）}
$$

即你需要大约 ~165 个 token 在飞，一个 projection GEMM 才打满算力单元。（激活流量会把精确交点略往上推；§4 会算。）*这就是 [continuous batching](../glossary.md) 兑现的那句「batching 抬吞吐」的定量表述。*

### 3.2 attention 的 roofline

Attention **不是**固定权重的 GEMM：它的操作数是 $Q$、$K$、$V$，而 decode 里 $K,V$ 就是 **KV cache**——其大小随上下文 $S$ 增长。每层，一个 decode 步让一个查询 token 在 $n_q$ 个查询头（每头 head_dim $d$）上关注 $S$ 个缓存的 key，读 $n_{\text{kv}}$ 个 KV 头的 $K,V$：

$$
\text{FLOPs}_{\text{attn}} \approx \underbrace{2 n_q S d}_{QK^\top} + \underbrace{2 n_q S d}_{\text{scores}\cdot V} = 4 n_q S d, \qquad
\text{bytes}_{\text{attn}} \approx \underbrace{2 n_{\text{kv}} S d\, b}_{\text{读 }K,V}
$$

$$
I_{\text{attn}}^{\text{decode}} = \frac{4 n_q S d}{2 n_{\text{kv}} S d\, b} = \frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}
$$

**上下文长度 $S$ 抵消了。** decode attention 强度是个由 [GQA](../glossary.md) 比例 $n_q/n_{\text{kv}}$ 决定的*常数*。对 `Qwen2.5-7B`（$n_q=28$、$n_{\text{kv}}=4$、BF16）：

$$
I_{\text{attn}}^{\text{decode}} = \frac{2}{2}\cdot\frac{28}{4} = 7 \ \text{FLOP/字节}
$$

两点随之而出。其一，GQA 把 attention 强度*抬高*恰好 $n_q/n_{\text{kv}} = 7\times$（每个 K/V 字节被 7 个查询头复用），相比 MHA 的 $I=2/b=1$——但 $7 \ll I^{*}\approx165$，所以 decode attention **仍牢牢 memory-bound**。（GQA 的头条战绩是 KV 字节*小* $7\times$，见 [Part 0](../part0/kv-cache.md)；强度提升是附赠。）其二，增大上下文**并不**能救它——更长的序列读更多 KV、也做更多 FLOPs。

**Prefill** 用 $S$ 个查询 token 对 $S$ 个 key，每个 KV 读一次、跨所有 $S$ 个查询复用——这是大 $M$ 的 attention 对应物：

$$
I_{\text{attn}}^{\text{prefill}} \approx \frac{4 n_q S^2 d}{2 n_{\text{kv}} S d\, b} = \frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}\cdot S = 7S \ \text{（Qwen，BF16）}
$$

强度**随 $S$ 线性增长**，在 $S \approx I^{*}b/(2\,n_q/n_{\text{kv}}) \approx 165/7 \approx 24$ 个 token 处越过拐点——所以哪怕短 prompt 也让 prefill attention compute-bound。（注意：这只数了 KV 读。*朴素* attention 会把 $S\times S$ 的 score 矩阵物化到 HBM，增加 $\sim S^2 b$ 字节，把强度又拽回来——这正是 [FlashAttention](../part0/gpu-hardware.md) 靠把 scores 留在 SRAM 所避开的陷阱。那是下一课、票 #7 的活儿。）

### 3.3 与 Part 0「decode $I\approx1$」对账

算子视角：decode 的 projection/FFN GEMM 坐在 $I\approx1$，attention 在 $I\approx7$。整模型的**按字节加权平均**仍是 $\approx1$，因为 ~14 GiB 的权重在中等上下文下压过不到 1 GiB 的 KV 读——于是 FFN 的 $I\approx1$ 主导了均值。算子视角不与 Part 0 冲突，而是*细化*它，并告诉你哪个旋钮动哪个 kernel。

## 4 · 完整可跑代码 + 逐行讲解

这个计算器把形状变成强度与状态——**纯 CPU、可离线运行**，无 GPU。这正是你*在租卡之前*用来判断什么值得优化的分析。

```python title="operator_intensity.py"
"""GEMM 与 attention 的按算子算术强度（纯 CPU，离线）。"""
RIDGE = 165.0   # 4090 拐点 I* = P/B ~= 165 FLOP/字节（示例）


def gemm_intensity(M: int, K: int, N: int, b: int = 2) -> float:
    flops = 2 * M * K * N                          # 2*M*K*N 次乘加
    bytes_ = (M * K + K * N + M * N) * b           # 读 X、读 W、写 Y
    return flops / bytes_


def attn_intensity(n_q: int, n_kv: int, phase: str, S: int, b: int = 2) -> float:
    if phase == "decode":                          # 1 个查询 token vs S 个缓存 KV
        flops = 4 * n_q * S                         # QK^T + scores*V（head_dim 因子抵消）
        bytes_ = 2 * n_kv * S * b                   # 读 K 和 V
    else:                                          # prefill：S 个查询 vs S 个 key
        flops = 4 * n_q * S * S
        bytes_ = 2 * n_kv * S * b                   # KV 读一次，跨 S 个查询复用
    return flops / bytes_                          # head_dim d 上下抵消


def regime(I: float) -> str:
    return "compute-bound" if I >= RIDGE else "memory-bound"


if __name__ == "__main__":
    # 已核实的 Qwen2.5-7B-Instruct 形状：hidden 3584、heads 28、kv_heads 4、head_dim 128。
    H, N_Q, N_KV, D = 3584, 28, 4, 128
    OPS = {                                         # 各线性层的 (K, N)
        "q_proj/o_proj": (H, N_Q * D),              # 3584 x 3584
        "k_proj/v_proj": (H, N_KV * D),             # 3584 x 512（GQA：瘦）
        "gate/up_proj":  (H, 18944),                # 3584 x 18944
        "down_proj":     (18944, H),                # 18944 x 3584
    }

    print(f"拐点 I* = {RIDGE:.0f} FLOP/字节\n")
    for M in (1, 256):
        print(f"-- M={M} 个 token 在飞的 GEMM --")
        for name, (K, N) in OPS.items():
            I = gemm_intensity(M, K, N)
            print(f"  {name:<14} I = {I:8.1f} FLOP/字节  ({regime(I)})")
        print()

    print("-- attention（Qwen GQA 28/4）--")
    for S in (128, 512, 2048):
        Id = attn_intensity(N_Q, N_KV, "decode", S)
        Ip = attn_intensity(N_Q, N_KV, "prefill", S)
        print(f"  S={S:>5}: decode I={Id:5.1f} ({regime(Id)})  |  "
              f"prefill I={Ip:8.1f} ({regime(Ip)})")
```

**逐行讲解：**

- `gemm_intensity` — §3.1 公式原样。分母三项是读 $X$、读 $W$、写 $Y$；$M=1$ 时 $KN$ 权重读淹没另两项，所以结果钉在 $\approx 1$。
- `attn_intensity` — decode 为**一个**查询 token 读整个 KV；prefill 把那份 KV 跨**$S$** 个查询复用（FLOPs 里的 $S^2$）。`head_dim` $d$ 在分子分母同时出现、抵消——强度取决于 **GQA 比例**，而非 $d$。
- `regime` — 与 4090 拐点（~165）的单次比较。它左边全是 bandwidth-bound。
- `__main__` — 代入**已核实的** Qwen2.5-7B 形状，对 GEMM 扫 $M\in\{1,256\}$、对 attention 扫 $S\in\{128,512,2048\}$。

预期输出（精确算术，非跑分）：

```text
拐点 I* = 165 FLOP/字节

-- M=1 个 token 在飞的 GEMM --
  q_proj/o_proj  I =      1.0 FLOP/字节  (memory-bound)
  k_proj/v_proj  I =      1.0 FLOP/字节  (memory-bound)
  gate/up_proj   I =      1.0 FLOP/字节  (memory-bound)
  down_proj      I =      1.0 FLOP/字节  (memory-bound)

-- M=256 个 token 在飞的 GEMM --
  q_proj/o_proj  I =    224.0 FLOP/字节  (compute-bound)
  k_proj/v_proj  I =    162.9 FLOP/字节  (memory-bound)
  gate/up_proj   I =    236.0 FLOP/字节  (compute-bound)
  down_proj      I =    236.0 FLOP/字节  (compute-bound)

-- attention（Qwen GQA 28/4）--
  S=  128: decode I=  7.0 (memory-bound)  |  prefill I=   896.0 (compute-bound)
  S=  512: decode I=  7.0 (memory-bound)  |  prefill I=  3584.0 (compute-bound)
  S= 2048: decode I=  7.0 (memory-bound)  |  prefill I= 14336.0 (compute-bound)
```

一张表三个教训：（1）$M=1$ 时**每个** GEMM 都在 $I\approx1$ memory-bound；（2）到 $M=256$，肥的 FFN GEMM 已 compute-bound，但*瘦的* GQA projection（`k_proj/v_proj`，$N=512$）仍在 162.9 落后——窄矩阵要更大 batch 才越线；（3）decode attention 无论上下文都卡在 **7**，而 prefill attention 冲过拐点——同一算子、相反状态。

### 在 vLLM 源码里读它（v0.26.0）

两种原型在 vLLM 里是两条不同的代码路径，找到它们正是这段 read-along 的意义：

- **GEMM** 是 [`vllm/model_executor/layers/linear.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/linear.py) 里的线性层——`QKVParallelLinear`（融合的 Q/K/V 投影）、`MergedColumnParallelLinear`（gate + up）、`RowParallelLinear`（`o_proj`、`down_proj`）。它们的强度就是你刚写的 `gemm_intensity(M, K, N)`；调度器塞进去的 batch 维 `M` 正是把它们推过拐点的那个量。
- **Attention 算子**不是其中之一——它经 [`vllm/v1/attention/backends/registry.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/backends/registry.py) 里的 `AttentionBackendEnum` 派发，其 `FLASH_ATTN` 项解析到 `vllm.v1.attention.backends.flash_attn.FlashAttentionImpl`。这就是那个 decode 强度被钉在 $2n_q/(n_{\text{kv}}b)$、与上下文无关的算子。

只读这两个文件，就足以看清引擎为何把「一批投影 GEMM」和「attention kernel」当作 roofline 上两种本质不同的东西——下一课就打开 attention 那一个。

## 5 · Lab —— 在你自己的卡上看见 GEMM roofline

!!! gpu "GPU Lab"
    - **最低显存：** 任意 CUDA GPU（分配几个 3584×3584 矩阵；不加载模型）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）——对齐基线
    - **预估耗时 / 花费：** ~5 分钟 · ~¥0.5 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** `torch.cuda.Event` 计时是 CUDA/ROCm 的；AMD 上同样的调用能跑但达到的数不同，CPU/TPU/Neuron 各有自己的计时器——请查你所在平台的文档。

$M$-旋钮不是理论——你能看着一个 matmul 随 $M$ 增大爬上 roofline。给 `q_proj` 形状的 GEMM 计时，把 wall-clock 变成达到的 TFLOP/s：

```python title="gemm_roofline.py"
import torch

assert torch.cuda.is_available()
K = N = 3584                                  # Qwen q_proj 形状
W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

def achieved_tflops(M: int, iters: int = 50) -> float:
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):                       # 预热（CUDA graphs、频率）
        _ = X @ W
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        _ = X @ W
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1e3 / iters
    return (2 * M * K * N) / secs / 1e12       # 达到的 TFLOP/s

for M in (1, 16, 64, 256, 1024):
    print(f"M={M:>5}:  {achieved_tflops(M):6.1f} TFLOP/s achieved")
```

**观察什么：** `M=1` 时你会看到峰值的一个*极小*零头——matmul 是 memory-bound，正是计算器预测的 $I\approx1$。$M$ 爬过 ~150–200 后，达到的 TFLOP/s 在算力屋顶附近走平：GEMM 越过了拐点。你用一个权重矩阵画出了 roofline 的左半边。进阶：给一个大张量的 `x.copy_()` 计时得到达到的 GB/s（你真实的 $B$），再确认 `M=1` GEMM 的 TFLOP/s ≈（那个 $B$）× 1 FLOP/字节。

## 6 · 常见坑 / 反直觉点

- **把整模型平均当成算子事实。** 「decode 是 $I\approx1$」是按字节加权的均值；单个算子各不相同（attention 在 ~7）。优化算子，别优化平均。
- **忘了权重每步读一次、不是每 token 读一次。** GEMM 分母里是 $KN$（权重矩阵）*而非* $MKN$——多塞 token（$M$）在权重读一侧是免费的。把权重按每 token 重复计，就抹掉了整个 batching 收益。
- **以为更长上下文让 decode attention 每字节更差。** 并不——$S$ 抵消；decode attention 强度固定在 $2n_q/(n_{\text{kv}}b)$。更长上下文花更多*总*字节（更大 KV），而非更低强度。
- **以为 GQA 修好了 memory-bound 状态。** GQA 把 attention 强度抬 $n_q/n_{\text{kv}}\times$ 并同倍缩小 KV 字节——对容量是巨大利好——但 $7 \ll 165$，decode attention *仍* memory-bound。状态由拐点定，不由 GQA 定。
- **瘦矩阵越线晚。** $N$ 小的 GEMM（GQA `k_proj`，$N=512$）要比肥 FFN GEMM 更大的 $M$ 才 compute-bound。「batch 256 让一切 compute-bound」对窄 projection 是错的。
- **把 prefill 的朴素 attention 当成 compute-bound。** 只有当 scores 留在 SRAM 时才是。把 $S\times S$ score 矩阵物化到 HBM 会增加 $S^2 b$ 字节，可能把 prefill attention 翻回 memory-bound——这正是 FlashAttention 存在的理由。

## 7 · 面试连线

- [GEMM 与 attention 的算术强度](../interview/arithmetic-intensity.md) —— 本课为你准备的高频题：*推导一个 decode matmul 与 decode attention 的强度，解释后者为何与上下文无关、GQA 对它做了什么，并求出一个 projection 变 compute-bound 的 batch 大小。*

## 8 · 小结 & 延伸阅读

**一句话：** 一个 decoder 层是 GEMM（强度 $\propto M$，那把 batching 旋钮 → 在 $M^{*}\approx I^{*}b/2$ 越过拐点）加 attention（decode 强度固定在 $2n_q/(n_{\text{kv}}b)$、与上下文无关且 memory-bound；prefill 强度 $\propto S$、compute-bound）——而从形状算出每一个，是你判断一个优化能买到、买不到什么的方法。

延伸阅读：

- Williams, Waterman, Patterson —— *Roofline: An Insightful Visual Performance Model* —— 按算子 roofline 的源头。
- Dao 等 —— *FlashAttention* —— 为什么 $S\times S$ score 矩阵必须留在 SRAM（上面那个 prefill-attention 注意点）。
- [GPU 硬件心智模型](../part0/gpu-hardware.md) 那节课 —— $\min(P, I\cdot B)$ 与拐点从哪来。
- [推理流程](../part0/inference-flow.md) 那节课 —— 本课细化的那个整模型平均。

## 9 · 自测小问

??? question "推导 `q_proj` matmul（K=N=3584，BF16）在 M=1 与 M=256 的算术强度。各是什么状态？"
    $I = 2MKN/((MK+KN+MN)b)$。$M=1$ 时：分子 $2\cdot3584^2$，分母 $\approx KN\cdot b = 3584^2\cdot2$，故 $I\approx 2/b = 1$ FLOP/字节——**memory-bound**。$M=256$ 时：$I = 2\cdot256\cdot3584^2/((256\cdot3584 + 3584^2 + 256\cdot3584)\cdot2) = 224$ FLOP/字节——高于 ~165 拐点，**compute-bound**。batch 维 $M$ 把同一个 matmul 挪过了拐点。

??? question "为什么 decode attention 的强度与上下文长度无关，GQA 又改变了什么？"
    FLOPs（$4n_qSd$）与读的字节（KV cache，$2n_{\text{kv}}Sd\,b$）都随上下文 $S$ 缩放，于是 $S$ 抵消：$I = \frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}$。GQA 把它抬高组比例 $n_q/n_{\text{kv}}$ 倍（Qwen：$7$，因为每个 K/V 被 7 个查询头复用）——但 $7 \ll 165$，decode attention 仍 **memory-bound**。GQA 更大的回报是 KV 字节小 $7\times$。

??? question "你 profile decode，发现 FFN `down_proj` 在 I≈1、attention 在 I≈7，但整模型报 I≈1。不矛盾——为什么？"
    整模型强度是**按字节加权的平均**。中等上下文下，每步读的 ~14 GiB 权重压过不到 1 GiB 的 KV 读，所以 FFN/projection GEMM（在 $I\approx1$ 搬走几乎所有字节）主导均值。attention 更高的 $7$ 是真的，但骑在总字节的一小部分上，几乎不动平均。算子视角细化——而非推翻——Part 0 的平均。
