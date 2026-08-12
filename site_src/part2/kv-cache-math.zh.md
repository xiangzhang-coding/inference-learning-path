# KV 缓存显存数学：为部署做容量规划

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    flag / 日志行（`gpu_memory_utilization`、启动时的「GPU KV cache size」报告）经 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。权重 / 激活的显存大小是**示例 / 量级参考**——真实值请查 `nvidia-smi` 与 vLLM 自己的启动日志。预算算术（一个减法加一个除法）是*精确*的。

---

## 1 · 直觉 & 为什么重要

「一张 4090 在 8k 上下文下能服务多少用户？」是个你用**算术、在租卡之前**就能回答的容量问题。[Part 0 的 KV 缓存课](../part0/kv-cache.md) 给了你每 token 大小 $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$ 与一个粗算（~22 序列）。本课把这个单一数字变成一份**部署计划**：拼出*完整*的显存预算——权重 + KV + 激活 + 框架开销——再解出你真正在意的数（最大并发，或你能承诺的最大上下文），并知道哪个旋钮买多少。

这是每个服务化决策背后的数学，也是[系统设计面试](../interview/vram-capacity-planning.md) 的半壁江山：给定模型、卡、SLO，能塞下多少并发请求、要改什么才够目标？算对了，你的机器守住延迟承诺；算错了，它在满队列的峰值负载下 OOM。→ 见[术语表](../glossary.md)的 *KV cache*、*PagedAttention*、*SLO*。

## 2 · 心智模型

显存是一根堆叠条，而 KV 缓存是*固定成本之后剩下的一切*：

```text
  24 GB 卡（gpu_memory_utilization 封住可用高度，默认 0.92）
  ┌──────────────────────────────────────────────┐  ── util · 24 GB ──┐
  │  CUDA context + 框架            ~1–2 GB         │  固定开销          │
  ├──────────────────────────────────────────────┤                    │
  │  模型权重        ~14 GB BF16 / ~5–6 GB AWQ      │  固定，一次付清    │  可用
  ├──────────────────────────────────────────────┤                    │
  │  激活 / workspace          随 batch 缩放        │  ~1 GB 上下       │
  ├──────────────────────────────────────────────┤                    │
  │  KV 缓存  ◄── 剩下的一切 = 你的并发预算                             │
  └──────────────────────────────────────────────┘  ───────────────────┘
        并发  =  (KV 预算)  /  (每序列 KV)
             =  (util·显存 − 权重 − 激活 − 开销) / (κ · S)
```

要握住的两个形状：

- **并发是剩余量，不是你直接设的目标。** 你不「设」能塞多少序列——你设权重（量化与否）、开销余量（`gpu_memory_utilization`）、每序列长度（`max_model_len`），而 KV 预算——进而并发——是剩下的那部分。每个容量杠杆都靠*扩大剩余量*或*缩小每序列成本*起作用。
- **最大的杠杆通常是权重，而非 KV dtype。** 在 24 GB 卡上，BF16 权重（~14 GiB）吃掉大半预算；把它量化到 4-bit（~5–6 GiB）直接释放 ~8 GiB 进 KV 预算——往往比把 KV 字节减半的并发收益更大。先上权重量化，再上 KV 量化。

## 3 · 原理与数学

由 [Part 0](../part0/kv-cache.md)，长度 $S$ 的一条序列花 $\kappa S$ 字节 KV，其中 $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$（对 `Qwen2.5-7B`：$\kappa = 2\cdot28\cdot4\cdot128\cdot2 = 57{,}344$ 字节/token = 56 KiB）。我们不重推它——我们*花*它。

**KV 预算**是可用显存减去一切固定项：

$$
\text{KV}_{\text{budget}} = \underbrace{u \cdot V}_{\text{可用}} - \underbrace{W}_{\text{权重}} - \underbrace{A}_{\text{激活}} - \underbrace{O}_{\text{开销}}
$$

其中 $u$ = `gpu_memory_utilization`（默认 0.92）、$V$ = 卡显存、$W$ = 权重字节、$A$ = 激活/workspace、$O$ = CUDA-context + 框架开销。然后你真正会问的两个问题：

$$
\boxed{N_{\text{seq}} = \left\lfloor \frac{\text{KV}_{\text{budget}}}{\kappa\,S} \right\rfloor}
\qquad
\boxed{S_{\max} = \left\lfloor \frac{\text{KV}_{\text{budget}}}{\kappa\,N} \right\rfloor}
$$

固定上下文 $S$ 下的最大并发，以及它的逆——若必须服务 $N$ 条并发流，你能承诺的最长上下文。

**实算计划 —— `Qwen2.5-7B` 在 24 GB 4090 上，$S = 8192$、$u=0.90$、开销+激活 $\approx 1.6$ GiB：**

| 权重 | KV dtype | $\kappa$ / token | 每序列 KV | KV 预算 | **最大并发** |
|---|---|---|---|---|---|
| BF16（~14.2 GiB） | BF16 | 56 KiB | 0.44 GiB | ~5.8 GiB | **~13** |
| AWQ 4-bit（~5.5 GiB） | BF16 | 56 KiB | 0.44 GiB | ~14.5 GiB | **~33** |
| AWQ 4-bit（~5.5 GiB） | FP8 | 28 KiB | 0.22 GiB | ~14.5 GiB | **~66** |

读这个递进：量化**权重**把并发大约抬 $2.5\times$（它释放最大的一块）；再量化 **KV** 又翻一倍（每序列字节减半）。两个旋钮把你从同一张卡上的 ~13 带到 ~66 条并发 8k-上下文流——*整个服务化吞吐的故事都写在这一张表里。*（数字均为示例；精确值由引擎报告——见 Lab。）

注意这比 Part 0 的 ~22 更保守：那个粗算忽略了激活/开销、并用满 24 GiB。减去真实开销、套上 `gpu_memory_utilization`，才是生产计划的做法。

## 4 · 完整可跑代码 + 逐行讲解

一个容量规划器——**纯 CPU、可离线运行**，无 GPU。这正是 vLLM 内部用来决定分配多少 KV block 的算术。

```python title="vram_planner.py"
"""显存容量规划器：最大并发 & 最大上下文（纯 CPU，离线）。"""
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass
class Card:
    vram_gib: float = 24.0        # RTX 4090
    util: float = 0.90            # gpu_memory_utilization（vLLM 默认 0.92）
    overhead_gib: float = 1.6     # CUDA context + 激活/workspace（示例）

    def kv_budget_gib(self, weight_gib: float) -> float:
        return self.util * self.vram_gib - weight_gib - self.overhead_gib


def kv_per_seq_gib(kappa_bytes: int, seq_len: int) -> float:
    return kappa_bytes * seq_len / GIB                 # kappa * S


def max_concurrency(card: Card, weight_gib: float, kappa_bytes: int, seq_len: int) -> int:
    return int(card.kv_budget_gib(weight_gib) / kv_per_seq_gib(kappa_bytes, seq_len))


def max_context(card: Card, weight_gib: float, kappa_bytes: int, n_seq: int) -> int:
    budget_bytes = card.kv_budget_gib(weight_gib) * GIB
    return int(budget_bytes / (kappa_bytes * n_seq))   #逆问题


if __name__ == "__main__":
    card = Card()
    KAPPA_BF16, KAPPA_FP8 = 57344, 28672               # Qwen2.5-7B KV 字节/token（Part 0）
    W_BF16 = 7.615e9 * 2 / GIB                          # ~14.2 GiB 稠密权重
    W_AWQ = 5.5                                         # ~5.5 GiB 4-bit（示例，实测）
    S = 8192

    plans = [
        ("BF16 权重，BF16 KV", W_BF16, KAPPA_BF16),
        ("AWQ  权重，BF16 KV", W_AWQ,  KAPPA_BF16),
        ("AWQ  权重，FP8  KV", W_AWQ,  KAPPA_FP8),
    ]
    print(f"S={S} tokens，util={card.util}，overhead={card.overhead_gib} GiB\n")
    for label, w, kappa in plans:
        n = max_concurrency(card, w, kappa, S)
        budget = card.kv_budget_gib(w)
        print(f"{label}：KV 预算 {budget:5.1f} GiB -> ~{n:>3} 并发序列")

    # 逆：在最优配置上服务 64 条并发流，能有多长上下文？
    s_max = max_context(card, W_AWQ, KAPPA_FP8, n_seq=64)
    print(f"\nAWQ + FP8 KV，目标 64 并发 -> 最大上下文 ~= {s_max} tokens")
```

**逐行讲解：**

- `Card.kv_budget_gib` — §3 的预算：可用显存（$u\cdot V$）减权重再减固定开销。这*就是* KV 缓存栖身的剩余量。
- `kv_per_seq_gib` — Part 0 的 $\kappa\cdot S$，单位 GiB。`kappa_bytes` 已含 K 与 V 的 ×2、且用 `n_kv`（GQA），别重复套。
- `max_concurrency` / `max_context` — 两个框起来的公式：把预算按每序列成本、或按每 token 成本 × 目标并发做整除。取下整，因为半条序列塞不下。
- `__main__` — 三个逐级量化（先权重后 KV）的计划，加逆问题（固定并发、解上下文）。`W_BF16` 从参数量算出；`W_AWQ` 是示例实测大小（4-bit 权重带 scale/zero，故不恰为 params/4）。

预期输出（精确算术，非跑分）：

```text
S=8192 tokens，util=0.9，overhead=1.6 GiB

BF16 权重，BF16 KV：KV 预算   5.8 GiB -> ~ 13 并发序列
AWQ  权重，BF16 KV：KV 预算  14.5 GiB -> ~ 33 并发序列
AWQ  权重，FP8  KV：KV 预算  14.5 GiB -> ~ 66 并发序列

AWQ + FP8 KV，目标 64 并发 -> 最大上下文 ~= 8484 tokens
```

前三行是 §3 的表变成代码；最后一行回答容量规划者真正会被问的逆问题（「我们要 64 条并发流——能承诺多少上下文？」）。改一个字段——`util`、`overhead_gib`、`S`——重读这份计划即可。

## 5 · Lab —— 拿你的计划对账 vLLM 自己的报告

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB（加载 `Qwen2.5-7B-Instruct-AWQ`）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）
    - **预估耗时 / 花费：** ~10 分钟 · ~¥1 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** 启动时的 KV-cache 报告与后端无关，但可达大小与 `kv_cache_dtype="fp8"` 支持在 ROCm/CPU 上不同——请查你所在平台的 vLLM 构建说明。

vLLM 在启动时算的正是这份预算，并把它**打印出来**。启动模型、读日志、拿它的数与你规划器的数对账：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8
```

启动时 vLLM 打印两行已核实的日志（v0.26.0）：

```text
GPU KV cache size: 524,288 tokens
Maximum concurrency for 8,192 tokens per request: 64.00x
```

**观察什么：** 「GPU KV cache size」（单位 *token*）恰是你的 $\text{KV}_{\text{budget}} / \kappa$，「Maximum concurrency for 8,192 tokens」是它除以 `max_model_len`——vLLM 自己版本的 `max_concurrency()`。你的规划器为此配置推出 ~66；引擎这里报 ~64，这点小差距就是你估的开销/激活项与 vLLM 实际预留之差——引擎的数落在理想预算规划器*略下方*，绝不在上方。现在扫参数：去掉 `--kv-cache-dtype fp8`（KV 翻倍 → token 数约减半），或调低 `--max-model-len`（更少上下文换更多并发）。每次改动都让报告的数按公式预测精确移动——预算数学变得可见。（上面的数字为示例；你的取决于 checkpoint 与驱动。）

## 6 · 常见坑 / 反直觉点

- **忽略激活与开销。** 朴素的「空闲显存 ÷ 每序列 KV」（Part 0 的 ~22）高估容量。真实部署在任何 KV *之前*就要丢 ~1–2 GiB 给 CUDA context + 激活/workspace——减掉它，否则负载下 OOM。
- **把 `gpu_memory_utilization` 拉到 0.99。** 激活随并发 prefill 飙升；余量太薄会启动正常、峰值 OOM。默认 0.92 是故意留余地。
- **在权重之前优化 KV dtype。** 24 GB 卡上权重是最大的一块；量化它释放 ~8 GiB——通常比把 KV 字节减半的并发收益更大。先权重。
- **重复套 ×2 或用 `n_heads`。** $\kappa$ 已含 2 的因子（K *和* V）、且用 `n_kv`（GQA 后），不是注意力头。任一重复计都会让估计翻倍/×7。
- **把 `max_model_len` 当免费。** 它封住每序列 KV，故为长上下文抬它会*直接*砍并发（$N \propto 1/S$）。这是权衡，不是要拉满的默认。
- **忘了 block 补齐。** [PagedAttention](../glossary.md) 按固定大小 block 分配 KV；每条序列的最后一块通常半空，所以真实并发略低于公式。分页拿一点补齐损耗换掉了外部碎片。

## 7 · 面试连线

- [显存预算与最大并发](../interview/vram-capacity-planning.md) —— 本课为你准备的高频题：*给定 Qwen2.5-7B 在 24 GB 卡、8k 上下文，走一遍完整显存预算、估最大并发，并说你如何达到 ~60 并发流的目标。*

## 8 · 小结 & 延伸阅读

**一句话：** 并发是个*剩余量*——$N_{\text{seq}} = \lfloor (u\cdot V - W - A - O)/(\kappa S)\rfloor$——所以容量规划就是「减去固定成本再除」，而最大的杠杆是量化权重（释放最大一块），再量化 KV。

延伸阅读：

- vLLM 文档 —— *Conserving Memory* / 引擎参数（`gpu_memory_utilization`、`max_model_len`、`kv_cache_dtype`）与启动 KV-cache 报告，基线 v0.26.0。
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* —— 为什么用 block、剩余量究竟去了哪。
- [KV 缓存](../part0/kv-cache.md) 那节课 —— 本课花掉的每 token $\kappa$。
- 姊妹课 [算子 Roofline](roofline-analysis.md) —— 为什么这份 KV 喂养的 decode 一开始就是 memory-bound。

## 9 · 自测小问

??? question "一张 24 GB 卡，util 0.90，~1.6 GiB 开销，BF16 权重 ~14.2 GiB，Qwen2.5-7B（κ=56 KiB/token），8192-token 上下文。能塞多少并发序列？"
    KV 预算 $= 0.90\times24 - 14.2 - 1.6 = 5.8$ GiB。每序列 $= 56\,\text{KiB}\times8192 = 0.4375$ GiB。$N = \lfloor 5.8/0.4375\rfloor = $ **~13**（示例；未计 block 补齐）。把权重量化到 AWQ（~5.5 GiB）把预算抬到 ~14.5 GiB → ~33。

??? question "你要在一张 4090 上 8k 上下文服务 ~60 条并发流。改什么，按什么顺序？"
    （1）**量化权重**（AWQ/GPTQ 4-bit）——释放 ~8 GiB，单项最大收益（~13 → ~33）。（2）**量化 KV**（`kv_cache_dtype=fp8`）——每序列字节减半（~33 → ~66），越过 60。（3）若还不够，**封 `max_model_len`** 到工作负载真实上下文（并发 $\propto 1/S$）。先权重，因为它是最大的固定块。

??? question "为什么真实最大并发低于 `(24 GB − 权重) / (κ·S)`？"
    那个公式忘了三件事：`gpu_memory_utilization` 把可用显存压到满 24 GB 以下（默认 0.92）、CUDA context + 激活/workspace 在任何 KV 之前就吃掉 ~1–2 GiB、以及 PagedAttention 的固定 block 让每序列最后一块半空。诚实的预算先减开销、再套利用率因子——这正是 vLLM 启动「GPU KV cache size」报告所反映的。
