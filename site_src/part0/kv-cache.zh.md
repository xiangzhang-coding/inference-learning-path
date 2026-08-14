# KV 缓存

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    本页所有 flag/API 均针对 vLLM 0.26.0 经 Context7 核实（ADR-0004）。性能数字为**示例 / 量级参考**——真实数字请在你自己的 AutoDL 环境实测。下文 KV 大小的算术是**精确**的（就是乘法），不是跑分。

---

## 1 · 直觉 & 为什么重要

LLM 逐个 token 生成文本。要产出第 *t* 个 token，注意力需要**此前每一个** token 的 **Key** 与 **Value** 向量。朴素做法是每步都重算全部——二次复杂度的浪费。

**KV 缓存**就是解法：每个 token 的 K、V 只算**一次**，存下来，供后续所有步复用。这把每步的工作从「对整段历史重新注意力」变成「用缓存的历史 + 一个新 token 做注意力」。

但有个关键——也正是为什么要先把这一课刻进脑子：**缓存不是免费的，它住在 GPU 显存里，且随每个 token、每个并发请求增长。** 在一张 24 GB 的卡上，通常是 KV 缓存——**而非**模型权重——决定你能同时服务多少请求。本路径后面的每个服务优化（PagedAttention、continuous batching、量化、prefix caching）本质上都是一种**在同一块 HBM 里塞下更多有用 KV 缓存**的办法。

## 2 · 心智模型

要记住的一张图：你那 24 GB 显存被劈成两块——**权重**（一次性付清）与 **KV 缓存**（按 token、按并发序列付费）——而先耗尽的总是 KV 那块：

```text
24 GB VRAM, split two ways   (BF16 weights; illustrative)

  |<------------------------------ 24 GB total ------------------------------>|
  +----------------------+------+------+------+------+------+ ~ +------+-------+
  |     weights ~14 GB   | KV#1 | KV#2 | KV#3 | KV#4 | KV#5 | ~ | KV#N | free  |
  +----------------------+------+------+------+------+------+ ~ +------+-------+
   paid ONCE, fixed        \_____ KV cache: ~0.44 GiB per 8k-token sequence _____/
                            each grows +1 token / step;
                            N ~= 10 GB / 0.44 ~= 22 sequences  =  concurrency ceiling
```

两个结论立刻浮现：

- **Decode 是 memory-bound（带宽受限）。** 每步都从 HBM 重新读取**整个** KV 缓存，却只做一个新 token 的少量计算。瓶颈在带宽，不在 FLOPs。→ 见[术语表](../glossary.md)的 *Memory-bound* 与 *Roofline*。
- **并发是一个显存预算问题。** 权重是一次性付清的固定成本；KV 缓存是随 `batch × sequence_length` 增长的按序列成本。服务吞吐很大程度上就是「能塞下多少 KV 缓存」——权重*没*占走的每一字节，都是上图中 KV 区域多出的一格。

## 3 · 原理与数学

对 decoder-only Transformer，KV 缓存按「每层、每 KV 头、每 token」各存一个 K 张量和一个 V 张量。其大小为：

$$
\text{KV bytes} = \underbrace{2}_{K,\,V} \times L \times n_{\text{kv}} \times d_h \times b_{\text{dtype}} \times S \times B
$$

其中 $L$ = 层数，$n_{\text{kv}}$ = **KV** 头数（分组后），$d_h$ = 头维度，$b_{\text{dtype}}$ = 每元素字节数（BF16/FP16 为 2），$S$ = 序列长度，$B$ = batch 大小。

单 token（$S = B = 1$）：

$$
\text{bytes/token} = 2\,L\,n_{\text{kv}}\,d_h\,b_{\text{dtype}}
$$

**算例——`Qwen2.5-7B-Instruct`**（$L=28$，$n_{\text{kv}}=4$，$d_h=128$，BF16 即 $b=2$）：

$$
2 \times 28 \times 4 \times 128 \times 2 = 57344 \text{ 字节} = 56 \text{ KiB/token}
$$

在其默认 32 768-token 上下文下，**单条**序列的 KV 缓存为 $56\,\text{KiB} \times 32768 \approx 1.75$ GiB。并发跑 8 条这样的序列，光 KV 缓存就吃掉 ~14 GiB——这还没算权重。

**GQA 为何重要。** Qwen2.5-7B 用 $n_{\text{kv}}=4$ 个 KV 头，但有 28 个*注意力*头。若它用朴素 MHA（$n_{\text{kv}}=28$），缓存会大 $28/4 = 7\times$——392 KiB/token。这就是 [GQA](../glossary.md) 给你换来的 7 倍更小 KV 缓存，也是几乎所有现代模型都用它的原因。

## 4 · 完整可跑代码 + 逐行讲解

这个计算器**可离线运行**——纯 CPU、无 GPU、无网络。它把上面的公式变成你能动手拨弄的东西。

```python title="kv_cache_size.py"
"""KV 缓存大小计算器（纯 CPU，可离线运行）。"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    num_layers: int       # transformer 层数 (L)
    num_kv_heads: int     # 分组后的 KV 头数；MHA 时等于注意力头数
    head_dim: int         # 每头维度 (d_h)
    dtype_bytes: int = 2  # BF16/FP16 = 2；FP8/INT8 = 1


def kv_bytes_per_token(cfg: ModelConfig) -> int:
    # 前导 2 = 一个 K 张量 + 一个 V 张量
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * cfg.dtype_bytes


def kv_bytes(cfg: ModelConfig, seq_len: int, batch: int = 1) -> int:
    return kv_bytes_per_token(cfg) * seq_len * batch


def gib(n: int) -> float:
    return n / (1024 ** 3)


if __name__ == "__main__":
    # 经核实的 Qwen2.5-7B-Instruct 配置：28 层、4 个 KV 头 (GQA)、head_dim 128。
    qwen = ModelConfig("Qwen2.5-7B-Instruct", num_layers=28, num_kv_heads=4, head_dim=128)

    per_tok = kv_bytes_per_token(qwen)
    print(f"{qwen.name}: {per_tok} bytes/token = {per_tok / 1024:.0f} KiB/token")
    for s in (2048, 8192, 32768):
        print(f"  seq_len={s:>6}: {gib(kv_bytes(qwen, s)):.2f} GiB (1 sequence)")

    # 同一模型，假设的朴素 MHA（num_kv_heads == 注意力头数 == 28）：
    mha = ModelConfig("Qwen2.5-7B (hypothetical MHA)", num_layers=28, num_kv_heads=28, head_dim=128)
    ratio = kv_bytes_per_token(mha) / per_tok
    print(f"MHA would be {ratio:.0f}x larger: {kv_bytes_per_token(mha) / 1024:.0f} KiB/token")
```

**逐行讲解：**

- `ModelConfig` — 驱动 KV 大小的四个架构数字，加 `dtype_bytes`。注意是 `num_kv_heads`，*不是*注意力头数：这个区别就是整个 GQA 的故事。
- `kv_bytes_per_token` — §3 的公式取 `S = B = 1`。前导 `2` 是 K **和** V；漏掉它是最常见的「差一个 2」bug。
- `kv_bytes` — 把单 token 成本按 `seq_len × batch` 放大，正是 $S \times B$ 项。
- `gib` — 字节 → GiB（`1024**3`），好和卡标称的 VRAM 对比。
- `__main__` — 代入**已核实**的 Qwen2.5-7B 配置，打印三种上下文长度下的单 token / 单序列大小，再和假设的 MHA 变体对比，暴露 7 倍的 GQA 收益。

预期输出（精确算术，非跑分）：

```text
Qwen2.5-7B-Instruct: 57344 bytes/token = 56 KiB/token
  seq_len=  2048: 0.11 GiB (1 sequence)
  seq_len=  8192: 0.44 GiB (1 sequence)
  seq_len= 32768: 1.75 GiB (1 sequence)
MHA would be 7x larger: 392 KiB/token
```

## 5 · Lab —— 在 vLLM 里看 KV 缓存如何决定容量

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）
    - **预估耗时 / 花费：** ~15 分钟 · ~¥1 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** vLLM 也能在 AMD ROCm 与 CPU 构建上跑；`kv_cache_dtype="fp8"` 的支持与确切显存行为随后端而异——请查你所在平台的 vLLM 构建说明。

vLLM 0.26.0 里控制 KV 缓存的旋钮（均经 Context7 核实）：

```python title="serve_kv.py"
from vllm import LLM, SamplingParams

# AWQ 4-bit 权重（~5–6 GB）把 24 GB 的大头留给 KV 缓存。
# vLLM 从该 checkpoint 自动识别 AWQ（量化本身是 Part 4）。
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    max_model_len=8192,           # 限制每序列 KV 长度 -> 限制其 KV 缓存
    gpu_memory_utilization=0.90,  # vLLM 可用的 24 GB 占比（默认 0.92）
    kv_cache_dtype="fp8",         # KV 字节数约减半 -> 更多并发序列（示例）
    enable_prefix_caching=True,   # 跨请求复用共享前缀的 KV
)

out = llm.generate(["Why is LLM decode memory-bound?"], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
```

等价的服务端 CLI：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

**观察什么：** 启动时 vLLM 会打印它分配的 KV 缓存 **block** 数。调低 `max_model_len` 或用 `kv_cache_dtype fp8` → 更多 block → 更多并发序列。那个 block 数**就是**你并发上限的具象化。

## 6 · 常见坑 / 反直觉点

- **是 KV 缓存、不是权重，决定并发上限。** 权重一次付清；KV 随 `序列数 × 长度` 增长。上下文翻倍大致让 KV 翻倍，而非计算翻倍。
- **「差一个 2」bug。** K *和* V。漏掉它，每个估算都只有真相的一半。
- **Decode 是 memory-bound 会让人别扭**（「这么大的模型，肯定是算力瓶颈吧！」）。但每个 decode 步从 HBM 重读整个 KV 缓存，只做一个 token 的运算——带宽才是墙。
- **`gpu_memory_utilization` 是把双刃剑。** 太高 → 真实负载下 OOM（激活会突增）；太低 → 白白浪费 KV 容量（吞吐）。
- **`kv_cache_dtype="fp8"` 在精度上不是免费的。** 它把 KV 字节减半，但可能改变输出——把质量差异当成要*测量*的东西，别想当然。
- **PagedAttention 仍有（少量）浪费。** vLLM 按固定大小 **block**（`--block-size` 个 token）分配 KV；序列最后一个 block 通常没填满。这种内部碎片相比朴素连续分配微不足道——分页的意义正在于此。

## 7 · 面试连线

- [KV 缓存与吞吐上限](../interview/kv-cache.md) —— 本课为你准备的高频题：*为什么瓶颈通常是 KV 缓存，而非算力？*

## 8 · 小结 & 延伸阅读

**一句话：** KV 缓存用显存换重算，而这块显存——随 token 与并发增长——正是每个服务优化都在设法放松的核心约束。

延伸阅读：

- vLLM 文档 —— [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/) 与 engine 参数（基线 v0.26.0）。
- *Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM 论文）。
- *GQA: Training Generalized Multi-Query Transformer Models* —— 为什么更少的 KV 头。
- [KV 缓存数学](../part2/kv-cache-math.md) 课（Part 2）—— 完整的显存估算（字节/token → 能塞多少条序列）。
- [PagedAttention](../part5/paged-attention.md) 课（Part 5）—— vLLM 实际如何以定长 block 存储与管理这块缓存。

## 9 · 自测小问

??? question "一个模型 L=32 层、n_kv=8 个 KV 头、head_dim=128、BF16。它的单 token KV 缓存是多少？"
    $2 \times 32 \times 8 \times 128 \times 2 = 131072$ 字节 $= 128$ KiB/token。（再乘序列长度和 batch 得到总量。）

??? question "为什么 decode 阶段是 memory-bound 而非 compute-bound？"
    每个 decode 步从 HBM 读取*整个* KV 缓存，却只算一个新 token 的注意力/FFN。每搬一字节所做的运算极少，于是 GPU 卡在显存带宽上，而非算力上。而一次处理很多 token 的 prefill，就是它 compute-bound 的对照面。

??? question "你有 24 GB、权重约 14 GB。忽略激活，大致能在 KV 缓存里塞下多少条 8192-token 的 Qwen2.5-7B 序列（BF16）？"
    单条 8192-token 序列约 0.44 GiB。剩余 ≈ 10 GiB / 0.44 ≈ **~22 条**（示例；计入激活与 block padding 后真实容量更低——而用 `kv_cache_dtype fp8` 或缩小权重的量化模型则更高）。
