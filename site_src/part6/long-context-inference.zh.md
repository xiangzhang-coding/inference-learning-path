# 长上下文推理：RoPE 缩放、Attention Sink 与 KV 墙

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    经 Context7 对 vLLM 0.26.0 核实（ADR-0004）：上下文扩展用 **`--hf-overrides`** 配一个 `rope_parameters` 字典（`rope_type: "yarn"`、`factor`、`original_max_position_embeddings`）加 **`--max-model-len`**——旧的 **`--rope-scaling` flag 已弃用**；KV 缓存量化用 **`kv_cache_dtype="fp8"`**（=`fp8_e4m3`；也有 `fp8_e5m2`），可选 `calculate_kv_scales`；长 prefill 由 [chunked prefill](../part5/scheduler-chunked-prefill-pd.md) 切块（`enable_chunked_prefill`，默认开）。本课建立在 [KV 缓存](../part0/kv-cache.md)的增长问题与 [PagedAttention block 池](../part5/paged-attention.md)之上。§4 的模型是**显存模型，不是 benchmark**；一切大小均为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

上下文窗口暴涨：两三年里 2K → 8K → 128K → 1M token。「把整个代码库 / 整本书 / 200 页合同直接喂给模型」就是卖点。但服务长上下文正是两堵很不同的墙同时撞上来的地方，而且面试官偏爱这一点——它是**两个**问题穿着同一件外套：

1. **质量墙——模型超出训练长度就崩。** 一个在 32K token 上训练的模型，从没见过位置 100,000。它的位置编码产出它从未训练过的旋转，注意力散掉，输出退化成重复或胡话——常常在标称上限*之前很多*就开始。让模型在超出训练长度后仍连贯，是 **RoPE 缩放**（position interpolation / NTK / YaRN）与流式背后的 **attention-sink** 洞察要干的活。
2. **显存与调度墙——[KV 缓存](../part0/kv-cache.md)随长度线性增长。** KV 每来一个 token 就长，所以一个 128K-token 请求可能要*数 GB* 的 KV——碾压普通请求，并饿死其他序列做[并发](../part5/continuous-batching.md)所需的[block 池](../part5/paged-attention.md)。而它的 prefill 巨大，不切块就冻住 decode。让长序列*装得下、排得开*，是 **KV 量化**（fp8）、**滑动窗口 / attention-sink** 逐出、与 **[chunked prefill](../part5/scheduler-chunked-prefill-pd.md)** 要干的活。

所以「长上下文推理」其实是：*在长度上保持连贯*（位置）**且** *在长度上装得下、排得开*（显存）。只修一个、无视另一个的方法上不了线。→ 见[术语表](../glossary.md)中 *Long-context inference、RoPE*。

## 2 · 心智模型

两条独立的轴——连贯与容量——各有各的失败、各有各的杠杆：

```text
轴 1 — 连贯（位置）：  "模型在第 100k 个 token 处还说得通吗？"
  RoPE 给每个位置每个频率一个旋转角 θ：
    训练区                │ 外推（未见）→ 注意力崩
    0 ─────────── 32k      ┊       100k ────────────► 1M
                    └ 训练长度墙 ┘
  Position Interpolation：把 0..100k 压进 0..32k 的角度里（留在分布内）
    0 ───────────────────────────────► 100k   按 s = L_train / L_target 缩放
    └───── 映射进已训练的 0..32k 角度范围 ─────┘   (YaRN 逐频率做 + 温度)

轴 2 — 容量（显存）：  "KV 缓存装得下吗，别人还跑得了吗？"
  KV 字节 = 2 · layers · kv_heads · head_dim · 长度 · 每元素字节     （对长度线性）
    4k ctx : ▓                          一个 128k 请求吃掉整个池：
   32k ctx : ▓▓▓▓▓▓▓▓                    128k : ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 饿死别人
  杠杆：  fp8 KV（字节减半）· GQA（更少 kv_heads）· 滑动窗口（限住长度）
          chunked prefill（别让一个 128k prefill 冻住每个 decode）

ATTENTION SINK（为何朴素滑动窗口会崩）：
  softmax 必须把权重放到某处 → 最前面几个 token 吸收「剩余」→ 它们是 SINK
  保留 [sink token] + [近期窗口] ──► 在有界显存下流式到无限长
  丢掉 sink ──► 质量崩（模型没地方倾倒注意力）
```

三个要记住的形状：

- **连贯与容量正交。** RoPE 缩放让模型*理解*位置 100K；它对 100K token KV 的*显存*毫无帮助。fp8 KV 让它*装得下*；它对模型*是否理解*那个位置毫无帮助。你几乎总是两者都要。
- **KV 代价对长度线性，且是真正的上限。** 上下文翻倍，KV 翻倍。长上下文服务被这条线主导，而非 FLOPs。
- **注意力需要一个 sink。** 你不能只留「最后 N 个 token」——模型依赖最前面几个 token 作为注意力倾倒处。sink *和*近期窗口都要留。

## 3 · 原理

### 3.1 RoPE 及它为何超出训练长度就崩

Rotary Position Embedding（RoPE）通过*旋转* query/key 向量来编码位置。对第 $i$ 个维度对（头维 $d$），旋转频率为

$$
\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2-1
$$

位置 $m$ 的 token 其第 $i$ 对被旋转 $m\theta_i$。注意力于是只依赖*相对*角 $(m-n)\theta_i$——优雅，也是 RoPE 能在**训练范围内**跨位置泛化的原因。问题：训练到长度 $L_\text{train}$ 的模型，只见过至多 $L_\text{train}\cdot\theta_i$ 的角。要位置 $m \gg L_\text{train}$，低频对就产出模型**从未见过**的旋转幅度——分布外——注意力退化。这就是质量墙：它不是硬截断，而是角度跑出了训练流形。

**修法通过重缩放位置，让角度留在分布内：**

- **Position Interpolation (PI)：** 把每个位置按 $s = L_\text{train}/L_\text{target}$ 缩放，$m\theta_i \to (m/s)\theta_i$——位置 $L_\text{target}$ 现在落在模型当年认作 $L_\text{train}$ 的角上。简单，但压缩了分辨率，通常需要短暂微调。
- **NTK-aware：** 不均匀缩放位置，而是增大 RoPE `base`（即 `rope_theta`），它对低频拉伸多于高频——更好地保住局部分辨率。
- **YaRN：** vLLM 配置的那个方法——逐频率插值（低频插值、高频外推）加一个注意力温度调整。它是扩展上下文的强默认，经 `rope_type: "yarn"` 配一个 `factor` 设定。

在 vLLM 0.26.0 里你在起服务时用 **`--hf-overrides`** 开启它（`--rope-scaling` flag 已弃用）：

```text
--hf-overrides '{"rope_parameters": {"factor": 4.0, "original_max_position_embeddings": 32768,
                                     "rope_theta": 1000000, "rope_type": "yarn"}}'
--max-model-len 131072
```

`factor` 是扩展倍数（$4.0 \times 32\text{K} = 128\text{K}$），`original_max_position_embeddings` 是模型的训练长度，`--max-model-len` 设新的服务上限。关键：这只在模型确实能用这份扩展时才有效——YaRN 买来的是可用长度，不是魔法般的理解力。

### 3.2 Attention sink（以及流式到无限长）

你以为「只留最后 $N$ 个 token 的 KV」能在固定显存下拿到无限上下文。并不能——质量会崩。**StreamingLLM** 的洞察解释了原因：注意力分数的 softmax 必须和为 1，所以即便没有哪个过去 token 真正相关，它也得把权重放到*某处*。模型学会把这份多余倾倒到**最前面几个 token**上——它们成了 **attention sink**。逐出它们，softmax 就没地方停放剩余质量，分布扭曲、输出退化。

修法很便宜：保留少数几个 **sink token**（常常就是前 4 个）**加**一个近期滑动窗口。这给出有界显存*且*稳定质量，适用于实际上无限的流。vLLM 通过**滑动窗口注意力**与它的**混合 KV 缓存管理器**实现这件事的显存侧：滑动窗口层只为最近的 token（窗口大小）预留块，而全注意力层为所有 token 预留——所以窗口化模型的 block 池无论流跑多久都保持有界。

### 3.3 显存墙与撬动它的杠杆

KV 缓存大小是（见 [KV 缓存数学课](../part2/kv-cache-math.md)）：

$$
\text{KV 字节} = 2 \times L_\text{layers} \times H_\text{kv} \times d_\text{head} \times \text{seq\_len} \times b
$$

唯一依赖长度的因子是 `seq_len`，且是**线性**——128K token 的 KV 是 4K 的 32×。在一张 24 GB 卡上，那单个请求就能吃掉大部分 [block 池](../part5/paged-attention.md)，于是*所有其他人的并发跌向 1*。杠杆：

- **fp8 KV 缓存**（`kv_cache_dtype="fp8"`）——KV 用 1 字节而非 2 存，把长度项大致减半。vLLM 校验器指出它「减少 GPU 显存占用、提升性能」但「无适当缩放因子时可能掉精度」——故有 `calculate_kv_scales`。这是长上下文最直接的容量杠杆。
- **GQA**（更少 $H_\text{kv}$，见[注意力变体](../interview/attention-variants.md)材料）——一开始就让长上下文可负担的架构性削减。
- **滑动窗口**（限住有效 `seq_len`）——§3.2，用于流式负载。
- **[Chunked prefill](../part5/scheduler-chunked-prefill-pd.md)**——*调度*杠杆。128K prefill 是一大块计算，会冻住每个进行中的 decode；chunked prefill（默认开）把它对着 `max_num_batched_tokens` 预算切块，让 decode 持续流动。`long_prefill_token_threshold` 标记一个 prefill 何时算「长」。

## 4 · 完整可跑代码 + 逐行讲解

一个纯 Python 的容量墙模型：KV 字节 vs 上下文长度，以及并发跌到多低——外加 fp8 KV 买回什么。无 GPU；这就是决定一个长上下文请求到底装不装得下的算术。

```python title="long_context_kv_wall.py"
"""长上下文 KV 显存墙：KV 随长度线性增长、碾压并发。
显存模型，不是 benchmark。纯 Python、离线。形状 ~ Qwen2.5-7B（GQA）。"""
LAYERS, KV_HEADS, HEAD_DIM = 28, 4, 128     # Qwen2.5-7B：28 层、4 个 KV 头（GQA）、head_dim 128
KV_BUDGET_GB = 16                            # 24GB 卡装完权重后留给 KV 的空间（示例）

def kv_bytes_per_token(bytes_per_elem):
    # 2（K 和 V）* layers * kv_heads * head_dim * 字节  —— 每 token、所有层
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * bytes_per_elem

def max_concurrent(seq_len, bytes_per_elem):
    per_req = kv_bytes_per_token(bytes_per_elem) * seq_len
    return (KV_BUDGET_GB * 1024**3) // per_req      # KV 预算里能装多少个这样的请求

for seq_len in [4_096, 32_768, 131_072]:
    fp16_per_req = kv_bytes_per_token(2) * seq_len / 1024**3     # 每请求 GB，FP16 KV
    fp16_conc    = max_concurrent(seq_len, 2)                    # 并发，FP16 KV
    fp8_conc     = max_concurrent(seq_len, 1)                    # 并发，fp8 KV（1 字节）
    print(f"ctx={seq_len:>7}: {fp16_per_req:6.2f} GB/req (FP16 KV)  "
          f"→ max concurrency {fp16_conc:>3} (FP16) | {fp8_conc:>3} (fp8, ~2x)")
```

**逐行讲解：**

- `LAYERS, KV_HEADS, HEAD_DIM` 是 `Qwen2.5-7B` 的真实形状；注意 `KV_HEADS=4`（GQA），不是 28——GQA 已经大幅缩小了 KV 项，这*正是*长上下文可行的原因。
- `kv_bytes_per_token()` 是 [KV 缓存公式](../part2/kv-cache-math.md)去掉 `seq_len`——每 token、跨所有层、K 和 V 的代价。
- `max_concurrent(seq_len, ...)` 用 KV 预算除以一个请求的 KV 占用——字面意义的「这么长的序列能装几个」。这就是长上下文压上来的并发上限。
- 循环扫 4K → 32K → 128K，打印每请求 GB 与 **FP16 vs fp8** KV 下的最大并发。fp8 把每元素字节减半，于是大致翻倍能装下多少长请求——把 §3.3 的杠杆变成数字。

预期输出（显存模型，示例）：

```text
ctx=   4096:   0.22 GB/req (FP16 KV)  → max concurrency  73 (FP16) | 146 (fp8, ~2x)
ctx=  32768:   1.75 GB/req (FP16 KV)  → max concurrency   9 (FP16) |  18 (fp8, ~2x)
ctx= 131072:   7.00 GB/req (FP16 KV)  → max concurrency   2 (FP16) |   4 (fp8, ~2x)
```

墙很陡：4K 时能装 ~73 个并发序列；128K 时只有 ~2 个——**32× 的跌落**（128K 是 4K 长度的 32×），恰好对长度线性。fp8 KV 大致把每一行翻倍（诚实的边界：这是显存收益，带可能的精度代价）。这就是为什么长上下文首先是个*容量*问题：RoPE 缩放能让模型理解位置 128K，但若只装得下 ~2 个这样的请求，你每 GPU 的吞吐就崩了。服务上的答案是 §3.3 的杠杆栈——把这套算术记熟就是面试。

## 5 · Lab —— 扩展上下文（YaRN）并缩小 KV（fp8）

!!! gpu "GPU Lab（单卡，完全可跑）"
    - **最低显存：** 中等上下文下 `Qwen2.5-7B-Instruct`（INT4/AWQ）约 ~16 GB；真正的 128K 运行需要 §4 的 KV 预算——撞 OOM 就降 `--max-model-len`（如 32K）。
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)；很长上下文（128K+）可能需要 A100（按 ADR-0001 开机即关）。
    - **预估耗时 / 花费：** 阅读 ~20 分钟（免费，无卡模式）· 选做运行 ~15 分钟 · ~¥1–3（示例）
    - **平台：** NVIDIA CUDA（默认）。**非 NVIDIA：** RoPE/YaRN 是架构层（与后端无关）；fp8 KV 需 CUDA 11.8+ 支持 `fp8_e4m3`/`fp8_e5m2`，ROCm 支持 `fp8_e4m3`——查你的后端。

用 YaRN 扩展服务上下文，用 fp8 缩小 KV 缓存：

```bash title="用 YaRN 上下文扩展 + fp8 KV 起服务（核实过的 0.26.0 flag）"
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --hf-overrides '{"rope_parameters": {"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}}' \
    --max-model-len 131072 \
    --kv-cache-dtype fp8            # KV 用 1 字节存 —— 大致翻倍能装下多少上下文
# 注意：--rope-scaling 已弃用；用如上的 --hf-overrides 配 rope_parameters。
```

```python title="离线等价 + fp8 KV"
from vllm import LLM, SamplingParams
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    hf_overrides={"rope_parameters": {"factor": 4.0, "original_max_position_embeddings": 32768,
                                      "rope_type": "yarn"}},
    max_model_len=131072,
    kv_cache_dtype="fp8",          # =fp8_e4m3；为精度可能需要 calculate_kv_scales
)
long_prompt = "Summarize the following document.\n" + ("lorem ipsum " * 20000)
print(llm.generate(long_prompt, SamplingParams(max_tokens=128))[0].outputs[0].text[:200])
```

**观察 / 动手：**

1. **确认扩展后的长度。** 不加 override 时，>32K 的 prompt 被拒（`--max-model-len` 超过模型默认）；加了 YaRN + `--max-model-len 131072` 就被接受。那就是 RoPE 缩放在干活。
2. **看 KV 显存减半。** 比较加与不加 `--kv-cache-dtype fp8` 时启动日志里的 `num_gpu_blocks`（[block 池](../part5/paged-attention.md)大小）——fp8 大致把块数翻倍，即 §4 的并发。
3. **感受并发悬崖。** 同时打几个 100K-token 请求，看能并发跑的有多少 vs 许多短请求——§4 的算术，活的。
4. **质量检查。** 在 4K vs 128K 跑一个「大海捞针」检索；注意超出训练长度的可用召回取决于扩展是否真的撑住了——YaRN 买的是长度，不是保证的理解。

## 6 · 常见坑 / 反直觉点

- **用被弃用的 `--rope-scaling`。** 0.26.0 里它被 `--hf-overrides` 配 `rope_parameters`（`rope_type: "yarn"`、`factor`、…）取代。粘贴旧的 `--rope-scaling '{...}'` 命令是头号翻车。
- **以为单靠 `--max-model-len` 就能扩展上下文。** 不配 RoPE 缩放 override 就抬高长度上限，只会让模型跑进它没训练过的位置、吐垃圾。你需要*同时*有缩放配置*和*长度。
- **以为长上下文是算力问题。** 它是**显存**问题——KV 对长度线性，早在 FLOPs 重要之前就碾压[并发](../part5/continuous-batching.md)。先估 KV 预算（§4）。
- **朴素滑动窗口丢掉 sink token。** 只留最后 $N$ 个 token 会崩，因为最前面几个 token 是 **attention sink**。保留 sink + 近期窗口（StreamingLLM）。
- **开 fp8 KV 却指望精度白给。** fp8 KV「无适当缩放因子时可能掉精度」——用 `calculate_kv_scales` / 校准过的 scale，并在你的任务上验证，尤其长上下文误差会随更多 token 累积。
- **让一个长请求冻住整台服务。** 128K prefill 是一大块计算；没有 [chunked prefill](../part5/scheduler-chunked-prefill-pd.md)（默认开）它会拖停每个 decode。保持 chunked prefill 开着并调 `max_num_batched_tokens`。
- **把「支持 1M token」当成「能在 1M token 上推理」。** 标称上下文 ≠ 有效上下文。通过大海捞针不等于跨整窗多跳推理；用你的真实任务基准。

## 7 · 面试连线

- [长上下文推理：位置、sink 与 KV 墙](../interview/long-context-inference.md) — 本课为你准备的高频题：*为什么模型超出训练长度就崩、RoPE 缩放（PI/NTK/YaRN）如何修、attention sink 是什么、以及为什么 KV 缓存——而非算力——是长上下文的上限。*

## 8 · 小结 & 延伸阅读

**一句话：** 长上下文推理是两个正交问题——**连贯**（RoPE 给每个位置一个角 $m\theta_i$，超出训练长度就跑出分布，靠 Position Interpolation / NTK / **YaRN** 重缩放位置来修，在 vLLM 里经 `--hf-overrides` `rope_parameters` + `--max-model-len` 设定；`--rope-scaling` flag 已弃用）与**容量**（[KV 缓存](../part0/kv-cache.md)对长度*线性*，所以一个 128K 请求碾压并发——靠 **fp8 KV**（`kv_cache_dtype="fp8"`）、GQA、滑动窗口 + **attention-sink** 逐出、与 **[chunked prefill](../part5/scheduler-chunked-prefill-pd.md)** 让大 prefill 不冻住 decode 来缓解）——真实系统两半都要。

延伸阅读：

- *RoFormer*（Su 等，2021）— 最初的 RoPE 表述，$\theta_i = \text{base}^{-2i/d}$。
- *Extending Context Window via Position Interpolation*（Chen 等，2023）与 *YaRN*（Peng 等，2023）— 缩放方法；YaRN 正是 `rope_type: "yarn"` 配置的东西。
- *StreamingLLM*（Xiao 等，2023）— attention-sink 现象与 sink + 滑动窗口流式。
- vLLM `docs/features/context_extension.md` 与 `docs/features/quantization/quantized_kvcache.md` — 这里引用的 `--hf-overrides`/`rope_parameters` 与 `kv_cache_dtype` 机制。
- [KV 缓存数学课](../part2/kv-cache-math.md)与 [PagedAttention 课](../part5/paged-attention.md) — 容量墙压上来的显存公式与 block 池。

## 9 · 自测小问

??? question "一个模型训练到 32K token。你设了 `--max-model-len 128000` 别的什么都没做，超过 ~32K 后输出变垃圾。为什么，vLLM 0.26.0 里正确的修法是什么？"
    RoPE 把位置编码成每个频率对一个旋转角 $m\theta_i$。模型只见过至多 $32\text{K}\cdot\theta_i$ 的角；要位置 128K 就产出**训练分布外**的低频旋转，注意力散掉、输出退化——单抬 `--max-model-len` 只是*让*模型跑进未见位置，并没教会它这些位置。修法是一份**RoPE 缩放**配置，把位置重缩放回已训练的角度范围——在 vLLM 0.26.0 里经 **`--hf-overrides`** 配 `rope_parameters`（`rope_type: "yarn"`、`factor: 4.0`、`original_max_position_embeddings: 32768`）**并**配 `--max-model-len 131072`。（旧的 `--rope-scaling` flag 已弃用。）即便如此，YaRN 买的是*可用长度*，不是保证的理解——在你的任务上验证。

??? question "为什么不能靠只保留最近 N 个 token 的 KV 来服务「无限」上下文？最小修法是什么？"
    因为 **attention sink**。注意力分数的 softmax 必须和为 1，所以即便近期窗口里没有真正相关的东西，模型也得把权重放到某处——它学会把这份多余倾倒到**最前面几个 token**上。若朴素滑动窗口逐出了这些起始 token，softmax 就失去它的「sink」，注意力分布扭曲、质量崩（StreamingLLM 的发现）。最小修法是保留几个 **sink token**（如前 4 个）**加**近期滑动窗口——有界显存、稳定质量、无限流长。vLLM 的滑动窗口 / 混合 KV 缓存管理器在窗口化层只为近期窗口预留块，让池保持有界。

??? question "你把一个 7B 模型扩到 128K 上下文。同事想调高 `max_num_seqs` 来保住高并发。为什么这多半会失败，真正管用的是什么？"
    因为 [KV 缓存](../part2/kv-cache-math.md)对**序列长度线性**，128K 时单个请求的 KV 可达*数 GB*——由 §4，~7.0 GB/req vs 4K 的 ~0.22 GB，所以 16 GB 的 KV 预算只装得下 ~2 个这样的请求，无论 `max_num_seqs` 设成多少。把 `max_num_seqs` 抬到超过 KV 预算只会导致抢占/OOM，而非更多真并发——绑定约束是 [block 池](../part5/paged-attention.md)，不是序列上限。长上下文真正管用的：**fp8 KV**（`kv_cache_dtype="fp8"`，~2× 请求数）、**GQA**（已内建）、若负载允许限住有效长度的**滑动窗口**、以及 **[chunked prefill](../part5/scheduler-chunked-prefill-pd.md)** 让巨大 prefill 不冻住 decode。先修好显存那条线；并发随之而来。
