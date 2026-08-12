# 长上下文推理：位置、sink 与 KV 墙

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 6 · 进阶推理专题   ·   **对应课程：** [长上下文推理：RoPE 缩放、Attention Sink 与 KV 墙](../part6/long-context-inference.md)

---

## Q：模型为何超出训练长度就崩，RoPE 缩放如何修，attention sink 是什么，为什么 KV 缓存——而非算力——是长上下文的上限？

### 直接答案

长上下文推理是**两个正交问题**：**连贯**（模型在位置 128K 处还说得通吗？）与**容量**（[KV 缓存](../part0/kv-cache.md)装得下吗，别人还跑得了吗？）。

**为何崩：** [RoPE](../glossary.md) 把位置编码成旋转角 $m\theta_i$（$\theta_i = \text{base}^{-2i/d}$）。训练到 $L_\text{train}$ 的模型只见过至多 $L_\text{train}\cdot\theta_i$ 的角；超过后低频对产出**分布外**旋转、注意力退化。这不是硬截断——是角度离开了训练流形。

**RoPE 缩放**把位置重缩放回训练范围：**Position Interpolation**（按 $s=L_\text{train}/L_\text{target}$ 缩放位置）、**NTK-aware**（抬高 base）、**YaRN**（逐频率 + 注意力温度——强默认）。vLLM 0.26.0 里：`--hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":4.0,...}}'` + `--max-model-len`（**`--rope-scaling` 已弃用**）。

**Attention sink：** softmax 必须和为 1，所以模型把多余注意力倾倒到最前面几个 token——它们是「sink」。丢掉它们（朴素滑动窗口）质量就崩；保留 **sink + 近期窗口**做有界显存的流式（StreamingLLM）。

**为何 KV 是上限：** KV 对长度**线性**——一个 128K 请求可能要数 GB，早在 FLOPs 重要之前就碾压[并发](../part5/continuous-batching.md)。

### 深入

- **容量算术。** $\text{KV 字节} = 2\,L\,H_\text{kv}\,d_\text{head}\,\text{seq\_len}\,b$；只有 `seq_len` 缩放，且线性。16 GB KV 预算下 7B 模型 4K 时装 ~73 个请求、128K 时只 ~2 个。
- **容量杠杆：** **fp8 KV**（`kv_cache_dtype="fp8"`，~2× 请求，「无适当缩放因子时可能掉精度」→ `calculate_kv_scales`）、**GQA**（更少 $H_\text{kv}$）、**滑动窗口**（限住有效长度）、**[chunked prefill](../part5/scheduler-chunked-prefill-pd.md)**（别让 128K prefill 冻住 decode）。
- **连贯 ≠ 容量。** YaRN 让模型*理解*位置 128K，但对显存毫无回补；fp8 让它*装得下*，但不帮理解。两者都要。
- **标称 ≠ 有效。** 通过大海捞针 ≠ 跨整窗多跳推理。

### 代码

容量墙的算术（Qwen2.5-7B 形状，GQA）：

```python
LAYERS, KV_HEADS, HEAD_DIM, KV_GB = 28, 4, 128, 16
per_tok = lambda b: 2*LAYERS*KV_HEADS*HEAD_DIM*b          # KV 字节/token，所有层
conc    = lambda n, b: (KV_GB*1024**3)//(per_tok(b)*n)    # 长度 n 的最大并发序列数
for n in [4096, 32768, 131072]:
    print(n, f"{per_tok(2)*n/1024**3:.2f} GB/req", "conc", conc(n,2), "→ fp8", conc(n,1))
# 4096 0.22 GB/req conc 73 → fp8 146 | 131072 7.00 GB/req conc 2 → fp8 4
```

### 面试官追问

- *「长上下文是算力还是显存问题？」* → **显存。** KV 对长度线性、限住并发；FLOPs 次要。先估 KV 预算。
- *「单靠 `--max-model-len 128000` 够吗？」* → 不。不配 RoPE 缩放 override，模型跑进未训练位置 → 垃圾。要缩放配置*和*长度。
- *「为什么不能只留最后 N 个 token？」* → attention sink。最前面的 token 吸收多余 softmax 质量；逐出它们质量崩。保留 sink + 窗口。
- *「怎么装下更多长请求？」* → fp8 KV（~2×）、GQA、滑动窗口、chunked prefill。不是抬 `max_num_seqs`——绑定的是 block 池，不是序列上限。
- *「0.26.0 里 `--rope-scaling` 是配 YaRN 的方式吗？」* → 不——已弃用。用 `--hf-overrides` 配 `rope_parameters`（`rope_type:"yarn"`）。

### 关联概念

- 课程：[长上下文推理](../part6/long-context-inference.md)
- 相关：[KV 缓存与吞吐上限](kv-cache.md)与[注意力变体：MHA/MQA/GQA](attention-variants.md)（GQA 缩小的 KV 项）、[PagedAttention：block manager 与碎片](kv-cache-block-manager.md)（长序列饿死的 block 池）、[Chunked prefill 与 PD 分离](chunked-prefill-pd.md)（切开巨大 prefill）、[量化方法：GPTQ/AWQ/SmoothQuant/FP8](quantization-methods.md)（fp8 KV）
- 术语：[Long-context inference、RoPE](../glossary.md)
