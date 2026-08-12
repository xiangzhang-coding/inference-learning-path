# Multi-LoRA serving：一基座，多 adapter

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 6 · 进阶推理专题   ·   **对应课程：** [Multi-LoRA Serving：一基座，多 adapter](../part6/multi-lora-serving.md)

---

## Q：什么是 multi-LoRA serving，为什么 LoRA 让 adapter 很小，vLLM 如何在一个 batch 里跑不同 adapter，哪些旋钮决定你能同服多少个？

### 直接答案

Multi-LoRA serving 托管**一份冻结基座 + 许多小 [LoRA](../glossary.md) adapter**，**按请求**选用哪个 adapter——甚至在*同一个* [continuous batch](../part5/continuous-batching.md) 的各行之间混用。它存在是因为服务 N 个全量微调意味着 N 份完整副本（7B 是 N × ~15 GB）——在一张 [24 GB 卡](../part0/gpu-hardware.md)上不可能。

**为什么 adapter 很小：** 微调是个**低秩**轻推，于是 LoRA 冻结 $W$、学 $\Delta W = BA$，$B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$、$r \ll d$。参数从 $d\cdot k$ 降到 $r(d+k)$——每矩阵约 100×——所以整模型 adapter 只有几到几十 MB。基座**只加载一次**，几十个 adapter 放货架。

**一个 batch 如何服务多个 adapter：** **基座 GEMM $Wx$ 对整个 batch 只跑一次**；每行的 adapter 增量 $B(Ax)$ 由一个按 adapter id 分组的 *grouped* kernel（SGMV/BGMV，来自 S-LoRA/Punica）加上。没有逐 adapter 的模型重放。

**旋钮：** `--max-lora-rank`（按你最高的 adapter 秩定槽位尺寸——不要更高）、`max_loras`（单 batch 活跃的不同 adapter 数）、`max_cpu_loras`（CPU 侧缓存）。

### 深入

- **选择是数据。** 请求用整数 id 指名 adapter（`LoRARequest(name, id, path)`），或在服务端把 adapter **名字放进 `model` 字段**。加一个微调 = 加载一个文件，而非新起服务。
- **静态 vs 动态。** 启动时声明（`--lora-modules name=path`），或运行时经 `POST /v1/load_lora_adapter` 热加载，受 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 门控——一个安全敏感、仅限可信环境的特性。
- **显存权衡。** adapter 槽位分享着本可作 [KV 缓存 block](../part5/paged-attention.md) 的显存；过大的 `--max-lora-rank` 会偷走并发。
- **batching 经济学。** 少数热门 adapter 好 batch；一条超过 `max_loras` 的冷 adapter 长尾无法在一步内并跑，掉吞吐。

### 代码

核实过的 0.26.0 离线 API——一次 `generate` 里的异构 adapter：

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True, max_lora_rank=64, max_loras=2)
outs = llm.generate(
    ["[user] SQL: airports in Malawi [/user] [assistant]", "What is a KV cache?"],
    SamplingParams(temperature=0, max_tokens=64),
    lora_request=[LoRARequest("sql", 1, sql_path), None],   # 行 0 → adapter，行 1 → 基座
)
```

### 面试官追问

- *「为什么不直接把 adapter 合并进基座？」* → 合并得到一个专用单模型、每请求零增量开销，但杀掉多租户。只服务一个微调时合并；服务多个时保持运行时。
- *「两行如何用不同 adapter 却不重跑模型？」* → 基座 GEMM 共享一次；逐行 $B(Ax)$ 由 grouped kernel 加上。基座那趟是代价，增量很便宜。
- *「你给秩 16 的 adapter 设了 `--max-lora-rank 256`——错在哪？」* → 浪费显存（vLLM 文档自己的「不必要地高」情形），那本可作 KV 缓存 = 更多并发。设成真正的最大秩。
- *「adapter 输出乱码——先查什么？」* → 基座/adapter 不匹配（错的基座模型或版本）。adapter 是*某个特定*基座上的增量。
- *「暴露 `/v1/load_lora_adapter` 的风险？」* → 它从任意路径加载权重 = 信任任意代码/数据。留在信任边界内；仅可信环境。

### 关联概念

- 课程：[Multi-LoRA Serving](../part6/multi-lora-serving.md)
- 相关：[Static vs continuous batching](continuous-batching.md)（adapter 所在的那个 batch）、[PagedAttention：block manager 与碎片](kv-cache-block-manager.md)（adapter 槽位分享的显存池）、[调参旋钮](tuning-knobs.md)（LoRA 旋钮 vs 吞吐/延迟曲线）
- 术语：[LoRA / Multi-LoRA serving](../glossary.md)
