# Guided / structured decoding：把 token 掩码到 schema

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 6 · 进阶推理专题   ·   **对应课程：** [Guided / Structured Decoding：让非法 token 不可能出现](../part6/structured-decoding.md)

---

## Q：structured decoding 如何强制合法的 JSON/正则/语法输出，为什么保证是硬的而非统计的，代价是什么，它*不*修什么？

### 直接答案

Structured decoding 通过在**每个 decode 步**掩码 logits，把生成约束到一个 **schema / 正则 / 语法 / 枚举**。schema 被一次性编译成一个**有限状态机**；在状态 $s$ 它产出一个允许下一 token 的二值掩码 $m^{(s)}\in\{0,1\}^{|V|}$。vLLM 在 softmax 前把 $\log m^{(s)}$ 加到 logits 上，于是被禁 token 的 logit 变成 $-\infty$、概率**恰为 0**——在任何 temperature/top-p 下都不可采样。采样后 FSM 推进，下一步用新状态的掩码。输出在**构造上**就 schema 合法。

**为何硬而非统计：** prompt 只是*降低*坏 token 的概率（规模上仍漏几个百分点）；掩码让它们*不可能*——不存在通往非法输出的路径。

**代价：** 一次性的 schema→自动机**编译**（可摊销；对全新 schema 可能表现为首 token 延迟）加上每步掩码，而默认 **`xgrammar`** 后端靠逐状态预计算的 token 集合把它压到近乎零。可选后端：**`guidance`**。

**它*不*修什么：** 它保证**形状，从不保证真值**——一个 schema 合法的答案仍可能是错的。

### 深入

- **掩码 softmax。** $z'_i = z_i + \log m^{(s)}_i$；因为 $e^{-\infty}=0$，被禁 token 无论采样设置如何都从分布中消失。
- **约束类型（`StructuredOutputsParams`）：** `json`（JSON Schema / Pydantic）、`regex`（后端用 Rust 风格正则）、`choice`（枚举）、`grammar`（EBNF CFG）、`structural_tag`。
- **要知道的 API 改名：** `guided_json`/`guided_regex`/… 已在 vLLM 0.12.0 **弃用并移除**；0.26.0 上用 `structured_outputs`（在线）/ `StructuredOutputsParams`（离线）。看到 `guided_*` = 早于 0.12.0 的代码。
- **后端选择：** `--structured-outputs-config.backend`（默认 `auto` → `xgrammar`）。

### 代码

核实过的 0.26.0 离线 API，JSON + 枚举：

```python
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
class Review(BaseModel):
    sentiment: str; score: float
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
so = StructuredOutputsParams(json=Review.model_json_schema())    # 或 regex=/choice=/grammar=
out = llm.generate("Rate: 'vLLM is wonderful!'",
                   SamplingParams(temperature=0, max_tokens=64, structured_outputs=so))
print(out[0].outputs[0].text)    # {"sentiment": "positive", "score": 0.95}
```

在线：`response_format={"type":"json_schema", "json_schema":{"name":..., "schema":...}}` 或 `extra_body={"structured_outputs": {"regex": r"\d{3}-\d{3}-\d{4}"}}`。

### 面试官追问

- *「prompt 就能 97% 拿到 JSON——何必？」* → prompt 偏置；掩码禁止。规模上那 3% 是一串解析错误。掩码让非法输出不可能，而非只是不太可能。
- *「它让答案正确吗？」* → 不。它修**形式，不修内容**。法国首都 `{"capital": "Lyon"}` 是 schema 合法且错的——值要单独校验。
- *「延迟在哪？」* → 一次性语法**编译**（在用该 schema 的请求间摊销）；每步掩码在 xgrammar 下近乎免费。冷、巨大、独一无二的 schema 会让首 token 延迟飙一下。
- *「`SamplingParams(guided_json=...)` 在 0.26.0 报错——为什么？」* → 0.12.0 移除了。用 `structured_outputs=StructuredOutputsParams(json=...)`。
- *「过松的 schema 仍给垃圾？」* → 是——`{"answer": "string"}` 几乎不约束。用枚举/正则/maxLength/必填字段收紧。

### 关联概念

- 课程：[Guided / Structured Decoding](../part6/structured-decoding.md)
- 相关：[Static vs continuous batching](continuous-batching.md)（掩码跑在每个序列的 decode 步内）、[数值格式与精度](number-formats.md)（掩码作用其上的 logits/softmax）、[调参旋钮](tuning-knobs.md)（后端选择作为一个服务旋钮）
- 术语：[Guided / Structured decoding](../glossary.md)
