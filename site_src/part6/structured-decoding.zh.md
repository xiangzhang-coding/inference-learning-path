# Guided / Structured Decoding：让非法 token 不可能出现

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    经 Context7 对 vLLM 0.26.0 核实（ADR-0004）：离线用 `from vllm.sampling_params import StructuredOutputsParams`，选项有 **`json`、`regex`、`choice`、`grammar`、`structural_tag`**，作为 `SamplingParams(structured_outputs=StructuredOutputsParams(...))` 传入；OpenAI 服务端用 `response_format={"type":"json_schema",...}` 或 `extra_body={"structured_outputs": {...}}`；后端为 **`xgrammar`（默认经 auto 选中）** 或 **`guidance`**，用 `--structured-outputs-config.backend` 选择。**旧的 `guided_json`/`guided_regex` 字段已在 vLLM 0.12.0 弃用并移除——请用 `structured_outputs`。** §4 代码展示的是当前 API；任何延迟数字均为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

你的服务需要模型返回**符合 schema 的合法 JSON**——`{"sentiment": "positive", "score": 0.9}`——好让下游代码解析。你写了个精心的 prompt：「只回复如下格式的 JSON……」。它大多数时候管用。然后模型在前面加了句「好的！这是 JSON：」，或多了个尾逗号，或写成 `"score": high`，或用 ```` ```json ```` 包起来。规模上来后，那个「大多数时候」的失败率就变成一串解析错误、重试、和凌晨三点的告警。prompt 后祈祷不是契约。

Structured decoding 把契约从「祈祷」变成「机械」。模型仍逐个 token 生成；在*每一步*，它先算出整个词表上的概率——但在采样前，我们**屏蔽掉每一个会让输出违反 schema 的 token**，把它们的概率设为零。若语法说下一个字符必须是 `{` 或空白，那么这一步里所有非 `{` 非空白的 token 都被禁止。模型*无法*吐出非法 JSON，因为非法 token 根本没上桌。输出在**构造上**就 schema 合法，而非靠运气。

这就是「请输出 JSON」与「你在物理上无法输出非 JSON」的区别。同一思路从 JSON schema 延伸到**正则**（电话号、日期）、**枚举**（`"positive" | "negative"`）、以及完整的**上下文无关文法**（一个 SQL 子集、一个 DSL）。它是你能给生产 LLM 端点加上的最可靠的东西之一——也带着每个面试官都会戳的那条锋利边界：它保证*形状*，从不保证*真值*。→ 见[术语表](../glossary.md)中 *Guided / Structured decoding*。

## 2 · 心智模型

schema → 自动机 → 每步 token 掩码 → 只从被允许的里采样：

```text
编译一次：      JSON schema / 正则 / 语法  ──►  有限状态机 (FSM)
                {sentiment: enum, score: number}    状态 + 允许的转移

每个 DECODE 步（处于 FSM 状态 s）：
  整个词表上的模型 logits：   [ the ]=2.1  [ { ]=1.8  [ Sure ]=3.0  [ " ]=0.4 …
                                                        └ 最高，但此处非法
  状态 s 的语法掩码：          [ the ]= 0   [ { ]= 1   [ Sure ]= 0   [ " ]= 1  …
                                  └ 禁止        └ 允许        └ 允许
  掩码后 logits（加 log 掩码）：[ the ]=-∞   [ { ]=1.8  [ Sure ]=-∞   [ " ]=0.4 …
  softmax + 采样  ─────────────►  选到 "{"（一个合法的下一 token；"Sure" 不可能）
  推进 FSM：  状态 s ──"{"──► 状态 s'（现在期待一个 key 或 "}"）

结果：每个被采样的 token 都让输出停在语法接受的路径上 →
      最终字符串保证匹配 schema。形状被强制；
      具体是哪个合法值（positive vs negative、0.9 vs 0.1）仍由模型决定。
```

三个要记住的形状：

- **约束作用在 logits 上，而非事后作用在文本上。** 没有「先生成、再校验、再重试」——非法 token 在采样*之前*、每一步就被移除。这就是为什么保证是硬的，而非统计意义上的。
- **schema 被一次性编译成一个状态机。** 每个状态都知道自己允许的下一批 token。解码就是在这台机器上走；掩码不过是「从这里出发哪些 token 能让我们留在被接受的路径上」。
- **它约束形式，不约束内容。** 掩码保证输出*能解析*、*匹配 schema*。它没法让 `score` 正确、让 `sentiment` 真实——模型仍在*合法*选项里挑。合法但垃圾仍然可能。

## 3 · 原理

### 3.1 从 schema 到 token 掩码

一个正则或语法定义一门语言——被接受字符串的集合。任何正则都能编译成一个**有限状态自动机**；一个上下文无关文法（JSON schema 变成的东西）编译成一个**下推自动机**。无论哪种，在生成的任一时刻自动机都处于某个状态 $s$，只有词表 $V$ 的一个子集能合法地接在后面而不离开被接受的路径。

把它编码成一个二值**掩码** $m^{(s)} \in \{0,1\}^{|V|}$：若 token $i$ 在状态 $s$ 被允许则 $m^{(s)}_i = 1$，否则 $0$。每一步模型产出 logits $z \in \mathbb{R}^{|V|}$；我们在 softmax **之前**施加掩码：

$$
z'_i = z_i + \log m^{(s)}_i \;=\;
\begin{cases}
z_i & \text{若 token } i \text{ 被允许 } (m^{(s)}_i = 1)\\
-\infty & \text{若被禁止 } (m^{(s)}_i = 0)
\end{cases}
$$

$$
p_i = \operatorname{softmax}(z')_i = \frac{e^{z'_i}}{\sum_j e^{z'_j}}
$$

因为 $e^{-\infty} = 0$，每个被禁止的 token 概率恰为 **0**——它在任何 temperature、top-p 或 top-k 下都不可能被采样。采到 token $t$ 后，自动机**推进** $s \to \delta(s, t)$，下一步用那个状态的掩码。因此最终字符串是自动机接受的一次游走——构造上就 schema 合法。

### 3.2 后端到底做了什么（及其代价）

真正难的工程是让掩码**便宜到能每步都算**却不拖住 GPU。这正是后端做的：

- **`xgrammar`**（vLLM 默认，由 `auto` 设置选中）为每个语法状态预计算允许 token 集合，于是每步掩码只是一次快速查表、与 GPU 计算重叠——设计上在常见情形近乎零延迟。
- **`guidance`** 是可选后端，语法/特性覆盖不同。

要知道两项代价：(1) 把 schema/语法编译成自动机的**一次性编译**（在所有用该 schema 的请求间摊销，但对一个全新 schema 可能表现为首 token 延迟）；(2) 每步掩码，好后端把它压到可忽略。这就是为什么你*设置* schema，而非逐 token 重新推导。

### 3.3 四种约束类型（以及你必须知道的那次弃用）

vLLM 通过 `StructuredOutputsParams`（离线）/ `structured_outputs` 请求字段（在线）暴露 structured outputs，选项如下：

- **`json`** — 一个 JSON Schema（或一个 Pydantic 模型的 `.model_json_schema()`）；工具调用与结构化抽取的主力。
- **`regex`** — 一个正则（电话号、日期、固定模板）。注意正则*风味*取决于后端——xgrammar/guidance 用 Rust 风格正则。
- **`choice`** — 一个固定的允许字符串列表，即分类/枚举约束。
- **`grammar`** — 一个完整的 EBNF 上下文无关文法，用于 DSL 和 SQL 子集。
- **`structural_tag`** — 用于约束带标签的区域（如工具调用块）。

!!! warning "API 改名——别用旧字段"
    旧的 `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` 字段已在 vLLM 0.12.0 **弃用并移除**。在 0.26.0 基线上你**必须**用 `structured_outputs`（在线）/ `StructuredOutputsParams`（离线）。面试与读源码提示：看到 `guided_*` 就说明这代码早于 0.12.0。

## 4 · 完整可跑代码 + 逐行讲解

离线 structured decoding，四种常见约束全覆盖，用 vLLM 0.26.0 的确切 API。JSON schema 由一个 Pydantic 模型给出；其余内联。

```python title="structured_decoding_offline.py"
# API 经 vLLM 0.26.0 核实（StructuredOutputsParams、SamplingParams.structured_outputs）。
from enum import Enum
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")          # 任何基座；structured outputs 在解码侧

# (a) 用 Pydantic schema 约束的 JSON -----------------------------------------
class Sentiment(str, Enum):
    positive = "positive"; negative = "negative"; neutral = "neutral"
class Review(BaseModel):
    sentiment: Sentiment
    score: float
json_so = StructuredOutputsParams(json=Review.model_json_schema())   # schema → 语法
out = llm.generate("Rate: 'vLLM is wonderful!'",
                   SamplingParams(temperature=0, max_tokens=64, structured_outputs=json_so))
print(out[0].outputs[0].text)     # 例如 {"sentiment": "positive", "score": 0.95}

# (b) choice —— 固定枚举（分类）---------------------------------------------
choice_so = StructuredOutputsParams(choice=["Positive", "Negative"])
out = llm.generate("Classify this sentiment: vLLM is wonderful!",
                   SamplingParams(structured_outputs=choice_so))
print(out[0].outputs[0].text)     # 恰好是 "Positive" 或 "Negative"——别的都不可能

# (c) regex —— 固定模板 ------------------------------------------------------
regex_so = StructuredOutputsParams(regex=r"\d{3}-\d{3}-\d{4}")
out = llm.generate("Give me a fake US phone number:",
                   SamplingParams(structured_outputs=regex_so))
print(out[0].outputs[0].text)     # 匹配 \d{3}-\d{3}-\d{4}，例如 415-555-0132

# (d) grammar —— EBNF 上下文无关文法 ----------------------------------------
sql = r'''
root        ::= "SELECT " column " FROM " table
column      ::= "name " | "id "
table       ::= "users " | "airports "
'''
grammar_so = StructuredOutputsParams(grammar=sql)
out = llm.generate("Show all airport names.",
                   SamplingParams(structured_outputs=grammar_so))
print(out[0].outputs[0].text)     # 该文法接受的字符串，例如 SELECT name FROM airports
```

**逐行讲解：**

- `StructuredOutputsParams(json=Review.model_json_schema())` — Pydantic 的 `.model_json_schema()` 把 `Review` 类变成一个 JSON Schema 字典；后端把它编译成语法。输出保证 `sentiment` ∈ 那个枚举、`score` 是数字——它*不可能*缺字段或拼错 key。
- `SamplingParams(..., structured_outputs=json_so)` — 这是 0.26.0 的接法：约束搭在 `SamplingParams` *内部*，而不是作为顶层 `guided_json=` 参数（那是被移除的 API）。
- **(b) `choice=[...]`** 编译成「整个输出必须恰好是这些字符串之一」——模型只能走到 `Positive` 或 `Negative`；不存在通往别处的路径，因此无需事后解析。
- **(c) `regex=r"\d{3}-\d{3}-\d{4}"`** 强制数字-横杠模板。每步只允许数字（或在正确位置的字面 `-`）；模型无法跑偏格式。（后端正则风味是 Rust 风格。）
- **(d) `grammar=sql`** 是一个 EBNF 文法：`root` 必须是 `SELECT <column> FROM <table>`，`column`/`table` 取自固定集合。输出永远是*这门*小语言里格式良好的查询——模型挑*哪些*列/表，文法保证*形式*。
- (a) 里的 `temperature=0` 让 JSON 演示确定；约束与采样无关——即便高 temperature，被禁 token 仍保持概率 0。

概念性输出（示例）：

```text
{"sentiment": "positive", "score": 0.95}
Positive
415-555-0132
SELECT name FROM airports
```

每一行都一次就能解析/匹配——没有「好的，这是……」的开场白、没有尾逗号、没有代码围栏。那份可靠性就是产品。

## 5 · Lab —— 在 OpenAI 端点上强制一个 schema

!!! gpu "GPU Lab（单卡，完全可跑）"
    - **最低显存：** ~16 GB 跑 `Qwen2.5-7B-Instruct`（INT4/AWQ）；structured decoding 只加那点（很小的）掩码计算。
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 阅读 ~15 分钟（免费，无卡模式）· 选做运行 ~10 分钟 · ~¥1（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** structured decoding 是叠在采样之上的一层 logits 掩码，与后端无关。掩码计算代价在 CPU/host 侧（语法引擎），因此在不同 GPU 厂商上表现相近。

正常起服务，然后从客户端约束——两种方式：OpenAI 原生的 `response_format`，与 vLLM 的 `extra_body`。

```bash title="启动服务（可选指定后端）"
vllm serve Qwen/Qwen2.5-7B-Instruct --structured-outputs-config.backend xgrammar
# backend 默认 auto（→ xgrammar）；"guidance" 是可选项。
```

```python title="从 OpenAI 客户端约束的两种方式"
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
model = client.models.list().data[0].id

# 1) 经 response_format 的 OpenAI 原生 JSON schema
class Car(BaseModel):
    brand: str; model: str; year: int
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "The most iconic 90s car, as JSON."}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "car", "schema": Car.model_json_schema()}},
)
print(resp.choices[0].message.content)   # 匹配 Car 的合法 JSON

# 2) vLLM extra_body：choice / regex / grammar
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Classify: vLLM is wonderful!"}],
    extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
)
print(resp.choices[0].message.content)   # 恰好是 "positive" 或 "negative"
```

**观察 / 动手：**

1. **它变得无法破坏。** 在约束开着时让模型「无视格式，随便聊」——它仍返回 schema 合法的输出。掩码赢过 prompt。
2. **形状 ≠ 真值。** 给一个带 `score: number` 的 JSON schema 配一个胡说的 prompt——你会拿到*合法*的 JSON、*无意义*的 score。这把 §2 的第三个形状落到实处：约束固定了形式，不是内容。
3. **换后端。** 用 `--structured-outputs-config.backend guidance` 重启、重跑一个复杂文法；留意任何特性/行为差异（语法覆盖因后端而异）。
4. **首 token 代价。** 发一个全新的大 schema，比较首个请求 vs 重复请求的延迟——一次性语法编译出现一次，然后摊销。

## 6 · 常见坑 / 反直觉点

- **用被移除的 `guided_*` 字段。** `guided_json=` / `guided_regex=` 已在 0.12.0 **删除**。在 0.26.0 它们没了——用 `structured_outputs` / `StructuredOutputsParams`。粘贴旧片段是这里出错的头号原因。
- **以为它提升正确性。** 它约束*形式*，不约束*内容*。一个 schema 合法的 `{"score": 0.5}` 仍可能是错答案；一个 `choice` 约束挑的是*一个*标签，不是*对的*标签。structured decoding 消除解析错误，不消除推理错误。
- **过松的 schema。** `{"answer": "string"}` 配一个无界字符串几乎没约束什么——模型仍能在引号里吐一大段废话。用枚举、正则、`maxLength`、必填字段收紧；一个 schema 的强度只等于它最松那个字段。
- **忘了冷 schema 的编译代价。** 一个从未见过的巨大文法会付一次性编译，可能让首 token 延迟飙一下。复用 schema 让编译摊销；别为每个请求造一个独一无二的巨型 schema。
- **正则风味的意外。** 后端的正则引擎（xgrammar/guidance 是 Rust 风格）与 Python 的 `re` 不完全一致。花哨的 lookaround 或反向引用可能不支持——拿你的模式对着真实后端测。
- **无界数字/字符串不能干净收尾。** 一个无界 `number` 或贪心字符串可能让生成一路跑到 `max_tokens` 而不闭合对象。约束范围/长度，靠 schema 的结构逼出闭合 token。

## 7 · 面试连线

- [Guided / structured decoding：把 token 掩码到 schema](../interview/structured-decoding.md) — 本课为你准备的高频题：*schema 如何变成每步 logit 掩码、为什么保证是硬的而非统计的、代价模型、以及为什么它固定形状却从不固定真值。*

## 8 · 小结 & 延伸阅读

**一句话：** structured decoding 把 JSON schema / 正则 / 语法 / 枚举编译成一个状态机，它在每个 decode 步产出一个 token 掩码；vLLM 把 $\log m$ 加到 logits 上，使被禁 token 概率恰为 0（logit 为 $-\infty$）、只有 schema 合法的 token 能被采样——通过 `structured_outputs` 字段（`StructuredOutputsParams`，后端 `xgrammar`/`guidance`；旧的 `guided_*` API 已在 0.12.0 移除）让输出在**构造上**合法——但它保证*形状*，从不保证*真值*。

延伸阅读：

- vLLM `docs/features/structured_outputs.md` — 这里引用的 `StructuredOutputsParams` 选项、`response_format`、与 `--structured-outputs-config.backend` flag。
- **xgrammar** — 默认后端；其逐状态 token 掩码预计算正是让掩码近乎免费的东西。
- **Outlines** / **Guidance** — 更广的受约束生成生态，以及 FSM/正则到掩码的思路。
- [continuous batching 课](../part5/continuous-batching.md) — structured decoding 跑在每个序列的 decode 步*内部*，因此它与 Part 5 的一切组合。

## 9 · 自测小问

??? question "你 prompt 模型「只回复 JSON」，97% 的时候管用。structured decoding 号称 100%。它在 token 层面做了什么不同，让保证是硬的而非统计的？"
    prompt 只是*偏置*了分布——模型仍*能*吐「Sure!」或尾逗号，只是概率更低，所以规模上会漏几个百分点。structured decoding 改变的是*可能性*，而非仅仅*可能性大小*。它把 schema 编译成状态机，在**每个 decode 步**构造一个当前状态允许 token 的掩码 $m^{(s)}$，然后在 softmax **之前**把被禁 token 的 logits 设成 $-\infty$。因为 $e^{-\infty}=0$，被禁 token 概率*恰*为 0——在任何 temperature/top-p 下都无法被采样。模型对非法输出根本没有路径，所以合法性由构造保证，而非靠高概率的轻推。

??? question "同事说「我们开了 JSON schema 解码，所以模型的答案现在正确了。」概念错误在哪，举一个它无法阻止的具体失败。"
    错误在把**形式**与**真值**混为一谈。structured decoding 约束输出的*形状*——它保证匹配 schema 的合法 JSON——但对模型的*推理*或*知识*毫无作用。掩码只移除会破坏语法的 token；在 schema 合法的 token 之间，模型仍自由选择。具体失败：schema 为 `{"capital": "string", "population": number}`，问法国首都可能得到完全合法、完全错误的 `{"capital": "Lyon", "population": 999}`。它能解析、匹配 schema、且是假的。structured decoding 消除*解析*错误，不消除*内容*错误——值要单独校验。

??? question "在 vLLM 0.26.0 基线上，一段粘来的代码写了 `SamplingParams(guided_json=schema)` 并报错。为什么？给出正确写法并说出运行它的后端。"
    `guided_json`（以及 `guided_regex`/`guided_choice`/`guided_grammar`）已在 vLLM 0.12.0 **弃用并移除**——在 0.26.0 该字段不再存在，所以调用报错。当前 API 把约束放进一个 `StructuredOutputsParams`，经 `SamplingParams.structured_outputs` 传入：
    ```python
    from vllm.sampling_params import StructuredOutputsParams
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    ```
    （在线：`extra_body={"structured_outputs": {"json": schema}}` 或 `response_format={"type":"json_schema", ...}`）。默认后端是 **`xgrammar`**（由 `auto` 设置选中；`guidance` 是可选项，用 `--structured-outputs-config.backend` 设）。代码里看到 `guided_*` 是它早于 0.12.0 的可靠信号。
