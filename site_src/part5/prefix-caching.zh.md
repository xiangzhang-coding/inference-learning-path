# Prefix Caching：复用共享前缀的 KV

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    经 Context7 针对 vLLM 0.26.0 核实（ADR-0004）：自动 prefix caching 把每个 KV **block** 用「其 token 的哈希**加上父块的哈希**」为键（所以位置有意义）、**只有整块可缓存**、复用**不改变输出**、`prefix_caching_hash_algo` 默认 **`"sha256"`**、由 `enable_prefix_caching` 控制（**V1 引擎默认开启**）。这是 [PagedAttention 课](paged-attention.md) 预告的块共享在 serving 侧的回报。§4 仿真是**省功模型，不是 benchmark**；加速为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

[PagedAttention 课](paged-attention.md) 结尾留了个引子：因为一个 KV [block](../part0/kv-cache.md) 由它持有的*内容*标识，两个以相同 token 开头的序列可以把 block table 指向**同一物理块**。Prefix caching 就是那个引子在生产里的样子——而且它常是你能部署的最便宜的大吞吐收益，因为大量真实流量**共享前缀**。

想想相同 token 一次次出现在哪：

- 一段 **system prompt**（「你是一个乐于助人的助手…[500 token 规则]」）前置到*每个*请求。
- 一批分类调用间共享的 **few-shot 示例**。
- 一段**多轮对话**，第 3 轮的 prompt 就是第 1–2 轮原文加新消息。
- 许多问题都问的一份长**文档**（RAG 风格）。

没有 prefix caching，那些请求里的每一个都对*整个*共享前缀重跑 [prefill](../part0/inference-flow.md)——重算相同的 KV，烧掉你已经花过的算力。Prefix caching 把共享前缀的 KV **算一次**、把那些块留在池里、让每个以相同 token 开头的后续请求**直接跳到它的独特后缀**。prefill 工作——以及 TTFT——坍缩到只剩新部分。因为复用的 KV 逐字节相同，输出完全一样；这是纯粹的「别重做」优化。→ 术语见 [Glossary](../glossary.md) 的 *Prefix caching、KV-cache aware routing*。

## 2 · 心智模型

同一前缀，算一次，按内容哈希复用：

```text
三个请求，都以同一段 512-token system prompt 开头：
  req A: [ SYSTEM PROMPT (512 tok) ][ "translate: hello"        ]
  req B: [ SYSTEM PROMPT (512 tok) ][ "summarize: the cat sat…" ]
  req C: [ SYSTEM PROMPT (512 tok) ][ "code: fizzbuzz"          ]

没有 prefix caching —— 各自 prefill 整段：
  A: prefill 512 + suffix   B: prefill 512 + suffix   C: prefill 512 + suffix
     └────────────── 512-token 前缀被算 3 次（每次 KV 相同） ─────┘

有 prefix caching —— 哈希块，复用匹配：
  A: prefill 512 + suffix  → 它的 32 个前缀块被缓存（token 哈希 + 父哈希）
  B: 块哈希与 A 匹配 → 把 block table 指向 A 的块（ref_cnt++），只 prefill suffix
  C: 同上 → 复用 A 的前缀块，只 prefill suffix
     └── 512-token 前缀只算一次；B、C 直接跳到各自 suffix ──┘

块哈希链（为何位置安全）：
  block0.hash = H(tokens[0:16])
  block1.hash = H(block0.hash, tokens[16:32])   ← 含父哈希 → 一个块只在
  block2.hash = H(block1.hash, tokens[32:48])     到它为止的整个前缀都相同时才匹配
```

三个要记的形状：

- **块是复用单位，且按内容为键。** 一个块的哈希把它的 token *和父块的哈希*都折进去，所以一个缓存块只在通向它的整个前缀都相同时才匹配——你永远不会误用来自不同上下文的 KV。**只有整块缓存**（半满的最后一块会重算）。
- **复用是免费的正确性。** 缓存的 KV 恰是 prefill 本会产出的，所以输出相同——prefix caching 只去掉冗余计算。若你开缓存后看到不同结果，那是 bug，不是特性。
- **收益随「前缀长度 × 共享率」放大。** 每个请求都共享的 500-token system prompt，从第一个之后每请求省 ~500 prefill token；5-token 的共享前缀几乎不省。它是流量形状的优化：前缀又长、命中又频时最高。

## 3 · 原理

### 3.1 哈希为键的块

vLLM 的自动 prefix caching（来自已核实的设计文档）「缓存已处理请求的 KV cache 块，为共享同一前缀的后续请求复用它们……不改变模型输出」。每个块得到一个哈希，由**它的 token、它父块的哈希**和少量元数据（LoRA id、多模态输入）算出。那个父链是关键的正确性巧招：块 *k* 只在块 0…*k* 全相同时才匹配，所以一次匹配就保证整个前置上下文相同。哈希算法是 `prefix_caching_hash_algo`（默认 `"sha256"`）。**只有整块**够格——半满尾块的 KV 依赖未来 token，还不能缓存。

### 3.2 命中如何服务

回忆[block manager](paged-attention.md)：池子保留一个 `cached_block_hash_to_block` 映射。新请求来时，引擎哈希 prompt 的块并查表：

- **命中：** 块已在池里。manager 调 `touch()` 把块的 `ref_cnt` 加一（它可能正躺在空闲队列里当驱逐候选），把新请求的 block table 指向它——**不重算**。prefill 从第一个*未*缓存块开始。
- **未命中：** 正常 prefill；产出的整块登记进哈希映射，供*未来*请求命中。

缓存块被引用计数，所以支撑某活跃前缀的块不会被驱逐；一旦没有请求引用它，它就成为驱逐候选（LRU 式，经空闲队列的驱逐顺序），其 VRAM 可被回收。所以稳态下 prefix caching **不花额外内存**——它复用同一块池，只是让有用的块多留一会儿。

### 3.3 开启它，并路由到命中

V1 引擎里 prefix caching **默认开启**（`enable_prefix_caching`）。值得知道的一个系统推论：命中只在请求落到**已持有那些块的实例**上才有用。在多副本规模上这催生 **KV-cache-aware routing**——把请求路由到最可能已缓存其前缀的副本（如按 system prompt 哈希），而非 round-robin。那是生产拓扑话题（Part 7/8），但它是本课的自然延伸：缓存制造命中，路由确保你落在命中上。

## 4 · 完整可跑代码 + 逐行讲解

一个纯 Python 模型，刻画有/无 prefix caching 时的 prefill 工作量，对一批共享 system prompt 的请求。它数计算的 prefill token——prefix caching 缩小的量。不用 GPU。

```python title="prefix_caching_sim.py"
"""Prefix caching：共享前缀的 KV 算一次，被后续请求复用。
省功模型，不是 benchmark。纯 Python、离线。"""
BLOCK = 16
PREFIX = 512                                   # 共享 system prompt / few-shot 前言（32 块）
SUFFIXES = [24, 40, 8, 32, 16, 48, 12, 20]     # 每个请求的独特尾巴（各异）

def prefill_tokens(prefix, suffixes, block, cache):
    """所有请求计算的 prefill token 总数，有或无 prefix caching。"""
    cached = 0                                  # 当前已缓存的前缀 token 数（只整块）
    total = 0
    for s in suffixes:
        reused = cached if cache else 0         # 命中复用已缓存的前缀块
        total += (prefix - reused) + s          # prefill 未缓存的前缀 + 本请求的 suffix
        if cache:
            cached = (prefix // block) * block  # 任一请求后，前缀的整块都已缓存
    return total

if __name__ == "__main__":
    no_cache   = prefill_tokens(PREFIX, SUFFIXES, BLOCK, cache=False)
    with_cache = prefill_tokens(PREFIX, SUFFIXES, BLOCK, cache=True)
    saved = 1 - with_cache / no_cache
    print(f"{len(SUFFIXES)} requests sharing a {PREFIX}-token prefix (block={BLOCK})")
    print(f"no prefix caching  : {no_cache} prefill tokens computed")
    print(f"with prefix caching: {with_cache} prefill tokens computed  ({saved:.1%} fewer)")
```

**逐行讲解：**

- `PREFIX` 是 512-token 的共享前言（system prompt / few-shot 块 = 32 个 16-token 块）；`SUFFIXES` 是每请求独特尾巴，故意各异。
- `prefill_tokens(..., cache=False)`——每个请求 prefill 完整 `prefix + suffix`；`reused` 恒为 0。前缀为全部 N 个请求重算。
- `prefill_tokens(..., cache=True)`——*第一个*请求填充缓存后，`cached` 变成前缀的整块数；每个后续请求减去 `reused`（跳过共享前缀）、只 prefill `(prefix - reused) + suffix`——即前缀完全缓存后只剩它的 suffix。`(prefix // block) * block` 建模**只整块**规则（非块对齐的前缀会留一小段未缓存）。
- 两次运行只差在是否减去已缓存的前缀 token——正是一次 prefix-cache 命中做的事。

预期输出（省功模型，不是 benchmark）：

```text
8 requests sharing a 512-token prefix (block=16)
no prefix caching  : 4296 prefill tokens computed
with prefix caching: 712 prefill tokens computed  (83.4% fewer)
```

八个请求、一段共享的 512-token system prompt：prefix caching 把前缀算**一次**而非八次，把 prefill 工作砍掉 **~83%**。那些算力不会凭空消失——它变成缓存请求更低的 TTFT 与[更多并发序列](continuous-batching.md)的腾出容量。省的量随「前缀长度 × 命中率」变化：前缀越长、越常共享，收益越大。（没有共享前缀的负载什么也拿不到——这是诚实的边界。）

## 5 · Lab——测你的命中率

!!! gpu "GPU Lab（单卡，完全可跑）"
    - **最低显存：** 读不需要；跑 `Qwen2.5-7B-Instruct`（INT4/AWQ）并观察缓存命中需 ~16 GB
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 读 ~15 分钟（免费，无卡模式）· 可选运行 ~10 分钟 · ~¥1（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** prefix caching 是 block-manager 特性，与后端无关——哈希为键的块复用在任何用 paged KV cache 的后端上都一样。

Prefix caching 默认开启；Lab 是关于*看到*它工作。

```python title="observe_prefix_cache.py"
# API 针对 vLLM 0.26.0 核实（LLM、enable_prefix_caching）。在有 GPU 的 AutoDL 上跑。
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="awq",
    enable_prefix_caching=True,       # V1 默认开启；这里显式写出以求清晰
)
SYSTEM = "You are a meticulous assistant. Follow the rules exactly.\n" * 20   # 一段长共享前缀
prompts = [SYSTEM + q for q in ["Translate 'hello' to French.",
                                "What is 2+2?",
                                "Name a primary color."]]
# 第一个请求填充前缀块；接下来两个应命中并跳过共享 prefill。
out = llm.generate(prompts, SamplingParams(max_tokens=16))
print([o.outputs[0].text[:30] for o in out])
```

**观察/动手：**

1. **看命中率。** vLLM 暴露 prefix-cache 命中指标（日志/metrics 里的 `gpu_prefix_cache_hit_rate`）。跑上面这批：第一个未命中（填充），其余命中。稍后再发*同一* system prompt——在那些块被驱逐前仍命中。
2. **改坏前缀，丢掉命中。** 对某个请求改 `SYSTEM` 开头**附近的一个 token**，看它的命中消失——块哈希链（父链）意味着早期改动使下游每个块失效。这让 §3.1 的「位置有意义」具象化。
3. **与无缓存对比。** 用 `enable_prefix_caching=False` 重跑，对比重复前缀请求的 TTFT——差额就是你之前在付的冗余 prefill。

## 6 · 常见坑 / 反直觉点

- **前缀不逐字节相同却指望命中。** 前缀里*任何一处*一个 token 不同——哪怕一个时间戳、打乱的 few-shot 顺序、尾随空格——都从那点起改变块哈希，杀掉下游一切的命中。让共享前缀真正恒定，把可变内容放*最后*。
- **把可变内容放最前。** 若独特部分（用户 id、时间戳）领头、共享 system prompt 在后，*什么都不缓存*——第一个块就已不同。顺序要紧：**稳定前缀在前，可变后缀在后。**
- **以为它改变输出。** 从不——复用的 KV 与重算的逐字节相同。Prefix caching 是纯省功；不同输出意味着 bug。
- **假设它花额外内存。** 它复用同一[块池](paged-attention.md)；缓存但未引用的块只是驱逐候选，有压力时被回收。稳态内存不变。
- **在低共享流量上高估它。** prompt 各异、无共享前缀时，prefix caching 几乎无用。它的价值完全是你流量前缀共享率的函数——先测命中率再声称收益。
- **多副本规模上忽略路由。** 命中只在持有块的副本上有用。round-robin 把请求打散、拉垮命中率；[KV-cache-aware routing](../glossary.md) 才能跨副本保住它。

## 7 · 面试连线

- [Prefix caching：复用共享前缀 KV](../interview/prefix-caching.md)——本课为你准备的高频题：*块哈希如何使安全复用成立、为何只整块缓存、何时有用、以及为何输出不变。*

## 8 · 小结 & 延伸阅读

**一句话：** 因为一个 KV 块由「它的 token *加上父块哈希*」为键，共享前缀的请求（system prompt、few-shot、多轮对话、RAG 文档）能复用同一物理块——于是 vLLM 把共享前缀的 KV 算一次、后续请求直接跳到独特后缀，砍掉 prefill 工作与 TTFT 且输出逐字节相同；收益随「前缀长度 × 命中率」放大，在多副本规模上 KV-cache-aware routing 才能让请求持续落在自己的缓存块上。

延伸阅读：

- vLLM `docs/design/prefix_caching.md`——此处引用的哈希块身份方案与只整块规则。
- [PagedAttention 课](paged-attention.md)——prefix caching 依托的 block manager 与 `ref_cnt`/`touch()` 机制。
- [调度器课](scheduler-chunked-prefill-pd.md)——chunked prefill（切分一个 prefill）是姊妹杠杆；prefix caching（跳过共享 prefill）常与它叠加。
- Part 7–8——KV-cache-aware routing 与多副本服务，命中率在此成为集群级议题。

## 9 · 自测小问

??? question "当两个 prompt 只是*看起来*相似时，vLLM 如何保证绝不复用错误的 KV？"
    每个 KV 块的哈希由**它自己的 token 加上父块的哈希**（一条链）算出，所以一个块只在它之前*每一个*块——即到那点为止的整个前缀——逐字节相同时才匹配缓存块。prompt 早处一个 token 不同就改变那个块的哈希，并通过父链改变下游每个块的哈希，所以不会有假匹配。此外，**只有整块可缓存**（半满尾块依赖尚未见到的 token），且复用的 KV 恰是 prefill 会算出的——所以命中可证安全、输出不变。

??? question "你的负载是聊天 API，每个请求带 600-token system prompt。估算 prefill 节省，并说出会悄悄毁掉它的那一件事。"
    600-token 共享前缀下，prefix caching 把它算**一次**而非每请求；第一个请求后，每个后续请求跳过 ~600 prefill token（减去任何非块对齐余量），只 prefill 它独特的用户轮——对前缀重的流量常是 **80–95% 的 prefill token 削减**（示例），表现为更低 TTFT 与更多并发的腾出容量。悄悄的杀手：把**可变内容放到 system prompt 之前**（或让前缀里任何 token 变化——时间戳、重排的 few-shot、改动的空格）。因为块哈希链从 token 0 开始，任何早期变化都改变所有下游哈希、把命中率降到 ~0。修法：让前缀恒定且逐字节相同，可变内容在最后。

??? question "你在 8 个副本、round-robin 负载均衡后开启 prefix caching，却几乎看不到命中率提升。为什么？怎么修？"
    prefix-cache 命中只在**真正持有那些块的副本**上有用。round-robin 把同前缀的请求打散到全部 8 个副本，所以每个副本只看到 ~1/8 的重复，其缓存在被驱逐前很少拿到第二次命中——聚合命中率保持低。修法是 **KV-cache-aware routing**：把请求路由到最可能已持有其前缀的副本（如把 system prompt / 会话 id 哈希到一个粘性副本），让重复前缀集中到同一实例、真正命中。缓存制造机会；路由在集群规模上实现它。
