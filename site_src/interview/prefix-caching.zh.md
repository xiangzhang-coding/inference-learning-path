# Prefix caching：复用共享前缀 KV

!!! info "基线：**vLLM 0.26.0** · 设计经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [Prefix Caching：复用共享前缀的 KV](../part5/prefix-caching.md)

---

## Q：什么是 prefix caching？vLLM 如何保证绝不复用错误的 KV？何时有用？为何不改变输出？

### 直接答案

Prefix caching 复用**共享前缀**的 [KV cache](../part0/kv-cache.md)，让它算一次而非每请求重 prefill。真实流量不断共享前缀——system prompt、few-shot 示例、多轮对话历史、RAG 文档——prefix caching 把那种重复变成跳过的 prefill。

**正确性来自对[块](../part5/paged-attention.md)做内容哈希。** 每个 KV 块的哈希把**它的 token *和父块的哈希***折进去（一条链），所以缓存块只在通向它的*整个*前缀逐字节相同时才匹配——匹配不会来自不同上下文。**只有整块缓存**（半满尾块依赖未来 token）。复用的 KV 恰是 prefill 会产出的，所以**输出不变**——纯粹的「别重做」优化。

**何时有用：** 高「前缀长度 × 命中率」。每个请求都带的 500-token system prompt，从第一个之后每请求省 ~500 prefill token（更低 TTFT、腾出容量）。无共享前缀的独特 prompt 一无所获。

### 深入原理

- **服务命中。** 块池保留 `cached_block_hash_to_block` 映射；匹配时 manager `touch()` 该块（加 `ref_cnt`、从驱逐队列救回）并把新请求 block table 指向它——prefill 从第一个*未*缓存块开始。
- **不花额外内存。** 缓存但未引用的块只是有压力时被回收的驱逐候选（经空闲队列顺序 LRU）；稳态内存不变。
- **V1 默认开启。** `enable_prefix_caching`；`prefix_caching_hash_algo` 默认 `"sha256"`。
- **顺序要紧。** 因为哈希链从 token 0 开始，共享前缀*之前*的可变内容（时间戳、用户 id、打乱的 few-shot）杀掉所有命中。稳定前缀在前，可变后缀在后。

### 代码

省功的纯算术——有/无缓存的 prefill token：

```python
BLOCK, PREFIX, SUFFIXES = 16, 512, [24, 40, 8, 32, 16, 48, 12, 20]
def prefill_tokens(prefix, suffixes, block, cache):
    cached = total = 0
    for s in suffixes:
        total += (prefix - (cached if cache else 0)) + s
        if cache: cached = (prefix // block) * block     # 前缀整块现已缓存
    return total
print(prefill_tokens(PREFIX, SUFFIXES, BLOCK, False))    # 4296 —— 前缀重算 8 次
print(prefill_tokens(PREFIX, SUFFIXES, BLOCK, True))     # 712  —— 前缀算一次（~83% 更少）
```

### 面试官追问

- *「如何防止假匹配？」* → 块哈希链入父哈希，所以一个块只在整个前置前缀相同时才匹配；早处一个 token 不同就改变下游每个哈希。
- *「它改变输出吗？」* → 不——复用的 KV 与重算的逐字节相同。不同输出 = bug。
- *「它花内存吗？」* → 无净代价——复用块池；未引用的缓存块是驱逐候选。
- *「为何在 round-robin LB 后命中率保持 ~0？」* → 命中只在持有块的副本上有用。用 **KV-cache-aware routing** 把共享前缀请求送到同一副本。
- *「什么悄悄杀掉命中？」* → 任何前缀变化（时间戳、空格、重排 few-shot）或把可变内容放最前；还有半满（非整）块。

### 关联概念

- 课程：[Prefix Caching](../part5/prefix-caching.md)
- 相关：[PagedAttention：block manager 与碎片](kv-cache-block-manager.md)（它依托的块/`ref_cnt`/`touch()`）、[Chunked prefill 与 PD](chunked-prefill-pd.md)（姊妹 prefill 杠杆）、[Static vs continuous batching](continuous-batching.md)（腾出的容量喂给谁）
- 术语：[Prefix caching、KV-cache aware routing](../glossary.md)
