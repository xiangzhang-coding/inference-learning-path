# KV 缓存与吞吐上限

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 0 · 基础   ·   **考察课程：** [KV 缓存](../part0/kv-cache.md)

---

## 问：在单张 24 GB GPU 上服务一个 7B 模型，为什么限制并发请求数的通常是 KV 缓存——而非模型权重或纯算力？

### 直接答案

权重是**固定、一次性**的显存成本；KV 缓存是随 `batch × sequence_length` 增长的**按序列**成本。权重加载后，**剩余** VRAM 是一个固定预算，每个并发请求按其上下文长度成比例地吃掉它。于是并发被「剩余显存能塞下多少条序列的 KV 缓存」卡住——这是一个**显存带宽与容量**问题，不是算力问题。加之 decode 本身也是 memory-bound（每步为极少计算从 HBM 重读整个缓存），所以你会在撞上 FLOPs 墙之前，先撞上显存墙。

### 深入原理

- **算术。** KV bytes $= 2 \times L \times n_{\text{kv}} \times d_h \times b_{\text{dtype}} \times S \times B$。对 `Qwen2.5-7B-Instruct`（28 层、4 KV 头、head_dim 128、BF16）就是 **56 KiB/token**——即一条 8192-token 序列约 0.44 GiB，满 32k 上下文约 1.75 GiB。
- **为什么不是权重？** ~14 GiB（BF16）或 ~5–6 GiB（AWQ 4-bit）——只付*一次*，不随负载增长。
- **为什么不是算力？** Decode 每步只做一个 token 的运算，却重读整个 KV 缓存。算术强度低 → 带宽受限。加 FLOPs 余量没用；加 KV 容量（或带宽）才有用。
- **对优化的含义。** 一切提吞吐的手段本质都是「塞下/服务更多 KV」：量化权重（腾 VRAM 给 KV）、`kv_cache_dtype=fp8`（KV 字节减半）、PagedAttention（消除碎片）、continuous batching（把 KV 预算填满）、prefix caching（避免重复 KV）。

### 代码

租 GPU 之前先把上限推出来：

```python
# Qwen2.5-7B-Instruct 单 token KV（BF16）：2 * 28 * 4 * 128 * 2 = 57344 字节 = 56 KiB
kib_per_token = 2 * 28 * 4 * 128 * 2 / 1024                 # 56.0

free_gib = 24 - 14                                          # 权重 ~14 GiB -> ~10 GiB 给 KV
seq_len = 8192
gib_per_seq = kib_per_token * seq_len / (1024 ** 2)         # ~0.44 GiB
max_concurrent = int(free_gib / gib_per_seq)                # ~22（示例，忽略激活）
print(kib_per_token, round(gib_per_seq, 2), max_concurrent)
```

### 面试官追问

- *「怎么把这个并发数翻三倍？」* → 缩小权重（AWQ/GPTQ，腾 VRAM）、缩小 KV（`kv_cache_dtype=fp8`；GQA 已经帮了 7×）、或把 `max_model_len` 压到工作负载真实的上下文。
- *「PagedAttention 在哪一环？」* → 它消除连续 KV 分配的*碎片*浪费，让更多剩余 VRAM 变成可用 KV——抬高的是有效容量，而非公式里的原始值。
- *「为什么把 `gpu_memory_utilization` 拉到 0.99 很危险？」* → 并发负载下激活会突增；余量太紧会在生产里 OOM，哪怕启动时没事。
- *「Prefill vs decode 哪个 memory-bound？」* → decode（memory-bound）；prefill 一次处理很多 token，通常 compute-bound。

### 关联知识点

- 课程：[KV 缓存](../part0/kv-cache.md)
- 术语表：[KV cache、GQA、Memory-bound、PagedAttention](../glossary.md)
