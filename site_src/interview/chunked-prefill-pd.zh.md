# Chunked prefill 与 PD 分离：平衡 TTFT 与吞吐

!!! info "基线：**vLLM 0.26.0** · flag 经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [调度器：Chunked Prefill 与 PD 分离](../part5/scheduler-chunked-prefill-pd.md)

---

## Q：其他请求流式时来了个长 prompt，会出什么问题？chunked prefill 如何修？`max_num_batched_tokens` 换什么？何时改用 PD 分离？

### 直接答案

[prefill 受算力约束、decode 受带宽约束](../part0/inference-flow.md)，它们抢一块 GPU。长 prompt 的 prefill 若作为一整步跑，会**霸占 GPU**——每个已在运行序列的 decode 停摆，它们的 inter-token latency（ITL）飙升（流式明显冻住）。

**Chunked prefill**（`enable_chunked_prefill=True`，默认）把长 prefill 切成块，在一步的 `max_num_batched_tokens` 预算内与进行中的 decode *一起*调度。decode 每步前进（ITL 平滑）；新请求首 token 稍晚（TTFT 略高）。这就是权衡。

**`max_num_batched_tokens`** 是旋钮：**更低** → 每步 prefill 更少 → ITL 更好、TTFT 更差；**更高**（文档建议 >8192 提吞吐）→ 每步 prefill 更多 → TTFT 更好、decode 干扰更多。

**PD 分离**把同一个「别混两阶段」逻辑跨 GPU：prefill 跑 producer 池、decode 跑 consumer 池（`--kv-transfer-config`，NixlConnector `kv_producer`/`kv_consumer`），把 KV cache 在其间流动——于是每个池为自己的瓶颈调优与扩缩。它是多实例、大规模技术；chunked prefill 是同一想法的单 GPU 版。

### 深入原理

- **一步是 token 预算，不是槽位数。** prefill 与 decode token 从同一 `max_num_batched_tokens` 取；chunked prefill 只是让 prefill 拿*一部分*。vLLM 默认策略优先 decode 以护 ITL。
- **Chunked prefill 不加速 prefill**——它可能让单个 prefill 略慢（多几步）。它改善*系统*：decode 不再停摆 → 总 ITL/吞吐更好。
- **禁用 chunked prefill 的注意。** 若 `enable_chunked_prefill=False`，`max_num_batched_tokens` 必须超过 `max_model_len`（整个 prompt 要放进一步），否则服务起不来。
- **PD 的代价。** 每请求一次 KV-cache 传输（网络带宽 + 延迟）加运维复杂度（producer/consumer 实例 + 路由 proxy）。只在两池独立扩缩/调优盖过传输代价时才值。

### 代码

调度权衡的纯 Python——有/无分块的 decode 停摆：

```python
BUDGET, DECODES, PREFILL = 16, 4, 48
def without_chunking(b, d, p):                      # prefill 独占；decode 挨饿
    steps = delayed = 0
    while p > 0: p -= min(b, p); steps += 1; delayed += d
    return steps, delayed
def with_chunking(b, d, p):                         # decode + prefill 块共享每步
    steps = delayed = 0; chunk = b - d
    while p > 0: p -= min(chunk, p); steps += 1
    return steps, delayed
print(without_chunking(BUDGET, DECODES, PREFILL))   # (3, 12)：早 1 步，冻住 12 个 decode-token
print(with_chunking(BUDGET, DECODES, PREFILL))      # (4, 0) ：+1 步 TTFT，0 decode 停摆
```

### 面试官追问

- *「你 ITL-bound——旋钮往哪调？」* → **调低** `max_num_batched_tokens`：每步 prefill 干扰更少、ITL 更平滑（代价是 TTFT）。
- *「TTFT-bound？」* → **调高**（朝 8192+）：每步更多 prefill、首 token 更快（代价是运行流 ITL）。
- *「为何不总是分离？」* → PD 需 ≥2 实例、每请求付一次 KV 传输；单 GPU 上没什么可分，小规模时传输/运维代价盖过收益。
- *「Chunked prefill vs prefix caching？」* → Chunked prefill 把*一个* prefill 跨步切分；prefix caching 为*共享*前缀跳过 prefill。不同杠杆，常叠加。
- *「两者共享的根本想法？」* → prefill 与 decode 想要的不同（算力 vs 带宽）；chunked prefill 在一块 GPU 上分时，PD 分到不同 GPU 上分空间。

### 关联概念

- 课程：[调度器：Chunked Prefill 与 PD 分离](../part5/scheduler-chunked-prefill-pd.md)
- 相关：[Static vs continuous batching](continuous-batching.md)（这个塑形的运行集）、[Prefix caching](prefix-caching.md)（姊妹 prefill 杠杆）、[Prefill vs decode](prefill-vs-decode.md)（算力/带宽之分）、[延迟与吞吐度量](latency-throughput-metrics.md)（TTFT/ITL）
- 术语：[Chunked prefill、PD disaggregation](../glossary.md)
