# 路由、自动扩缩与 KV 感知路由

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 8 · 生产部署与系统设计   ·   **对应课程：** [路由、自动扩缩与 KV 感知路由（多实例）](../part8/routing-autoscaling.md)

---

## Q：你已经扩到一个实例之外。为什么前缀感知路由胜过 round-robin？为什么按请求队列而非 GPU 利用率自动扩缩？冷启动与排空怎么塑造一个安全的扩缩策略？

### 直接答案

**前缀感知 > round-robin**，因为**缓存是每实例的**：每个副本有自己的 KV / [prefix cache](../part5/prefix-caching.md)，对其他副本不可见。round-robin 打散共享长 prompt（system prompt、RAG 前言、对话）的请求，于是*每个*副本都重跑共享 **prefill**——每次一次缓存**未命中**。**KV 感知（前缀感知）路由**哈希前缀、把请求发给已持它的副本 → prefill 跳过、**TTFT 降**。vLLM 的 DP 文档直说：独立的每引擎 KV cache 意味智能路由能最大化 prefix-caching 收益。

**按队列自动扩缩**（`vllm:num_requests_waiting`），**不按 GPU 利用率**。decode 是 [memory-bound](../part0/inference-flow.md) 的：一个副本可以显示高利用却有 batch 余量，或适中利用却 KV cache 满、在排队。队列深度对任何 workload 都一个意思——>0 且涨 = 过了 knee → 加副本。（SkyPilot 的 `target_qps_per_replica`、设到[测出的 knee](load-testing-knee.md)，是可接受的更粗代理。）

**冷启动**——新副本要数十秒加载权重 + 热 CUDA graphs，所以扩容**滞后**：趁早扩 / 留余量。**排空**——缩容前，停止路由给该 pod、等到 `num_requests_running` **和** `num_requests_waiting` 都到 0，否则丢在途请求。

### 深入原理

- **路由是缓存落点。** 因为默认副本间什么都不共享，请求发*去哪*决定命中还是未命中。前缀感知路由把落点变成一个避免 prefill 的收益。
- **热点权衡。** 纯前缀亲和会把一个流行前缀堆到一个副本上；真实 router **揉合**亲和与负载均衡。
- **共享 KV 层。** production stack 能把 KV offload 到 **LMCache**，让跨副本未命中是一次*取*、而非完整重算。
- **为何不用利用率。** 它混淆 compute- 与 memory-bound 状态；knee 是排队现象，所以队列才是忠实信号。
- **就绪 ≠ 存活。** `/health` 是存活性；一个真实就绪探针（极小生成请求）在 LB 发流量前证明 pod 能服务。

### 代码

```yaml
# production stack (Helm)：2 副本挂在前缀感知 router 后（字段名为示意——见 chart 的 values.yaml）
routerSpec: { routingLogic: "prefixaware" }        # round-robin 会重 prefill 共享 prompt
servingEngineSpec: { modelSpec: [{ modelURL: "Qwen/Qwen2.5-7B-Instruct", replicaCount: 2 }] }
---
# HPA：按队列扩缩，不按 GPU 利用率
metrics: [{ type: Pods, pods: { metric: { name: vllm_num_requests_waiting },
            target: { type: AverageValue, averageValue: "5" } } }]   # 平均队列 > 5 → 扩容
# SkyPilot 替代：replica_policy.target_qps_per_replica = 测出的每实例 knee
```

### 面试官追问

- *「4 副本、round-robin、共享 2k-token system prompt、TTFT 高——为什么？」* → 每副本重 prefill 共享前缀（独立缓存）；round-robin 打败 prefix cache。修：前缀感知路由。
- *「按 GPU 利用率 > 80% 扩缩——好主意？」* → 不；decode 是 memory-bound，利用率不跟踪 knee。用 `num_requests_waiting`。
- *「缓存跨副本共享吗？」* → 不，每实例。跨副本复用需共享 KV 层（LMCache）。
- *「扩容触发了但飙升期 SLO 仍违反——为什么？」* → 冷启动滞后（权重 + CUDA-graph 热身）。趁早扩 / 预热 / 定好 min-replicas。
- *「缩容丢请求——怎么修？」* → 先排空：停止路由、等 running 与 waiting 都到 0，再终止。
- *「纯前缀亲和的坏处？」* → 流行前缀热点；与负载均衡揉合。

### 关联知识点

- 课程：[路由、自动扩缩与 KV 感知路由（多实例）](../part8/routing-autoscaling.md)
- 相关：[Prefix caching](prefix-caching.md)（路由保住的每实例收益）、[压测与并发拐点](load-testing-knee.md)（你扩过去的每实例天花板）、[HTTP 服务化：OpenAI 兼容 server](openai-server-deployment.md)（路由/扩缩读的 `/metrics` 与 `/health` 信号）
- 术语表：[KV 感知路由、Knee、SLO](../glossary.md)
