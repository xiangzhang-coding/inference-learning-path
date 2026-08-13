# Routing, autoscaling & KV-aware routing

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [Routing, Autoscaling & KV-Aware Routing (Multi-Instance)](../part8/routing-autoscaling.md)

---

## Q: You've scaled past one instance. Why does prefix-aware routing beat round-robin, why do you autoscale on the request queue rather than GPU utilization, and how do cold-start and drain shape a safe scaling policy?

### Direct answer

**Prefix-aware > round-robin** because **caches are per-instance**: each replica has its own KV / [prefix cache](../part5/prefix-caching.md), invisible to the others. Round-robin scatters requests that share a long prompt (system prompt, RAG preamble, conversation), so *every* replica re-runs the shared **prefill** — a cache **miss** each time. **KV-cache-aware (prefix-aware) routing** hashes the prefix and sends the request to the replica that already holds it → prefill skipped, **TTFT drops**. vLLM's DP docs say it directly: independent per-engine KV caches mean intelligent routing maximizes prefix-caching benefit.

**Autoscale on the queue** (`vllm:num_requests_waiting`), **not GPU utilization**. Decode is [memory-bound](../part0/inference-flow.md): a replica can show high util with batch headroom, or modest util while its KV cache is full and queuing. Queue depth means the same thing for any workload — >0 and rising = past the knee → add replicas. (SkyPilot's `target_qps_per_replica`, set to the [measured knee](load-testing-knee.md), is an acceptable coarser proxy.)

**Cold start** — a new replica takes tens of seconds to load weights + warm CUDA graphs, so scale-up **lags**: scale early / keep headroom. **Drain** — before scale-down, stop routing to the pod and wait until `num_requests_running` **and** `num_requests_waiting` hit 0, or you drop in-flight requests.

### Deep dive

- **Routing is cache placement.** Because nothing is shared across replicas by default, *where* a request goes decides hit vs miss. Prefix-aware routing turns placement into a prefill-avoidance win.
- **Hotspot trade-off.** Pure prefix-affinity can pile a popular prefix onto one replica; real routers **blend** affinity with load-balancing.
- **Shared KV tier.** The production stack can offload KV to **LMCache**, so a cross-replica miss is a *fetch* rather than a full recompute.
- **Why not util.** It conflates compute- and memory-bound regimes; the knee is a queueing phenomenon, so the queue is the faithful signal.
- **Readiness ≠ liveness.** `/health` is liveness; a real readiness probe (a tiny generation request) proves the pod can serve before the LB sends traffic.

### Code

```yaml
# production stack (Helm): 2 replicas behind a prefix-aware router (field names illustrative — see the chart's values.yaml)
routerSpec: { routingLogic: "prefixaware" }        # round-robin would re-prefill shared prompts
servingEngineSpec: { modelSpec: [{ modelURL: "Qwen/Qwen2.5-7B-Instruct", replicaCount: 2 }] }
---
# HPA: scale on the QUEUE, not GPU util
metrics: [{ type: Pods, pods: { metric: { name: vllm_num_requests_waiting },
            target: { type: AverageValue, averageValue: "5" } } }]   # avg queue > 5 → scale out
# SkyPilot alternative: replica_policy.target_qps_per_replica = the measured per-instance knee
```

### Interviewer follow-ups

- *"4 replicas, round-robin, shared 2k-token system prompt, high TTFT — why?"* → Each replica re-prefills the shared prefix (independent caches); round-robin defeats the prefix cache. Fix: prefix-aware routing.
- *"Autoscale on GPU util > 80% — good idea?"* → No; decode is memory-bound, util doesn't track the knee. Use `num_requests_waiting`.
- *"Are caches shared across replicas?"* → No, per-instance. Cross-replica reuse needs a shared KV tier (LMCache).
- *"Scale-up fires but SLO still misses during the spike — why?"* → Cold-start lag (weights + CUDA-graph warmup). Scale early / pre-warm / size min-replicas.
- *"Scale-down drops requests — fix?"* → Drain first: stop routing, wait until running and waiting both hit 0, then terminate.
- *"Downside of pure prefix-affinity?"* → Hotspots on popular prefixes; blend with load-balancing.

### Linked concepts

- Lesson: [Routing, Autoscaling & KV-Aware Routing (Multi-Instance)](../part8/routing-autoscaling.md)
- Related: [Prefix caching](prefix-caching.md) (the per-instance win routing preserves), [Load-testing & the concurrency knee](load-testing-knee.md) (the per-instance ceiling you scale past), [Serving over HTTP: the OpenAI-compatible server](openai-server-deployment.md) (the `/metrics` and `/health` signals routing/scaling read)
- Glossary: [KV-cache aware routing, Knee, SLO](../glossary.md)
