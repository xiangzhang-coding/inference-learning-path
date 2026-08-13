# Routing, Autoscaling & KV-Aware Routing (Multi-Instance)

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · multi-instance (single- or multi-GPU)"
    Verified against vLLM 0.26.0 via Context7 (ADR-0004): to scale past one instance vLLM points you at the **production stack** (`helm repo add vllm https://vllm-project.github.io/production-stack`, `helm install vllm vllm/vllm-stack -f values.yaml`), which deploys a **router** pod in front of **engine** pods and offers **model-aware and prefix-aware routing** plus KV-cache offload via **LMCache**. For data-parallel deployments the docs are explicit: *each engine keeps an independent KV cache, so intelligent routing maximizes prefix-caching benefit*. Autoscaling reads load signals — SkyPilot's `replica_policy` uses **`target_qps_per_replica`**; a Kubernetes HPA reads the engine's own **`vllm:num_requests_waiting`** gauge from `/metrics`. All numbers here are **illustrative / order-of-magnitude references**.

---

## 1 · Intuition & why it matters

The [last lesson](load-testing-knee.md) gave you a number: the **knee** of one instance — the arrival rate past which latency runs away. Real traffic exceeds one instance's knee. So you do the obvious thing: run **several instances** and put a **router** in front that spreads requests across them. That's horizontal scaling, and it's where "an engine" becomes "a service."

But two decisions turn a pile of instances into a *good* service, and both are interview favorites:

1. **How you route matters as much as how many you run.** The naive answer — round-robin — is often the *wrong* one, because it ignores the single biggest free win you built in Part 5: the **prefix cache**. If two requests share a long system prompt, sending them to the **same** instance lets the second reuse the first's cached KV; round-robin scatters them and every instance re-does the prefill. **KV-cache-aware (prefix-aware) routing** turns request placement into a cache-hit optimization.
2. **How you scale — and on what signal.** More traffic → more instances, automatically. But the signal you autoscale on decides whether it works. GPU utilization is a **trap** (a memory-bound decode workload can be "100% busy" and still have queue headroom, or "low util" while the KV cache is full). The signal that actually tracks "am I past the knee" is the **queue depth** — `vllm:num_requests_waiting` — which is why vLLM exports it and why autoscalers key on it.

So: routing across replicas (and why prefix-aware beats round-robin), then autoscaling (and why the queue, not GPU util, is the trigger). → see the [Glossary](../glossary.md) for *KV-cache aware routing, SLO, Knee*.

## 2 · Mental model

A **router** sits in front of N independent engine replicas — *each with its own KV cache* — and an **autoscaler** watches the load signal and changes N.

```text
                                  ┌─────────── AUTOSCALER ───────────┐
                                  │ reads vllm:num_requests_waiting  │
                                  │ (queue depth) → scale N up/down  │
                                  └───────────────┬──────────────────┘
                                                  │ sets N
        requests                                  ▼
   ───────────────▶  ┌───────────────┐     ┌────────────────────────────────┐
                     │    ROUTER     │────▶ │ replica 0   [KV cache A]        │
                     │ round-robin?  │────▶ │ replica 1   [KV cache B]  ← independent
                     │ prefix-aware? │────▶ │ replica 2   [KV cache C]    caches!      │
                     └───────────────┘      └────────────────────────────────┘
                       │
   ROUND-ROBIN: same-prefix requests scatter → every replica re-prefills the shared prompt (cache MISS)
   PREFIX-AWARE: route by cached prefix → the replica that has it serves it (cache HIT, prefill skipped)
```

Three shapes to keep:

- **Caches are per-instance, not shared.** Replica 0's prefix cache and KV blocks are invisible to replica 1. So *where* you send a request determines whether it hits a warm cache. Routing is therefore a **cache-placement** problem, not just a load-spreading one.
- **Prefix-aware routing = "send it where its prefix already lives."** If a request's leading tokens (system prompt, few-shot preamble, conversation history) are already cached on some replica, route it there and skip the prefill. Round-robin throws that away; prefix-aware routing is the whole reason the production stack advertises it.
- **Autoscale on the queue, not on utilization.** `num_requests_waiting > 0` and rising means requests are past the knee *right now* — the direct, model-agnostic overload signal. GPU utilization conflates compute-bound and memory-bound regimes and lies about headroom for decode. Scale N on the queue (or on a proxy like QPS-per-replica calibrated to the knee).

## 3 · Principle

### 3.1 Multiple instances, independent caches

The unit of scaling is a full engine instance (a `vllm serve` process, possibly itself TP/PP across GPUs from [Part 7](../part7/index.md)). Run several — on several GPUs, several nodes, or several pods — behind one address. Each instance is **independent**: its own weights (or a shared read-only copy), its own KV-cache block pool, its own [prefix cache](../part5/prefix-caching.md). Nothing is shared across instances unless you add an external KV store (LMCache, below). This independence is why routing is not neutral: two identical requests can hit a warm cache or a cold one purely based on which replica the router picked.

### 3.2 Load balancing: round-robin vs KV-aware

- **Round-robin / least-loaded.** Spread requests evenly by count or by current load. Simple, and correct when requests are independent and short. But it **ignores prefix reuse**: requests sharing a long system prompt get scattered, so each replica pays the shared prefill again. You lose the prefix-cache win exactly when it matters most (long shared prefixes).
- **KV-cache-aware / prefix-aware routing.** Route by *content*: hash the request's prefix and send it to the replica that already has that prefix cached, subject to load limits. Now the shared system prompt is prefilled **once** per replica and reused; the [DP-deployment docs](../part7/index.md) state the rationale directly — *because each engine maintains an independent KV cache, intelligent request routing can maximize the benefits of prefix caching.* The trade-off is a hotspot risk (a popular prefix piles onto one replica), so real routers blend prefix-affinity with load-balancing.

The **production stack** router bakes this in: it offers **model-aware** routing (pick the replica serving the right model) and **prefix-aware** routing (pick the replica with the warm prefix), and can offload KV to **LMCache** so a prefix evicted from one replica's GPU can be re-fetched instead of recomputed.

### 3.3 Autoscaling — and the signal

Autoscaling changes the replica count N with load. The whole game is **which signal** you scale on:

- **Queue depth — `vllm:num_requests_waiting`.** The direct read of "requests are past the knee." vLLM exports it on `/metrics` precisely so a Kubernetes **HPA** (or a KEDA Prometheus scaler) can target it: waiting climbs → add replicas; waiting stays 0 with low running → remove them. This is the recommended signal because it means the same thing for prefill-heavy and decode-heavy workloads.
- **QPS per replica.** SkyPilot's serve `replica_policy` scales on **`target_qps_per_replica`** between `min_replicas` and `max_replicas` — a coarser but effective proxy *if you've calibrated the target to the measured knee* from the last lesson. Set `target_qps_per_replica` at (or just below) the knee and the autoscaler keeps each replica in its linear region.
- **GPU utilization — the trap.** Tempting and wrong. Decode is [memory-bound](../part0/inference-flow.md): a replica can report high "utilization" while still having batch headroom, or report modest util while its KV cache is full and it's queuing. Util doesn't track the knee; the queue does.

Two operational realities: **cold start** — a new replica must load weights and warm CUDA graphs (tens of seconds), so scale-up **lags**; provision headroom or pre-warm. And **scale-down safety** — before killing a replica, let it **drain** (both `vllm:num_requests_running` and `vllm:num_requests_waiting` reach 0) so in-flight requests aren't dropped.

### 3.4 Putting it together

The shape of a production deployment: a **router** (prefix-aware, model-aware) in front of a **Deployment** of engine pods, an **autoscaler** watching `num_requests_waiting`, `/health` for liveness and `/metrics` for load, and — optionally — **LMCache** as a shared KV tier so cache misses across replicas cost a fetch, not a full prefill. The production stack ships all of this as a Helm chart; you can also assemble it from a plain load balancer + HPA + your own routing rule.

## 4 · Complete runnable code + line-by-line

Deploy the production stack (router + replicas), then the two autoscaling signals: a Kubernetes HPA on the queue gauge, and SkyPilot's QPS policy.

```yaml title="values.yaml — production stack: 2 replicas behind a prefix-aware router"
# helm repo add vllm https://vllm-project.github.io/production-stack
# helm install vllm vllm/vllm-stack -f values.yaml
# NOTE: the Helm install, the router pod + engine pods, and model-/prefix-aware routing are
#       verified; the exact values.yaml field names below are illustrative — confirm the schema
#       against the production-stack chart's own values.yaml for your chart version.
servingEngineSpec:
  modelSpec:
    - name: "qwen"
      repository: "vllm/vllm-openai"
      modelURL: "Qwen/Qwen2.5-7B-Instruct"   # the served model
      replicaCount: 2                         # start with 2 engine pods, each an independent cache
      requestGPU: 1                           # 1 GPU per replica (TP>1 would request more)
routerSpec:
  routingLogic: "prefixaware"                 # KV-aware: send a request to the replica with its warm prefix
                                              # (round-robin scatters shared prefixes → cache misses)
```

```yaml title="hpa.yaml — autoscale on the QUEUE, not GPU utilization"
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-engine
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: vllm-engine }
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting     # the /metrics gauge = queue depth = "past the knee"
        target:
          type: AverageValue
          averageValue: "5"                   # add replicas when avg queue > 5 (tune to your SLO/knee)
```

```yaml title="skypilot: QPS-based autoscaling (calibrate the target to the measured knee)"
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 8                 # = the per-instance KNEE from the load-test lesson
  readiness_probe:
    path: /v1/chat/completions                # a real request, not just /health — proves it can serve
    post_data: { model: qwen2.5-7b, messages: [{role: user, content: "ping"}], max_completion_tokens: 1 }
```

**Line-by-line:**

- **`replicaCount: 2` + independent caches** — two engine pods, each with its **own** KV/prefix cache. That independence is exactly why the router's policy matters: the same request is a hit on one pod and a miss on the other.
- **`routingLogic: "prefixaware"`** — the router hashes each request's prefix and sends it to the replica that already holds that prefix's KV, turning a shared system prompt into **one** prefill per replica instead of one per request. Switch it to round-robin and you forfeit the [prefix-caching](../part5/prefix-caching.md) win for shared-prefix traffic.
- **HPA `metric: vllm_num_requests_waiting`** — the autoscaler targets the **queue-depth gauge** vLLM exports on `/metrics`. Average queue > 5 → scale out; drains toward 0 → scale in. This is the model-agnostic overload signal; it means the same thing whether the load is prefill- or decode-heavy — unlike GPU util.
- **`averageValue: "5"`** — the threshold is your policy dial: set it against the SLO and the knee (a small standing queue you tolerate before adding capacity). Too low → flapping; too high → SLO misses during the scale-up lag.
- **`target_qps_per_replica: 8`** — SkyPilot's coarser knob: keep each replica at ~8 req/s, which should be the **knee you measured** last lesson. Below the knee each replica stays in its linear region; the autoscaler adds replicas as total QPS grows.
- **`readiness_probe` posting a real chat request** — readiness must prove the pod can actually *serve* (weights loaded, engine warm), which `/health` (liveness only) does not. A tiny `max_completion_tokens: 1` request is the honest readiness check.

## 5 · Lab — see prefix-aware routing beat round-robin

!!! gpu "GPU Lab (multi-instance — 2 GPUs, or 2 small instances on one)"
    - **Min VRAM / GPUs:** two engine instances. Cleanest on **2 GPUs** (one replica each). On a single 24 GB 4090 you can approximate it with **two instances of a small model** (e.g. `Qwen2.5-0.5B-Instruct`) on different ports, or two MPS slices — enough to *see the routing behavior*, not to benchmark 7B throughput.
    - **Suggested AutoDL card:** **2× 4090 or 2× A100** on a **"power-on-then-off" basis** (ADR-0001) for a real 7B multi-instance test; or a single 4090 with two tiny instances for the routing demo. Tear multi-GPU down when done.
    - **Est. time / cost:** ~30–45 min · **~¥3–15** depending on card (illustrative). The deliverable is the **cache-hit difference**, not a throughput record.
    - **Platform:** NVIDIA CUDA (default). **Non-NVIDIA:** the router and autoscaler are infrastructure (Helm/K8s/HTTP) and are hardware-agnostic; only the engine pods differ per backend.

Steps:

1. **Two replicas, one shared prefix.** Start two instances. Craft requests that share a long system prompt (e.g. a 1k-token preamble) with different user questions.
2. **Round-robin baseline.** Route the shared-prefix requests round-robin. On each replica, `curl /metrics | grep -i prefix` (the prefix-cache hit/total counters) — hits stay low because the prefix scatters and each replica re-prefills.
3. **Prefix-aware.** Switch the router to prefix-aware. Send the same requests. Now the shared prefix lands on one replica repeatedly: its prefix-cache **hit counter climbs**, its **TTFT drops** (prefill skipped). That gap is the whole point.
4. **Autoscale on the queue.** Drive load past one replica's knee and watch `vllm:num_requests_waiting` rise; confirm your HPA/policy adds a replica and the queue drains. Then **power off** any multi-GPU instance.

## 6 · Common pitfalls / counter-intuitive points

- **Round-robin routing that silently kills the prefix cache.** The single most common waste: shared-prefix traffic (same system prompt, RAG preamble, or conversation) scattered evenly means every replica re-prefills the shared part. For shared-prefix workloads, **prefix-aware routing** can cut TTFT and prefill cost dramatically — round-robin throws it away.
- **Autoscaling on GPU utilization.** Decode is memory-bound: a replica can sit at "100% util" with batch headroom, or "40% util" while its KV cache is full and it's queuing. Util doesn't track the knee. Scale on **`vllm:num_requests_waiting`** (queue depth) or a knee-calibrated QPS target.
- **Ignoring cold-start lag.** A new replica takes tens of seconds to load weights and warm CUDA graphs. If you scale up only *after* the queue is deep, the SLO is already blown by the time the pod is ready. Provision headroom, pre-warm, or scale on a leading indicator.
- **Scaling down without draining.** Killing a pod with in-flight requests drops them. Before terminating, stop routing new work to it and wait until **both** `num_requests_running` and `num_requests_waiting` hit 0 — then remove it.
- **Assuming caches are shared across replicas.** They aren't. A prefix warm on replica 0 is cold on replica 1. Cross-replica reuse needs an explicit shared KV tier (**LMCache** in the production stack); otherwise routing is your only lever on cache hits.
- **Prefix-affinity hotspots.** Pure prefix-aware routing can pile a popular prefix onto one replica while others idle. Real routers **blend** prefix-affinity with load-balancing; tune the balance, don't route on prefix alone.
- **Readiness = `/health`.** `/health` is liveness (engine alive), not readiness (can serve now). Load balancers that add a pod to rotation on `/health` send traffic to a still-loading engine. Use a real readiness probe (a tiny generation request), as in §4.
- **Streaming through a buffering load balancer.** As in the [server lesson](openai-server.md), an LB that buffers responses collapses SSE streaming into one late chunk. Configure the router/LB to pass streaming responses through unbuffered.

## 7 · Interview links

- [Routing, autoscaling & KV-aware routing](../interview/routing-autoscaling.md) — the high-frequency question this lesson prepares you for: *why prefix-aware routing beats round-robin (independent per-replica caches), why you autoscale on `num_requests_waiting` rather than GPU utilization, and how cold-start and drain-before-scale-down shape a safe policy.*

## 8 · Summary & further reading

**One line:** Past one instance's [knee](load-testing-knee.md) you scale horizontally — N independent engine replicas behind a **router** — and the two decisions that make it good are **KV-cache-aware (prefix-aware) routing** (send a request to the replica whose cache already holds its prefix, because caches are per-instance, so round-robin re-prefills shared prompts) and **autoscaling on the queue** (`vllm:num_requests_waiting`, the model-agnostic overload signal, not GPU utilization) with cold-start lag and drain-before-scale-down handled; vLLM ships this as the **production stack** (Helm: prefix-aware + model-aware router, engine pods, LMCache KV offload), and SkyPilot autoscales on `target_qps_per_replica` calibrated to your measured knee.

Further reading:

- vLLM `docs/deployment/integrations/production-stack.md` — the Helm chart, the router (model-aware + prefix-aware), and LMCache KV offload.
- vLLM `docs/serving/data_parallel_deployment.md` — why independent per-engine KV caches make intelligent routing pay off.
- vLLM `docs/deployment/frameworks/skypilot.md` — `replica_policy` / `target_qps_per_replica` autoscaling and readiness probes.
- vLLM `docs/design/metrics.md` — `vllm:num_requests_waiting` and the prefix-cache hit/total counters you route and scale on.
- The [prefix-caching lesson](../part5/prefix-caching.md) — the per-instance win that prefix-aware routing preserves across instances.

## 9 · Self-check

??? question "You run 4 replicas behind a round-robin load balancer. All requests share a 2000-token system prompt. TTFT is high and GPU cost is worse than you expected. What's happening, and what's the fix?"
    Round-robin scatters the shared-prefix requests across all 4 replicas, and **each replica has its own independent prefix cache** — so all four re-run the 2000-token **prefill** for the shared system prompt instead of reusing it. You've defeated the [prefix cache](../part5/prefix-caching.md): the shared preamble is prefilled once *per replica per request* rather than cached. The fix is **KV-cache-aware (prefix-aware) routing** — hash the prefix and send matching requests to the replica that already holds that prefix's KV, so the preamble is prefilled once per replica and then reused (prefill skipped → TTFT drops, GPU work falls). Blend it with load-balancing so a hot prefix doesn't overload one replica; optionally add a shared KV tier (LMCache) so cross-replica misses are a fetch, not a recompute.

??? question "An SRE proposes autoscaling vLLM on GPU utilization > 80%. Why is that a poor signal for LLM serving, and what would you use instead?"
    GPU utilization is a **trap** for LLM inference because decode is **memory-bound**, not compute-bound: a replica running a decode-heavy batch can report high "utilization" while still having batch-width headroom (it's waiting on HBM, not maxing FLOPs), or report *modest* utilization while its **KV cache is full** and requests are already queuing. Util doesn't correlate with "am I past the knee." The direct signal is **queue depth** — `vllm:num_requests_waiting` from `/metrics` — which means the same thing regardless of prefill/decode mix: >0 and rising = overloaded, add replicas; ~0 with low running = idle, scale in. A calibrated **`target_qps_per_replica`** (set to the measured [knee](load-testing-knee.md)) is an acceptable coarser proxy; GPU util is not.

??? question "Your autoscaler adds a replica the instant the queue spikes, yet the SLO still gets violated during the spike. Separately, scaling *down* occasionally drops user requests. Diagnose both."
    **Scale-up SLO miss = cold-start lag.** A fresh replica must load weights and warm CUDA graphs — tens of seconds — so it isn't serving when the queue spikes; by the time it's ready the SLO is already blown. Fixes: keep **headroom** (scale on a leading indicator / lower threshold so you add capacity *before* saturation), **pre-warm** replicas, or accept that scale-up is not instantaneous and size min-replicas for the baseline peak. **Scale-down dropping requests = no drain.** Terminating a pod with in-flight work kills those requests. Before removing a replica, **stop routing new requests to it** and wait until **both** `vllm:num_requests_running` and `vllm:num_requests_waiting` reach 0 (graceful drain), then terminate. Together: scale up early (lag), scale down gently (drain).
