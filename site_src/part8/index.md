# Part 8 · Production & System Design

> From a working engine to a production service — and to answering the system-design interview.

## What this part covers

- **Serving over HTTP**: the OpenAI-compatible server, **load-testing** to find the concurrency **knee**, and **routing / autoscaling / KV-cache-aware routing** across multiple instances
- **Observability & profiling** and **SLO tuning**; framework trade-offs — **TensorRT-LLM / TGI / SGLang / LMDeploy** — for selection questions
- **Capacity planning**: estimate VRAM / throughput given a model + hardware
- **System design** drills: "design an inference service for X QPS at Y latency"

The **[Capstone](../capstone/index.md)** pulls everything here together on a single 4090.

## Lessons

- **[Serving vLLM over HTTP: the OpenAI-Compatible Server](openai-server.md)** — `vllm serve` wraps the engine core in a thin FastAPI frontend that speaks the **OpenAI API**, so any OpenAI client retargets with one `base_url` line. The endpoints (`/v1/chat/completions` applies the chat template, `/v1/completions` is raw, `/v1/models` lists the served id + LoRA adapters, `/health` is **liveness** 200/503, `/metrics` is the Prometheus feed), auth (`--api-key` / `VLLM_API_KEY`, repeatable for rotation), streaming (SSE), and why the **interface** knobs (`--port` / `--served-model-name`) are separate from the **capacity** knobs (`--max-num-seqs` / `--gpu-memory-utilization`) that set the ceiling — verified on vLLM 0.26.0.
- **[Load-Testing to Find the Concurrency Knee](load-testing-knee.md)** — the **knee** is where a single instance's batch fills and `vllm:num_requests_waiting` climbs off zero; by **Little's Law** ($L=\lambda W$), pushing arrival rate past the max completion rate makes the queue and latency run away. You find it by sweeping **`vllm bench serve --request-rate`** upward (open-loop Poisson arrivals — *not* `--request-rate inf` and *not* closed-loop `--max-concurrency`), reading **p99** TTFT/E2EL and **goodput** against your SLO, and reporting the last passing rate as the instance's honest capacity.
- **[Routing, Autoscaling & KV-Aware Routing (Multi-Instance)](routing-autoscaling.md)** — past one instance's knee you scale to N independent replicas behind a **router**; the two decisions that make it good are **KV-cache-aware (prefix-aware) routing** (caches are per-instance, so round-robin re-prefills shared prompts) and **autoscaling on the queue** (`vllm:num_requests_waiting`, not GPU utilization), with cold-start lag and drain-before-scale-down handled. vLLM ships this as the **production stack** (Helm: prefix-aware + model-aware router, engine pods, LMCache KV offload); SkyPilot autoscales on `target_qps_per_replica`.

!!! note "Scaffolding status"
    Three lessons are in — the [OpenAI-compatible server](openai-server.md), [load-testing the knee](load-testing-knee.md), and [routing / autoscaling](routing-autoscaling.md) (ticket #19, the production-serving hands-on) — each two-way-linked to its interview question ([the server & endpoints](../interview/openai-server-deployment.md), [the knee & Little's Law](../interview/load-testing-knee.md), [routing & autoscaling](../interview/routing-autoscaling.md)). **Observability / profiling, SLO tuning, framework comparison** (TensorRT-LLM / TGI / SGLang / LMDeploy), and **capacity planning + system-design** long-form land in later tickets. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**, and every performance number is an **illustrative / order-of-magnitude reference**. See the **[Glossary](../glossary.md)** and the [Interview Bank](../interview/index.md).
