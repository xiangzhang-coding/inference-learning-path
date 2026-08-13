# System design: sizing & designing an inference service

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004). All numbers are illustrative / order-of-magnitude references."

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [Capacity Planning: From One GPU's Throughput to a Fleet](../part8/capacity-planning.md)

---

These are **long-form system-design questions** — the 30–45-minute kind that close a senior inference-infra loop. They reward a *method*, not a memorized diagram: clarify the ask, do the napkin math out loud, draw a defensible architecture, then name the bottlenecks and failure modes before the interviewer does. Three complete worked designs follow a shared framework.

## The framework (use it on every "design an inference service" question)

1. **Clarify the requirements → an SLO.** Pin down: model & context length, **peak** QPS (not mean), the latency SLO (**p99 TTFT** and **p99 TPOT/E2EL**), the input/output length mix, and any multi-tenancy / quality / cost constraints. Everything downstream is judged against this. If they won't give numbers, state your assumptions.
2. **Napkin math, out loud** (the [capacity lesson](../part8/capacity-planning.md)): **(a) feasibility** — is the SLO above decode's TPOT floor $W/\beta_{\text{eff}}$? **(b) per-instance capacity** — VRAM gate ($N_{\text{seq}}$, [Part 2](../part2/kv-cache-math.md)) and the measured knee $r_{\text{inst}}=T_{\text{out}}/\bar{o}$; capacity is the `min`. **(c) fleet** — $N_{\text{inst}}=\lceil \lambda_{\text{peak}}/(\rho\,r_{\text{inst}})\rceil$, $N_{\text{GPU}}=N_{\text{inst}}\times\text{TP}$.
3. **Architecture.** Client → gateway (auth, rate-limit) → **router** → N vLLM replicas ([OpenAI-compatible](../part8/openai-server.md)) → shared weights storage. Name the router policy, the autoscaling signal, and the KV/prefix strategy.
4. **Bottlenecks & failure modes.** Where does it break first, and what happens at 2× load, a dead replica, a cold start, or a bad deploy? A design without failure modes is incomplete.
5. **Trade-offs & follow-ups.** Every choice has a cost; say it. "Default X; switch to Y when constraint Z."

---

## Q1: Design a chat completion API — 50 QPS peak, p99 TTFT ≤ 300 ms, p99 TPOT ≤ 50 ms, Qwen2.5-7B, ~512-in / ~256-out

**Type:** System design (long-form)

### Clarify → SLO

Assume: `Qwen2.5-7B-Instruct`, RTX 4090 (24 GB) class GPUs, peak **50 QPS**, **p99 TTFT ≤ 300 ms**, **p99 TPOT ≤ 50 ms**, mean 512-in/256-out, chat (shared system prompt across users), cost-sensitive. SLO is the contract.

### Napkin math

- **Feasibility:** decode TPOT floor for 7B BF16 ≈ $15.2/(0.7\times1008)\approx21.5$ ms < 50 ms ✅ — even BF16 clears the TPOT SLO on one stream. (AWQ would give ~8 ms of headroom.)
- **Per-instance capacity:** VRAM gate — AWQ weights + BF16 KV fit ~33 streams at 8k, plenty for 512+256 (from [Part 2](../part2/kv-cache-math.md)). Speed gate — measure the knee: say $T_{\text{out}}\approx2000$ tok/s at the SLO → $r_{\text{inst}}=2000/256\approx7.8$ req/s. Capacity = min(fits, fast) ≈ **7.8 req/s**.
- **Fleet:** $N_{\text{inst}}=\lceil 50/(0.7\times7.8)\rceil = \lceil 9.2\rceil = \mathbf{10}$ instances = **10 GPUs** (TP=1; 7B fits one card). Round up; keep 1–2 as burst/failure headroom on top if budget allows.

### Architecture

```text
  clients ─▶ API gateway ─▶ ROUTER ─▶ [ vLLM replica × 10 ]  (Qwen2.5-7B-AWQ,
             (auth, rate-       │        each: OpenAI server,   --gpu-memory-utilization 0.90,
              limit, TLS)       │        /metrics, /health)     --enable-prefix-caching)
                                │
                    prefix-aware routing (shared system prompt ⇒ hit the same replica's cache)
                    autoscale on Σ vllm:num_requests_waiting   (queue, NOT gpu-util)
                    Prometheus /metrics ─▶ Grafana + alerts;  OTel traces ─▶ Jaeger
```

- **Engine config:** AWQ weights (frees VRAM, lowers TPOT floor), `--enable-prefix-caching` (chat shares a system prompt → skip its prefill → lower TTFT), `--max-model-len` set to the real context, `--gpu-memory-utilization 0.90`.
- **Router:** [prefix-aware](../part8/routing-autoscaling.md), because per-replica KV caches mean round-robin re-prefills the shared system prompt on every replica.
- **Autoscaling:** on `vllm:num_requests_waiting` summed across replicas (the [knee](../part8/load-testing-knee.md) signal), not GPU utilization; scale up fast, **drain before scale-down**, and account for cold-start (model load) lag.
- **Observability:** [`/metrics` → Grafana](../part8/observability-profiling.md) to detect, OTel traces to localize, torch/Nsight profiles on demand.

### Bottlenecks & failure modes

- **First wall = the queue.** At >50 QPS `num_requests_waiting` climbs and p99 TTFT blows the SLO — that's the knee; the fix is *more replicas*, not tuning ([SLO lesson](../part8/slo-driven-tuning.md)).
- **Dead replica:** load redistributes to 9 → each now past its safe point; the 1–2 headroom replicas and fast scale-up absorb it. Without headroom, one failure cascades.
- **Cold start:** a new replica takes tens of seconds to load weights + warm caches; scale on a *leading* signal (queue depth) and pre-warm, or the scale-up lands after the spike passed.
- **Bad deploy:** roll one replica at a time behind the router with `/health` gating; keep the old version until the new one passes.

### Interviewer follow-ups

- *"Traffic 10×'s overnight to 500 QPS."* → Same math: ~100 instances; now capacity/cost dominate — revisit AWQ+FP8 KV to raise $r_{\text{inst}}$, consider PD disaggregation, and make autoscaling + multi-region real.
- *"p99 TTFT is fine but p99 TPOT creeps up under load."* → Decode-bound; the batch is bandwidth-limited. `--max-num-seqs`, weight/KV quant, or speculative decoding — but check it's not actually a queue (diagnose first).
- *"Cut cost 30% at the same SLO."* → Shorter outputs (cap `max_tokens`), prefix caching wins for the shared prompt, off-peak autoscaling (pay the mean, not the peak), and AWQ+FP8 KV to pack more per GPU.
- *"Why not one giant instance instead of 10?"* → One GPU has a hard knee; you can't exceed one card's throughput by tuning. Horizontal replicas are how you raise the ceiling.

---

## Q2: Design a multi-tenant platform — one base model, 100s of per-customer LoRA fine-tunes, mixed traffic

**Type:** System design (long-form)

### Clarify → SLO

Assume: one base `Qwen2.5-7B`, **hundreds of LoRA adapters** (one per tenant), long-tail traffic (a few hot tenants, many cold), a shared SLO (p99 TTFT ≤ 500 ms), and strict **tenant isolation** (no cross-tenant leakage). Key question: co-serve adapters cheaply without a GPU per tenant.

### Napkin math

- **Why LoRA makes this cheap:** an adapter is tiny (rank-16 ≈ tens of MB vs ~15 GB base), so one loaded base + many adapters ([Part 6](../part6/multi-lora-serving.md)) means the **base weights are paid once**; adapters add negligible VRAM. The VRAM gate is set by base + KV, essentially as Q1.
- **Throughput:** vLLM batches heterogeneous adapters in one step via grouped GEMM, so $r_{\text{inst}}$ is close to the single-model number — a small overhead for the per-adapter GEMM. Size the fleet as in Q1 on *aggregate* QPS across tenants.
- **Adapter capacity:** `--max-loras` caps how many adapters are *resident* per step; `--max-cpu-loras` + dynamic loading swaps the long tail in/out from CPU. Hot set resident, cold set swapped.

### Architecture

```text
  tenants ─▶ gateway (authN, maps tenant ⇒ adapter_id) ─▶ ROUTER ─▶ [ vLLM replica × N ]
                                                            │          base Qwen2.5-7B +
                                                            │          LoRA pool (--enable-lora,
                        adapter-aware routing:               │          --max-loras, --max-cpu-loras)
                        pin hot adapters to replicas         │
                        (avoid reload churn)                 └─▶ adapter registry / object store
```

- **Adapter-aware routing:** route a tenant's requests to the replica(s) that already have its adapter resident — like prefix-aware routing but for adapters — to avoid constant load/evict churn.
- **Hot/cold tiering:** keep the top-K adapters resident (`--max-loras`), swap the tail from CPU/registry on demand (`--max-cpu-loras`, dynamic loading endpoints).
- **Isolation:** adapter_id is derived server-side from the authenticated tenant, never client-supplied; requests can't select another tenant's weights. KV is per-request, so no cross-tenant KV sharing.

### Bottlenecks & failure modes

- **Adapter thrash:** if the working set of active adapters exceeds `--max-loras`, replicas churn loading/evicting → latency spikes. Fix with adapter-aware routing + raising the resident cap (VRAM permitting) or sharding tenants across replica groups.
- **A hot tenant starves others:** one tenant's burst fills shared batches. Per-tenant rate limits / fair-queuing at the gateway.
- **Cold-tenant TTFT:** first request for a cold adapter pays a load. Acceptable if rare; pre-warm known-active tenants.

### Interviewer follow-ups

- *"One tenant needs a full fine-tune, not LoRA."* → It no longer shares the base — it's a separate model deployment (its own replicas, its own fleet math). LoRA co-serving only works because the base is shared.
- *"Guarantee tenant A's p99 regardless of tenant B."* → Isolation of *latency* needs either reserved capacity (dedicated replica group) or strict fair-scheduling; shared batching alone gives best-effort. Trade cost vs guarantee.
- *"How many adapters can one replica really serve?"* → Resident count is `--max-loras` (VRAM-bound); total addressable is that plus the CPU-swappable pool — but the *working set* per step is what matters for latency.

---

## Q3: Design long-context RAG serving — 32k context, large shared knowledge-base prefix, p99 TTFT ≤ 2 s

**Type:** System design (long-form)

### Clarify → SLO

Assume: `Qwen2.5-7B` at **32k context**, RAG where many requests share a **large retrieved-document prefix** (or a big fixed system corpus), moderate QPS, **p99 TTFT ≤ 2 s** (long prompts make TTFT the hard SLO), short outputs. Key tensions: the [KV wall](../part6/long-context-inference.md) and prefill cost.

### Napkin math

- **KV wall dominates VRAM:** one 32k sequence costs $\kappa\cdot32\text{k}$; for Qwen2.5-7B at 56 KiB/token that's ~1.75 GiB of KV *per stream* (BF16 KV). Concurrency collapses vs short context — the VRAM gate, not the speed gate, binds. FP8 KV halves it; weight quant frees the base.
- **Prefill dominates TTFT:** 32k input tokens is a large prefill; naive per-request prefill blows the 2 s TTFT. **Prefix caching** is the lever — if the KB prefix is shared, its KV is computed once and reused, so TTFT drops to the cost of the *unique* suffix only.
- **Fleet:** per-instance capacity is low (few concurrent 32k streams); size on that reduced $r_{\text{inst}}$, and lean on prefix reuse to raise effective throughput.

### Architecture

```text
  clients ─▶ gateway ─▶ PREFIX-AWARE ROUTER ─▶ [ vLLM replica × N ]  (--enable-prefix-caching,
                             │                     │                    --max-model-len 32768,
     route by KB-prefix hash │                     │                    --kv-cache-dtype fp8,
     so shared corpus hits    │                    │                    chunked prefill on)
     the SAME replica's cache  │                   └─▶ (optional) LMCache KV offload to CPU/NVMe
                              │                          for a KB prefix too big to keep hot
```

- **Prefix caching is the core decision:** shared KB/system prefix → compute its KV once per replica, reuse across requests → TTFT becomes the unique-suffix prefill only. Route by prefix hash so the same corpus lands on the same replica ([routing lesson](../part8/routing-autoscaling.md)).
- **KV footprint control:** FP8 KV (`--kv-cache-dtype fp8`) to survive the KV wall; cap `--max-model-len` to the real max; consider KV offload (LMCache in the production stack) for a KB prefix too large to keep resident.
- **Prefill scheduling:** [chunked prefill](../part5/scheduler-chunked-prefill-pd.md) so a 32k prefill doesn't stall other requests' decode; consider PD disaggregation if prefill and decode contend badly.

### Bottlenecks & failure modes

- **KV OOM / preemption:** too many concurrent long streams exhaust the KV pool → preemption/recompute → latency spikes. Cap concurrency, FP8 KV, and admission-control long requests.
- **Cache miss storm:** if routing isn't prefix-aware, each replica re-prefills the shared corpus → TTFT SLO violated. Prefix-aware routing is not optional here.
- **A unique-per-request prefix** (no sharing) removes the main lever → you're back to paying full 32k prefill every time; then the honest answer is fewer concurrent streams and more GPUs, or a smaller context.

### Interviewer follow-ups

- *"Context grows to 128k."* → KV per stream ~7 GiB (BF16) — a handful saturate a 24 GB card. Needs aggressive FP8 KV + weight quant, very low concurrency, or sharding the model (TP) / offloading KV; revisit whether 128k is truly required.
- *"Prefix caching hit rate is low in practice."* → Measure it; if the shared prefix isn't actually shared (per-user retrieval), the design premise fails — re-architect around what *is* shared (a fixed system corpus) or accept the prefill cost.
- *"TTFT is fine but throughput is terrible."* → Long-context is KV-bound; you traded concurrency for context. Raise it with FP8 KV / weight quant, or separate long-context traffic onto its own pool so it doesn't starve short requests.

---

## Linked concepts

- Lesson: [Capacity Planning: From One GPU's Throughput to a Fleet](../part8/capacity-planning.md) — the napkin math (feasibility → per-instance → fleet) all three designs open with.
- Capstone: [Max Out Qwen2.5-7B Throughput on One 4090](../capstone/index.md) — the hands-on other half of these designs: actually climb the optimization ladder and produce the before→after report you'd bring to this interview.
- Related questions: [VRAM budget & max concurrency](vram-capacity-planning.md) (the memory gate), [Load-testing & the concurrency knee](load-testing-knee.md) ($r_{\text{inst}}$), [Routing, autoscaling & KV-aware routing](routing-autoscaling.md) (the fleet at runtime), [SLO-driven tuning](slo-driven-tuning.md) (tune vs scale), [Multi-LoRA serving](multi-lora-serving.md) (Q2), [Long-context inference](long-context-inference.md) (Q3), [Parallelism: TP/PP/DP/EP](parallelism-strategies.md) (TP degree).
- Glossary: [SLO, Knee, Goodput, TP degree, KV-cache aware routing, Prefix caching](../glossary.md)
