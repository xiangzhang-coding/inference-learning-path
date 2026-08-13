# Observability & Profiling: Metrics, Traces, and the Kernel Timeline

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090"
    Verified against vLLM 0.26.0 via Context7 (ADR-0004): metrics come from the Prometheus **`/metrics`** endpoint (`vllm:num_requests_running` / `num_requests_waiting`, `vllm:gpu_cache_usage_perc`, histograms `vllm:time_to_first_token_seconds` / `vllm:request_prefill_time_seconds` / `vllm:request_decode_time_seconds`, counters `vllm:generation_tokens_total`, `vllm:prompt_tokens_cached`, `vllm:request_success_total{finished_reason}`), with a ready-made **Prometheus + Grafana** example (`examples/observability/prometheus_grafana`, `docker compose up`). **OpenTelemetry** request tracing exports spans to an OTLP collector such as **Jaeger** (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://…:4317`). Profiling: the **PyTorch profiler** (`vllm serve … --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'` + the **`/start_profile`** / **`/stop_profile`** endpoints, or the client's `vllm bench serve --profile`) and **Nsight Systems** (`nsys profile … --capture-range=cudaProfilerApi … vllm serve … --profiler-config.profiler cuda`). All numbers here are **illustrative / order-of-magnitude references**.

---

## 1 · Intuition & why it matters

You [found the knee](load-testing-knee.md) in a controlled sweep. Production is not controlled: traffic shifts, prompts get longer, a deploy regresses, p99 latency creeps up at 3am. The question stops being "what's the ceiling" and becomes **"it's slow right now — *where*?"** Answering that fast is observability, and it's a core on-call and system-design skill.

The trap is reaching for the wrong tool. If you fire up a kernel profiler because "latency is high," you'll drown in a gigabyte of GPU timeline and still not know whether the problem is a full queue, a long prefill, or a slow kernel. The discipline is **three zoom levels**, cheapest first:

1. **Metrics** — aggregate, always-on, nearly free. *Is* there a problem, and roughly where (queue? prefill? decode? cache misses?). This is what your dashboard and alerts watch 24/7.
2. **Traces** — per-request, sampled. *Which* stage of *which* request's lifecycle ate the time (waiting in queue vs prefill vs decode).
3. **Profiles** — the kernel/operator timeline, on-demand, expensive. *Why* a stage is slow, down to individual CUDA kernels.

Two things an interviewer expects: that you **start at metrics and only zoom in when the cheaper tier points you there**, and that you know vLLM ships all three so you're not inventing infrastructure. → see the [Glossary](../glossary.md) for *SLO, Goodput*.

## 2 · Mental model

Three tiers, each a deeper zoom at a higher cost — reach for the next only when the current one localizes the problem.

```text
   ZOOM LEVEL          WHAT IT ANSWERS                COST / CADENCE            vLLM SURFACE
   ──────────────────────────────────────────────────────────────────────────────────────────
   ① METRICS      "is there a problem, roughly where?"  ~free, always-on        /metrics (Prometheus)
      (fleet)      queue depth · TTFT p99 · cache hit    scrape every 15s        + Grafana dashboard
        │          prefill vs decode time · finished     alerts fire here
        │  points you at a stage / request class
        ▼
   ② TRACES       "which stage of which request?"       cheap, sampled          OpenTelemetry → Jaeger
      (request)    queue → prefill → decode, per req     per-request spans       (OTLP :4317)
        │  points you at a slow stage
        ▼
   ③ PROFILES     "why is this stage slow — which        expensive, on-demand    torch profiler + /start_profile
      (kernel)     kernel/op?"  the GPU timeline          seconds of capture      Nsight Systems (nsys)
   ──────────────────────────────────────────────────────────────────────────────────────────
   RULE: detect at ①, localize at ②, explain at ③. Never start at ③.
```

Three shapes to keep:

- **Metrics detect; profiles explain.** A dashboard tells you p99 TTFT doubled and the queue is deep — that's an *alert*, not a *diagnosis*. The profiler tells you a specific kernel regressed — that's the diagnosis, but it costs real overhead to collect, so you only run it once metrics point you at the box.
- **The histograms already split prefill from decode.** vLLM exports `request_prefill_time_seconds` and `request_decode_time_seconds` separately. That single fact often ends the investigation: a TTFT spike with rising *prefill* time is a prompt-length / batching problem; a TPOT spike with rising *decode* time is a memory-bandwidth / batch-width problem. You rarely need the profiler for that split.
- **Profiling is a scalpel, not monitoring.** The torch profiler and Nsight are **on-demand**: you turn them on for a few seconds, capture, turn them off. Leaving them on in production adds overhead and fills disk. Metrics and (sampled) traces are the always-on layers.

## 3 · Principle

### 3.1 Tier ① — Metrics (Prometheus + Grafana)

The `/metrics` endpoint exposes a Prometheus text feed with the `vllm:` prefix. The ones that carry an on-call investigation:

| Metric | Type | Reads as |
|---|---|---|
| `vllm:num_requests_running` / `num_requests_waiting` | gauge | batch width in use / **queue depth** (the [knee](load-testing-knee.md) signal) |
| `vllm:gpu_cache_usage_perc` | gauge | how full the KV-cache block pool is (near 1.0 → preemptions, the [PagedAttention](../part5/paged-attention.md) ceiling) |
| `vllm:time_to_first_token_seconds` | histogram | **TTFT** distribution — alert on p99 |
| `vllm:request_prefill_time_seconds` / `request_decode_time_seconds` | histogram | time split into **prefill** vs **decode** — the first fork in any latency investigation |
| `vllm:generation_tokens_total` | counter | output-token throughput (rate over time) |
| `vllm:prompt_tokens_cached` | counter | prefix-cache reuse — pair with prompt-token counters for a **hit rate** |
| `vllm:request_success_total{finished_reason}` | counter | completions by `stop` / `length` / `abort` — a spike in `abort` = clients timing out |

Because histograms and counters are exported (not pre-averaged gauges), you compute rates and percentiles in **PromQL** over any window — e.g. prefix-cache hit rate as `rate(vllm:prompt_tokens_cached[5m]) / rate(prompt_tokens_total[5m])`. vLLM ships a **Prometheus + Grafana** example (`examples/observability/prometheus_grafana`, `docker compose up`) and an importable dashboard, so the fleet view is a copy-paste away.

### 3.2 Tier ② — Traces (OpenTelemetry)

Metrics aggregate across requests; a **trace** follows *one* request and time-stamps each stage — time in queue, prefill, decode. vLLM supports **OpenTelemetry**: point it at an OTLP collector (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://<collector>:4317`, e.g. a **Jaeger** all-in-one), and each request emits spans you can open in the Jaeger UI. This answers "the p99 is bad — is it a few requests stuck *in queue*, or is *prefill* itself slow for long prompts?" without a profiler. Caveat from the docs: **detailed** tracing can involve costly/blocking work, so sample it — don't trace 100% of production traffic.

### 3.3 Tier ③ — Profiles (torch profiler & Nsight)

When a *stage* is slow and you need the *kernel*, profile — on-demand:

- **PyTorch profiler.** Launch with `--profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'`, then bracket the window with the **`POST /start_profile`** and **`POST /stop_profile`** endpoints (or let the benchmark client drive it with `vllm bench serve … --profile`). It writes traces you open in **TensorBoard** or Perfetto — operator times, shapes, and the Python→kernel mapping.
- **Nsight Systems (`nsys`).** For the CUDA-level timeline (kernel durations, gaps, CUDA-graph replay, NCCL): `nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end repeat vllm serve … --profiler-config.profiler cuda`, with the client's `--profile` flag marking the capture range. `--capture-range=cudaProfilerApi` is what keeps the trace to the window you care about instead of the whole run.

The rule: metrics say *there's a problem in decode*; the profiler says *this attention kernel is the cost* — you only pay tier ③'s overhead after tiers ① / ② have narrowed the target.

## 4 · Complete runnable code + line-by-line

Read the metrics (with the PromQL you'd alert on), then capture a torch profile on demand.

```bash
# (a) Tier ① — scrape the live metrics feed and pull the investigation signals
curl -s http://localhost:8000/metrics | grep -E \
  "num_requests_(running|waiting)|gpu_cache_usage_perc|request_(prefill|decode)_time_seconds_count"
#   vllm:num_requests_waiting{...} 12.0         # queue depth — is work backing up?
#   vllm:gpu_cache_usage_perc{...} 0.97         # KV pool ~full → preemptions imminent
# PromQL you'd put on a Grafana panel / alert:
#   histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))   # p99 TTFT
#   rate(vllm:request_success_total{finished_reason="abort"}[5m])                                # clients timing out
```

```bash
# (b) Tier ③ — capture a PyTorch profile for a few seconds, on demand
# 1) launch the server with the torch profiler enabled (writes traces to ./vllm_profile)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
# 2) bracket ONLY the window you care about — start, send load, stop
curl -X POST http://localhost:8000/start_profile          # begin capture
#    (send a handful of representative requests here)
curl -X POST http://localhost:8000/stop_profile           # end capture → trace written to ./vllm_profile
# 3) open ./vllm_profile in TensorBoard / Perfetto to read operator + kernel times
```

```bash
# (c) Nsight Systems — the CUDA kernel timeline, scoped to the client's capture range
# server:
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end repeat \
    vllm serve Qwen/Qwen2.5-7B-Instruct --profiler-config.profiler cuda
# client marks the capture window:
vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --dataset-name sharegpt --dataset-path sharegpt.json --profile --num-prompts 2
```

**Line-by-line:**

- **`grep num_requests_waiting | gpu_cache_usage_perc`** — the two-gauge triage: a deep **queue** means you're past the knee (add capacity / [route + autoscale](routing-autoscaling.md)); **`gpu_cache_usage_perc`** near 1.0 means the KV pool is about to force preemptions (lower `--max-num-seqs` or raise `--gpu-memory-utilization`).
- **`histogram_quantile(0.99, … time_to_first_token_seconds_bucket …)`** — because TTFT is exported as a **histogram**, you compute any percentile over any window in PromQL. Alert on p99, not the mean (the [knee lesson](load-testing-knee.md)'s tail rule).
- **`request_success_total{finished_reason="abort"}`** — a rising abort rate is the machine-visible symptom of clients hitting their timeout: the SLO is already violated. It's often the *first* alert to fire.
- **`--profiler-config '{"profiler":"torch",…}'`** — arms the PyTorch profiler but captures nothing until you call `/start_profile`. This is the on-demand contract: enabled ≠ recording.
- **`/start_profile` … `/stop_profile`** — bracket the smallest representative window. Profiling all traffic is the classic mistake (§6); a few seconds of representative load is enough to see the hot operators.
- **`nsys … --capture-range=cudaProfilerApi`** — scopes the CUDA trace to the client-marked window (`--profile`) instead of the whole process, so you get a readable timeline, not a multi-GB dump. `--cuda-graph-trace=node` unpacks the [CUDA-graph](../part2/kernel-fusion-cuda-graphs.md) replay decode uses.

## 5 · Lab — dashboard, then a profile

!!! gpu "GPU Lab (single-GPU)"
    - **Min VRAM:** the [same server](openai-server.md) — `Qwen2.5-7B-Instruct` on a **24 GB RTX 4090**. Prometheus/Grafana/Jaeger run in Docker on CPU; the profiler needs the GPU only during the short capture.
    - **Suggested AutoDL card:** single **RTX 4090 (24 GB)** (ADR-0001). No multi-GPU needed.
    - **Est. time / cost:** ~25–40 min · **~¥1–4** (illustrative). Bring up the monitoring stack in **无卡模式** first; power on the GPU only to serve + profile.
    - **Platform:** NVIDIA CUDA (default). **Non-NVIDIA:** metrics/traces are hardware-agnostic (pure HTTP/Prometheus/OTel). **Nsight Systems is NVIDIA-only**; on AMD ROCm use `rocprof` / the PyTorch profiler instead (same `/start_profile` flow, different kernel viewer).

Steps:

1. **Stand up tier ①.** From `examples/observability/prometheus_grafana`, `docker compose up`; import the dashboard. Point Prometheus at your server's `/metrics`. Watch `num_requests_waiting` and `gpu_cache_usage_perc` while you drive load.
2. **Make a problem visible.** Push load past the knee (from the [previous lesson](load-testing-knee.md)); watch p99 TTFT climb on the dashboard and `finished_reason="abort"` tick up. **Detection, from metrics alone.**
3. **Localize with a trace (optional).** Bring up Jaeger, set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, send a few requests, and read one request's queue/prefill/decode spans.
4. **Explain with a profile.** Arm the torch profiler, `/start_profile`, send ~2 requests, `/stop_profile`; open `./vllm_profile` and find the top operators. **Power off** when done.

## 6 · Common pitfalls / counter-intuitive points

- **Starting at the profiler.** Firing up `nsys` because "it's slow" gives you a giant timeline and no direction. Detect at metrics, localize at traces, *then* profile the narrowed target.
- **Leaving profiling on in production.** The torch profiler and Nsight add real overhead and write large traces; `--profiler-config` + `/start_profile` is meant for short, bracketed captures. Monitoring is metrics (+ sampled traces), not a permanent profiler.
- **Reading the mean instead of the tail.** A healthy mean TTFT hides a bad p99. The SLO lives on p99 — use `histogram_quantile(0.99, …)` on the exported buckets, not the average.
- **Ignoring the prefill/decode split.** vLLM already separates `request_prefill_time_seconds` from `request_decode_time_seconds`. Skipping that split and jumping to a profiler wastes the cheapest, most decisive signal you have.
- **`nsys` without `--capture-range`.** Tracing the whole process produces an unreadable multi-GB file. Scope it to the client's `--profile` window with `--capture-range=cudaProfilerApi`.
- **Tracing 100% of traffic.** Detailed OpenTelemetry spans can involve costly/blocking work; sample them. Full-fidelity tracing on every request can itself become the latency problem.
- **Alerting on GPU utilization.** As in [routing/autoscaling](routing-autoscaling.md), util misleads for memory-bound decode. Alert on **queue depth**, **p99 TTFT/TPOT**, **`gpu_cache_usage_perc`**, and **abort rate** — signals that track the SLO.

## 7 · Interview links

- [Observability & profiling: metrics, traces, and the kernel timeline](../interview/observability-profiling.md) — the high-frequency question this lesson prepares you for: *the three tiers and when to use each, which vLLM metrics you alert on, how the prefill/decode split short-circuits a latency investigation, and how to capture a torch/Nsight profile without drowning in data.*

## 8 · Summary & further reading

**One line:** Production debugging is **three zoom levels** — **metrics** (Prometheus `/metrics`: `num_requests_waiting`, `gpu_cache_usage_perc`, the `time_to_first_token` / `request_prefill_time` / `request_decode_time` histograms, `request_success_total{finished_reason}`; Grafana dashboard included) to **detect**; **OpenTelemetry traces** (spans to Jaeger over OTLP `:4317`) to **localize** a stage in a request's lifecycle; and the **PyTorch profiler** (`--profiler-config` + `/start_profile` / `/stop_profile`) or **Nsight Systems** (`nsys … --capture-range=cudaProfilerApi`) to **explain** it at the kernel — always cheapest-tier-first, and profiling is an on-demand scalpel, never always-on monitoring.

Further reading:

- vLLM `docs/design/metrics.md` — the full metric list, the histograms, and the OpenTelemetry tracing configuration.
- vLLM `examples/observability/prometheus_grafana` and `examples/observability/opentelemetry` — the copy-paste monitoring and tracing stacks quoted here.
- vLLM `docs/contributing/profiling.md` — the torch-profiler `--profiler-config` and the Nsight `nsys` recipe.
- The [next lesson](slo-driven-tuning.md) — turning these signals into an SLO-driven tuning loop.

## 9 · Self-check

??? question "p99 TTFT just doubled in production. Walk through the tiers you'd use, in order, and say what each would tell you."
    **Tier ① metrics first** (free, already on): check `vllm:num_requests_waiting` — if the **queue** is deep, you're past the [knee](load-testing-knee.md) (traffic exceeded capacity → route/autoscale, not a code bug). Check `gpu_cache_usage_perc` — near 1.0 means KV-pool pressure and preemptions. Then read the **prefill vs decode split** (`request_prefill_time_seconds` vs `request_decode_time_seconds`): a TTFT spike with rising *prefill* time points at longer prompts / batching; rising *decode* time points at batch width / bandwidth. **Tier ② traces** (sampled OTel → Jaeger) if you need to confirm *where in a single request's lifecycle* the time goes — queue vs prefill vs decode. **Tier ③ profile** only if a stage is slow for no obvious reason: arm the torch profiler, `/start_profile` for a few seconds, read the hot operators (or `nsys` for the CUDA timeline). The point: metrics usually answer it; you escalate to a profiler only when the cheap tiers point at a kernel-level cause.

??? question "Why is `vllm:gpu_cache_usage_perc` sitting near 1.0 an actionable signal, and what would you change?"
    It means the **KV-cache block pool is nearly full**, so vLLM is about to (or already has to) **preempt / recompute** running sequences to make room — which shows up as latency spikes and stalled decode. It's the [PagedAttention](../part5/paged-attention.md) capacity ceiling made visible. Actions: raise **`--gpu-memory-utilization`** if there's spare VRAM (bigger block pool), *lower* **`--max-num-seqs`** or **`--max-model-len`** to reduce concurrent KV demand, enable/verify **prefix caching** to reuse KV, or — if this is steady-state load — add an [instance](routing-autoscaling.md). Watching this gauge lets you act *before* preemptions tank the p99 rather than after.

??? question "Your teammate keeps a full Nsight capture running on the production server 'so we always have data.' Why is that wrong, and what's the right posture?"
    Profilers are **on-demand scalpels**, not monitoring. A continuous Nsight/torch capture adds real per-op overhead (skewing the very latencies you're trying to measure) and writes huge trace files that fill disk — it can *become* the incident. The right posture is layered: **always-on** = cheap **metrics** (Prometheus, ~free) plus **sampled** OpenTelemetry traces; **on-demand** = bracket a *few seconds* of representative load with `/start_profile` → `/stop_profile` (or `nsys --capture-range=cudaProfilerApi` scoped to the client's `--profile` window) only after metrics/traces have localized the problem. Detect and localize cheaply and continuously; profile expensively and briefly.
