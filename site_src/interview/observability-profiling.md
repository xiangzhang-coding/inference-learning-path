# Observability & profiling: metrics, traces, and the kernel timeline

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [Observability & Profiling: Metrics, Traces, and the Kernel Timeline](../part8/observability-profiling.md)

---

## Q: In production, p99 latency is rising. What observability tiers does vLLM give you, which do you use in what order, which metrics do you alert on, and how do you capture a profile without drowning in data?

### Direct answer

**Three zoom levels, cheapest first:**

1. **Metrics** (Prometheus `/metrics`, always-on, ~free) — *is there a problem, roughly where?* Alert on **`vllm:num_requests_waiting`** (queue depth), **`gpu_cache_usage_perc`** (KV pool), the **`vllm:time_to_first_token_seconds`** histogram (p99 TTFT), the **`request_prefill_time_seconds`** vs **`request_decode_time_seconds`** split, and **`request_success_total{finished_reason="abort"}`** (clients timing out). Grafana dashboard shipped.
2. **Traces** (OpenTelemetry → Jaeger over OTLP `:4317`, sampled) — *which stage of which request's lifecycle* (queue vs prefill vs decode).
3. **Profiles** (PyTorch profiler via `--profiler-config` + `/start_profile` / `/stop_profile`; **Nsight Systems** `nsys … --capture-range=cudaProfilerApi`, on-demand, expensive) — *why a stage is slow, at the kernel*.

**Order:** detect at ①, localize at ②, explain at ③ — never start at ③.

**Without drowning:** profiling is a **scalpel** — bracket a few seconds of representative load with `/start_profile`/`/stop_profile`; scope `nsys` to the client's `--profile` window. Never leave it on in production.

### Deep dive

- **The prefill/decode split short-circuits most investigations.** vLLM exports `request_prefill_time_seconds` and `request_decode_time_seconds` separately: TTFT spike + rising *prefill* = prompt-length/batching; TPOT spike + rising *decode* = batch-width/bandwidth. You rarely need a profiler for the fork.
- **Histograms, not gauges, for latency.** TTFT/prefill/decode are histograms → any percentile over any window in PromQL (`histogram_quantile(0.99, …)`). Alert on the tail.
- **Metrics detect, profiles explain.** A dashboard is an alert, not a diagnosis; the profiler names the kernel but costs overhead, so you pay only after ①/② narrow the target.
- **Tracing overhead is real.** Detailed OTel spans can be costly/blocking — sample, don't trace 100%.

### Code

```bash
# ① detect — the two-gauge triage + a p99 alert expression
curl -s localhost:8000/metrics | grep -E "num_requests_waiting|gpu_cache_usage_perc"
# PromQL:  histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
# ③ explain — on-demand torch profile, bracketed
vllm serve Qwen/Qwen2.5-7B-Instruct --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'
curl -X POST localhost:8000/start_profile     # … send a few requests …
curl -X POST localhost:8000/stop_profile      # trace → ./vllm_profile (open in TensorBoard/Perfetto)
# nsys (CUDA timeline): nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node vllm serve … --profiler-config.profiler cuda
```

### Interviewer follow-ups

- *"First move when p99 TTFT doubles?"* → Metrics: is the **queue** deep (past the knee → capacity, not a bug)? Then the prefill/decode split. Profile last.
- *"Alert on GPU utilization?"* → No — memory-bound decode makes util misleading. Alert on queue depth, p99 TTFT/TPOT, `gpu_cache_usage_perc`, abort rate.
- *"`gpu_cache_usage_perc` ≈ 1.0 means?"* → KV pool nearly full → preemptions imminent. Raise `--gpu-memory-utilization`, lower `--max-num-seqs`/`--max-model-len`, or add capacity.
- *"Why not always-on profiling?"* → Overhead skews the latencies and fills disk; profiling is bracketed/on-demand. Monitoring = metrics + sampled traces.
- *"`nsys` produced a 4 GB trace — why?"* → No `--capture-range`; scope it to the client's `--profile` window.

### Linked concepts

- Lesson: [Observability & Profiling: Metrics, Traces, and the Kernel Timeline](../part8/observability-profiling.md)
- Related: [SLO-driven tuning](slo-driven-tuning.md) (the metrics feed the tuning loop), [Load-testing & the concurrency knee](load-testing-knee.md) (`num_requests_waiting` = the knee), [Routing, autoscaling & KV-aware routing](routing-autoscaling.md) (why you alert on the queue, not util)
- Glossary: [SLO, Goodput](../glossary.md)
