# 可观测性与 profiling：指标、trace 与 kernel 时间线

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 8 · 生产部署与系统设计   ·   **对应课程：** [可观测性与 profiling：指标、trace 与 kernel 时间线](../part8/observability-profiling.md)

---

## Q：生产里 p99 延迟在涨。vLLM 给你哪些可观测性层级？你按什么顺序用？对哪些指标告警？怎么捕获 profile 而不淹死在数据里？

### 直接答案

**三个缩放层级，先便宜的：**

1. **指标**（Prometheus `/metrics`，常开、~免费）—— *有没有问题、大致在哪？* 对 **`vllm:num_requests_waiting`**（队列深度）、**`gpu_cache_usage_perc`**（KV pool）、**`vllm:time_to_first_token_seconds`** 直方图（p99 TTFT）、**`request_prefill_time_seconds`** vs **`request_decode_time_seconds`** 分叉、**`request_success_total{finished_reason="abort"}`**（客户端超时）告警。自带 Grafana 仪表盘。
2. **Trace**（OpenTelemetry → Jaeger，经 OTLP `:4317`，采样）—— *哪个请求生命周期的哪一段*（排队 vs prefill vs decode）。
3. **Profile**（PyTorch profiler 经 `--profiler-config` + `/start_profile` / `/stop_profile`；**Nsight Systems** `nsys … --capture-range=cudaProfilerApi`，按需、昂贵）—— *某段为什么慢，细到 kernel*。

**顺序：** 在 ① 检测、在 ② 定位、在 ③ 解释——永远别从 ③ 开始。

**不淹死：** profiling 是**手术刀**——用 `/start_profile`/`/stop_profile` 框住几秒代表性负载；用 `nsys` 限在客户端的 `--profile` 窗口。生产里绝不常开。

### 深入原理

- **prefill/decode 分叉短路多数调查。** vLLM 分别导出 `request_prefill_time_seconds` 与 `request_decode_time_seconds`：TTFT 尖峰 + *prefill* 上升 = prompt 长度/批处理；TPOT 尖峰 + *decode* 上升 = batch 宽度/带宽。这个分叉你很少需要 profiler。
- **延迟用直方图、不用 gauge。** TTFT/prefill/decode 是直方图 → PromQL 里对任意窗口取任意分位（`histogram_quantile(0.99, …)`）。对尾巴告警。
- **指标检测、profile 解释。** 仪表盘是告警、不是诊断；profiler 点出 kernel 但有开销，所以只在 ①/② 缩窄目标后才付。
- **tracing 开销真实。** 详细 OTel span 可能昂贵/阻塞——采样，别 trace 100%。

### 代码

```bash
# ① 检测——两个 gauge 分诊 + 一个 p99 告警表达式
curl -s localhost:8000/metrics | grep -E "num_requests_waiting|gpu_cache_usage_perc"
# PromQL:  histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
# ③ 解释——按需 torch profile，框住
vllm serve Qwen/Qwen2.5-7B-Instruct --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'
curl -X POST localhost:8000/start_profile     # … 发几个请求 …
curl -X POST localhost:8000/stop_profile      # trace → ./vllm_profile（TensorBoard/Perfetto 打开）
# nsys（CUDA 时间线）: nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node vllm serve … --profiler-config.profiler cuda
```

### 面试官追问

- *「p99 TTFT 翻倍第一步？」* → 指标：**队列**深吗（过了 knee → 容量，不是 bug）？再看 prefill/decode 分叉。profile 放最后。
- *「按 GPU 利用率告警？」* → 不——memory-bound 的 decode 让利用率误导。对队列深度、p99 TTFT/TPOT、`gpu_cache_usage_perc`、abort 率告警。
- *「`gpu_cache_usage_perc` ≈ 1.0 意味？」* → KV pool 快满 → 抢占在即。升 `--gpu-memory-utilization`、降 `--max-num-seqs`/`--max-model-len`、或加容量。
- *「为何不常开 profiling？」* → 开销扭曲延迟、填满磁盘；profiling 是框住/按需的。监控 = 指标 + 采样 trace。
- *「`nsys` 出了 4 GB trace——为什么？」* → 没 `--capture-range`；限在客户端的 `--profile` 窗口。

### 关联知识点

- 课程：[可观测性与 profiling：指标、trace 与 kernel 时间线](../part8/observability-profiling.md)
- 相关：[SLO 驱动调优](slo-driven-tuning.md)（指标喂给调优闭环）、[压测与并发拐点](load-testing-knee.md)（`num_requests_waiting` = knee）、[路由、自动扩缩与 KV 感知路由](routing-autoscaling.md)（为何对队列而非利用率告警）
- 术语表：[SLO、Goodput](../glossary.md)
