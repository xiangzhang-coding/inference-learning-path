# SLO 驱动调优：goodput、绑定约束与闭环

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 8 · 生产部署与系统设计   ·   **对应课程：** [SLO 驱动调优：从指标到调优闭环](../part8/slo-driven-tuning.md)

---

## Q：别人跟你说「让它更快」。你怎么把它变成一个有纪律的调优闭环——优化什么、怎么找该调什么、哪个旋钮对哪个约束、纪律是什么？

### 直接答案

**对着 SLO 优化 goodput、不是裸吞吐。** 先写 SLO（例如 p99 TTFT ≤ 300 ms、p99 TPOT ≤ 50 ms @ 20 请求/秒）。**goodput** = 满足*全部*目标的请求/秒；一个打满 tok/s 却打穿 p99 的配置得**零**。

**闭环：** 定义 SLO → **测** goodput（`vllm bench serve`、p99 指标）→ 从 `/metrics` **诊断**绑定约束 → 拧**一个**缓解它的旋钮 → 重测 → goodput 升就保留 → 走平就停。

**约束 → 旋钮：**

- **队列绑定**（`num_requests_waiting` 深）→ **不是调优问题**：加副本 / [路由](routing-autoscaling.md)。
- **prefill / TTFT** → `--max-num-batched-tokens`（chunked-prefill 旋钮）、`--enable-prefix-caching`。
- **decode / TPOT** → `--max-num-seqs`、[量化](../part4/quantization-methods.md)、投机解码。
- **KV 绑定**（`gpu_cache_usage_perc`→1.0）→ `--gpu-memory-utilization`↑、`--max-model-len`↓、KV 量化。

**纪律：** 一次一个旋钮、重测、对着**类生产** workload。

### 深入原理

- **为何 goodput。** 吞吐与延迟相权衡；只有 goodput 把多目标收成一个分（框*内*的吞吐）。
- **先诊断再调。** 指标告诉你哪面墙绑定。队列是墙时调 decode 毫无用处——那是容量、不是配置。
- **一次一个旋钮。** 旋钮相互作用、多在 TTFT↔吞吐间权衡；一次改两个让增量无法归因。
- **workload 相关。** prefill-heavy（长入短出）与 decode-heavy（短入长出）绑定不同资源 → 不同获胜配置。对着你的真实分布调。
- **平台。** goodput 不再升就是撞到这个 workload 的硬件极限——更多收益需不同硬件或更多实例、不是更多拧旋钮。

### 代码

```python
SLO = {"p99_ttft_ms": 300, "p99_tpot_ms": 50}       # 成功在这里定义、不是靠旋钮
# 对一个旋钮的每个候选值：重启 server、在目标 QPS 跑 vllm bench serve，
r = json.load(open("r.json"))
meets = r["p99_ttft_ms"] <= SLO["p99_ttft_ms"] and r["p99_tpot_ms"] <= SLO["p99_tpot_ms"]
goodput = r["request_throughput"] if meets else 0.0  # 吞吐仅在 SLO 成立时才算数
# 保留 SLO 通过下 goodput 最高的值；走平就停。
```

### 面试官追问

- *「A：1500 tok/s @ p99 TTFT 900 ms。B：1100 @ 250 ms。SLO ≤ 300 ms——哪个？」* → B。A 违反 SLO → goodput 0。以 goodput 打分、不是吞吐。
- *「你调了几小时 p99 几乎不动，而队列全程都深？」* → 调错约束：队列深 = 容量，加副本；没有 decode 旋钮能排掉它。
- *「为何一次一个旋钮？」* → 旋钮相互作用/权衡；两个一起无法归因。
- *「哪个旋钮降 TTFT 而吞吐损失不大？」* → 更小 `--max-num-batched-tokens`（chunked prefill 让 decode 更早交织）；共享 prompt 用 prefix caching。
- *「何时停止调优？」* → goodput 走平时——这个 workload 的硬件极限；横向扩或换硬件。
- *「量化提 decode goodput 的坑？」* → 在评测集上验证质量；延迟收益可能折损精度。

### 关联知识点

- 课程：[SLO 驱动调优：从指标到调优闭环](../part8/slo-driven-tuning.md)
- 相关：[调参旋钮：哪个对哪个 SLO](tuning-knobs.md)（每个旋钮的曲线）、[压测与并发拐点](load-testing-knee.md)（goodput 与 knee）、[可观测性与 profiling](observability-profiling.md)（揭示约束的指标）
- 术语表：[SLO、Goodput、Knee](../glossary.md)
