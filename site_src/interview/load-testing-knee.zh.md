# 压测与并发拐点（Little 定律）

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 8 · 生产部署与系统设计   ·   **对应课程：** [压测找并发拐点（knee）](../part8/load-testing-knee.md)

---

## Q：什么是并发「knee」？延迟曲线为何在那折弯？Little 定律怎么解释过 knee 后的失控？开环与闭环负载差在哪？你到底报哪个指标？

### 直接答案

**knee** 是单实例运行 batch 填满、请求开始**排队**（`vllm:num_requests_waiting` 离开零往上爬）的那个施加负载。**之下**：加负载抬吞吐、延迟几乎不动（batch 有余量）。**之上**：GPU 饱和，吞吐**走平**，而延迟**失控**，因为每个新请求都排在越来越长的积压后面。

**Little 定律**（$L = \lambda W$，稳态：系统中平均请求 = 到达率 × 在系统中时间）解释它。knee 之下 $W$ 大致恒定，故 $L$ 随 $\lambda$ 线性增长。到 knee，到达率 $\lambda$ 达到最大完成率 $\mu$；推到 $\lambda > \mu$ 就**没有稳态**——$L$ 与 $W$ 无界增长。knee 恰是 $\lambda \approx \mu$。

**开环**（固定**到达率**，`--request-rate λ`、Poisson 间隔）模拟真实流量、*会*过载 → 揭示 knee。**闭环**（固定**并发**，`--max-concurrency N`）自限（完一个才起一个）→ 永不显示失控。默认 `--request-rate inf` 是饱和测试（最大吞吐、忽略延迟）。

**报 goodput**——满足**全部** SLO 的请求/秒（读 **p99**）——不是裸吞吐。靠**扫** `--request-rate` 往上、取最后一个满足 SLO 的档，找到 knee。

### 深入原理

- **队列就是 knee。** 一个 gauge，`vllm:num_requests_waiting` 变正，*就是* knee 实时的。它是「到达超过完成」的直接读数。
- **过 knee 的吞吐是个谎。** 平台真实（GPU 100% 忙）但无用——每个用户排队好几秒深。所以容量必须*连同*一个延迟 SLO 一起说。
- **SLO 定义天花板。** 同硬件、不同 SLO（「p99 TTFT < 200 ms」vs「< 2 s」）→ 不同 knee。上线 **你 SLO 下的 goodput**。
- **尾巴，不是 median。** 好 median 配烂 p99 会坑掉 1% 用户；SLO 写在尾巴上（`--percentile-metrics "ttft,tpot,itl,e2el"`）。
- **workload 形状是答案的一部分。** 512-in/128-out（decode-heavy）与 4k-in/1k-out（prefill-heavy）饱和不同资源 → 不同 knee。固定并报告长度/数据集。

### 代码

```bash
# 固定到达率的开环 run（不是 'inf'）；Poisson 间隔到达
vllm bench serve --backend vllm --model qwen2.5-7b --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 500 --request-rate 8 \
  --percentile-metrics "ttft,tpot,itl,e2el" --save-result
# 扫 --request-rate 2,4,8,16,32 → knee = p99 TTFT ≤ SLO 且 goodput 仍涨的最后一档。
# 在 knee 处看队列:  curl -s localhost:8000/metrics | grep num_requests_waiting
```

### 面试官追问

- *「你把 `--request-rate inf` 的吞吐当容量报告——问题在哪？」* → 饱和测试：最大吞吐、忽略延迟；每个用户在排队。容量 = SLO 下的 knee，来自有限速率 sweep。
- *「为什么过 knee 后延迟垂直？」* → Little 定律：$\lambda > \mu$ → 无稳态 → 队列 $L$ 与等待 $W$ 无界增长；吞吐不能超 $\mu$。
- *「开环 vs 闭环——各何时？」* → 开环 `--request-rate` 找真天花板/knee（会过载）；闭环 `--max-concurrency` 刻画固定客户端池（自限、无失控）。
- *「median TTFT 很好——上线？」* → 不——SLO 在 p99；坏尾巴坑用户。读分位。
- *「用一个数概括容量？」* → SLO 下的 goodput，外加它测量时的 workload 形状（输入/输出长度）。
- *「压 localhost 延迟很抖。」* → 用 `127.0.0.1`（vLLM 工具提示）避 IPv6 解析卡顿；先热身；用足 prompt 到稳态。

### 关联知识点

- 课程：[压测找并发拐点（knee）](../part8/load-testing-knee.md)
- 相关：[延迟与吞吐度量](latency-throughput-metrics.md)（TTFT/TPOT/ITL/goodput 定义）、[调参旋钮：哪个对哪个 SLO](tuning-knobs.md)（*移动* knee 的旋钮）、[路由、自动扩缩与 KV 感知路由](routing-autoscaling.md)（撞上 knee 之后怎么办）
- 术语表：[Knee、SLO、Goodput](../glossary.md)
