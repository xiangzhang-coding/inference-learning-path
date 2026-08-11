# 延迟与吞吐度量

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 0 · 基础   ·   **考察课程：** [推理性能度量](../part0/metrics.md)

---

## Q：定义 TTFT、TPOT/ITL、throughput、goodput。你会怎么测每一个？为什么增大 batch size 抬高 throughput 却伤害 TTFT/TPOT，goodput 相对裸 throughput 又多给了什么？

### 直接答案

四个数，一次全上：

- **TTFT**（Time To First Token）= 从请求到达到首个输出 token 的时间；由 **prefill** 主导。
- **TPOT**（Time Per Output Token）= 之后每个 token 的平均时间 = $(t_{\text{last}}-t_{\text{first}})/(N-1)$；由 **decode** 主导。**ITL**（Inter-Token Latency）是*逐间隔*版本——抖动看 ITL，平均看 TPOT。
- **Throughput（吞吐）** = 整个系统每秒处理的 token（或请求）数。
- **Goodput（有效吞吐）** = 只算满足其延迟 **SLO** 的那些请求的吞吐（例如 TTFT ≤ 0.5 s *且* TPOT ≤ 50 ms）。恒 ≤ throughput。

**测量：** 客户端从单请求时间戳（到达 + 每个 token 的到达时刻）算，或用 vLLM 自带压测器 `vllm bench serve`（按百分位报 TTFT/TPOT/ITL/throughput），或抓服务端 Prometheus `/metrics`（`vllm:time_to_first_token_seconds`、`vllm:request_prefill_time_seconds`、`vllm:request_decode_time_seconds`、`vllm:generation_tokens_total`）。永远报 **p50/p90/p99**，绝不只报均值。

**更大的 batch** 抬高 throughput，因为 decode 的权重读取摊到更多序列上（continuous batching 的要点）——但每个请求现在要与更多工作共享 GPU，于是它的 TTFT 与 TPOT 上升。不存在单一的「更快」。**Goodput** 就是抓住这点的：batch size 扫描让 throughput 单调上升，而 goodput *先升后降*，那个峰值才是真正的工作点——越过它的裸吞吐是没人按时收到的 token。

### 深入原理

- **两个延迟映射到两个阶段。** TTFT 是 prefill 度量（任何输出前先消化整段 prompt），TPOT 是 decode 度量（每 token 一个 memory-bound 步）。一个请求的端到端延迟 ≈ $\text{TTFT} + (N-1)\cdot\text{TPOT}$。用户感受不同：TTFT 慢 = 画面冻住；TPOT 慢 = 流式卡顿。
- **Little's Law 把一切绑起来。** $L = \lambda W$（并发 = 到达率 × 延迟）。要在延迟 $W$ 下服务 $\lambda$，你需要 $L$ 个常驻请求（你的 batch/KV 预算）；在 GPU 受限时抬 $\lambda$ 会逼 $W$ 上升；吞吐-延迟曲线的**拐点**就是 $W$ 比 $\lambda$ 涨得更快的地方。
- **throughput vs goodput，精确地。** goodput $=\frac{\sum_r N_r\,\mathbb{1}(\text{SLO}_r)}{W}$——同一窗口，分子限制为满足 SLO 的请求。两者之差就是你靠违背承诺「挣」来的吞吐。
- **尾部，而非均值。** SLO 是对着 p99 写的。均值很好但 p99 违反 SLO，意味着你最倒霉的 1% 被稳定辜负——在均值里隐形。

### 代码

一切都从单请求时间戳导出——纯 CPU：

```python
arrival, tok = 0.20, [0.90, 1.00, 1.10]     # 一个请求：.20 发出，3 个 token
ttft = tok[0] - arrival                       # 0.70（prefill 等待）
tpot = (tok[-1] - tok[0]) / (len(tok) - 1)    # 0.10（平均 token 间间隔）
e2e  = tok[-1] - arrival                      # 0.90 = ttft + (N-1)*tpot
met  = ttft <= 0.5 and tpot <= 0.05           # False -> 计入 throughput，不计入 goodput
print(round(ttft,2), round(tpot,2), round(e2e,2), met)   # 0.7 0.1 0.9 False
```

### 面试官追问

- *「batch 翻倍后 throughput 涨了 20% 但用户说更慢——如何自洽？」* → 两者都对：聚合吞吐上升（权重跨 batch 摊薄），单请求 TTFT/TPOT 上升（各自共享 GPU）。若延迟上升越过 SLO，则即便 throughput 涨，goodput 很可能*降了*——那才是真正的回归。
- *「客户端测的 TTFT 比服务端直方图高。为什么？」* → 客户端 TTFT = 服务端计算 + 排队 + 网络 RTT。拿它对照 `vllm:time_to_first_token_seconds`，把「模型慢」与「链路慢」分开。
- *「为什么 p99 而非均值？」* → SLO 是尾部承诺；均值藏住那稳定违反的 1%。
- *「continuous batching 如何改变你观测到的 ITL？」* → 它让 token 间间隔*不均*——接纳新请求的那一步更重——所以均值 TPOT 很好也可能感觉卡顿。看 ITL 百分位。
- *「信任这些数字前你绝不能忘的一件事。」* → 预热：第一个请求付 CUDA graph 捕获 + 缓存预热；丢掉它，否则你的 p99 其实是「第一个请求」。

### 关联知识点

- 课程：[推理性能度量](../part0/metrics.md)
- 相关课程：[推理流程：Prefill 与 Decode](../part0/inference-flow.md)（为何 TTFT↔prefill、TPOT↔decode）
- 术语：[TTFT、TPOT / ITL、Throughput、Goodput、SLO、Knee](../glossary.md)
