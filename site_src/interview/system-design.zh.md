# 系统设计：给推理服务定容与设计

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）。所有数字均为示例 / 量级参考。"

**模块：** Part 8 · 生产部署与系统设计   ·   **考察课程：** [容量规划：从一张卡的吞吐到一个集群](../part8/capacity-planning.md)

---

这些是**系统设计长题**——那种 30–45 分钟、收尾一场高级推理 Infra 面试的题。它们奖励的是一套*方法*，不是背下来的图：厘清需求、把草稿纸算术念出来、画一张站得住脚的架构，然后在面试官开口之前先自己点出瓶颈与失败模式。下面三道完整带解设计题共用一套框架。

## 框架（每道「设计推理服务」题都用它）

1. **厘清需求 → 一个 SLO。** 钉死：模型与上下文长度、**峰值** QPS（不是均值）、延迟 SLO（**p99 TTFT** 与 **p99 TPOT/E2EL**）、输入/输出长度分布，以及任何多租户 / 质量 / 成本约束。下游一切都按它评判。对方不给数就把你的假设说出来。
2. **草稿纸算术，念出来**（[容量课](../part8/capacity-planning.md)）：**(a) 可行性** —— SLO 是否高于 decode 的 TPOT 地板 $W/\beta_{\text{eff}}$？**(b) 单实例容量** —— 显存闸（$N_{\text{seq}}$，[Part 2](../part2/kv-cache-math.md)）与测出的 knee $r_{\text{inst}}=T_{\text{out}}/\bar{o}$；容量取 `min`。**(c) 集群** —— $N_{\text{inst}}=\lceil \lambda_{\text{peak}}/(\rho\,r_{\text{inst}})\rceil$、$N_{\text{GPU}}=N_{\text{inst}}\times\text{TP}$。
3. **架构。** client → 网关（鉴权、限流）→ **router** → N 个 vLLM 副本（[OpenAI 兼容](../part8/openai-server.md)）→ 共享权重存储。点名 router 策略、autoscaling 信号、KV/prefix 策略。
4. **瓶颈与失败模式。** 先在哪崩？2× 负载、死一个副本、冷启动、坏发布时会怎样？没有失败模式的设计不完整。
5. **取舍与追问。** 每个选择都有代价，说出来。「默认 X；当约束 Z 时切到 Y。」

---

## Q1：设计一个 chat completion API —— 峰值 50 QPS，p99 TTFT ≤ 300 ms、TPOT ≤ 50 ms，Qwen2.5-7B，~512-in / ~256-out

**类型：** 系统设计（长题）

### 厘清 → SLO

假设：`Qwen2.5-7B-Instruct`，RTX 4090（24 GB）级 GPU，峰值 **50 QPS**，**p99 TTFT ≤ 300 ms**、**p99 TPOT ≤ 50 ms**，平均 512-in/256-out，chat（用户间共享系统 prompt），成本敏感。SLO 是契约。

### 草稿纸算术

- **可行性：** 7B BF16 的 decode TPOT 地板 ≈ $15.2/(0.7\times1008)\approx21.5$ ms < 50 ms ✅ —— 单流上连 BF16 都越过 TPOT SLO。（AWQ 还能再给 ~8 ms 余量。）
- **单实例容量：** 显存闸 —— AWQ 权重 + BF16 KV 在 8k 下装得下 ~33 条流，对 512+256 绰绰有余（见 [Part 2](../part2/kv-cache-math.md)）。速度闸 —— 测 knee：设 SLO 下 $T_{\text{out}}\approx2000$ tok/s → $r_{\text{inst}}=2000/256\approx7.8$ req/s。容量 = min(装得下, 跑得快) ≈ **7.8 req/s**。
- **集群：** $N_{\text{inst}}=\lceil 50/(0.7\times7.8)\rceil = \lceil 9.2\rceil = \mathbf{10}$ 实例 = **10 GPU**（TP=1；7B 装得下一张卡）。向上取整；预算允许就再留 1–2 个作突发/失效余量。

### 架构

```text
  clients ─▶ API 网关 ─▶ ROUTER ─▶ [ vLLM 副本 × 10 ]  (Qwen2.5-7B-AWQ,
             (鉴权、限流、    │        每个：OpenAI server,   --gpu-memory-utilization 0.90,
              TLS)           │        /metrics, /health)     --enable-prefix-caching)
                            │
                    前缀感知路由（共享系统 prompt ⇒ 命中同一副本的 cache）
                    按 Σ vllm:num_requests_waiting 自动扩缩（队列，不是 gpu-util）
                    Prometheus /metrics ─▶ Grafana + 告警；OTel trace ─▶ Jaeger
```

- **引擎配置：** AWQ 权重（腾 VRAM、降 TPOT 地板）、`--enable-prefix-caching`（chat 共享系统 prompt → 跳过其 prefill → 降 TTFT）、`--max-model-len` 设成真实上下文、`--gpu-memory-utilization 0.90`。
- **Router：** [前缀感知](../part8/routing-autoscaling.md)，因为每副本 KV cache 意味着 round-robin 会在每个副本上重 prefill 共享系统 prompt。
- **Autoscaling：** 按跨副本求和的 `vllm:num_requests_waiting`（[knee](../part8/load-testing-knee.md) 信号），不是 GPU 利用率；扩容要快，**缩容前排空**，并计入冷启动（加载模型）滞后。
- **可观测性：** [`/metrics` → Grafana](../part8/observability-profiling.md) 检测、OTel trace 定位、torch/Nsight profile 按需。

### 瓶颈与失败模式

- **第一堵墙 = 队列。** >50 QPS 时 `num_requests_waiting` 上爬、p99 TTFT 冲破 SLO —— 那就是 knee；解法是*加副本*，不是调参（[SLO 课](../part8/slo-driven-tuning.md)）。
- **死一个副本：** 负载重分到 9 个 → 每个都越过安全点；那 1–2 个余量副本与快速扩容吸收它。没有余量，一次失效就雪崩。
- **冷启动：** 新副本要几十秒加载权重 + 预热 cache；按*领先*信号（队列深度）扩容并预热，否则扩容落地时尖峰已过。
- **坏发布：** 在 router 后一次滚一个副本、用 `/health` 把关；新版本通过前保留旧版本。

### 面试官追问

- *「流量一夜 10× 到 500 QPS。」* → 同样的算术：~100 实例；此时容量/成本主导 —— 重新考虑 AWQ+FP8 KV 抬高 $r_{\text{inst}}$、考虑 PD 分离、把 autoscaling + 多区域做实。
- *「p99 TTFT 没事，但负载下 p99 TPOT 慢慢涨。」* → decode-bound；batch 受带宽限。`--max-num-seqs`、权重/KV 量化或投机解码 —— 但先确认它不是队列（先诊断）。
- *「同 SLO 砍成本 30%。」* → 更短输出（限 `max_tokens`）、共享 prompt 的 prefix caching 大赢、低谷 autoscaling（按均值付费而非峰值）、AWQ+FP8 KV 让每 GPU 装更多。
- *「为什么不用一个巨型实例而要 10 个？」* → 一张 GPU 有硬 knee；调参无法超过单卡吞吐。横向副本才是抬高天花板的办法。

---

## Q2：设计多租户平台 —— 一个基座模型、数百个按客户的 LoRA 微调、混合流量

**类型：** 系统设计（长题）

### 厘清 → SLO

假设：一个基座 `Qwen2.5-7B`，**数百个 LoRA adapter**（每租户一个），长尾流量（少数热租户、大量冷租户），共享 SLO（p99 TTFT ≤ 500 ms），且严格**租户隔离**（无跨租户泄漏）。核心问题：便宜地共服 adapter，而不给每租户一张 GPU。

### 草稿纸算术

- **为何 LoRA 让这变便宜：** adapter 很小（rank-16 ≈ 几十 MB vs 基座 ~15 GB），所以一个已加载基座 + 多个 adapter（[Part 6](../part6/multi-lora-serving.md)）意味着**基座权重只付一次**；adapter 增加的 VRAM 可忽略。显存闸由基座 + KV 决定，基本同 Q1。
- **吞吐：** vLLM 靠 grouped GEMM 在一步里 batch 异构 adapter，所以 $r_{\text{inst}}$ 接近单模型的数——每 adapter GEMM 有点小开销。按 Q1 用跨租户*聚合* QPS 定容。
- **adapter 容量：** `--max-loras` 限制每步*常驻*多少 adapter；`--max-cpu-loras` + 动态加载把长尾从 CPU 换入换出。热集常驻、冷集换入换出。

### 架构

```text
  租户 ─▶ 网关（认证，租户 ⇒ adapter_id）─▶ ROUTER ─▶ [ vLLM 副本 × N ]
                                                       │         基座 Qwen2.5-7B +
                                                       │         LoRA 池 (--enable-lora,
                      adapter 感知路由：                │         --max-loras, --max-cpu-loras)
                      把热 adapter 钉到副本             │
                      （避免反复加载抖动）              └─▶ adapter 注册表 / 对象存储
```

- **adapter 感知路由：** 把某租户的请求路由到已常驻其 adapter 的副本——像前缀感知路由但针对 adapter——避免不停加载/淘汰抖动。
- **热/冷分层：** 常驻 top-K adapter（`--max-loras`），按需从 CPU/注册表换入长尾（`--max-cpu-loras`、动态加载端点）。
- **隔离：** adapter_id 由已认证租户在服务端推出、绝不由 client 提供；请求无法选别的租户权重。KV 是每请求的，无跨租户 KV 共享。

### 瓶颈与失败模式

- **adapter 抖动：** 若活跃 adapter 工作集超过 `--max-loras`，副本反复加载/淘汰 → 延迟尖峰。用 adapter 感知路由 + 抬高常驻上限（VRAM 允许）或把租户分片到副本组解决。
- **一个热租户饿死别人：** 一个租户的突发填满共享 batch。在网关做每租户限流 / 公平队列。
- **冷租户 TTFT：** 冷 adapter 的首请求要付一次加载。若罕见则可接受；对已知活跃租户预热。

### 面试官追问

- *「一个租户要全量微调，不是 LoRA。」* → 它不再共享基座——这是独立的模型部署（自己的副本、自己的集群算术）。LoRA 共服之所以行，正因为基座共享。
- *「不管租户 B 怎样都保租户 A 的 p99。」* → *延迟*隔离要么靠预留容量（专属副本组）、要么靠严格公平调度；仅共享 batch 只给尽力而为。权衡成本 vs 保证。
- *「一个副本真能服多少 adapter？」* → 常驻数是 `--max-loras`（VRAM 界）；可寻址总数是它加上 CPU 可换池——但对延迟真正要紧的是每步的*工作集*。

---

## Q3：设计长上下文 RAG 服务 —— 32k 上下文、大的共享知识库前缀、p99 TTFT ≤ 2 s

**类型：** 系统设计（长题）

### 厘清 → SLO

假设：`Qwen2.5-7B` 在 **32k 上下文**，RAG 中许多请求共享一个**大的检索文档前缀**（或一个大的固定系统语料），QPS 中等，**p99 TTFT ≤ 2 s**（长 prompt 让 TTFT 成为硬 SLO），输出短。核心张力：[KV 墙](../part6/long-context-inference.md)与 prefill 成本。

### 草稿纸算术

- **KV 墙主导 VRAM：** 一条 32k 序列花 $\kappa\cdot32\text{k}$；Qwen2.5-7B 56 KiB/token 即*每条流* ~1.75 GiB KV（BF16 KV）。相比短上下文并发骤降——绑定的是显存闸，不是速度闸。FP8 KV 减半；权重量化腾出基座。
- **prefill 主导 TTFT：** 32k 输入 token 是一次大 prefill；朴素的每请求 prefill 会冲破 2 s TTFT。**prefix caching** 是杠杆——若 KB 前缀共享，其 KV 只算一次、复用，于是 TTFT 降到只剩*独特*后缀的成本。
- **集群：** 单实例容量低（并发的 32k 流没几条）；按这个降低的 $r_{\text{inst}}$ 定容，并靠前缀复用抬高有效吞吐。

### 架构

```text
  clients ─▶ 网关 ─▶ 前缀感知 ROUTER ─▶ [ vLLM 副本 × N ]  (--enable-prefix-caching,
                        │                   │                  --max-model-len 32768,
     按 KB 前缀 hash 路由 │                  │                  --kv-cache-dtype fp8,
     让共享语料命中同一   │                  │                  开 chunked prefill)
     副本的 cache         │                 └─▶ (可选) LMCache 把 KV offload 到 CPU/NVMe
                        │                        用于大到留不住的 KB 前缀
```

- **prefix caching 是核心决策：** 共享 KB/系统前缀 → 每副本只算一次其 KV、跨请求复用 → TTFT 变成只剩独特后缀的 prefill。按前缀 hash 路由让同一语料落到同一副本（[路由课](../part8/routing-autoscaling.md)）。
- **KV 足迹控制：** FP8 KV（`--kv-cache-dtype fp8`）熬过 KV 墙；`--max-model-len` 卡到真实上限；对大到留不住的 KB 前缀考虑 KV offload（production stack 里的 LMCache）。
- **prefill 调度：** [chunked prefill](../part5/scheduler-chunked-prefill-pd.md) 让 32k prefill 不阻塞其他请求的 decode；若 prefill 与 decode 争抢严重则考虑 PD 分离。

### 瓶颈与失败模式

- **KV OOM / 抢占：** 太多并发长流耗尽 KV 池 → 抢占/重算 → 延迟尖峰。限并发、FP8 KV、对长请求做准入控制。
- **cache miss 风暴：** 若路由非前缀感知，每副本都重 prefill 共享语料 → 违反 TTFT SLO。这里前缀感知路由不是可选项。
- **每请求独特前缀**（无共享）抽掉主杠杆 → 又回到每次付满 32k prefill；那时诚实答案是更少并发流 + 更多 GPU，或更小上下文。

### 面试官追问

- *「上下文涨到 128k。」* → 每条流 KV ~7 GiB（BF16）——几条就占满一张 24 GB 卡。需激进 FP8 KV + 权重量化、极低并发，或分片模型（TP）/ offload KV；重新审视 128k 是否真必要。
- *「实际中 prefix caching 命中率低。」* → 测它；若共享前缀其实不共享（按用户检索），设计前提就垮了——围绕*真正*共享的东西（固定系统语料）重构，或接受 prefill 成本。
- *「TTFT 没事但吞吐很差。」* → 长上下文是 KV-bound；你用并发换了上下文。用 FP8 KV / 权重量化抬它，或把长上下文流量分到自己的池，别饿死短请求。

---

## 关联概念

- 课程：[容量规划：从一张卡的吞吐到一个集群](../part8/capacity-planning.md) —— 三道设计都以之开场的草稿纸算术（可行性 → 单实例 → 集群）。
- Capstone：[在单张 4090 上把 Qwen2.5-7B 吞吐拉满](../capstone/index.md) —— 这些设计动手的另一半：真去爬优化阶梯、产出你会带进这场面试的优化前→后报告。
- 相关题目：[显存预算与最大并发](vram-capacity-planning.md)（显存闸）、[压测与并发拐点](load-testing-knee.md)（$r_{\text{inst}}$）、[路由、自动扩缩与 KV 感知路由](routing-autoscaling.md)（运行时的集群）、[SLO 驱动调优](slo-driven-tuning.md)（调参 vs 扩容）、[Multi-LoRA serving](multi-lora-serving.md)（Q2）、[长上下文推理](long-context-inference.md)（Q3）、[并行策略：TP/PP/DP/EP](parallelism-strategies.md)（TP degree）。
- 术语：[SLO、Knee、Goodput、TP degree、KV-cache aware routing、Prefix caching](../glossary.md)
