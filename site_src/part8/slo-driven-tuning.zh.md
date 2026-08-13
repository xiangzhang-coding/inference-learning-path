# SLO 驱动调优：从指标到调优闭环

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090"
    已按 ADR-0004 用 Context7 对照 vLLM 0.26.0 核实：旋钮就是你见过的引擎 flag——**`--max-num-seqs`**、**`--max-num-batched-tokens`**（chunked-prefill 旋钮）、**`--gpu-memory-utilization`**、**`--max-model-len`**、**`--enable-prefix-caching`**、投机解码（`--speculative-config`）与[量化](../part4/quantization-methods.md)。你用 **`vllm bench serve`** 对着 SLO 测量（分位经 `--percentile-metrics "ttft,tpot,itl,e2el"`；它按 SLO 校验 **goodput**），并从 **`/metrics`**（`vllm:num_requests_waiting`、`gpu_cache_usage_perc`、prefill/decode 直方图）读出绑定约束。本节所有数字均为**示例 / 量级参考**——获胜配置与 workload 相关，请自测。

---

## 1 · 直觉 & 为什么重要

你能[测 knee](load-testing-knee.md)、能[读指标](observability-profiling.md)。现在 on-call/系统设计的问题：**给定延迟目标，你怎么调引擎、在仍满足它的前提下服务最多流量？** 随手拧旋钮是瞎折腾。调优是一个*闭环*，而它从业务给你的一个数字开始、不是从旋钮开始。

那个数字是 **SLO**——例如「p99 TTFT ≤ 300 ms 且 p99 TPOT ≤ 50 ms，@ 20 请求/秒」。一切随它而来：

1. **SLO 定义成功，指标是 goodput。** 裸吞吐是虚荣；**goodput**——满足*全部* SLO 目标的请求/秒——才是分。一个做 1500 tok/s 却打穿 p99 TTFT 的配置，对 TTFT SLO 的 goodput 是*零*。
2. **你调的是*绑定*约束、不是随便一个旋钮。** 任一时刻有一样东西在限制你：prefill 太慢、decode 太慢、KV pool 满、或队列深。指标告诉你哪个。当**队列**是问题时去拧 decode 旋钮毫无用处——那是容量问题，靠[加副本](routing-autoscaling.md)、不靠调优。

所以：定义 SLO、从指标找绑定约束、拧那*一个*能移动它的旋钮、重测 goodput、重复。本节就是这个闭环的具体化。→ 术语见 [术语表](../glossary.md) 的 *SLO、Goodput、Knee*。

## 2 · 心智模型

SLO 在延迟/吞吐平面上圈出一个**目标框**。调优 = 在框*内*把 goodput 尽量推高，靠放松当前绑定的那面墙。

```text
   SLO 是一个框；调优把你移到框内
                                          闭环：
   p99 TTFT                               ┌───────────────────────────────────────────┐
     ▲   ✗ 超 TTFT SLO                    │ 1. 定义 SLO   p99 TTFT≤300ms, TPOT≤50ms     │
     │ ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ ← TTFT 上限        │ 2. 测量      vllm bench serve → goodput     │
     │ ░░░░░░░░░░░░ │                      │ 3. 诊断      哪个约束在绑定？                │
     │ ░ 目标框  ░ │  在这里面最大化       │      队列？ prefill？ decode？ KV pool？     │
     │ ░         ░ │  goodput              │ 4. 拧一个旋钮 移动那个约束                   │
     │ ░░░░░░░░░░░░ │                      │ 5. 重测 goodput → 更好就保留                │
     └─────────────┴──────────▶ QPS       │    重复直到 goodput 不再改善                │
                    ✗ 超 QPS = 队列        └───────────────────────────────────────────┘

   诊断 → 旋钮：
     队列深 (num_requests_waiting↑)  → 不是调优问题：加副本 / 路由（Part 8 路由）
     prefill 慢 (request_prefill_time↑, TTFT↑) → --max-num-batched-tokens（chunked prefill）、prefix caching
     decode 慢  (request_decode_time↑, TPOT↑)  → --max-num-seqs、量化、投机解码
     KV 满 (gpu_cache_usage_perc→1.0)          → --gpu-memory-utilization↑、--max-model-len↓、KV 量化
```

三个要记住的形状：

- **SLO 先行；goodput 是分。** 没有目标你无法说一个配置「更好」——一个轴更快总意味另一个轴更慢。SLO 把多目标混乱收成一个数：框内的 goodput。
- **一个旋钮移一面墙。** 多数旋钮在 TTFT 与吞吐间权衡（`--max-num-batched-tokens` 这个 chunked-prefill 旋钮是最干净的例子）。**一次一个**、重测，否则你无法归因。
- **不是每个问题都是调优问题。** 队列深意味你过了 [knee](load-testing-knee.md)——没有引擎旋钮能把天花板抬高多少；你需要*更多实例*。正确诊断出这点，能省下你对着容量墙调好几小时。

## 3 · 原理

### 3.1 从 SLO 起步

把目标明确写下：延迟分位（**p99 TTFT**、**p99 TPOT/ITL**）、你必须撑住的**吞吐/QPS**，有时还有成本上限。goodput 于是定义清楚：*全部*目标都成立的请求率。每个调优决定都以它是否抬高 goodput 来判——别无其他。

### 3.2 诊断绑定约束

读 `/metrics`（上一节）找此刻真正限制你的东西：

- **队列绑定** —— `num_requests_waiting` 深且在涨。你过了 knee；**这不是调优问题**。加副本 / 扩缩（[路由那节](routing-autoscaling.md)）。
- **prefill 绑定** —— `request_prefill_time_seconds` 与 TTFT 高；长 prompt 拖住 batch。调 prefill。
- **decode 绑定** —— `request_decode_time_seconds` 与 TPOT 高；运行 batch 受带宽限。调 decode。
- **KV 绑定** —— `gpu_cache_usage_perc` 接近 1.0、在抢占。腾出或扩大 KV。

### 3.3 对症的旋钮

每个旋钮移一面墙（这些是 [Part 5 的调参旋钮](../part5/tuning-knobs-sweep.md)，现在按它缓解哪个约束来框）：

| 绑定约束 | 旋钮 | 效果 / 权衡 |
|---|---|---|
| prefill / TTFT | **`--max-num-batched-tokens`**（chunked prefill） | 更小的块 → 更低 TTFT（decode 更早交织），但 prefill 吞吐略降；最干净的 TTFT↔吞吐旋钮 |
| prefill（共享 prompt） | **`--enable-prefix-caching`** | 复用共享前缀 KV → 跳过重复 prefill → RAG/chat 的 TTFT 更低 |
| decode / TPOT | **`--max-num-seqs`** | 更宽 batch → 更多吞吐，但过点后 TPOT 更长且 KV 压力 |
| decode / TPOT | **[量化](../part4/quantization-methods.md)**（W4A16、FP8） | 每 token 更少权重带宽 → decode 更快；注意质量 |
| decode / TPOT | 投机解码（`--speculative-config`） | draft-and-verify → memory-bound 时更少前向；高负载下反效果 |
| KV pool | **`--gpu-memory-utilization`** ↑ | 更大 KV block pool → 更多并发余量（若 VRAM 有余） |
| KV pool | **`--max-model-len`** ↓ | 封顶单请求 KV → 更多并发序列装得下 |

### 3.4 闭环

**定义 SLO → 测 goodput（`vllm bench serve`）→ 从 `/metrics` 诊断绑定约束 → 拧那一个对症旋钮 → 重测。** 只在一个配置*抬高 goodput 且仍满足 SLO* 时保留它。当 goodput 走平就停——你撞上了这个 workload 在这硬件上的真实极限，更多收益需要不同硬件或更多副本。关键：对着匹配生产的 workload（输入/输出长度分布）调；512-in/128-out 的获胜配置不是 4k-in/1k-out 的获胜者。

## 4 · 完整可跑代码 + 逐行讲解

一个 SLO 门控的调优闭环：扫一个旋钮，保留在 SLO *约束下*使 goodput 最大的值。

```python title="slo_tune.py"
"""对一个旋钮（这里：--max-num-seqs）做 SLO 驱动的调优闭环。
对每个候选值：（重）启 server、跑 vllm bench serve、以 GOODPUT 打分——同时满足 p99 SLO 的吞吐。
保留最优配置。逻辑只读；run 需要 GPU + server。数字为示例。"""
import json, subprocess, time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SLO = {"p99_ttft_ms": 300, "p99_tpot_ms": 50}    # 那个 SLO——成功在这里定义、不是靠旋钮
CANDIDATES = [64, 128, 256, 384]                 # 一次扫一个旋钮
TEST_QPS = 20                                    # SLO 必须撑住的负载

def measure(max_num_seqs):
    server = subprocess.Popen(["vllm", "serve", MODEL,   # 用这个旋钮值重启
        "--max-num-seqs", str(max_num_seqs), "--gpu-memory-utilization", "0.90"])
    wait_until_ready("http://localhost:8000/health")     # 发负载前轮询 /health (200)
    subprocess.run(["vllm", "bench", "serve", "--backend", "vllm", "--model", MODEL,
        "--endpoint", "/v1/completions", "--dataset-name", "random",
        "--random-input-len", "512", "--random-output-len", "128",
        "--num-prompts", "500", "--request-rate", str(TEST_QPS),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result", "--result-filename", "r.json"], check=True)   # --result-filename 为示意
    r = json.load(open("r.json"))
    server.terminate(); time.sleep(5)
    meets_slo = r["p99_ttft_ms"] <= SLO["p99_ttft_ms"] and r["p99_tpot_ms"] <= SLO["p99_tpot_ms"]
    goodput = r["request_throughput"] if meets_slo else 0.0     # goodput = 吞吐，仅当 SLO 成立
    return goodput, r["p99_ttft_ms"], r["p99_tpot_ms"], meets_slo

best = (0.0, None)
for v in CANDIDATES:                              # 一个旋钮，几个值
    goodput, ttft, tpot, ok = measure(v)
    print(f"max_num_seqs={v:>3} | p99 TTFT {ttft:6.1f} | p99 TPOT {tpot:5.1f} | "
          f"goodput {goodput:5.2f} req/s | {'MEETS SLO' if ok else 'VIOLATES SLO'}")
    if goodput > best[0]:
        best = (goodput, v)                       # 保留 SLO 通过下 goodput 最高的配置
print(f"\n最优 --max-num-seqs = {best[1]}，goodput {best[0]:.2f} req/s（示例——请自测）")
```

**逐行讲解：**

- **`SLO = {...}`** —— 闭环的北极星。成功是「满足这些 p99 目标」；错过就得**零 goodput**，不管它均值多快。
- **`CANDIDATES`（一个旋钮）** —— 单扫 `--max-num-seqs`。每 run 改多个旋钮会让结果无法归因（§6）；隔离一个。
- **`wait_until_ready(/health)`** —— [server 那节](openai-server.md)的存活性门：引擎起来后才压，否则你测到冷启动。
- **`--percentile-metrics "ttft,tpot,itl,e2el"`** —— 拉**尾巴**；SLO 在 p99 上，均值会误导。
- **`goodput = throughput if meets_slo else 0.0`** —— 整套纪律的一行：吞吐*只在* SLO 成立时才算数。这就是阻止你「优化」进一个更快但违反 SLO 配置的东西。
- **`best = max goodput`** —— 保留服务最多 SLO 合规流量的值。sweep 走平时，你找到了这个 workload 在这硬件上的天花板——更多收益需[副本](routing-autoscaling.md)或不同硬件，不是更多调优。
- **`--result-filename` / JSON key** —— 为示意（如 [knee 那节](load-testing-knee.md)）；请对你的版本确认确切结果文件 flag 与字段名。

## 5 · Lab —— 在 4090 上按 SLO 调优

!!! gpu "GPU Lab（单卡）"
    - **最低显存：** `Qwen2.5-7B-Instruct` 在 **24 GB RTX 4090**（BF16，或 INT4 换更多 KV 余量）。闭环每候选重启 server，各留几分钟。
    - **建议 AutoDL 卡型：** 单张 **RTX 4090 (24 GB)**（ADR-0001）。
    - **预估耗时 / 花费：** 4 点单旋钮 sweep 约 30–50 分钟 · **约 ¥2–6**（示例）。`--num-prompts` 保持适中；goodput 曲线的*形状*是产出。
    - **平台：** NVIDIA CUDA（默认）。**非 NVIDIA：** 闭环是 HTTP + benchmark 客户端——与硬件无关；只是获胜值按后端不同。

步骤：

1. **把 SLO 写下来。** 挑反映你用例的 p99 TTFT / TPOT 目标与测试 QPS。一切以它们为准。
2. **先诊断。** 跑一个基线、读 `/metrics`：队列深（→ 这是路由/容量问题，停止调优）、prefill 绑定、decode 绑定、还是 KV 绑定？调对症旋钮。
3. **扫一个旋钮。** 跑 `slo_tune.py`（从 `--max-num-seqs` 起，或 prefill 绑定时用 `--max-num-batched-tokens`）。看 goodput 先升后走平；记下 p99 越过 SLO 之处。
4. **确认，别堆。** 取获胜值，再从那个基线调*第二个*旋钮——一次一个。goodput 走平就停。**关机。**

## 6 · 常见坑 / 反直觉点

- **无 SLO 调优。**「更快」没有答案——更快的 TTFT 通常以吞吐为代价、反之亦然。没有目标框，你无法称任何配置更好。先写 SLO。
- **优化裸吞吐。** 一个打满 tok/s 却违反 p99 TTFT 的配置，对延迟 SLO 的 goodput 是**零**。以 goodput 打分、不是吞吐。
- **一次拧几个旋钮。** 同时改 `--max-num-seqs` 与 `--max-num-batched-tokens`，你分不清哪个帮了（或它们抵消了）。每 run 一个旋钮、重测、再换。
- **调错约束。** 若 `num_requests_waiting` 深，瓶颈是**容量**、不是引擎配置——没有 decode 旋钮能修队列。加[副本](routing-autoscaling.md)。调之前先诊断。
- **在错的 workload 上调。** 短 prompt 的获胜配置在长 prompt 上输（prefill- vs decode-heavy 饱和不同资源）。对着匹配生产的 prompt/长度分布调。
- **越过平台还追。** goodput 不再改善时，你撞上了这个 workload 的硬件极限。再拧是噪声；真杠杆是不同硬件或更多实例。
- **量化/投机时无视质量。** [量化](../part4/quantization-methods.md)与投机解码抬高 decode goodput，但可能移动质量；在你的[评测集](../eval/index.md)上验证，别只看延迟数。

## 7 · 面试连线

- [SLO 驱动调优：goodput、绑定约束与闭环](../interview/slo-driven-tuning.md) —— 本节为你准备的高频题：*为何从 SLO 起步、以 goodput 打分，怎么从指标读绑定约束（队列 / prefill / decode / KV）、哪个旋钮缓解哪个、以及为何一次一个旋钮、对着类生产 workload 才是唯一诚实的闭环。*

## 8 · 小结 & 延伸阅读

**一句话：** 调优是锚在 **SLO** 上的闭环：定义延迟/吞吐目标，以 **goodput**（*满足* SLO 的吞吐——用 `vllm bench serve` 测）给每个配置打分，从 `/metrics` 读**绑定约束**（队列 → 加副本、别调；prefill → `--max-num-batched-tokens` / prefix caching；decode → `--max-num-seqs` / 量化 / 投机解码；KV → `--gpu-memory-utilization` / `--max-model-len`），拧那**一个**对症旋钮、重测、goodput 走平就停——始终对着类生产 workload。

延伸阅读：

- [调参旋钮 sweep](../part5/tuning-knobs-sweep.md) —— 每个旋钮的机制与其吞吐/延迟曲线。
- [压测那节](load-testing-knee.md) —— 这个闭环优化对象的 knee 与 goodput。
- [可观测性那节](observability-profiling.md) —— 揭示绑定约束的指标。
- vLLM `docs/configuration/optimization.md` —— `max_num_batched_tokens` 与 chunked-prefill 的 TTFT/吞吐权衡。

## 9 · 自测小问

??? question "配置 A 做 1500 tok/s、p99 TTFT 900 ms；配置 B 做 1100 tok/s、p99 TTFT 250 ms。你的 SLO 是 p99 TTFT ≤ 300 ms。哪个更好？通用原则是什么？"
    **B。** 对着 p99 TTFT ≤ 300 ms 的 SLO，配置 A **违反**目标，所以它的 *goodput* 是 **0**、不管裸吞吐多高——每个请求都太慢、不算数。配置 B 满足 SLO，所以它的 1100 tok/s 是真实、可用的 goodput。原则：**以 goodput 打分，不是裸吞吐。** 没有延迟预算的吞吐是虚荣；唯一重要的数是你在 SLO *内*服务了多少流量。A 在吞吐图上总更好看，在这个 SLO 的生产里却无用。

??? question "你花一下午调 `--max-num-seqs`、量化、投机解码，但 p99 几乎不动。然后你注意到 `vllm:num_requests_waiting` 全程都深。哪里错了？"
    你调错了约束。深且持续的 **`num_requests_waiting`** 意味请求在**排队**——你过了 [knee](load-testing-knee.md)，所以瓶颈是**容量**、不是引擎配置。decode/KV 旋钮在一个实例的天花板*内*移动吞吐与 TPOT，但没有一个能把天花板抬到足以排掉由「到达超过最大完成率」造成的常驻队列。修法是**更多实例**（横向扩）加[前缀感知路由 / 自动扩缩](routing-autoscaling.md)，不是又一个旋钮。教训：**先从指标诊断绑定约束**——若队列是墙，停止调优、加容量。

??? question "为什么一次调一个旋钮，为什么对着镜像生产的 workload 重测、而非通用 benchmark？"
    **一次一个旋钮**，因为旋钮相互作用且彼此权衡（多数在 TTFT↔吞吐间）：一次改两个你无法归因增量——一个的收益可能被另一个掩盖或抵消，你会「学到」错误教训。隔离、重测 goodput、保留或回退、再换下一个。**类生产 workload**，因为绑定约束取决于 prompt/输出长度分布：prefill-heavy（长入短出）与 decode-heavy（短入长出）受不同资源限，所以获胜的 `--max-num-batched-tokens` / `--max-num-seqs` 不同。在 512-in/128-out 上调、给 4k-in/1k-out 部署，就上线了一个为错误瓶颈优化的配置。让 benchmark 的输入/输出分布（与缓存命中模式）匹配生产，否则你测到的 goodput 不是你会拿到的 goodput。
