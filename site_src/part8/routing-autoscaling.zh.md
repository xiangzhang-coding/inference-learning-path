# 路由、自动扩缩与 KV 感知路由（多实例）

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 多实例（单卡或多卡）"
    已按 ADR-0004 用 Context7 对照 vLLM 0.26.0 核实：要扩到一个实例之外，vLLM 指向 **production stack**（`helm repo add vllm https://vllm-project.github.io/production-stack`、`helm install vllm vllm/vllm-stack -f values.yaml`），它在 **engine** pod 前部署一个 **router** pod，提供**模型感知与前缀感知路由 (model-aware and prefix-aware routing)**，外加经 **LMCache** 的 KV-cache offload。对数据并行部署，文档说得很明确：*每个引擎各自维护独立 KV cache，所以智能路由能最大化 prefix-caching 收益*。自动扩缩读负载信号——SkyPilot 的 `replica_policy` 用 **`target_qps_per_replica`**；Kubernetes HPA 读引擎自身 `/metrics` 的 **`vllm:num_requests_waiting`** gauge。本节所有数字均为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

[上一节](load-testing-knee.md)给了你一个数：一个实例的 **knee**——过了它延迟就失控的到达率。真实流量会超过一个实例的 knee。于是你做那件显然的事：跑**若干实例**，在前面放一个 **router** 把请求铺开。那就是横向扩展，也是「一个引擎」变成「一个服务」的地方。

但有两个决定，把一堆实例变成一个*好*服务，两者都是面试宠儿：

1. **怎么路由和开几个一样重要。** 朴素答案——round-robin——常常是*错的*，因为它无视你在 Part 5 造好的最大免费收益：**prefix cache**。若两个请求共享一段长 system prompt，把它们发到**同一个**实例，第二个就能复用第一个缓存的 KV；round-robin 把它们打散、每个实例都重做一遍 prefill。**KV 感知（前缀感知）路由**把请求落点变成一个缓存命中优化。
2. **怎么扩——以及按什么信号。** 更多流量 → 更多实例，自动地。但你按之扩缩的信号，决定它成不成。GPU 利用率是个**陷阱**（memory-bound 的 decode workload 可以「100% 忙」却仍有队列余量，或「低利用」却 KV cache 已满）。真正跟踪「我过 knee 了吗」的信号是**队列深度**——`vllm:num_requests_waiting`——这就是 vLLM 导出它、自动扩缩器挂它的原因。

所以：先是跨副本路由（以及前缀感知为何胜过 round-robin），再是自动扩缩（以及为何队列、而非 GPU 利用率，是触发信号）。→ 术语见 [术语表](../glossary.md) 的 *KV 感知路由、SLO、Knee*。

## 2 · 心智模型

一个 **router** 坐在 N 个独立引擎副本前——*每个各有自己的 KV cache*——一个**自动扩缩器**看负载信号、改 N。

```text
                                  ┌─────────── 自动扩缩器 ───────────┐
                                  │ 读 vllm:num_requests_waiting     │
                                  │ (队列深度) → 增/减 N              │
                                  └───────────────┬──────────────────┘
                                                  │ 设 N
        请求                                       ▼
   ───────────────▶  ┌───────────────┐     ┌────────────────────────────────┐
                     │    ROUTER     │────▶ │ 副本 0   [KV cache A]           │
                     │ round-robin？ │────▶ │ 副本 1   [KV cache B]  ← 各自独立 │
                     │ 前缀感知？     │────▶ │ 副本 2   [KV cache C]    的缓存！  │
                     └───────────────┘      └────────────────────────────────┘
                       │
   ROUND-ROBIN: 同前缀请求打散 → 每个副本都重 prefill 共享 prompt（缓存未命中）
   前缀感知:     按缓存前缀路由 → 已有它的副本来服务（缓存命中，跳过 prefill）
```

三个要记住的形状：

- **缓存是每实例的，不共享。** 副本 0 的 prefix cache 与 KV block 对副本 1 不可见。所以你把请求发*去哪*决定它命不命中热缓存。路由因而是个**缓存落点**问题，不只是铺负载。
- **前缀感知路由 =「发到它前缀已在的那台」。** 若一个请求的前导 token（system prompt、few-shot 前言、对话历史）已缓存在某副本上，就路由过去、跳过 prefill。round-robin 把这个丢掉；前缀感知路由正是 production stack 标榜它的全部理由。
- **按队列扩缩，不按利用率。** `num_requests_waiting > 0` 且在涨，意味请求*此刻*过了 knee——直接、与模型无关的过载信号。GPU 利用率混淆 compute-bound 与 memory-bound 两种状态、对 decode 的余量撒谎。按队列（或按校准到 knee 的 QPS 代理）扩 N。

## 3 · 原理

### 3.1 多实例，独立缓存

扩展单元是一个完整引擎实例（一个 `vllm serve` 进程，它自己可能跨 GPU 做 [Part 7](../part7/index.md) 的 TP/PP）。跑若干个——在若干 GPU、若干节点或若干 pod 上——挂在一个地址后面。每个实例都**独立**：自己的权重（或一份只读共享副本）、自己的 KV-cache block pool、自己的 [prefix cache](../part5/prefix-caching.md)。除非你加一个外部 KV 存储（LMCache，见下），实例间什么都不共享。这个独立性正是路由不中立的原因：两个相同请求，纯凭 router 挑了哪个副本，就命中热缓存或冷缓存。

### 3.2 负载均衡：round-robin vs KV 感知

- **round-robin / 最少负载。** 按数量或当前负载均匀铺请求。简单，且在请求独立且短时是对的。但它**无视前缀复用**：共享长 system prompt 的请求被打散，于是每个副本再付一遍共享 prefill。你恰在它最要紧时（长共享前缀）丢掉 prefix-cache 收益。
- **KV 感知 / 前缀感知路由。** 按*内容*路由：哈希请求前缀、在负载上限内发给已有该前缀缓存的副本。现在共享 system prompt 每副本**只 prefill 一次**并复用；[DP 部署文档](../part7/index.md)直说了理由——*因为每个引擎维护独立 KV cache，智能请求路由能最大化 prefix-caching 收益*。代价是热点风险（一个流行前缀堆到一个副本上），所以真实 router 会把前缀亲和与负载均衡揉在一起。

**production stack** 的 router 把这个内建：它提供**模型感知**路由（挑服务对应模型的副本）与**前缀感知**路由（挑有热前缀的副本），并能把 KV offload 到 **LMCache**，让一个副本 GPU 上被逐出的前缀被重取、而非重算。

### 3.3 自动扩缩——以及那个信号

自动扩缩随负载改副本数 N。整个游戏是**按哪个信号**扩缩：

- **队列深度——`vllm:num_requests_waiting`。** 「请求过了 knee」的直接读数。vLLM 在 `/metrics` 导出它，正是为了让 Kubernetes **HPA**（或 KEDA Prometheus scaler）挂它：waiting 涨 → 加副本；waiting 停在 0 且 running 低 → 撤掉。这是推荐信号，因为它对 prefill-heavy 与 decode-heavy workload 意思一样。
- **每副本 QPS。** SkyPilot serve 的 `replica_policy` 在 `min_replicas` 与 `max_replicas` 间按 **`target_qps_per_replica`** 扩缩——一个更粗但有效的代理，*前提是你把目标校准到上一节测出的 knee*。把 `target_qps_per_replica` 设在（或略低于）knee，扩缩器就把每副本保持在线性区。
- **GPU 利用率——陷阱。** 诱人且错。decode 是 [memory-bound](../part0/inference-flow.md) 的：一个副本可以报高「利用率」却仍有 batch 余量，或报适中利用率却 KV cache 已满、正在排队。利用率不跟踪 knee；队列跟踪。

两个运维现实：**冷启动**——新副本要加载权重、热 CUDA graphs（数十秒），所以扩容**滞后**；备好余量或预热。以及**缩容安全**——杀一个副本前，让它**排空 (drain)**（`vllm:num_requests_running` 与 `vllm:num_requests_waiting` 都到 0），别丢在途请求。

### 3.4 拼起来

一个生产部署的形状：一个 **router**（前缀感知、模型感知）在一个引擎 pod **Deployment** 前，一个**自动扩缩器**盯 `num_requests_waiting`，`/health` 做存活性、`/metrics` 做负载，以及——可选——**LMCache** 作为共享 KV 层，让跨副本未命中花一次取、而非一次完整 prefill。production stack 把这一切作为一个 Helm chart 发布；你也可以用一个普通负载均衡器 + HPA + 你自己的路由规则拼出来。

## 4 · 完整可跑代码 + 逐行讲解

部署 production stack（router + 副本），再是两种自动扩缩信号：一个挂队列 gauge 的 Kubernetes HPA，和 SkyPilot 的 QPS 策略。

```yaml title="values.yaml —— production stack：2 副本挂在前缀感知 router 后"
# helm repo add vllm https://vllm-project.github.io/production-stack
# helm install vllm vllm/vllm-stack -f values.yaml
# 注：Helm 安装、router pod + 引擎 pod、模型/前缀感知路由均已核实；下面 values.yaml 的确切
#     字段名为示意——请对照 production-stack chart 自带的 values.yaml（按你的 chart 版本）确认 schema。
servingEngineSpec:
  modelSpec:
    - name: "qwen"
      repository: "vllm/vllm-openai"
      modelURL: "Qwen/Qwen2.5-7B-Instruct"   # 所服务的模型
      replicaCount: 2                         # 起步 2 个引擎 pod，每个各是独立缓存
      requestGPU: 1                           # 每副本 1 张 GPU（TP>1 会请求更多）
routerSpec:
  routingLogic: "prefixaware"                 # KV 感知：把请求发给有其热前缀的副本
                                              # （round-robin 会打散共享前缀 → 缓存未命中）
```

```yaml title="hpa.yaml —— 按队列扩缩，不按 GPU 利用率"
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-engine
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: vllm-engine }
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting     # /metrics 的 gauge = 队列深度 =「过了 knee」
        target:
          type: AverageValue
          averageValue: "5"                   # 平均队列 > 5 就加副本（按你的 SLO/knee 调）
```

```yaml title="skypilot：基于 QPS 的自动扩缩（把目标校准到测出的 knee）"
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 8                 # = 压测那节测出的每实例 knee
  readiness_probe:
    path: /v1/chat/completions                # 一个真实请求，不只是 /health——证明它能服务
    post_data: { model: qwen2.5-7b, messages: [{role: user, content: "ping"}], max_completion_tokens: 1 }
```

**逐行讲解：**

- **`replicaCount: 2` + 独立缓存** —— 两个引擎 pod，每个有**自己**的 KV/prefix cache。那份独立性正是 router 策略要紧的原因：同一请求在一个 pod 命中、在另一个未命中。
- **`routingLogic: "prefixaware"`** —— router 哈希每个请求的前缀、发给已持该前缀 KV 的副本，把一段共享 system prompt 从每请求一次 prefill 变成每副本**一次** prefill。换成 round-robin，你就为共享前缀流量放弃了 [prefix-caching](../part5/prefix-caching.md) 收益。
- **HPA `metric: vllm_num_requests_waiting`** —— 扩缩器盯 vLLM 在 `/metrics` 导出的**队列深度 gauge**。平均队列 > 5 → 扩容；排空趋 0 → 缩容。这是与模型无关的过载信号；无论负载 prefill-heavy 还是 decode-heavy 都一个意思——不像 GPU 利用率。
- **`averageValue: "5"`** —— 阈值是你的策略旋钮：按 SLO 与 knee 设（加容量前你能容忍的一小段常驻队列）。太低 → 抖动；太高 → 扩容滞后期错过 SLO。
- **`target_qps_per_replica: 8`** —— SkyPilot 更粗的旋钮：把每副本保持在约 8 请求/秒，那应是你上一节**测出的 knee**。knee 之下每副本停在线性区；总 QPS 涨时扩缩器加副本。
- **`readiness_probe` 打一个真实 chat 请求** —— 就绪必须证明 pod 真能*服务*（权重加载完、引擎热），而 `/health`（只存活性）做不到。一个极小 `max_completion_tokens: 1` 的请求是诚实的就绪检查。

## 5 · Lab —— 看前缀感知路由打赢 round-robin

!!! gpu "GPU Lab（多实例——2 张 GPU，或单卡上 2 个小实例）"
    - **最低显存 / 卡数：** 两个引擎实例。**2 张 GPU**（各一副本）最干净。单张 24 GB 4090 上可用**两个小模型实例**（如 `Qwen2.5-0.5B-Instruct`）在不同端口、或两个 MPS 分片来近似——够*看路由行为*，不够压 7B 吞吐。
    - **建议 AutoDL 卡型：** 真实 7B 多实例测试用 **2× 4090 或 2× A100**、**「开机即关」**（ADR-0001）；或单张 4090 配两个小实例做路由演示。做完把多卡拆掉。
    - **预估耗时 / 花费：** 约 30–45 分钟 · 视卡 **约 ¥3–15**（示例）。产出是**缓存命中差异**，不是吞吐纪录。
    - **平台：** NVIDIA CUDA（默认）。**非 NVIDIA：** router 与自动扩缩器是基础设施（Helm/K8s/HTTP）、与硬件无关；只有引擎 pod 按后端不同。

步骤：

1. **两副本，一个共享前缀。** 起两个实例。造一批共享长 system prompt（如 1k-token 前言）、user 问题各异的请求。
2. **round-robin 基线。** 把共享前缀请求 round-robin 路由。在每个副本上 `curl /metrics | grep -i prefix`（prefix-cache 命中/总数计数器）——命中很低，因为前缀被打散、每个副本都重 prefill。
3. **前缀感知。** 把 router 切到前缀感知。发同样的请求。现在共享前缀反复落在一个副本上：它的 prefix-cache **命中计数器爬升**、它的 **TTFT 下降**（prefill 被跳过）。那个差距就是全部要点。
4. **按队列扩缩。** 把负载推过一个副本的 knee，看 `vllm:num_requests_waiting` 上升；确认你的 HPA/策略加了一个副本、队列排空。然后把任何多卡实例**关机**。

## 6 · 常见坑 / 反直觉点

- **round-robin 路由悄悄杀死 prefix cache。** 最常见的浪费：共享前缀流量（同 system prompt、RAG 前言或对话）被均匀打散，意味每个副本都重 prefill 共享部分。对共享前缀 workload，**前缀感知路由**能大幅削减 TTFT 与 prefill 成本——round-robin 把它丢了。
- **按 GPU 利用率自动扩缩。** decode 是 memory-bound 的：一个副本可以坐在「100% 利用」却有 batch 余量，或「40% 利用」却 KV cache 满、在排队。利用率不跟踪 knee。按 **`vllm:num_requests_waiting`**（队列深度）或校准到 knee 的 QPS 目标扩缩。
- **无视冷启动滞后。** 新副本要数十秒加载权重、热 CUDA graphs。若你只在队列已深*之后*才扩容，等 pod 就绪时 SLO 已被打穿。备余量、预热、或按前瞻指标扩缩。
- **不排空就缩容。** 杀一个有在途请求的 pod 会丢掉它们。终止前，停止把新活路由给它、等到 `num_requests_running` 与 `num_requests_waiting` **都**到 0——再撤。
- **以为缓存跨副本共享。** 不共享。副本 0 上热的前缀在副本 1 上是冷的。跨副本复用需一个显式共享 KV 层（production stack 里的 **LMCache**）；否则路由是你对缓存命中的唯一杠杆。
- **前缀亲和热点。** 纯前缀感知路由会把一个流行前缀堆到一个副本上、其他闲着。真实 router **揉合**前缀亲和与负载均衡；调这个平衡，别只按前缀路由。
- **把就绪当 `/health`。** `/health` 是存活性（引擎活着），不是就绪性（现在能服务）。在 `/health` 上就把 pod 加入轮转的负载均衡器，会把流量发给还在加载的引擎。用真实就绪探针（一个极小生成请求），如 §4。
- **流式穿过一个缓冲的负载均衡器。** 如[server 那节](openai-server.md)，一个缓冲响应的 LB 会把 SSE 流式塌成一个迟到的 chunk。配置 router/LB 不缓冲地透传流式响应。

## 7 · 面试连线

- [路由、自动扩缩与 KV 感知路由](../interview/routing-autoscaling.md) —— 本节为你准备的高频题：*前缀感知路由为何胜过 round-robin（每副本独立缓存）、为何按 `num_requests_waiting` 而非 GPU 利用率自动扩缩、以及冷启动与「缩容前排空」怎么塑造一个安全策略。*

## 8 · 小结 & 延伸阅读

**一句话：** 过了一个实例的 [knee](load-testing-knee.md)，你横向扩到 N 个独立引擎副本、挂在一个 **router** 后；让它变好的两个决定是 **KV 感知（前缀感知）路由**（缓存是每实例的，所以 round-robin 会重 prefill 共享 prompt）与**按队列自动扩缩**（`vllm:num_requests_waiting`，与模型无关的过载信号，不是 GPU 利用率），并处理好冷启动滞后与「缩容前排空」；vLLM 把它作为 **production stack** 发布（Helm：前缀感知 + 模型感知 router、引擎 pod、LMCache KV offload），SkyPilot 按 `target_qps_per_replica` 自动扩缩。

延伸阅读：

- vLLM `docs/deployment/integrations/production-stack.md` —— Helm chart、router（模型感知 + 前缀感知）、LMCache KV offload。
- vLLM `docs/serving/data_parallel_deployment.md` —— 为何每引擎独立 KV cache 让智能路由划算。
- vLLM `docs/deployment/frameworks/skypilot.md` —— `replica_policy` / `target_qps_per_replica` 自动扩缩与就绪探针。
- vLLM `docs/design/metrics.md` —— 你用来路由与扩缩的 `vllm:num_requests_waiting` 与 prefix-cache 命中/总数计数器。
- [prefix-caching 那节](../part5/prefix-caching.md) —— 前缀感知路由跨实例保住的那份每实例收益。

## 9 · 自测小问

??? question "你在一个 round-robin 负载均衡器后跑 4 个副本。所有请求共享一段 2000-token 的 system prompt。TTFT 很高、GPU 成本比预期差。发生了什么？怎么修？"
    round-robin 把共享前缀请求打散到全部 4 个副本，而**每个副本有自己独立的 prefix cache**——于是四个都为共享 system prompt 重跑那 2000-token 的 **prefill**、而不复用。你打败了 [prefix cache](../part5/prefix-caching.md)：共享前言被*每副本每请求*重 prefill、而非缓存。修法是 **KV 感知（前缀感知）路由**——哈希前缀、把匹配请求发给已持该前缀 KV 的副本，于是前言每副本 prefill 一次再复用（prefill 跳过 → TTFT 降、GPU 活减少）。把它与负载均衡揉在一起，别让热前缀压垮一个副本；可选加共享 KV 层（LMCache），让跨副本未命中是一次取、而非重算。

??? question "一个 SRE 提议按 GPU 利用率 > 80% 自动扩缩 vLLM。为什么这对 LLM 服务是个差信号？你会改用什么？"
    GPU 利用率对 LLM 推理是**陷阱**，因为 decode 是 **memory-bound** 而非 compute-bound：跑 decode-heavy batch 的副本可以报高「利用率」却仍有 batch 宽度余量（它在等 HBM、不是打满 FLOPs），或报*适中*利用率却 **KV cache 已满**、请求已在排队。利用率不与「我过 knee 了吗」相关。直接信号是**队列深度**——`/metrics` 的 `vllm:num_requests_waiting`——它无论 prefill/decode 配比都一个意思：>0 且涨 = 过载、加副本；≈0 且 running 低 = 空闲、缩容。一个校准过的 **`target_qps_per_replica`**（设到测出的 [knee](load-testing-knee.md)）是可接受的更粗代理；GPU 利用率不是。

??? question "你的扩缩器在队列一飙就加副本，SLO 却仍在飙升期被违反。另外，缩容偶尔丢用户请求。分别诊断。"
    **扩容期 SLO 违反 = 冷启动滞后。** 新副本要加载权重、热 CUDA graphs——数十秒——所以队列飙升时它还没在服务；等它就绪，SLO 已被打穿。修法：留**余量**（按前瞻指标/更低阈值扩缩，好在饱和*之前*加容量）、**预热**副本、或接受扩容非即时并把 min-replicas 按基线峰值定。**缩容丢请求 = 没排空。** 终止一个有在途活的 pod 会杀掉那些请求。撤一个副本前，**停止把新请求路由给它**、等到 `vllm:num_requests_running` 与 `vllm:num_requests_waiting` **都**到 0（优雅排空），再终止。合起来：扩容趁早（滞后），缩容从容（排空）。
