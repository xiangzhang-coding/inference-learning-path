# Part 8 · 生产部署与系统设计

> 从一个能跑的引擎，到一个生产服务——再到答好系统设计面试。

## 本 Part 覆盖

- **HTTP 服务化**：OpenAI 兼容 server、**压测**找并发 **knee**、以及跨多实例的**路由 / 自动扩缩 / KV 感知路由**
- **可观测性与 profiling** 及 **SLO 调优**；框架取舍——**TensorRT-LLM / TGI / SGLang / LMDeploy**——在选型题上有观点
- **容量规划**：给定模型 + 硬件估 VRAM / 吞吐
- **系统设计**演练：「为 X QPS、Y 延迟设计推理服务」

**[Capstone](../capstone/index.md)** 会把这里的一切在单张 4090 上串起来。

## 课程

- **[用 HTTP 服务化 vLLM：OpenAI 兼容 server](openai-server.md)** —— `vllm serve` 把引擎核心包进一层薄 FastAPI 前端、说 **OpenAI API**，任何 OpenAI client 改一行 `base_url` 就能重定向。讲 endpoints（`/v1/chat/completions` 套 chat template、`/v1/completions` 纯文本、`/v1/models` 列服务 id + LoRA adapter、`/health` 是**存活性** 200/503、`/metrics` 是 Prometheus 数据源）、鉴权（`--api-key` / `VLLM_API_KEY`，可重复以轮换）、流式（SSE），以及为何**接口**旋钮（`--port` / `--served-model-name`）与设天花板的**容量**旋钮（`--max-num-seqs` / `--gpu-memory-utilization`）相互独立——均在 vLLM 0.26.0 上核实。
- **[压测找并发拐点（knee）](load-testing-knee.md)** —— **knee** 是单实例 batch 填满、`vllm:num_requests_waiting` 离开零往上爬之处；由 **Little 定律**（$L=\lambda W$），把到达率推过最大完成率会让队列与延迟失控。你靠**把 `vllm bench serve --request-rate` 往上扫**（开环 Poisson 到达——*不是* `--request-rate inf`、*不是*闭环 `--max-concurrency`）找到它，按 SLO 读 **p99** TTFT/E2EL 与 **goodput**，把最后一个仍通过的档作为实例诚实容量报告。
- **[路由、自动扩缩与 KV 感知路由（多实例）](routing-autoscaling.md)** —— 过了一个实例的 knee，你扩到 N 个独立副本、挂在一个 **router** 后；让它变好的两个决定是 **KV 感知（前缀感知）路由**（缓存是每实例的，所以 round-robin 会重 prefill 共享 prompt）与**按队列自动扩缩**（`vllm:num_requests_waiting`，不是 GPU 利用率），并处理好冷启动滞后与「缩容前排空」。vLLM 把它作为 **production stack** 发布（Helm：前缀感知 + 模型感知 router、引擎 pod、LMCache KV offload）；SkyPilot 按 `target_qps_per_replica` 自动扩缩。

!!! note "脚手架状态"
    三节课已在——[OpenAI 兼容 server](openai-server.md)、[压测找 knee](load-testing-knee.md)、[路由 / 自动扩缩](routing-autoscaling.md)（票 #19，生产服务化上手篇）——各自与其面试题双向链接（[server 与 endpoints](../interview/openai-server-deployment.md)、[knee 与 Little 定律](../interview/load-testing-knee.md)、[路由与自动扩缩](../interview/routing-autoscaling.md)）。**可观测性 / profiling、SLO 调优、框架对比**（TensorRT-LLM / TGI / SGLang / LMDeploy）与**容量规划 + 系统设计**长题在后续票落地。所有 vLLM flag/API 均经 Context7 核实（ADR-0004）；基线是 **vLLM 0.26.0**，一切性能数字均为**示例 / 量级参考**。见 **[术语表](../glossary.md)** 与 [面试题库](../interview/index.md)。
