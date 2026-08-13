# HTTP 服务化：OpenAI 兼容 server 及其 endpoints

!!! info "基线：**vLLM 0.26.0** · API 经 Context7 核实（ADR-0004）"

**模块：** Part 8 · 生产部署与系统设计   ·   **对应课程：** [用 HTTP 服务化 vLLM：OpenAI 兼容 server](../part8/openai-server.md)

---

## Q：你怎么用 HTTP 服务化一个 vLLM 模型？走一遍 `vllm serve`、它的主要 endpoints、`/v1/chat/completions` 与 `/v1/completions` 的区别、`/health` 保证与不保证什么、鉴权怎么工作、以及哪些 flag 塑造接口 vs 容量。

### 直接答案

**启动：** `vllm serve <model>` 启动引擎、起一个说 **OpenAI API** 的 **FastAPI/uvicorn** 前端——任何 OpenAI client（`openai` SDK、LangChain、聊天 UI）改一行 `base_url="http://host:8000/v1"` 就能重定向。

**endpoints：**

- **`/v1/chat/completions`** —— 角色消息；server **套用模型的 chat template**。instruct 模型的主 endpoint。
- **`/v1/completions`** —— **纯**文本进/出；**不套**模板。
- **`/v1/models`** —— 服务 id（`--served-model-name`）+ 任何已加载 **LoRA adapter**。
- **`/health`** —— **存活性**：引擎活着 200、死了 503。*不是*就绪性、*不是*负载信号。
- **`/metrics`** —— Prometheus 数据源（`vllm:num_requests_running` / `num_requests_waiting`、KV 用量、延迟直方图）。
- 工具：`/ping`（SageMaker）、`/version`、`/load`、`/tokenize`、`/detokenize`。

**鉴权：** `--api-key KEY`（或 `VLLM_API_KEY`）；**多次**传它做 key 轮换。没设 key 时，server **开放**（任意非空 key 都过）。

**接口旋钮**（`--host`/`--port`/`--uds`、`--api-key`、`--served-model-name`）塑造*客户端怎么跟你说话*；**容量旋钮**（`--max-num-seqs`、`--max-num-batched-tokens`、`--max-model-len`、`--gpu-memory-utilization`）塑造*你能服务多少*——你接着去[测](load-testing-knee.md)的天花板。

### 深入原理

- **前端 vs 引擎核心。** HTTP 层做鉴权、JSON 校验、chat-template 渲染、SSE 流式；[引擎核心](../part5/vllm-architecture-map.md)（scheduler + workers）做批处理与 GPU 计算。延迟问题住在引擎队列、不是 FastAPI。
- **chat template。** `/v1/chat/completions` 套用 instruct 模型被调时那套精确的特殊 token 格式。把裸文本发给 instruct 模型的 `/v1/completions` 会跳过它 → 分布外 prompt → 悄悄掉质量。
- **流式。** `"stream": true` 切到 **Server-Sent Events**：第一个 `data:` chunk 约在 **TTFT** 落地、其余由 **TPOT** 配速。这就是流式让 TTFT 用户可见的原因。
- **`--served-model-name`。** 把对外 id 与 checkpoint 路径解耦；客户端必须发那个 id、否则*模型未找到*。
- **存活性 ≠ 就绪性。** `/health` 说进程活着，不说权重加载完/热好、也不说有余量——就绪与自动扩缩挂 `/metrics`。

### 代码

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-demo-key")  # 那一行
# chat：server 对角色消息套 chat template
r = client.chat.completions.create(model="qwen2.5-7b",                       # = --served-model-name
    messages=[{"role": "user", "content": "hi"}], stream=True)               # SSE：第一个 chunk ≈ TTFT
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
# 运维: curl -s localhost:8000/health          -> 200（活）/ 503（死）
#       curl -s localhost:8000/v1/models -H "Authorization: Bearer sk-demo-key"
```

### 面试官追问

- *「客户端把 HF 路径当 `model` 发、得到*未找到*——为什么？」* → `--served-model-name` 把对外 id 设成了别的；`model` 字段须等于服务名。
- *「`/health` == 能接流量？」* → 不——只存活性（200 活 / 503 死）。就绪/负载来自 `/metrics`（`num_requests_waiting`）。
- *「`/v1/chat/completions` vs `/v1/completions`？」* → chat 套模板（角色消息）；completions 纯文本。给 instruct 模型用错那个会悄悄掉质量。
- *「怎么不停机轮换 API key？」* → 多次传 `--api-key`，切换期间新旧 key 都有效。
- *「没设 `--api-key`——server 安全吗？」* → 不，开放；任意非空 key 都过。暴露 `0.0.0.0` 前绑 `127.0.0.1` 或设 key + 防火墙。
- *「流式看起来是成批、不是逐字——为什么？」* → 中间代理/LB 在缓冲 SSE 响应；给那条路由关缓冲。

### 关联知识点

- 课程：[用 HTTP 服务化 vLLM：OpenAI 兼容 server](../part8/openai-server.md)
- 相关：[追踪一个请求穿过 vLLM 架构](vllm-architecture.md)（前端背后的引擎核心）、[压测与并发拐点](load-testing-knee.md)（测 server 暴露的容量）、[Multi-LoRA serving](multi-lora-serving.md)（`/v1/models` 列的 adapter）
- 术语表：[SLO、Goodput](../glossary.md)
