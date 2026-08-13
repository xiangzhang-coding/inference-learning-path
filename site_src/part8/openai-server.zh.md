# 用 HTTP 服务化 vLLM：OpenAI 兼容 server

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090"
    已按 ADR-0004 用 Context7 对照 vLLM 0.26.0 核实：用 **`vllm serve <model>`** 启动 server（`--host` / `--port` / `--uds`，`--api-key` 或环境变量 `VLLM_API_KEY`——可传多个 key 做轮换，`--served-model-name` 设置对外的 model id）。它暴露 OpenAI 路由 **`/v1/chat/completions`**、**`/v1/completions`**、**`/v1/models`**，外加工具路由 **`/health`**（引擎活着返回 200、死了返回 503）、**`/ping`**、**`/version`**、**`/load`**、**`/tokenize`** / **`/detokenize`**，以及 Prometheus 的 **`/metrics`**。容量由你已熟悉的引擎参数塑造——`--max-num-seqs`、`--max-num-batched-tokens`、`--max-model-len`、`--gpu-memory-utilization`。本节所有数字均为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

到 Part 7 为止，我们造出并调优了一个**引擎**——一个把 prompt 变成 token、快到硬件极限的 Python 对象。但没人上线一个 Python 对象。生产服务是一个 **HTTP server**：客户端从网络发请求，server 把它们复用到引擎上，再把 token 流式发回。本节把引擎变成那个 server。

vLLM 的 server 说 **OpenAI API**。就这一个决定，让 vLLM 极易上手：任何早已针对 `api.openai.com` 写好的工具、SDK 或 app——`openai` Python client、LangChain、LlamaIndex、一个聊天 UI——只要改**一行** `base_url`，就能对着你的 vLLM server 跑。你不是在发明协议；你是在扮演那个支持最广的协议。

面试官期望你真的懂、而非只会比划的两件事：

1. **server *是*什么。** 它是同一个引擎核心（scheduler + workers，见[架构地图](../part5/vllm-architecture-map.md)）前面一层薄薄的 **FastAPI/uvicorn 前端**。前端做 HTTP、鉴权、请求校验，并套用**聊天模板 (chat template)**；引擎核心做批处理与 GPU 计算。知道哪个盒子干哪件事，就知道 bug 或瓶颈住在哪。
2. **哪些旋钮是 server 旋钮、哪些是引擎旋钮。** `--port`、`--api-key`、`--served-model-name` 塑造*接口*。`--max-num-seqs`、`--gpu-memory-utilization`、`--max-model-len` 塑造*容量*——它们设的天花板，你会在[下一节](load-testing-knee.md)去测。同一个二进制，两类 flag。

所以：server 暴露什么（endpoints 与鉴权），以及你怎么跟它说话（OpenAI client + 流式）。→ 术语见 [术语表](../glossary.md) 的 *SLO、Goodput*。

## 2 · 心智模型

一个二进制，两半：一个从 `curl` 就能看到的 **HTTP 前端**，和你在 Part 4–7 调过的**引擎核心**。

```text
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
   ─────────────────────────────────────────────────────────────────────────────

   client ──HTTP──▶  API SERVER (FastAPI/uvicorn)  ──▶  ENGINE CORE（架构地图）
   (openai SDK,      HTTP 前端：                          SCHEDULER
    curl, app)       • /v1/chat/completions（chat template）  │  continuous batching
        ▲            • /v1/completions       （纯文本）        ▼
        │            • /v1/models            （服务 id+LoRA）  WORKERS ──▶ GPU
        │            • /health /metrics /ping /version /load  （PagedAttention）
        └───────────────  token 流式返回 (SSE)  ◀──────────────┘

   接口旋钮 → --port  --api-key  --served-model-name              （客户端怎么跟你说话）
   容量旋钮 → --max-num-seqs  --max-num-batched-tokens
              --max-model-len  --gpu-memory-utilization           ← 设你要测的天花板
```

三个要记住的形状：

- **server 是前端；引擎是后端。** HTTP 层便宜、近乎无状态（鉴权、JSON 解析、chat-template 渲染、SSE 流式）。昂贵、有状态的部分——KV cache、运行中的 batch——住在引擎核心。延迟飙升时，几乎从不是 FastAPI 层；是 scheduler 前面的队列（[下一节](load-testing-knee.md)）。
- **OpenAI 兼容意味着即插即用。** 把 `openai` client 的 `base_url` 指向 `http://your-host:8000/v1`，传任意非空 `api_key`，同一份调 GPT 的代码就在调你的 Qwen。`/v1/chat/completions` 套用模型的 chat template；`/v1/completions` 是纯文本进、纯文本出。
- **`/health` ≠「能接流量了」。** `/health` 告诉你*引擎进程活着*（200）还是*死了*（503）。它**不**说「模型加载完了」或「还有余量」。那些是不同的信号——其中一个，你会在两节课后接进自动扩缩器。

## 3 · 原理

### 3.1 启动 server

那一条命令：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

`vllm serve <model>` 启动引擎（下载/加载权重、profile KV-cache block pool），然后起一个 uvicorn server。好用的接口 flag（均在 0.26.0 上核实）：

- **`--host` / `--port`** —— 绑到哪里。`0.0.0.0` 暴露在所有接口上（见 §6 的安全坑）；只给本机用的 dev server 用 `127.0.0.1`。`--uds /tmp/vllm.sock` 绑 Unix 域套接字而非 TCP。
- **`--api-key KEY`**（或环境变量 **`VLLM_API_KEY`**）—— 要求每个请求带这个 bearer token。可**多次**传这个 flag 一次授权多个 key，这就是不停机轮换 key 的做法。
- **`--served-model-name NAME`** —— 客户端要在 `"model"` 字段里发、且 `/v1/models` 会报告的 model id。默认是 HF 路径（`Qwen/Qwen2.5-7B-Instruct`）；设一个稳定别名，客户端就不用硬编码 checkpoint 路径。

### 3.2 endpoints

面试官可能让你列举的路由：

| Endpoint | 方法 | 干什么 |
|---|---|---|
| `/v1/chat/completions` | POST | 用**角色消息 (role messages)** 对话；server 套用模型的 **chat template**。主 endpoint。 |
| `/v1/completions` | POST | 纯**文本进、文本出**——不套 chat template。 |
| `/v1/models` | GET | 列出所服务的模型（id = `--served-model-name`）**+ 任何已加载的 LoRA adapter**（[multi-LoRA](../part6/multi-lora-serving.md)）。 |
| `/health` | GET | 引擎活着返回 **200**、死了（`EngineDeadError`）返回 **503**。存活性 (liveness)。 |
| `/ping` | GET/POST | SageMaker 期望的健康检查名。 |
| `/version` | GET | vLLM 版本——把你的文档/复现钉在它上（ADR-0004）。 |
| `/load` | GET | server **负载指标**（比全套 Prometheus 更轻的一瞥）。 |
| `/tokenize`、`/detokenize` | POST | token↔文本、不生成——方便客户端侧数 token。 |
| `/metrics` | GET | **Prometheus** 指标（`vllm:num_requests_running`、`vllm:num_requests_waiting`、KV-cache 用量、TTFT/ITL 直方图）。可观测性 + 自动扩缩的数据源。 |

### 3.3 流式

默认一次 completion 完整返回一次。传 `"stream": true`，server 切到 **Server-Sent Events (SSE)**：每个生成的 token（或小 chunk）作为独立 `data:` 事件到达，以 `data: [DONE]` 结束。这就是聊天 UI 逐字打印的原理，也是 **TTFT**（Part 0）成为用户可见数字的原因——有了流式，用户在 prefill 一完成就看到第一个 token，而不是等整段答复完成。

### 3.4 接口旋钮 vs 容量旋钮

同一条 `vllm serve` 命令带两族 flag。上面的**接口**旋钮塑造*客户端怎么跟你说话*。**容量**旋钮——你调引擎时见过的——塑造*你能服务多少*：

- **`--max-num-seqs`** —— 一个 batch 里的最大并发序列数（运行 batch 宽度）。
- **`--max-num-batched-tokens`** —— 每个 scheduler step 的 token 预算（[chunked-prefill](../part5/scheduler-chunked-prefill-pd.md) 的旋钮）。
- **`--max-model-len`** —— 最大上下文长度；封顶单请求 KV cache。
- **`--gpu-memory-utilization`** —— vLLM 可占的 VRAM 比例；越大 → KV-cache block pool 越大 → 并发越高。

这四个设一个实例的**天花板**。[下一节](load-testing-knee.md)完全在讲*测*天花板在哪；再下一节讲靠多开实例*抬高*它。

## 4 · 完整可跑代码 + 逐行讲解

启动 server，再用三种方式跟它说话：OpenAI Python client（非流式 + 流式），以及 `curl` 打运维 endpoint。

```bash
# 1) 启动 server（在一个终端里留着运行）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --api-key sk-demo-key \                 # 要求这个 bearer token；重复该 flag 可加更多
    --served-model-name qwen2.5-7b \        # 客户端必须发的对外 model id
    --max-num-seqs 256 \                    # 容量旋钮：运行 batch 宽度
    --gpu-memory-utilization 0.90           # 容量旋钮：多少 VRAM → KV cache
```

```python title="client.py"
"""用标准 openai SDK 跟 vLLM OpenAI 兼容 server 说话。
读代码是离线安全的；要命中网络需上面的 server。"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",   # 那「一行」：把 OpenAI client 指向 vLLM
    api_key="sk-demo-key",                 # 必须匹配某个 --api-key 值（没设时任意非空串即可）
)

# (a) 非流式对话——server 对这些角色消息套用 Qwen 的 chat template
resp = client.chat.completions.create(
    model="qwen2.5-7b",                    # 必须等于 --served-model-name，不是 HF 路径
    messages=[
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user",   "content": "Name three GPUs good for LLM inference."},
    ],
    max_tokens=64,
    temperature=0.7,
)
print(resp.choices[0].message.content)     # 完整 completion，一次返回

# (b) 流式——token 作为 SSE 事件到达；第一个 chunk 约在 TTFT 落地
stream = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,                           # 切到 Server-Sent Events
)
for chunk in stream:                       # 每个 chunk 带下一个 token
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)   # 像聊天 UI 一样逐字打印
print()
```

```bash
# (c) 运维 endpoint——health/metrics 不需鉴权；/v1/* 需要鉴权
curl -s http://localhost:8000/health              # -> HTTP 200（引擎活着）或 503（死了）
curl -s http://localhost:8000/v1/models \
     -H "Authorization: Bearer sk-demo-key"       # -> 列出 "qwen2.5-7b"（+ 任何 LoRA adapter）
curl -s http://localhost:8000/metrics | grep -E "num_requests_(running|waiting)"
#   vllm:num_requests_running{...} 3.0            # 此刻在运行 batch 里
#   vllm:num_requests_waiting{...} 0.0            # 排队、等一个槽位
```

**逐行讲解：**

- **`--api-key sk-demo-key`** —— server 现在拒绝任何不带 `Authorization: Bearer sk-demo-key` 的 `/v1/*` 请求。重复该 flag（`--api-key k1 --api-key k2`）可在轮换期间接受多个 key。`/health` 和 `/metrics` 保持免鉴权，好让探针和 Prometheus 够得着。
- **`--served-model-name qwen2.5-7b`** —— 把对外 id 与 checkpoint 路径解耦。客户端发 `"model": "qwen2.5-7b"`；不匹配就返回*模型未找到*错误（头号首请求错误，§6）。
- **`base_url=".../v1"`** —— 整件事的关键。`openai` SDK 默认打 `api.openai.com/v1`；这一行把它重定向到你的机器。`api_key` 须匹配某个 `--api-key` 值；没设 `--api-key` 时，任意非空串（惯例 `"EMPTY"`）都行。
- **`chat.completions.create(..., stream=True)`** —— 同一个调用，切到 SSE。`for chunk in stream` 循环收到 `delta.content` 片段；第一个约在 **TTFT** 到达，其余由 **TPOT**（Part 0）配速。非流式会等整段答复完成才返回。
- **`/v1/models`** —— 返回所服务的 id 和任何已加载的 LoRA adapter；客户端靠它发现能问什么。
- **`grep num_requests_(running|waiting)`** —— 概括负载的两个 gauge：现在 batch 里有多少序列、排了多少。`waiting > 0` 且在涨，就是实例饱和的签名——[knee](load-testing-knee.md) 与[自动扩缩信号](routing-autoscaling.md)。

## 5 · Lab —— 立起 server 并把每个 endpoint 都练一遍

!!! gpu "GPU Lab（单卡）"
    - **最低显存：** `Qwen2.5-7B-Instruct` BF16、适中 `--max-model-len` 约 18–20 GB；**24 GB 的 RTX 4090** 从容装下。紧张？服务 [INT4 量化](../part4/quantization-lab.md) checkpoint 并调高 `--gpu-memory-utilization`。
    - **建议 AutoDL 卡型：** 单张 **RTX 4090 (24 GB)**——默认主线卡（ADR-0001）。本节不需多卡。
    - **预估耗时 / 花费：** 上手约 15–25 分钟 · 4090 常见时价下 **约 ¥1–3**（示例）。先在 **无卡模式**（纯 CPU）下把模型下好，再开 GPU 服务。
    - **平台：** NVIDIA CUDA（默认）。**非 NVIDIA：** server 是纯 Python/FastAPI，各处一致；只有引擎后端不同（AMD ROCm 版 vLLM 暴露相同的 endpoints 和 flag）。

从启动做到可观测：

1. **启动。** 跑上面的 `vllm serve …`。看日志报告模型加载与 **KV-cache block pool 大小**——那个数就是你的并发预算。
2. **先存活、后就绪。** `curl /health` 在模型完全就绪、能服务好*之前*就返回 200；注意它是**存活性**探针，不是「热好了」信号。确认 `/v1/models` 列出 `qwen2.5-7b`。
3. **先对话，再流式。** 跑 `client.py`。确认 (a) 返回一整块、(b) 逐字打印。留意第一个流式 token 前那段可感知的延迟——那就是 **TTFT**。
4. **看负载。** 循环 `curl /metrics | grep num_requests`。并发打几个请求（同时开几个 `client.py`），看 `num_requests_running` 上升、超过 batch 宽度时 `num_requests_waiting` 变正。做完**关机**。

## 6 · 常见坑 / 反直觉点

- **设了 `--served-model-name` 却发 HF 路径。** 若你以 `--served-model-name qwen2.5-7b` 启动，带 `"model": "Qwen/Qwen2.5-7B-Instruct"` 的请求返回*模型未找到*。`"model"` 字段必须等于你所选的**服务名**。（没设 `--served-model-name` 时才是 HF 路径。）
- **忘了 API key——或以为有。** 设了 `--api-key`，每个 `/v1/*` 调用都需 `Authorization: Bearer <key>`，否则 401。**没设** `--api-key` 时，server 是**开放**的——任意非空 key 串都过。别把「客户端发了 `EMPTY`」当成「server 已加固」。
- **无鉴权就在不可信网络上绑 `0.0.0.0`。** `--host 0.0.0.0` 把 server 暴露在每个接口。在共享/租来的机器上、无 `--api-key` 时，谁够得着端口谁就能烧你的 GPU。本地开发绑 `127.0.0.1`，或在暴露前设 API key（加防火墙）。
- **把 `/health` 当成「能接流量」。** `/health` 是**存活性**（引擎活/死），不是**就绪性**（加载完 + 有余量），也不是负载信号。把负载均衡就绪与自动扩缩挂在 **`/metrics`** 的 gauge（`num_requests_waiting`）上，别挂 `/health`。
- **用 `/v1/completions` 却纳闷 chat 格式去哪了。** `/v1/completions` 是**纯**文本——它*不*套 chat template。角色消息与 system prompt 只在 **`/v1/chat/completions`** 上有效。把裸指令发给 `/completions` 会跳过模型被调过的模板，质量下降。
- **把延迟怪到 FastAPI。** HTTP 前端是微秒级开销。若 p99 延迟差，那是引擎队列（并发超过 `--max-num-seqs`，或 prefill 饿死 decode）——去[测 knee](load-testing-knee.md)，别 profile uvicorn。
- **负载均衡/代理缓冲破坏流式。** 中间代理（nginx、部分云 LB）若**缓冲**响应，会把所有 SSE 事件收齐后一次发出——毁掉逐字效果、抬高感知 TTFT。给流式路由关掉响应缓冲。

## 7 · 面试连线

- [HTTP 服务化：OpenAI 兼容 server 及其 endpoints](../interview/openai-server-deployment.md) —— 本节为你准备的高频题：*`vllm serve` 暴露什么、`/v1/chat/completions` 与 `/v1/completions` 的区别、`/health` 保证与不保证什么、鉴权怎么工作、哪些 flag 塑造接口 vs 容量。*

## 8 · 小结 & 延伸阅读

**一句话：** `vllm serve <model>` 把引擎核心包进一层薄 FastAPI 前端、说 **OpenAI API**——一行 `base_url` 就能重定向任何 OpenAI client——暴露 `/v1/chat/completions`（套 chat template）、`/v1/completions`（纯文本）、`/v1/models`（服务 id + LoRA adapter）、`/health`（存活性：200 活 / 503 死）、`/metrics`（Prometheus 数据源）与工具路由；鉴权是 `--api-key`/`VLLM_API_KEY`（可重复以轮换）；**接口**旋钮（`--port`、`--api-key`、`--served-model-name`）与设天花板的**容量**旋钮（`--max-num-seqs`、`--max-num-batched-tokens`、`--max-model-len`、`--gpu-memory-utilization`）相互独立，后者的天花板你接着去测。

延伸阅读：

- vLLM `docs/serving/openai_compatible_server.md` —— 完整 endpoint 列表、请求字段、采样参数。
- vLLM `docs/cli/README.md` —— 每个 `vllm serve` flag（host/port/uds、api-key、served-model-name）。
- vLLM `docs/usage/security.md` —— 工具 endpoint（`/health`、`/ping`、`/version`、`/load`、`/tokenize`）与加固说明。
- [vLLM 架构地图](../part5/vllm-architecture-map.md) —— 这个前端背后的「引擎核心」到底干什么。
- [下一节](load-testing-knee.md) —— 把这个 server 变成一条测出来的吞吐曲线。

## 9 · 自测小问

??? question "客户端发 `POST /v1/chat/completions`、带 `{\"model\": \"Qwen/Qwen2.5-7B-Instruct\", ...}`，却得到*模型未找到*——即便那正是你在服务的 checkpoint。为什么？"
    因为 server 启动时把 **`--served-model-name`** 设成了别的（比如 `qwen2.5-7b`），而 `"model"` 字段必须匹配**服务名**、不是 Hugging Face checkpoint 路径。`--served-model-name` 刻意把对外 id 与 checkpoint 解耦，好让客户端不硬编码路径；代价是你*对外公布*的 id（在 `/v1/models` 可见）才是唯一能解析的。修法：发 `"model": "qwen2.5-7b"`，或去掉 `--served-model-name` 让 HF 路径成为 id。

??? question "你的 uptime 监控轮询 `/health`、看到 200、就标记实例就绪——但头几个真实请求慢得离谱或永远排队。误解在哪？"
    `/health` 是**存活性**探针：引擎进程活着返回 200、死了（`EngineDeadError`）返回 503。它对就绪性（权重加载完并热好）与余量（batch 可能已满、队列很长）**只字不提**。拿它当就绪/流量信号，会把负载发给一个服务不好的实例。就绪与自动扩缩应读 **`/metrics`** 的 gauge——尤其 `vllm:num_requests_waiting`（队列深度）与 `vllm:num_requests_running`——它们真正反映实例能不能接更多活。那个队列深度信号，正是驱动[路由与自动扩缩](routing-autoscaling.md)的东西。

??? question "`/v1/chat/completions` 与 `/v1/completions` 的实际区别是什么？发错哪个会悄悄伤到质量？"
    `/v1/chat/completions` 收**角色消息**（`system`/`user`/`assistant`），server 套用**模型的 chat template**——即 instruct 模型被微调时那套精确的特殊 token 格式。`/v1/completions` 是**纯文本进、文本出**、**不套模板**。若你把一条裸指令发给 instruct 模型的 `/v1/completions`，它在没有模板 system 框架与轮次标记的情况下运行，模型看到分布外的 prompt，输出质量悄悄劣化——不报错，只是答得更差。instruct/chat 模型用 `/v1/chat/completions`；`/v1/completions` 留给 base 模型、或你有意自己控制原始 prompt 时。
