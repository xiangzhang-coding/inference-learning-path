# vLLM 架构地图：你学的一切都住在哪

!!! info "基线：**vLLM 0.26.0**（V1 引擎）· 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    这里的组件名与职责引自 vLLM **V1** 的 `docs/design/arch_overview.md` 与源码，经 Context7 核实（ADR-0004）：多进程拆分（**API server → engine core → GPU workers → DP coordinator**）、**engine core** 的 busy loop 拥有 **scheduler + KV cache manager**（每个 data-parallel rank 一个）、**worker**（每 GPU 一个，共 `TP×PP` 个）拥有权重 + forward、以及 **model runner**（`GPUModelRunner.execute_model`）做输入张量准备 + CUDA-graph 捕获，然后 **sampler**。V1 重构了 scheduler、KV-cache manager、worker、sampler、API server；保留了 V0 的 models、GPU kernels、utilities。这是一节**读懂+会导航**的课（ADR-0002）——§4 的地图是纯 Python 分类，不是运行的引擎。

---

## 1 · 直觉 & 为什么重要

你在 Part 5 学了一堆*机制*——[continuous batching](continuous-batching.md)、[PagedAttention](paged-attention.md)、[chunked prefill](scheduler-chunked-prefill-pd.md)、[prefix caching](prefix-caching.md)、[speculative decoding](speculative-decoding.md)。本课是**说清每一个物理上住在哪**的地图，这样当真实系统出问题——TTFT 飙升、吞吐见顶、启动就 OOM——你知道该打开*哪个盒子*。「带我过一遍 vLLM 架构、追踪一个请求」是高频的资深 infra 面试题，正因为它证明你能*导航*系统，而不只是背功能。

要内化的一件事：**vLLM V1 是一条多进程流水线，职责干净分离。** 一个 **API server** 处理 HTTP 与 tokenize。一个 **engine core** 跑*调度*和*KV cache 管理*的 busy loop。**GPU worker**（每 GPU 一个）*执行模型*。这种分离不是偶然——它正是为什么 scheduler 能在上一步的 token 还在 GPU 上飞时就决定下一步的批，也是每个优化嵌入的框架。一旦你能把「continuous batching = scheduler」「PagedAttention = KV-cache manager / block pool」「CUDA graphs = model runner」对上号，一个费解的症状就变成有方向的搜索。→ 组件术语见 [Glossary](../glossary.md)。

## 2 · 心智模型

V1 进程流水线，以及每个 Part 5 概念住在哪：

```text
        HTTP 请求
             │
   ┌─────────▼──────────┐   进程 1（随 data parallelism 扩展）
   │    API SERVER      │   HTTP 收发、tokenize / detokenize、输入处理
   └─────────┬──────────┘
             │  (IPC)
   ┌─────────▼──────────────────────────────────────────┐   进程：每 DP rank 一个
   │              ENGINE CORE  (busy loop)               │
   │  ┌───────────────┐   ┌──────────────────────────┐   │
   │  │  SCHEDULER    │   │  KV-CACHE MANAGER          │   │   ◄── continuous batching (scheduler)
   │  │ admit/evict、 │   │  BlockPool：alloc/free、   │   │   ◄── PagedAttention (block manager)
   │  │ chunked       │   │  prefix-cache 哈希映射     │   │   ◄── prefix caching (哈希映射)
   │  │ prefill、     │   │                            │   │
   │  │ token 预算    │   │                            │   │
   │  └───────────────┘   └──────────────────────────┘   │
   └─────────┬───────────────────────────────────────────┘
             │  下发 scheduler_output
   ┌─────────▼──────────────────────────────────────────┐   进程：每 engine core 有 TP × PP 个
   │              GPU WORKER  (每 GPU 一个)              │   权重、forward、GPU 内存
   │  ┌────────────────────────────────────────────┐    │
   │  │  MODEL RUNNER (GPUModelRunner.execute_model) │    │   ◄── CUDA graphs / enforce_eager
   │  │  输入张量 → nn.Module fwd → logits           │    │   ◄── speculative decoding (校验)
   │  │            → SAMPLER → token                 │    │
   │  └────────────────────────────────────────────┘    │
   └─────────────────────────────────────────────────────┘
   (+ 数据并行时的 DP COORDINATOR 进程做负载均衡)
```

三个要记的形状：

- **三种职责、三类进程。** *说*（API server）/ *决策*（engine core：调度 + 管 KV）/ *计算*（GPU workers）。每个 Part 5 优化都是对其中一个盒子的改动——知道是哪个盒子，就知道该拧哪个旋钮、对哪个症状。
- **engine core 是大脑，且它拥有你学得最多的两样东西。** **scheduler**（continuous batching、chunked prefill）与 **KV-cache manager / block pool**（PagedAttention、prefix caching）都住这儿，在一个 busy loop 里，每 data-parallel rank 一个。人们说的「vLLM 的魔法」大半是这个循环。
- **worker 只管执行；model runner 是最内层的盒子。** 一个 worker 拥有一块 GPU（共 `TP×PP` 个）；它内部的 **model runner** 准备输入张量、捕获/重放 CUDA graph、跑 `nn.Module` forward、把 logits 交给 **sampler**。Speculative-decoding 校验与 `enforce_eager` 作用在这里。

## 3 · 原理——组件与请求路径

### 3.1 五个组件

- **API server**——接 HTTP（OpenAI 兼容）、tokenize 输入、detokenize 输出、做输入处理。随 data parallelism 横向扩展。它*不*做调度或 GPU 工作。
- **Engine core**——核心。跑一个 **busy loop**，每迭代问 scheduler 该跑什么、让 KV-cache manager 分配块、把这步下发给 workers、收集输出。**每 data-parallel rank 一个 engine-core 进程。** 它拥有：
    - **Scheduler**——决定每步的批：准入等待请求、驱逐已完（[continuous batching](continuous-batching.md)）、切分长 prefill、花 `max_num_batched_tokens` 预算（[chunked prefill](scheduler-chunked-prefill-pd.md)）。
    - **KV-cache manager**——从共享 **BlockPool** 发出并回收 KV block（[PagedAttention](paged-attention.md)），持有 `cached_block_hash_to_block` 映射（[prefix caching](prefix-caching.md)）。
- **GPU worker**——**每 GPU 一个**进程；每 engine core 有 `tensor_parallel_size × pipeline_parallel_size` 个。加载模型权重、跑 forward、管那块 GPU 的内存。
- **Model runner**（`GPUModelRunner`）——在每个 worker 内。准备输入张量、捕获与重放 **CUDA graph**、跑模型 `nn.Module` forward 得到 logits。`enforce_eager=True` 关掉它的 CUDA-graph 捕获。
- **Sampler**——把 logits 变成下一个 token（贪心或采样），应用 logits processors。

### 3.2 请求路径（一个 decode 步）

```text
HTTP → API server（tokenize）→ engine core busy loop：
   scheduler.schedule()  → 挑这步的请求 + token 预算
   kv_cache_manager      → 确保它们有块（分配 / prefix 命中）
   下发给 worker(s)      → model runner：构建输入张量、跑 fwd（CUDA graph）→ logits
   sampler               → 下一个 token
   ← 输出回 engine core → API server（detokenize）→ HTTP 响应块
```

让它快的微妙之处：engine core 能在*上一步*的 token 还在 GPU 上处理时就调度*下一步*（scheduler 追踪在途的 "output placeholders"）。这种 CPU/GPU 重叠正是 GPU 很少等 scheduler 的原因。

### 3.3 读源码

地图告诉你去哪看。设计文档 `docs/design/arch_overview.md` 是叙事；然后在 V1 树里：`vllm/v1/core/sched/`（scheduler）、`vllm/v1/core/block_pool.py` + `kv_cache_manager`（block manager）、`vllm/v1/worker/gpu_worker.py` + `gpu_model_runner.py`（worker + model runner）、`vllm/v1/sample/sampler.py`。遇到症状，打开地图指向的那个盒子——别从头读到尾。

## 4 · 完整可跑代码 + 逐行讲解

一个纯 Python **地图即代码**：组件、职责、请求穿过它们的路径，以及每个盒子由哪节 Part 5 课程管辖。这是你能重新生成的心智模型——离线、无 GPU、无 vLLM。

```python title="vllm_architecture_map.py"
"""vLLM V1 架构即地图：哪个组件拥有什么、以及请求的路径。
纯 Python、离线——分类，不是运行的引擎。"""

# 组件: (职责, 覆盖其行为的 Part-5 课程)
COMPONENTS = {
    "APIServer":     ("HTTP in/out, tokenize & detokenize, input processing",        "—"),
    "EngineCore":    ("busy loop; owns the Scheduler + KVCacheManager, one per DP rank", "—"),
    "Scheduler":     ("admit/evict each step, chunked prefill, token budget",         "continuous-batching, scheduler"),
    "KVCacheManager":("KV blocks: allocate/free from BlockPool, prefix-cache hashes", "paged-attention, prefix-caching"),
    "Worker":        ("owns ONE GPU: weights, forward pass, GPU memory (TP*PP of them)", "tuning-knobs (TP)"),
    "ModelRunner":   ("input tensors, CUDA-graph capture, runs the nn.Module",         "tuning-knobs (enforce_eager)"),
    "Sampler":       ("logits -> next token (greedy / sampling)",                      "—"),
}

# 请求穿过 V1 流水线的路径（一个 decode 步）
PATH = ["APIServer", "EngineCore", "Scheduler", "KVCacheManager",
        "Worker", "ModelRunner", "Sampler", "APIServer"]

if __name__ == "__main__":
    print("component owners:")
    for name, (role, _lesson) in COMPONENTS.items():
        print(f"  {name:<15} {role}")
    print("\nrequest path (one step):")
    print("  " + " -> ".join(PATH))
    print("\nwhere each Part-5 optimization lives:")
    for name, (_role, lesson) in COMPONENTS.items():
        if lesson != "—":
            print(f"  {name:<15} <- {lesson}")
```

**逐行讲解：**

- `COMPONENTS`——这七行是五个组件（§3.1）加上 engine core 的两个子盒子拆开：*Scheduler* 与 *KVCacheManager* 单列，因为机制挂在那里。读一行就把一个概念定位：*Scheduler* 行是 continuous batching 与 chunked prefill 住的地方；*KVCacheManager* 行是 PagedAttention + prefix caching。这就是整个 Part 5 索引，按*组件*而非按*功能*重排。
- `PATH`——请求一步内穿过的有序流水线：从 API server 进、经 engine core 的 scheduler 与 KV-cache manager、出到 worker 的 model runner 与 sampler、再回。把这个列表叙述出来*就是*面试答案。
- `__main__`——打印所有权表、路径，然后反向索引（组件 → 解释其行为的课程），让你从一个症状的组件跳到机制。

预期输出（分类，不是运行的引擎）：

```text
component owners:
  APIServer       HTTP in/out, tokenize & detokenize, input processing
  EngineCore      busy loop; owns the Scheduler + KVCacheManager, one per DP rank
  Scheduler       admit/evict each step, chunked prefill, token budget
  KVCacheManager  KV blocks: allocate/free from BlockPool, prefix-cache hashes
  Worker          owns ONE GPU: weights, forward pass, GPU memory (TP*PP of them)
  ModelRunner     input tensors, CUDA-graph capture, runs the nn.Module
  Sampler         logits -> next token (greedy / sampling)

request path (one step):
  APIServer -> EngineCore -> Scheduler -> KVCacheManager -> Worker -> ModelRunner -> Sampler -> APIServer

where each Part-5 optimization lives:
  Scheduler       <- continuous-batching, scheduler
  KVCacheManager  <- paged-attention, prefix-caching
  Worker          <- tuning-knobs (TP)
  ModelRunner     <- tuning-knobs (enforce_eager)
```

价值不在这段打印——而在你现在能为任何 Part 5 概念回答「X 在哪发生？」、把症状变成一个要打开的组件。那个映射就是架构地图的全部意义。

## 5 · Lab——在活引擎的日志里追踪它

!!! gpu "GPU Lab（主要是读；可选单卡运行）"
    - **最低显存：** 读地图/源码不需要；启动 vLLM 看组件打日志需 ~16 GB
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 读 ~25 分钟（免费，无卡模式）· 可选运行 ~10 分钟 · ~¥1（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** 进程架构与后端无关；GPU worker/model runner 才是后端（ROCm/CPU）不同之处。

核心 Lab 是读 + 追踪，无卡模式免费：

```text
阅读清单——把源码对上 §4 的盒子：
1. docs/design/arch_overview.md —— 读 "V1 Process Architecture"：API server / engine core / workers / DP coordinator。
2. vllm/v1/core/sched/  —— 找 scheduler 的 schedule()：准入/驱逐 + token 预算（continuous batching）。
3. vllm/v1/core/block_pool.py —— 找 BlockPool.get_new_blocks / free_blocks（PagedAttention）与哈希映射（prefix caching）。
4. vllm/v1/worker/gpu_model_runner.py —— 找 GPUModelRunner.execute_model：输入张量 → fwd → Sampler。
5. 从你关心的每个症状（TTFT / OOM / 低吞吐）画一根箭头指向拥有它的盒子。
```

可选 GPU 运行——起服务，看组件报到：

```python title="observe_engine.py"
# API 针对 vLLM 0.26.0 核实（LLM）。在 GPU 上跑；读启动日志。
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", quantization="awq")
# 启动日志展示 §3 的各部件：engine core init、KV-cache profiling
#（"# GPU blocks: N" —— BlockPool 大小）、CUDA-graph 捕获（model runner）、
# 以及每步 "Running/Waiting" 计数（scheduler）。把每行日志对上一个 §4 盒子。
print(llm.generate(["Name the parts of a car engine in one line."])[0].outputs[0].text[:80])
```

**观察什么：** 启动序列*就是*架构——engine-core init，然后 KV-cache profiling（[`num_gpu_blocks`](paged-attention.md) 数字 = BlockPool 大小），然后 CUDA-graph 捕获（model runner，若 `enforce_eager=True` 则没有），然后服务循环的 Running/Waiting 计数（scheduler）。每行都对上 §2 的一个盒子。

## 6 · 常见坑 / 反直觉点

- **以为它是单进程。** V1 是*多进程*——API server、engine core(s)、GPU workers 是经 IPC 通信的独立进程。那种分离正是使能 CPU/GPU 重叠的东西；混为一谈会搞乱你对延迟来自哪里的心智模型。
- **把 scheduler 放进 worker。** scheduler 与 KV-cache manager 住在 **engine core**，不是 GPU worker。worker 只执行交给它的东西。Continuous batching 是*调度*决策，不是 kernel。
- **把 worker 数当成你拥有的 GPU 数。** 每 engine core 有 `tensor_parallel_size × pipeline_parallel_size` 个 worker；单张 4090 上就是一个 worker。多卡改变 worker 数，不改变流水线形状。
- **假设 V1 == V0。** V1 重构了 scheduler、KV-cache manager、worker、sampler、API server。描述 V0 单进程 `LLMEngine.step()` 的旧博客对不上你要读的代码。确认你在读 V1 树。
- **从头读到尾。** 地图的存在就是让你*别*这样。从症状的组件起步（TTFT → scheduler；启动 OOM → KV-cache profiling / block pool；decode 慢 → model runner / CUDA graphs），打开那个盒子。
- **忘了 model runner 捕获 CUDA graph。** 若 decode 莫名慢，查是不是开了 `enforce_eager`（无 graph）——那是 *model runner* 设置，[tuning-knobs 课](tuning-knobs-sweep.md) 讲。

## 7 · 面试连线

- [追踪一个请求穿过 vLLM 架构](../interview/vllm-architecture.md)——本课为你准备的高频题：*说出组件、端到端追踪一个请求、并说哪个优化住在哪个盒子。*

## 8 · 小结 & 延伸阅读

**一句话：** vLLM V1 是一条多进程流水线——一个 **API server**（HTTP + tokenize）、一个或多个 **engine core** 跑 busy loop 拥有 **scheduler**（continuous batching、chunked prefill）与 **KV-cache manager / block pool**（PagedAttention、prefix caching）、以及 **GPU worker**（每 GPU 一个、共 `TP×PP` 个）其 **model runner** 准备张量、重放 CUDA graph、跑 `nn.Module`、采样——于是你学的每个优化都对应恰好一个盒子，任何症状都指向要打开的那个盒子。

延伸阅读：

- vLLM `docs/design/arch_overview.md`——本课映射的 V1 进程架构（开着 §2 一起读）。
- [continuous-batching](continuous-batching.md) 与 [PagedAttention](paged-attention.md) 课——engine core 里那两个盒子的深入。
- [tuning-knobs 课](tuning-knobs-sweep.md)——每个盒子上的旋钮如何移动吞吐/延迟曲线（自然的下一步）。
- vLLM `docs/usage/v1_guide.md`——V1 相对 V0 重构了什么、保留了什么，好让你读对代码。
- [PagedAttention kernel](../part3/paged-attention-kernel.md) 课（Part 3）—— KV-cache 管理器那个盒子里的代码。
- [OpenAI server](../part8/openai-server.md) 课（Part 8）—— API-server 那个盒子作为生产端点暴露出来。

## 9 · 自测小问

??? question "追踪一个 HTTP 请求穿过 vLLM V1，说出每步做事的组件。"
    （1）**API server** 进程接 HTTP 请求、tokenize prompt、做输入处理。（2）它把请求（经 IPC）交给一个 **engine core** 进程，其 busy loop 驱动其余。（3）**scheduler**（在 engine core）决定是否准入它、这步跑它多少 token（chunked prefill、token 预算）。（4）**KV-cache manager**（也在 engine core）从 BlockPool 为它分配 KV block——或 prefix 命中时复用缓存块。（5）engine core 把这步下发给一个 **GPU worker**，其 **model runner** 构建输入张量、跑 `nn.Module` forward（重放 CUDA graph，除非 `enforce_eager`）、产出 logits。（6）**sampler** 把 logits 变成下一个 token。（7）token 流回 engine core、经 API server（detokenize）作为 HTTP 响应块出去。scheduler 能在这些 token 还在途时就开始下一步——CPU/GPU 重叠。

??? question "同事说「continuous batching 是 GPU worker 里的 kernel 优化」。错在哪？它实际住在哪？"
    Continuous batching 是**调度**决策、不是 kernel，且它住在 **engine core** 里的 **scheduler**——不是 GPU worker。每个 engine-core 循环迭代，scheduler 决定这步的批成员（准入等待请求、驱逐已完、切分长 prefill、花 token 预算）；GPU worker 只是*执行*交给它的批。worker 的 kernel（attention、GEMM）不论 batching 是否连续都一样——变的是*每步谁在批里*，在上游 engine core 决定。把它放进 worker 会错失收益来自 CPU 侧调度与 GPU 执行重叠这一点。

??? question "你在启动时（任何请求之前）遇到 OOM，另外 decode 比预期慢。各该打开哪个组件/盒子？"
    **启动 OOM** → engine core 里的 **KV-cache manager / BlockPool profiling**：启动时 vLLM profiling 内存、由 `gpu_memory_utilization`（默认 0.92）减去权重/激活/CUDA-graph 来定 `num_gpu_blocks`。这里 OOM 意味着预算装不下——调低 `gpu_memory_utilization`、量化权重、或缩小 `max_model_len`/CUDA-graph 内存。**decode 慢** → GPU worker 里的 **model runner**：查 CUDA-graph 捕获是否被关（`enforce_eager=True` 失去在 memory-bound decode 里最要紧的启动开销摊薄）。两者都由地图定位：启动内存定尺是 KV-cache-manager 的事；每步执行速度是 model-runner 的事。（两个旋钮的细节在 [tuning-knobs 课](tuning-knobs-sweep.md)。）
