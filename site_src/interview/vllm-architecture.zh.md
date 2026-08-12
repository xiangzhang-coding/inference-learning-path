# 追踪一个请求穿过 vLLM 架构

!!! info "基线：**vLLM 0.26.0**（V1）· 组件经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [vLLM 架构地图](../part5/vllm-architecture-map.md)

---

## Q：带我过一遍 vLLM V1 架构。说出组件、端到端追踪一个请求、并说哪个优化住在哪个盒子。

### 直接答案

vLLM V1 是一条**多进程流水线**，职责干净分离：

- **API server**——HTTP（OpenAI 兼容）、tokenize/detokenize、输入处理。不调度、不做 GPU 工作。随 data parallelism 扩展。
- **Engine core**——大脑；一个 **busy loop**，每 data-parallel rank 一个进程，拥有两样东西：
    - **Scheduler**——决定每步的批：准入/驱逐（[continuous batching](../part5/continuous-batching.md)）、切分长 prefill、花 `max_num_batched_tokens` 预算（[chunked prefill](../part5/scheduler-chunked-prefill-pd.md)）。
    - **KV-cache manager**——从 **BlockPool** 分配/释放 KV block（[PagedAttention](../part5/paged-attention.md)）、持有 prefix-cache 哈希映射（[prefix caching](../part5/prefix-caching.md)）。
- **GPU worker**——**每 GPU 一个**进程（每 engine core `TP×PP` 个）；加载权重、跑 forward、管 GPU 内存。
- **Model runner**（`GPUModelRunner`）——在 worker 内；输入张量、**CUDA-graph** 捕获/重放、跑 `nn.Module` → logits（[`enforce_eager`](../part5/tuning-knobs-sweep.md) 关掉 graph；spec-decode 校验在此）。
- **Sampler**——logits → 下一个 token。

**追踪：** HTTP → API server（tokenize）→ engine core[scheduler 挑这步 + KV-cache manager 确保块] → GPU worker 的 model runner（张量 → fwd → logits）→ sampler（token）→ 回经 engine core → API server（detokenize）→ 响应。scheduler 能在这些 token 还在途时开始下一步（CPU/GPU 重叠）。

### 深入原理

- **V1 ≠ V0。** V1 重构了 scheduler、KV-cache manager、worker、sampler、API server（保留 V0 的 models、kernels、utils）。旧的单进程 `LLMEngine.step()` 描述对不上代码。
- **为何多进程。** 分离说/决策/计算让 engine core 在 GPU 计算时调度——保持 GPU 忙的 CPU/GPU 重叠。
- **地图是调试工具。** TTFT → scheduler；启动 OOM → KV-cache profiling / BlockPool 定尺（`num_gpu_blocks` 由 `gpu_memory_utilization` 得出）；decode 慢 → model runner / CUDA graphs。症状 → 盒子。

### 代码

架构即地图（纯 Python）：

```python
COMPONENTS = {  # 组件: (职责, 管辖的 Part-5 课程)
    "APIServer":     ("HTTP + tokenize/detokenize", "—"),
    "Scheduler":     ("admit/evict, chunked prefill", "continuous-batching, scheduler"),
    "KVCacheManager":("BlockPool alloc/free + prefix hashes", "paged-attention, prefix-caching"),
    "ModelRunner":   ("tensors, CUDA graphs, nn.Module fwd", "tuning-knobs (enforce_eager)"),
}
PATH = "APIServer → EngineCore(Scheduler → KVCacheManager) → Worker(ModelRunner → Sampler) → APIServer"
```

### 面试官追问

- *「continuous batching 住在哪？」* → engine core 里的 **scheduler**——它是调度决策，不是 worker kernel。
- *「多少个 GPU worker？」* → 每 engine core `tensor_parallel_size × pipeline_parallel_size` 个；单张 4090 上一个。
- *「V1 为何多进程？」* → 为把 CPU 调度与 GPU 计算重叠——engine core 规划下一步、worker 执行当前步。
- *「启动 OOM——哪个盒子？」* → KV-cache manager profiling：`num_gpu_blocks` 由 `gpu_memory_utilization` 减权重/激活/CUDA-graph 得出。调低它、量化、或缩小 `max_model_len`。
- *「decode 慢——哪个盒子？」* → model runner：查 `enforce_eager`（关 CUDA graphs 抬高 decode 延迟）。

### 关联概念

- 课程：[vLLM 架构地图](../part5/vllm-architecture-map.md)
- 相关：[Static vs continuous batching](continuous-batching.md)（scheduler 盒子）、[PagedAttention：block manager](kv-cache-block-manager.md)（KV-cache-manager 盒子）、[调参旋钮](tuning-knobs.md)（哪个旋钮拧哪个盒子）、[CUDA graphs 与 kernel fusion](cuda-graphs-fusion.md)（model runner）
- 术语：[PagedAttention、continuous batching、KV cache](../glossary.md)
