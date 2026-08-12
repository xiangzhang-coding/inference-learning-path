# Trace a request through vLLM's architecture

!!! info "Baseline: **vLLM 0.26.0** (V1) · components verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [The vLLM Architecture Map](../part5/vllm-architecture-map.md)

---

## Q: Walk me through vLLM's V1 architecture. Name the components, trace a request end-to-end, and say which optimization lives in which box.

### Direct answer

vLLM V1 is a **multi-process pipeline** with cleanly separated concerns:

- **API server** — HTTP (OpenAI-compatible), tokenize/detokenize, input processing. No scheduling, no GPU work. Scales with data parallelism.
- **Engine core** — the brain; a **busy loop**, one process per data-parallel rank, that owns two things:
    - **Scheduler** — decides each step's batch: admit/evict ([continuous batching](../part5/continuous-batching.md)), chunk long prefills, spend the `max_num_batched_tokens` budget ([chunked prefill](../part5/scheduler-chunked-prefill-pd.md)).
    - **KV-cache manager** — allocates/frees KV blocks from the **BlockPool** ([PagedAttention](../part5/paged-attention.md)) and holds the prefix-cache hash map ([prefix caching](../part5/prefix-caching.md)).
- **GPU worker** — one process **per GPU** (`TP×PP` per engine core); loads weights, runs forward, manages GPU memory.
- **Model runner** (`GPUModelRunner`) — inside the worker; input tensors, **CUDA-graph** capture/replay, runs the `nn.Module` → logits ([`enforce_eager`](../part5/tuning-knobs-sweep.md) disables the graphs; spec-decode verification happens here).
- **Sampler** — logits → next token.

**Trace:** HTTP → API server (tokenize) → engine core [scheduler picks the step + KV-cache manager ensures blocks] → GPU worker's model runner (tensors → fwd → logits) → sampler (token) → back through the engine core → API server (detokenize) → response. The scheduler can start the next step while these tokens are still in flight (CPU/GPU overlap).

### Deep dive

- **V1 ≠ V0.** V1 re-architected the scheduler, KV-cache manager, worker, sampler, and API server (kept V0's models, kernels, utils). Old single-process `LLMEngine.step()` descriptions don't match the code.
- **Why multi-process.** Separating talk/decide/compute lets the engine core schedule while the GPU computes — the CPU/GPU overlap that keeps the GPU busy.
- **The map is a debugging tool.** TTFT → scheduler; OOM at startup → KV-cache profiling / BlockPool sizing (`num_gpu_blocks` from `gpu_memory_utilization`); slow decode → model runner / CUDA graphs. Symptom → box.

### Code

The architecture as a map (pure Python):

```python
COMPONENTS = {  # component: (responsibility, owning Part-5 lesson)
    "APIServer":     ("HTTP + tokenize/detokenize", "—"),
    "Scheduler":     ("admit/evict, chunked prefill", "continuous-batching, scheduler"),
    "KVCacheManager":("BlockPool alloc/free + prefix hashes", "paged-attention, prefix-caching"),
    "ModelRunner":   ("tensors, CUDA graphs, nn.Module fwd", "tuning-knobs (enforce_eager)"),
}
PATH = "APIServer → EngineCore(Scheduler → KVCacheManager) → Worker(ModelRunner → Sampler) → APIServer"
```

### Interviewer follow-ups

- *"Where does continuous batching live?"* → The **scheduler**, in the engine core — it's a scheduling decision, not a worker kernel.
- *"How many GPU workers?"* → `tensor_parallel_size × pipeline_parallel_size` per engine core; one on a single 4090.
- *"Why is V1 multi-process?"* → To overlap CPU scheduling with GPU compute — the engine core plans the next step while the worker executes the current one.
- *"OOM at startup — which box?"* → KV-cache manager profiling: `num_gpu_blocks` is sized from `gpu_memory_utilization` minus weights/activations/CUDA-graph. Lower it, quantize, or shrink `max_model_len`.
- *"Slow decode — which box?"* → Model runner: check `enforce_eager` (CUDA graphs off raises decode latency).

### Linked concepts

- Lesson: [The vLLM Architecture Map](../part5/vllm-architecture-map.md)
- Related: [Static vs continuous batching](continuous-batching.md) (the scheduler box), [PagedAttention: block manager](kv-cache-block-manager.md) (the KV-cache-manager box), [Tuning knobs](tuning-knobs.md) (which knob turns which box), [CUDA graphs & kernel fusion](cuda-graphs-fusion.md) (the model runner)
- Glossary: [PagedAttention, continuous batching, KV cache](../glossary.md)
