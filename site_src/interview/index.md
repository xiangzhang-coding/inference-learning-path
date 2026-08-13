# Interview Bank

> A growing bank of **high-frequency** interview questions, organized by module (Part 0–8). Each question follows one schema: **direct answer → deep dive → code (if applicable) → interviewer follow-up → linked concept**.

Every question links back to the lesson it tests, and every lesson's "Interview links" section links here — a closed learn-and-practice loop.

## By module

- **Part 0 · Foundations**
    - [Prefill vs decode](prefill-vs-decode.md) — which phase is compute- vs memory-bound, and why.
    - [Attention variants: MHA/MQA/GQA](attention-variants.md) — how KV heads set the KV cache and the throughput ceiling.
    - [KV cache & throughput ceiling](kv-cache.md) — why the KV cache, not compute, is usually the bottleneck.
    - [GPU memory hierarchy & roofline](gpu-memory-hierarchy.md) — walk the memory tiers and use the roofline to explain why decode is memory-bound.
    - [Latency vs throughput metrics](latency-throughput-metrics.md) — TTFT/TPOT/ITL/throughput/goodput, how to measure, and the batch-size trade.
    - [Number formats & precision](number-formats.md) — FP16/BF16/FP8/INT8/INT4, range vs precision, and why low-bit speeds up decode.
- **Part 2 · Single-GPU Inference Performance**
    - [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md) — derive an operator's intensity from its shapes, why decode attention is context-independent, and the batch that crosses the ridge.
    - [VRAM budget & max concurrency](vram-capacity-planning.md) — walk the full VRAM budget and size max concurrency; the knobs that hit a concurrency target.
    - [FlashAttention & IO-aware attention](flash-attention.md) — why it's faster at the same FLOPs, online softmax, and where it does/doesn't help.
    - [CUDA graphs & kernel fusion](cuda-graphs-fusion.md) — decode launch overhead, why it hits decode not prefill, and what `enforce_eager` trades.
- **Part 3 · GPU Programming (Triton)**
    - [CUDA execution model: warps, SIMT & occupancy](cuda-execution-model.md) — what a warp is, the cost of SIMT divergence, and why maxing occupancy isn't always faster.
    - [Memory coalescing, shared memory & bank conflicts](memory-coalescing.md) — what makes an access coalesced, what uncoalesced costs, and what shared memory and bank conflicts are.
    - [Triton programming model](triton-programming.md) — what a Triton program maps to, `program_id`/offsets/masks, FP32 accumulation, and when to reach for Triton.
    - [PagedAttention kernel & block tables](paged-attention-kernel.md) — why KV lives in blocks, what a block table does, how the kernel gathers KV, and why it equals dense attention.
- **Part 4 · Quantization**
    - [Quantization: why it speeds up inference](quantization-basics.md) — why quantization raises throughput (memory, not compute), the affine map, and what bounds the error.
    - [Quantization schemes: granularity, symmetry, PTQ/QAT](quantization-schemes.md) — per-tensor/channel/group, symmetric vs asymmetric, W4A16 vs W8A8, and why inference uses PTQ.
    - [Quantization methods: GPTQ/AWQ/SmoothQuant/FP8](quantization-methods.md) — place each method on the axes, its anti-outlier trick, and which to pick for a bottleneck.
    - [Quantizing & serving in practice](quantization-serving.md) — quantize → serve → validate: the tool, the settings, and what to measure.
- **Part 5 · Serving & Throughput (vLLM Core)**
    - [Static vs continuous batching](continuous-batching.md) — why static batching wastes the GPU, what iteration-level scheduling means, and what actually limits the batch size.
    - [PagedAttention: block manager & fragmentation](kv-cache-block-manager.md) — why contiguous KV fragments, what the block manager does, how `num_gpu_blocks` is set, and how paging becomes throughput.
    - [Chunked prefill & PD disaggregation](chunked-prefill-pd.md) — why a long prefill stalls decode, what chunked prefill trades, the `max_num_batched_tokens` dial, and when to disaggregate.
    - [Prefix caching](prefix-caching.md) — how block hashing makes reuse safe, why only full blocks cache, when it helps, and why outputs are unchanged.
    - [Speculative decoding](speculative-decoding.md) — guess-and-verify, why it's free only because decode is memory-bound, what sets the speedup, and when it backfires.
    - [Trace a request through vLLM's architecture](vllm-architecture.md) — the V1 components (API server / engine core / worker), an end-to-end trace, and which optimization lives in which box.
    - [Tuning knobs: which one for which SLO](tuning-knobs.md) — which knob moves which end of the throughput/latency curve, its trade, and the sweep to run.
- **Part 6 · Advanced Inference Topics**
    - [Multi-LoRA serving: one base, many adapters](multi-lora-serving.md) — why a LoRA adapter is tiny, how vLLM batches heterogeneous adapters via grouped GEMM, and the knobs (`max_lora_rank`, `max_loras`, dynamic loading) that cap how many you can co-serve.
    - [Guided / structured decoding](structured-decoding.md) — how a schema becomes a per-step logit mask, why the guarantee is hard rather than statistical, its cost, and why it fixes shape but never truth.
    - [Long-context inference: positions, sinks & the KV wall](long-context-inference.md) — why models break past training length and how RoPE scaling (PI/NTK/YaRN) fixes it, what the attention sink is, and why the KV cache — not compute — is the long-context ceiling.
- **Part 7 · Multi-GPU & Distributed**
    - [Parallelism: TP/PP/DP/EP & when to use each](parallelism-strategies.md) — the two reasons to parallelize, what each of TP/PP/DP/EP splits and costs to communicate, why TP stays within a node while PP crosses them, and how to pick a strategy from model size and topology.
    - [NCCL collectives & launching TP/PP](nccl-collective-communication.md) — what all-reduce / all-gather / reduce-scatter each move, why ring all-reduce is ~2× the message independent of GPU count, which collective TP uses and how often, and how vLLM launches TP/PP single- vs multi-node (mp vs ray) — including debugging an init hang.
- **Part 8 · Production & System Design**
    - [Serving over HTTP: the OpenAI-compatible server & its endpoints](openai-server-deployment.md) — what `vllm serve` exposes, `/v1/chat/completions` vs `/v1/completions`, what `/health` does and doesn't promise, how auth works, and interface vs capacity flags.
    - [Load-testing & the concurrency knee (Little's Law)](load-testing-knee.md) — what the knee is and why the curve bends there, open- vs closed-loop load, how Little's Law explains the runaway past it, and why you report goodput (not raw throughput).
    - [Routing, autoscaling & KV-aware routing](routing-autoscaling.md) — why prefix-aware routing beats round-robin (per-instance caches), why you autoscale on `num_requests_waiting` rather than GPU utilization, and how cold-start and drain shape a safe policy.
    - [Observability & profiling: metrics, traces & the kernel timeline](observability-profiling.md) — the three zoom levels (metrics → traces → profiles), which vLLM metrics you alert on, the prefill/decode split, and capturing a torch/Nsight profile without drowning in data.
    - [SLO-driven tuning: goodput, the binding constraint & the loop](slo-driven-tuning.md) — why you optimize goodput against an SLO, reading the binding constraint (queue/prefill/decode/KV) from metrics, which knob relieves which, and the one-knob-at-a-time loop.
    - [The serving ecosystem: choosing vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](framework-comparison.md) — the shared baseline vs the divergence axes, a defensible default with exceptions, and deciding by benchmarking OpenAI-compatibly on your own workload at your SLO.
- **Part 1** — questions land alongside their lessons in later tickets.

!!! note "Scaffolding status"
    Part 0 (tickets #2, #4, #5), Part 2 (tickets #6, #7), Part 3 (tickets #8, #9), Part 4 (tickets #10, #11), Part 5 (tickets #12, #13, #14), Part 6 (tickets #15, #16), Part 7 (tickets #17, #18), and the first six Part 8 questions (tickets #19, #20) are in, each two-way-linked to the lesson it tests. The full ~100-question bank grows as parts land. Difficulty tiers / frequency tags / weighting are intentionally out of scope for now.
