# Glossary

> A bilingual mirror of the repository's ubiquitous-language file (`CONTEXT.md`). Definitions only — no implementation detail, decisions, or budget. Cross-references use →. Terms are kept in English throughout the site, even in Chinese pages, to match interview and source-reading contexts.

## Inference Flow

- **Prefill** — the stage that processes the whole input prompt at once, computes its KV, and emits the first output token; usually → compute-bound. → Decode, → TTFT
- **Decode** — the autoregressive stage that generates subsequent tokens one at a time, computing a single new token per step and appending its KV; usually → memory-bound. → Prefill, → TPOT
- **Autoregressive** — each new token is generated conditioned on all previous tokens.

## Memory & Cache

- **KV cache** — cached Key/Value tensors of already-generated tokens, avoiding step-by-step recomputation of historical attention; the central tension behind inference memory footprint and the throughput ceiling. → PagedAttention, → GQA
- **HBM / SRAM** — the GPU's high-bandwidth memory (HBM) versus on-chip cache/registers (SRAM); their bandwidths differ by an order of magnitude, the premise of IO-aware optimization.
- **Memory-bound / Compute-bound** — whether the bottleneck is data movement or computation. → Roofline

## Architecture

- **MHA / MQA / GQA** — Multi-Head / Multi-Query / Grouped-Query Attention; the number of KV heads decreases in turn, directly shrinking the → KV cache and raising the throughput ceiling.
- **FFN / MLP** — the feed-forward layers that carry most of the FLOPs and weight memory in a Transformer.
- **RoPE** — rotary position embedding; its extrapolation properties underpin → long-context inference.
- **MoE** — Mixture-of-Experts, a sparse structure that activates only some experts per token. → Expert parallelism

## Metrics

- **TTFT** — Time To First Token; dominated by → Prefill.
- **TPOT / ITL** — Time Per Output Token / Inter-Token Latency; dominated by → Decode.
- **Throughput** — tokens or requests processed per unit time.
- **Goodput** — effective throughput under → SLO constraints, not raw throughput.

## Single-GPU Performance

- **Roofline / Arithmetic Intensity** — a model that judges whether an operator is compute- or bandwidth-limited from its "compute / memory-traffic" ratio.
- **FlashAttention** — an IO-aware attention algorithm that cuts HBM reads/writes with tiling + online softmax.
- **CUDA graphs** — record a sequence of kernel launches into a graph and replay it, amortizing launch overhead in the → Decode stage.
- **Kernel fusion** — merge multiple operators into one kernel to reduce memory traffic and launch overhead.

## GPU Programming

- **SM / Warp / Occupancy** — Streaming Multiprocessor / a 32-thread scheduling unit / occupancy.
- **Coalescing / Shared memory / Bank conflict** — memory-access coalescing / on-chip shared memory / bank conflicts.
- **Triton** — a Python-based GPU kernel language.

## Quantization

- **PTQ / QAT** — Post-Training Quantization / Quantization-Aware Training.
- **Weight-only vs Weight+Activation** — quantizing only weights vs quantizing both weights and activations.
- **Per-tensor / per-channel / per-group** — quantization granularity.
- **GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** — mainstream quantization method families.
- **KV-cache quantization** — quantizing the → KV cache itself to save memory.

## Serving & Throughput

- **Static / Dynamic / Continuous batching** — continuous batching (Orca-style) admits requests as they arrive and evicts them as they finish; the key lever for inference throughput.
- **PagedAttention** — manages the → KV cache in blocks, allocated like virtual-memory paging, eliminating fragmentation and raising utilization.
- **Chunked prefill** — split a long prefill into chunks and interleave it with → Decode scheduling, balancing TTFT and throughput.
- **PD disaggregation** — split → Prefill and → Decode onto different resources to optimize each separately.
- **Prefix caching** — reuse the KV of a shared prefix, skipping repeated prefill.
- **Speculative decoding** — a small draft model guesses several tokens, the large model verifies them in one pass, speeding up → Decode.

## Advanced Topics

- **LoRA / Multi-LoRA serving** — low-rank adapters; a serving form with one base + multiple adapters swapped dynamically.
- **Guided / Structured decoding** — constrain output with JSON / regex / grammar.
- **Long-context inference** — → RoPE extrapolation, attention sink, KV compression, and the memory/scheduling problems of long sequences.

## Distributed

- **Tensor / Pipeline / Data / Expert Parallelism** — the four parallelism axes. → MoE
- **Collective communication** — all-reduce / all-gather / reduce-scatter and other primitives (NCCL).
- **TP degree** — the number of GPUs a tensor-parallel split spans.

## Production

- **SLO** — Service Level Objective; drives → Goodput and tuning.
- **Knee** — the point where the throughput-vs-concurrency curve starts to degrade; the key spot load tests hunt for.
- **KV-cache aware routing** — route requests by which instance already caches their prefix, improving prefix-cache hit rate.
