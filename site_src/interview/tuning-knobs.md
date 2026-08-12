# Tuning knobs: which one for which SLO

!!! info "Baseline: **vLLM 0.26.0** · flags verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [The Tuning Knobs: Sweeping the Throughput/Latency Curve](../part5/tuning-knobs-sweep.md)

---

## Q: Given a TTFT / throughput / OOM problem on vLLM, name the knob, its direction on the throughput↔latency curve, and its trade — and describe the sweep you'd run.

### Direct answer

There's **no universally fast config** — only one tuned to an SLO. Every knob moves one end of the throughput↔latency curve and trades something:

| Knob | Direction | Trade |
|---|---|---|
| `gpu_memory_utilization ↑` | throughput (more KV blocks → bigger batch) | VRAM headroom → OOM risk |
| `max_num_seqs ↑` | throughput (wider batch) | per-req latency at saturation |
| `max_num_batched_tokens ↑` | TTFT + throughput | worse ITL (prefill interference) |
| `quantization` INT4/AWQ | **both** (frees VRAM *and* speeds decode) | some output quality |
| `kv_cache_dtype=fp8` | throughput (~2× KV capacity) | some KV precision |
| `enable_prefix_caching` | **both** on shared prefixes | ~nothing (V1 default on) |
| `enforce_eager=True` | ↓ throughput/latency (no CUDA graphs) | saves VRAM/startup |
| `tensor_parallel_size ↑` | **both** (headroom + split compute) | multi-GPU + comm cost |

**Diagnose → knob:** TTFT high → `max_num_batched_tokens ↑` or prefix caching; OOM at startup → `gpu_memory_utilization ↓` / quantize / `max_model_len ↓`; low throughput → capacity knobs (quant, FP8 KV, `gpu_memory_utilization ↑`); slow single-stream decode → keep CUDA graphs (not `enforce_eager`) or speculative decoding.

**The sweep:** fix an [eval set](../eval/index.md), change **one** knob across a few values with fixed sampling (`temperature=0`, `seed`), measure the (quality, throughput, latency) **triple**, keep the change only if the trade is worth it.

### Deep dive

- **Capacity knobs are the master lever.** Anything fitting more KV — `gpu_memory_utilization`, [quantization](../part4/index.md), [FP8 KV](../part4/quantization-methods.md) — raises the concurrency ceiling, boosting throughput *and* cutting queueing latency.
- **The near-free knobs** (quant, prefix caching, TP) help *both* ends — reach for them first. Pure trade-offs (`max_num_seqs`, `max_num_batched_tokens`) come after.
- **Direction transfers, magnitude doesn't.** The *way* a knob pushes is a property of the mechanism; the *how far* is a property of your model/GPU/traffic — so you measure, never copy someone's numbers.

### Code

The knob→direction map (pure Python):

```python
KNOBS = {  # knob: (throughput, latency)
    "gpu_memory_utilization↑": ("↑", "≈"), "max_num_batched_tokens↑": ("↑", "↑ITL"),
    "quantization": ("↑", "↓"), "kv_cache_dtype=fp8": ("↑", "≈"),
    "enable_prefix_caching": ("↑", "↓"), "enforce_eager": ("↓", "↑"),
}
both = [k for k,(t,l) in KNOBS.items() if t=="↑" and l=="↓"]  # ['quantization','enable_prefix_caching']
```

### Interviewer follow-ups

- *"TTFT too high — first move?"* → If prompts share a prefix, `enable_prefix_caching` (free). Else `max_num_batched_tokens ↑` (trades ITL). If it's queueing, add capacity.
- *"OOM at startup?"* → `gpu_memory_utilization ↓`, quantize weights, or `max_model_len ↓` — all shrink the KV-pool sizing that overflowed.
- *"Which knobs help both ends?"* → Quantization, prefix caching, FP8 KV, TP — they don't sit on the trade; they cost quality/hardware instead.
- *"Why not just publish optimal values?"* → They depend on model/GPU/traffic; only directions transfer. Copy the sweep method, not the numbers.
- *"What must every sweep measure besides speed?"* → **Quality** — a faster config that tanks accuracy is a regression. Measure the triple.

### Linked concepts

- Lesson: [The Tuning Knobs: Sweeping the Throughput/Latency Curve](../part5/tuning-knobs-sweep.md)
- Related: [Chunked prefill & PD](chunked-prefill-pd.md) (`max_num_batched_tokens`), [PagedAttention: block manager](kv-cache-block-manager.md) (`gpu_memory_utilization`/`num_gpu_blocks`), [Prefix caching](prefix-caching.md), [Speculative decoding](speculative-decoding.md), [vLLM architecture](vllm-architecture.md) (which box each knob turns), [VRAM budget & max concurrency](vram-capacity-planning.md)
- Glossary: [SLO, Goodput, TTFT, TPOT/ITL](../glossary.md)
