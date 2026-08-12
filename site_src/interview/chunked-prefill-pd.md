# Chunked prefill & PD disaggregation: balancing TTFT and throughput

!!! info "Baseline: **vLLM 0.26.0** · flags verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [The Scheduler: Chunked Prefill & PD Disaggregation](../part5/scheduler-chunked-prefill-pd.md)

---

## Q: A long prompt arrives while other requests stream. What goes wrong, how does chunked prefill fix it, what does `max_num_batched_tokens` trade, and when would you reach for PD disaggregation instead?

### Direct answer

[Prefill is compute-bound, decode is memory-bound](../part0/inference-flow.md), and they compete for one GPU. A long prompt's prefill, run as one monolithic step, **monopolizes the GPU** — every already-running sequence's decode stalls, so their inter-token latency (ITL) spikes (streams visibly freeze).

**Chunked prefill** (`enable_chunked_prefill=True`, default) slices the long prefill into chunks and co-schedules each chunk *alongside* the ongoing decodes within one step's `max_num_batched_tokens` budget. Decodes advance every step (smooth ITL); the new request's first token comes a bit later (slightly higher TTFT). That's the trade.

**`max_num_batched_tokens`** is the dial: **lower** → less prefill per step → better ITL, worse TTFT; **higher** (docs suggest >8192 for throughput) → more prefill per step → better TTFT, more decode interference.

**PD disaggregation** takes the same "don't mix the phases" logic across GPUs: run prefill on a producer pool and decode on a consumer pool (`--kv-transfer-config`, NixlConnector `kv_producer`/`kv_consumer`), streaming the KV cache between them — so each pool is tuned and scaled for its own bottleneck. It's a multi-instance, large-scale technique; chunked prefill is the single-GPU version of the same idea.

### Deep dive

- **A step is a token budget, not a slot count.** Prefill and decode tokens draw from the same `max_num_batched_tokens`; chunked prefill just lets a prefill take *part* of it. vLLM's default policy prioritizes decode to protect ITL.
- **Chunked prefill doesn't speed up prefill** — it may make one prefill slightly slower (extra steps). It improves the *system*: decodes stop stalling → better aggregate ITL/throughput.
- **Disable-chunked-prefill caveat.** If `enable_chunked_prefill=False`, `max_num_batched_tokens` must exceed `max_model_len` (a whole prompt must fit one step) or the server won't start.
- **PD's cost.** A KV-cache transfer per request (network bandwidth + latency) plus operational complexity (producer/consumer instances + routing proxy). Worth it only when independent scaling/tuning of the two pools beats the transfer cost.

### Code

The scheduling trade as pure Python — decode stalls with vs without chunking:

```python
BUDGET, DECODES, PREFILL = 16, 4, 48
def without_chunking(b, d, p):                      # prefill runs alone; decodes starve
    steps = delayed = 0
    while p > 0: p -= min(b, p); steps += 1; delayed += d
    return steps, delayed
def with_chunking(b, d, p):                         # decode + prefill chunk share each step
    steps = delayed = 0; chunk = b - d
    while p > 0: p -= min(chunk, p); steps += 1
    return steps, delayed
print(without_chunking(BUDGET, DECODES, PREFILL))   # (3, 12): 1 step sooner, 12 decode-tokens frozen
print(with_chunking(BUDGET, DECODES, PREFILL))      # (4, 0) : +1 step TTFT, 0 decode stall
```

### Interviewer follow-ups

- *"You're ITL-bound — which way do you move the dial?"* → **Lower** `max_num_batched_tokens`: less prefill interference per step, smoother ITL (at the cost of TTFT).
- *"TTFT-bound?"* → **Raise** it (toward 8192+): more prefill per step, faster first token (at the cost of running-stream ITL).
- *"Why not always disaggregate?"* → PD needs ≥2 instances and pays a KV transfer per request; on one GPU there's nothing to split, and at small scale the transfer/ops cost outweighs the benefit.
- *"Chunked prefill vs prefix caching?"* → Chunked prefill splits *one* prefill across steps; prefix caching skips prefill for a *shared* prefix. Different levers, often stacked.
- *"What's the root idea both share?"* → Prefill and decode want different things (compute vs bandwidth); chunked prefill time-shares one GPU, PD space-separates onto different GPUs.

### Linked concepts

- Lesson: [The Scheduler: Chunked Prefill & PD Disaggregation](../part5/scheduler-chunked-prefill-pd.md)
- Related: [Static vs continuous batching](continuous-batching.md) (the running set this shapes), [Prefix caching](prefix-caching.md) (the sibling prefill lever), [Prefill vs decode](prefill-vs-decode.md) (the compute/memory split), [Latency vs throughput metrics](latency-throughput-metrics.md) (TTFT/ITL)
- Glossary: [Chunked prefill, PD disaggregation](../glossary.md)
