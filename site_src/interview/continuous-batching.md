# Static vs continuous batching: the throughput lever

!!! info "Baseline: **vLLM 0.26.0** · flags verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [From Static to Continuous Batching](../part5/continuous-batching.md)

---

## Q: Explain static vs continuous batching. Why is continuous batching the biggest lever on inference throughput, what does "iteration-level scheduling" mean, and what actually limits the batch size?

### Direct answer

**Static batching** collects a fixed set of requests and runs them together until **every** member finishes, then takes the next set. Because output lengths vary widely, sequences that finish early leave their GPU slots **idle** until the longest one is done (bubbles), and queued requests can't start until the whole batch drains (**head-of-line blocking**).

**Continuous batching** (Orca's *iteration-level scheduling*) changes the scheduling granularity from a whole request to a **single decode iteration**: after every forward step it **evicts** finished sequences (freeing their [KV cache](../part0/kv-cache.md)) and **admits** waiting ones into the freed slots. The batch membership changes every step, so no slot sits idle while work waits.

It's *the* throughput lever because **decode is [memory-bound](../part0/inference-flow.md)**: each step reads the model weights from HBM once and reuses them across every sequence in the batch, so an extra sequence is nearly free on compute. Keeping the batch as full as possible maximizes that amortization. What limits the batch is almost always **KV-cache capacity** (free blocks to admit another sequence), not compute — until you eventually cross the [roofline](../part2/roofline-analysis.md) compute ridge.

### Deep dive

- **The two loops.** Static: `admit N → step until all done → evict all`. Continuous: `admit → step (one token each) → evict → repeat`. The "step until all done" is exactly what creates the bubbles.
- **Why memory-bound makes it work.** Batch of 1 reads all weights to make one token — almost pure waste. Batch of 32 reads the same weights once for 32 tokens — ~32× the useful work for the same HBM traffic. Continuous batching keeps you near that sweet spot every step.
- **The limiting walls.** (1) KV-cache capacity — the common one; more sequences need more KV blocks, and admission stalls when the pool is empty. This is why [PagedAttention](../part5/paged-attention.md) (kill fragmentation → more blocks) and KV-cache quant matter. (2) The compute ridge — enough batched GEMM work saturates the tensor cores; past it, latency rises without throughput.
- **The knobs.** `max_num_seqs` (default **128**) caps the running-set width; `max_num_batched_tokens` (default **2048**, auto-tuned) caps tokens processed per step. Neither *is* the batch size — the batch floats below them, bounded by KV room.
- **You don't enable it.** Continuous batching *is* vLLM's scheduler — there's no flag. Sending concurrent requests is enough.

### Code

The scheduling difference as pure Python — only the refill timing differs:

```python
from collections import deque
def continuous(requests, slots):          # iteration-level: evict done, admit waiting, every step
    waiting, running, step, busy = deque(requests), {}, 0, 0
    while waiting or running:
        while waiting and len(running) < slots:      # ADMIT into any free slot
            rid, n = waiting.popleft(); running[rid] = n
        step += 1
        for rid in running: running[rid] -= 1; busy += 1   # STEP: one token each
        running = {r: n for r, n in running.items() if n > 0}  # EVICT finished
    return step, busy
# vs static, which would step the same batch "until all done" before refilling — leaving bubbles.
```

A freed slot is backfilled on the next iteration, not held until the batch drains.

### Interviewer follow-ups

- *"Throughput is low and the GPU isn't saturated — where do you look?"* → Admission, not batching. The batch is likely starved for KV room: is the model quantized to free VRAM for blocks? Is `gpu_memory_utilization` (0.92) leaving headroom? Would FP8 KV cache fit more sequences? The lever is capacity ([PagedAttention](../part5/paged-attention.md), quantization).
- *"Does a bigger batch always help?"* → Only up to KV capacity or the compute ridge. Before the ridge (memory-bound decode), extra sequences are nearly free; past it, more sequences add latency without throughput.
- *"How could you accidentally get static batching in your own code?"* → A loop that hands vLLM a fixed list and *waits for all of it* before the next list barriers on whole batches — you've thrown away continuous scheduling. Stream requests in instead.
- *"What's the padding difference?"* → Static batching typically pads all sequences to the longest and computes the pad tokens (double waste); continuous batching over a paged KV cache has no padding — each sequence uses exactly the blocks it needs.
- *"Where does chunked prefill fit?"* → It's the knob (`max_num_batched_tokens`) that lets prefill and decode share a step, balancing TTFT vs throughput — a scheduler refinement on top of continuous batching (next Part 5 topic).

### Linked concepts

- Lesson: [From Static to Continuous Batching](../part5/continuous-batching.md)
- Related: [PagedAttention: KV cache as virtual memory](kv-cache-block-manager.md) (the capacity that limits admission), [Prefill vs decode](prefill-vs-decode.md) (why decode is memory-bound), [KV cache & throughput ceiling](kv-cache.md), [VRAM budget & max concurrency](vram-capacity-planning.md), [Arithmetic intensity](arithmetic-intensity.md) (the compute ridge)
- Glossary: [Static / Dynamic / Continuous batching](../glossary.md)
