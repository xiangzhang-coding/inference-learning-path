# Preemption: recompute vs swap when the KV pool is exhausted

!!! info "Baseline: **vLLM 0.26.0** · flags verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [PagedAttention: Managing the KV Cache Like Virtual Memory](../part5/paged-attention.md)

---

## Q: When vLLM can't find KV blocks for the running batch, what does it do? Explain preemption, recompute vs swap, and vLLM's V1 behavior.

### Direct answer

When the KV pool can't grow a running sequence (or admit a needed one), the scheduler **preempts**: it evicts a running sequence, frees its KV blocks, and re-queues it to resume when space frees up — the paging equivalent of the OS swapping a page out. It's **graceful degradation**, not a crash.

Two classic recovery modes on resume:

- **Recompute** — discard the evicted KV; when the sequence resumes, re-run prefill over its tokens to rebuild the KV. Cost = recompute FLOPs; **no CPU transfer**.
- **Swap** — copy the evicted KV out to CPU RAM (swap space) and copy it back on resume. Cost = PCIe transfer **both ways**; needs a `swap_space`.

**vLLM V1 (the 0.26.0 default engine) uses `RECOMPUTE` by default** — lower overhead than swap. In fact V1 **removed swap**: the `--swap-space` flag is gone and the swapped-preemption metrics (`vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc`) are no longer relevant; **prefix caching** gives recompute a near-zero-overhead path by reusing already-cached blocks. You observe preemption as a log warning:

```text
WARNING scheduler.py Sequence group N is preempted by PreemptionMode.RECOMPUTE
mode because there is not enough KV cache space. ... total_cumulative_preemption_cnt=1
```

### Deep dive

- **Why recompute wins in V1:** recompute *is* prefill — compute-bound and fast on the GPU — whereas swap moves megabytes of KV over PCIe, slow and bandwidth-contended. With prefix caching, the recompute can hit cached blocks, so it's often nearly free.
- **Preemption is a symptom, not a knob.** It signals under-provisioned KV. The fix is **capacity**: ↑ `gpu_memory_utilization`, quantize (weights or FP8 KV), ↓ `max_num_seqs` / `max_num_batched_tokens`, or ↑ `tensor_parallel_size` / `pipeline_parallel_size` to spread KV across GPUs.
- **The running set can shrink, not just grow.** A sequence you think is "running" can be evicted and re-admitted under pressure — steady low throughput with churn in the running count is the tell (the [continuous-batching](continuous-batching.md) `schedule()` preempt path).
- **Preemption ≠ OOM.** Preemption keeps the server alive and just adds latency; OOM is a hard crash. Preemption is exactly the mechanism that lets vLLM stay robust when demand exceeds KV.

### Code

A minimal scheduler sim — fixed block pool; admit requests; when blocks run out, preempt the newest running sequence and re-queue it (pure Python, illustrative — counts are 示例):

```python
from collections import deque

def run(pool_blocks, arrivals, blocks_per_req):
    free, running, waiting, preemptions = pool_blocks, [], deque(arrivals), 0
    while waiting or running:
        # admit while there's room
        while waiting and free >= blocks_per_req:
            running.append(waiting.popleft()); free -= blocks_per_req
        # a running seq needs one more block but the pool is empty -> preempt
        if running and free == 0:
            victim = running.pop()            # evict newest (LIFO)
            free += blocks_per_req            # free its KV (RECOMPUTE: just drop it)
            waiting.append(victim)            # re-queue to resume later
            preemptions += 1
        # a step of progress: retire the oldest running seq
        if running:
            running.pop(0); free += blocks_per_req
    return preemptions

print(run(pool_blocks=4, arrivals=list(range(6)), blocks_per_req=2))  # 示例
```

### Interviewer follow-ups

- *"Recompute vs swap — which is cheaper?"* → Recompute (prefill is fast, compute-bound); swap pays PCIe both ways. V1 defaults to recompute and dropped swap entirely.
- *"Cumulative preemption count is climbing — what do you change?"* → Capacity, not the scheduler: ↑ `gpu_memory_utilization`, quantize / FP8 KV, ↓ `max_num_seqs` / `max_num_batched_tokens`, or ↑ TP/PP.
- *"Is preemption the same as OOM?"* → No. Preemption is graceful (evict + resume, latency cost); OOM crashes the engine. Preemption is what prevents the crash under KV pressure.
- *"How does prefix caching interact with recompute?"* → A recomputed sequence can hit the prefix cache, so the "recompute" is often near-free — part of why V1 could drop swap.
- *"What tells you it's happening?"* → The cumulative preemption count (Prometheus metrics, or `disable_log_stats=false`) and the `PreemptionMode.RECOMPUTE` warning.

### Linked concepts

- Lesson: [PagedAttention: Managing the KV Cache Like Virtual Memory](../part5/paged-attention.md) — where the block manager and the preempt-under-pressure path live.
- Related: [PagedAttention: block manager & fragmentation](kv-cache-block-manager.md), [Static vs continuous batching](continuous-batching.md) (the `schedule()` preempt path), [Tuning knobs](tuning-knobs.md) (the capacity knobs that stop it), [Prefix caching](prefix-caching.md), [VRAM budget & max concurrency](vram-capacity-planning.md)
- Glossary: [Preemption, PagedAttention](../glossary.md)
