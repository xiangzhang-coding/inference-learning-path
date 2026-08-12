# VRAM budget & max concurrency

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 2 · Single-GPU Inference Performance   ·   **Tests the lesson:** [KV Cache Memory Math: Sizing a Deployment](../part2/kv-cache-math.md)

---

## Q: Qwen2.5-7B on a single 24 GB RTX 4090 at 8k context. Walk the full VRAM budget and estimate max concurrent sequences. Then: your SLO needs ~60 concurrent streams — how do you get there?

### Direct answer

Concurrency is a **leftover**: usable VRAM minus the fixed costs, divided by KV per sequence.

$$
N_{\text{seq}} = \left\lfloor \frac{u\cdot V - W - A - O}{\kappa\, S} \right\rfloor
$$

with $u$ = `gpu_memory_utilization` (default 0.92), $V=24$ GiB, $W$ = weights, $A$ = activations, $O$ = CUDA/framework overhead, and $\kappa=56$ KiB/token for Qwen2.5-7B (from [Part 0](../part0/kv-cache.md)). Taking $u=0.90$, $A+O\approx1.6$ GiB, $S=8192$ (so $\kappa S=0.44$ GiB/seq):

- **BF16 weights (~14.2 GiB):** budget $=0.9\cdot24-14.2-1.6=5.8$ GiB → $\approx$ **13** concurrent.
- **AWQ 4-bit weights (~5.5 GiB):** budget $=14.5$ GiB → $\approx$ **33**.
- **AWQ weights + FP8 KV** ($\kappa\to28$ KiB): $\approx$ **66**.

**To hit ~60:** (1) quantize the **weights** (AWQ/GPTQ) — frees ~8 GiB, the biggest single block (~13→~33); (2) quantize the **KV** (`kv_cache_dtype=fp8`) — halves per-seq bytes (~33→~66, clearing 60); (3) if still short, **cap `max_model_len`** to the real workload context (concurrency $\propto 1/S$). Weights first, because they're the largest fixed cost.

### Deep dive

- **Why weights before KV.** On 24 GB the weights dominate the budget; quantizing them frees ~8 GiB straight into KV, usually a bigger concurrency gain than halving KV bytes. KV quantization is the second lever, not the first.
- **The overhead you can't skip.** `gpu_memory_utilization` caps usable VRAM below 24 GB, and CUDA context + activation/workspace eats ~1–2 GiB before any KV. Part 0's naive "~22" ignored these; the honest number is lower.
- **`max_model_len` is a trade.** It bounds per-sequence KV, so raising it for long context cuts concurrency proportionally. Set it to the workload, not the model's 32k max.
- **PagedAttention vs the formula.** Paging allocates fixed-size blocks; a sequence's last block is partly empty, so real concurrency is a hair under the formula. But it *eliminates external fragmentation*, so far more of the leftover is usable than with contiguous allocation — the reason vLLM's reported capacity is close to the arithmetic at all.

### Code

The planner, and the inverse question:

```python
GIB = 1024**3
def kv_budget(util, vram, weight_gib, overhead=1.6):
    return util*vram - weight_gib - overhead
def max_conc(util, vram, weight_gib, kappa, S):
    return int(kv_budget(util, vram, weight_gib)*GIB / (kappa*S))

print(max_conc(0.90, 24, 14.2, 57344, 8192))   # ~13  BF16 weights, BF16 KV
print(max_conc(0.90, 24, 5.5,  57344, 8192))   # ~33  AWQ  weights, BF16 KV
print(max_conc(0.90, 24, 5.5,  28672, 8192))   # ~66  AWQ  weights, FP8  KV
```

Verify against vLLM's own startup log (v0.26.0): `GPU KV cache size: N tokens` ($=$ budget$/\kappa$) and `Maximum concurrency for 8,192 tokens per request: Xx` ($=$ that $/$ `max_model_len`).

### Interviewer follow-ups

- *"Why not just set `gpu_memory_utilization=0.99`?"* → Activations spike under concurrent prefills; too thin a margin boots fine then OOMs at peak. The 0.92 default is deliberate headroom.
- *"How does PagedAttention change the number?"* → It doesn't change the per-sequence formula; it makes more of the leftover *usable* by killing external fragmentation (minus small last-block padding). It raises effective capacity, not $\kappa$.
- *"What's the cost of `kv_cache_dtype=fp8`?"* → It halves KV bytes (more concurrency) but can shift outputs — treat the quality delta as something to measure on your eval set, not assume free.
- *"Serve 128k context on this card — feasible?"* → One 128k sequence is $\kappa\cdot128\text{k}\approx7$ GiB (BF16 KV); a handful saturate the budget. You'd need FP8 KV, aggressive weight quant, and accept very low concurrency — or shard across GPUs.

### Linked concepts

- Lesson: [KV Cache Memory Math: Sizing a Deployment](../part2/kv-cache-math.md)
- Related: [KV cache & throughput ceiling](kv-cache.md) (why KV, not compute, caps concurrency), [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md)
- Glossary: [KV cache, PagedAttention, GQA, SLO](../glossary.md)
