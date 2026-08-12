# CUDA execution model: warps, SIMT & occupancy

!!! info "Baseline: **vLLM 0.26.0** · `torch.cuda` device-query API verified via Context7 (ADR-0004)"

**Module:** Part 3 · GPU Programming (Triton)   ·   **Tests the lesson:** [The CUDA Execution Model: Threads, Warps, and Occupancy](../part3/cuda-execution-model.md)

---

## Q: Walk me through how a GPU actually runs a kernel. What's a warp, what does SIMT divergence cost, how does the SM hide memory latency, and does maximizing occupancy always make a kernel faster?

### Direct answer

A kernel launches a **grid of blocks**; each block is assigned to **one SM** and is chopped into **warps of 32 threads**. The warp is the real unit of execution: the scheduler issues **one instruction per cycle for all 32 lanes** — this is SIMT (Single Instruction, Multiple Threads).

- **Divergence**: if the 32 lanes of a warp take *different* sides of a data-dependent branch, the warp **serializes both paths** (running each with the other lanes masked off) — roughly $T_{if}+T_{else}$ instead of one side. The cost is per-warp: if all 32 lanes agree, there's no penalty. Different warps diverge for free.
- **Latency hiding**: an HBM load costs hundreds of cycles. The SM keeps many warps resident and, when one stalls, switches to a ready one at zero cost (all resident warps' registers stay live). Latency is *hidden*, never removed.
- **Occupancy** = resident warps / max warps per SM (48 on compute capability 8.9). It's the *slack* that enables latency hiding — capped by whichever per-SM resource runs out first: registers/thread, shared-mem/block, or block-count limits.

Does maxing occupancy always help? **No.** You want *enough* occupancy to hide the latency, then more buys nothing — and forcing it (e.g. cutting registers) can cause spills that hurt. A kernel can also be memory-bound at 100% occupancy.

### Deep dive

- **Why 32 everywhere.** The 32-thread warp is a hardware constant (on NVIDIA). Launch a block of 40 threads and you still occupy two warps — 24 lanes in the second sit idle. Block sizes are multiples of 32 to avoid wasting lanes.
- **Occupancy is a resource-limited max.** With a 65,536-register file per SM (cc 8.9), a kernel using 64 registers/thread caps at $65536/64 = 1024$ threads = 32 warps → ≤67% occupancy. Shared memory does the same to co-resident blocks. The CUDA occupancy calculator just solves for the binding constraint.
- **"Enough" is workload-dependent.** Memory-bound kernels need more resident warps (more stalls to hide); compute-bound kernels saturate at lower occupancy. This is why blindly maximizing occupancy is the wrong target — the [roofline](../part2/roofline-analysis.md) tells you which regime you're in.
- **Tie to LLM inference.** Decode at batch 1 launches tiny kernels that can't fill the SMs — too few warps to hide latency, so the GPU idles. That underutilization is the whole motivation for [continuous batching](../part5/index.md) (pack sequences → one fat launch) and for [CUDA graphs](cuda-graphs-fusion.md) (kill the per-launch overhead).

### Code

The SIMT divergence rule as a pure-CPU model — cost is per-warp, and only split warps pay double:

```python
WARP = 32
def branch_bodies(conditions):                      # conditions[i]: does lane i take 'if'?
    bodies = 0
    for s in range(0, len(conditions), WARP):
        warp = conditions[s:s + WARP]
        bodies += 1 if (all(warp) or not any(warp)) else 2   # uniform: 1 side; divergent: both
    return bodies

n = 32 * 8                                           # 8 warps
interleaved = [i % 2 == 0 for i in range(n)]         # every warp split -> 16 (2.0x)
aligned     = [(i // WARP) % 2 == 0 for i in range(n)]  # each warp uniform -> 8 (1.0x)
print(branch_bodies(interleaved), branch_bodies(aligned))   # 16 8
```

Same 50/50 workload; the interleaved layout costs 2× purely because the branch splits every warp.

### Interviewer follow-ups

- *"Why must launch configs be multiples of 32?"* → The hardware chops blocks into 32-thread warps regardless; a non-multiple wastes lanes in the last, under-filled warp (they're masked but still occupy a warp slot).
- *"How would you fix a kernel that's slow due to divergence?"* → Make warps internally uniform — sort/bucket data so lanes in a warp take the same branch, or restructure so the branch aligns to 32-thread boundaries. The tax is from lanes *within* a warp disagreeing.
- *"Occupancy went from 50% to 100% but runtime didn't improve — why?"* → It was already memory-bandwidth-bound (or 50% was already enough to hide latency). Occupancy is slack for hiding stalls, not a throughput multiplier; check the roofline / achieved bandwidth.
- *"What limits occupancy?"* → The binding per-SM resource: registers/thread, shared-memory/block, threads/block, or the resident-block cap — whichever is exhausted first.
- *"Is `__syncthreads()` a global barrier?"* → No — it synchronizes threads within *one block* only. There's no cheap in-kernel grid-wide barrier; global sync means a new kernel launch.

### Linked concepts

- Lesson: [The CUDA Execution Model: Threads, Warps, and Occupancy](../part3/cuda-execution-model.md)
- Related: [Memory coalescing, shared memory & bank conflicts](memory-coalescing.md) (how those warps should touch memory), [GPU memory hierarchy & roofline](gpu-memory-hierarchy.md) (SM/warp/HBM tiers), [CUDA graphs & kernel fusion](cuda-graphs-fusion.md) (why tiny decode launches underfill the GPU)
- Glossary: [SM / Warp / Occupancy](../glossary.md)
