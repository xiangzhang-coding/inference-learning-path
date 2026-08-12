# CUDA graphs & kernel fusion

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 2 · Single-GPU Inference Performance   ·   **Tests the lesson:** [Kernel Fusion & CUDA Graphs: Killing Decode Launch Overhead](../part2/kernel-fusion-cuda-graphs.md)

---

## Q: A decode step runs at low GPU utilization even though the model is loaded. Beyond being memory-bound, what other tax does decode pay? Explain kernel launch overhead, why it hits decode but not prefill, how kernel fusion and CUDA graphs address it, and what `enforce_eager` trades.

### Direct answer

The extra tax is **kernel launch overhead**. A decode step runs *hundreds* of small kernels (per layer: norms, QKV/O projections, attention, gate/up/down, activations, residual adds — ×~28 layers). Each launch costs the CPU a few microseconds to dispatch, and because decode kernels are tiny (batch 1, memory-bound), the GPU finishes each and **idles waiting for the CPU to launch the next**. Model a step as $T_{\text{eager}} \approx T_{\text{compute}} + N\tau$: with $N\approx430$ kernels and $\tau\approx5\,\mu s$, that's ~2 ms of pure launch tax per step.

It **hits decode not prefill** because prefill kernels are big (many tokens, compute-bound), so $N\tau \ll T_{\text{compute}}$ — the launches hide behind real work. In decode the per-kernel GPU work is tiny, so launches dominate.

- **Kernel fusion** merges ops into one kernel: fewer launches ($N$ down) *and* fewer HBM round-trips (intermediates stay in SRAM/registers).
- **CUDA graphs** record the whole kernel sequence once and replay it with a *single* CPU submit: $T_{\text{graph}} \approx T_{\text{compute}} + \tau$, collapsing $N\tau$.

**`enforce_eager=True`** disables CUDA-graph capture (and torch.compile): it **frees the VRAM** the captured graphs would occupy but **pays the launch tax** every decode step — a throughput-for-memory trade.

### Deep dive

- **Quantized models benefit *more*.** The launch tax $N\tau$ is fixed, so shrinking compute (AWQ weights → smaller $T_{\text{compute}}$) makes it a bigger fraction: ~36% of a 6 ms step vs ~14% of a 15 ms step. Optimize compute → graphs matter more, not less.
- **The static-shape constraint.** A graph captures fixed shapes and memory addresses; vLLM captures a graph per batch-size bucket and **pads** the running batch up to it. Novel shapes / data-dependent control flow fall back to eager.
- **Warmup + static buffers.** You must run the exact workload before capture (settle allocations/autotuning) and copy inputs into the same static buffers each replay, since the graph reuses captured pointers.
- **Fusion ≠ graphs.** Fusion lowers the kernel *count*; graphs lower the *launch cost* of the remaining kernels. `torch.compile` does fusion; vLLM does both.

### Code

The launch-overhead model — why quantized decode gains more (pure CPU):

```python
def step_ms(weight_gib, n_kernels=430, tau_us=5.0, bw=1e12):
    compute = weight_gib * 1024**3 / bw * 1e3      # bytes / bandwidth (ms)
    eager = compute + n_kernels * tau_us / 1e3
    graph = compute + tau_us / 1e3
    return compute, eager, graph, eager / graph
for w in (5.5, 14.2):                              # AWQ vs BF16 weights
    c, e, g, spd = step_ms(w)
    print(f"{w:>4} GiB: compute {c:4.1f}ms eager {e:4.1f}ms graph {g:4.1f}ms speedup {spd:.2f}x")
# 5.5 GiB: compute  5.9ms eager  8.1ms graph  5.9ms speedup 1.36x
# 14.2 GiB: compute 15.2ms eager 17.4ms graph 15.3ms speedup 1.14x
```

### Interviewer follow-ups

- *"Why does CUDA graph help decode but not prefill?"* → Launch overhead's impact = $N\tau$ relative to $T_{\text{compute}}$. Prefill kernels are big/compute-bound ($N\tau$ negligible); decode kernels are tiny/memory-bound (launches dominate).
- *"You enabled graphs but throughput didn't move — why?"* → Capture may not have engaged: dynamic shapes outside the captured buckets, unsupported ops, or fell back to eager. Check that shapes hit a captured bucket.
- *"When would you *want* `enforce_eager`?"* → When VRAM-starved (reclaim the graph buffers for [KV cache](vram-capacity-planning.md)) or when debugging — accepting lower decode throughput.
- *"Does fusion alone give you one launch per step?"* → No — fusion reduces the count but you still launch each remaining kernel. Only a CUDA graph collapses the whole sequence to a single submit.

### Linked concepts

- Lesson: [Kernel Fusion & CUDA Graphs: Killing Decode Launch Overhead](../part2/kernel-fusion-cuda-graphs.md)
- Related: [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md) (why decode's per-kernel work is tiny), [FlashAttention & IO-aware attention](flash-attention.md), [VRAM budget & max concurrency](vram-capacity-planning.md) (the memory `enforce_eager` frees)
- Glossary: [CUDA graphs, Kernel fusion, Memory-bound / Compute-bound](../glossary.md)
