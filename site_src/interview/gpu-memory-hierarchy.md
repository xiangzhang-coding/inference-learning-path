# GPU memory hierarchy & roofline

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 0 · Foundations   ·   **Tests the lesson:** [GPU Hardware Mental Model](../part0/gpu-hardware.md)

---

## Q: Walk through the GPU memory hierarchy and the SM/warp execution model, then use the roofline model to explain why LLM decode is memory-bound. What is the ridge point, and where do decode and prefill sit relative to it?

### Direct answer

A GPU has a **memory pyramid**: registers and SRAM (L1/shared memory) on-chip at ~TB/s to tens of TB/s, an L2 cache, and **HBM off-chip at ~1 TB/s** — roughly an order of magnitude slower than SRAM. Weights and the KV cache live in HBM, so every decode step drags them across that slow line. Compute happens on **SMs** (Streaming Multiprocessors, ~128 on a 4090), each running **warps** of 32 threads in lockstep; an SM hides memory latency by swapping warps (**occupancy** = how many are resident to swap between).

The **roofline** says attainable throughput $= \min(P,\ I\cdot B)$, where $P$ = peak FLOP/s, $B$ = bandwidth, $I$ = arithmetic intensity (FLOP/byte). The **ridge point** $I^{*} = P/B$ is where the sloped bandwidth roof meets the flat compute roof — for a 4090 (~165 TFLOP/s BF16 dense, ~1 TB/s) that's **≈165 FLOP/byte**. **Decode** runs at $I\approx1$, ~two orders of magnitude *below* the ridge, so it's pinned to the bandwidth roof at ~1 TFLOP/s — **< 1% of peak, memory-bound**. **Prefill** runs at $I$ in the thousands, *above* the ridge, on the flat compute roof — **compute-bound**. Same GPU, opposite regimes.

### Deep dive

- **Why HBM is the villain.** The bandwidth gap between HBM and SRAM (an order of magnitude) means an algorithm that re-reads from HBM when it could have stayed in SRAM pays that penalty every access. That's the entire motivation for IO-aware kernels like FlashAttention (keep the attention working set in SRAM, avoid HBM round-trips).
- **Occupancy hides latency, not bandwidth.** Once enough warps are resident to keep the memory pipe saturated, adding more buys nothing for a memory-bound kernel — you're limited by bytes/second, not idle SMs. This is why "just raise occupancy" doesn't fix decode.
- **Achievable vs peak.** Real kernels hit ~70–85% of spec-sheet HBM bandwidth, and the headline peak TFLOPS often assumes 2:4 sparsity or FP8 (a 4090's "≈330 TFLOPS" is the sparse figure; dense BF16 is ~half). Always qualify "dense or sparse? which dtype?"
- **How the levers move the roofline.** Quantization cuts bytes (higher $I$ *and* fewer bytes to move → the direct decode win); batching reuses weights across requests (higher $I$); IO-aware kernels dodge HBM (effectively more usable $B$). All three attack the *bandwidth* side because that's the binding constraint.

### Code

The regime falls out of `min(P, I·B)` — no GPU needed:

```python
P, B = 165e12, 1.0e12          # 4090: ~165 TFLOP/s BF16 dense, ~1 TB/s (illustrative)
ridge = P / B                  # I* = 165 FLOP/byte
for I in (1, 1000):            # decode ~1, prefill ~1000
    got = min(P, I * B)
    print(f"I={I:>4}: {got/1e12:6.1f} TFLOP/s ({got/P:5.1%} of peak)")
# I=   1:    1.0 TFLOP/s ( 0.6% of peak)   <- decode, memory-bound
# I=1000:  165.0 TFLOP/s (100.0% of peak)  <- prefill, compute-bound
```

### Interviewer follow-ups

- *"A vendor doubles TFLOPS but keeps bandwidth. Does decode speed up?"* → No — decode is below the ridge, pinned to $I\cdot B$, and $B$ is unchanged. Prefill would (it rides the higher $P$ roof). Decode wants bandwidth, prefill wants FLOPs.
- *"Why is FlashAttention faster if it does the same FLOPs?"* → It's IO-aware: tiling + online softmax keep the attention working set in SRAM instead of round-tripping the big intermediate scores through HBM, cutting bytes moved — a win on the bandwidth-bound side.
- *"Where does quantization move you on the roofline?"* → Rightward (higher $I$, fewer bytes/param) *and* it lowers the bytes that must cross HBM — a direct throughput multiplier for the memory-bound decode phase.
- *"What limits a kernel at 100% occupancy that's still slow?"* → Bandwidth. Occupancy hid all the latency it could; the kernel is now bound by $B$ (bytes/second), so you need fewer bytes or higher intensity, not more warps.

### Linked concepts

- Lesson: [GPU Hardware Mental Model](../part0/gpu-hardware.md)
- Related lesson: [Inference Flow: Prefill & Decode](../part0/inference-flow.md) (where decode's $I\approx1$ comes from)
- Glossary: [SM / Warp / Occupancy, HBM / SRAM, Roofline, FlashAttention](../glossary.md)
