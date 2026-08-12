# Triton programming model

!!! info "Baseline: **vLLM 0.26.0** · Triton API verified via Context7 (ADR-0004)"

**Module:** Part 3 · GPU Programming (Triton)   ·   **Tests the lesson:** [Triton: Writing Your First GPU Kernels](../part3/triton-basics.md)

---

## Q: Explain Triton's programming model. What does a Triton "program" map to, how do `program_id` / offsets / masks work, why accumulate a matmul in FP32, and when would you reach for Triton over PyTorch or CUDA C++?

### Direct answer

Triton is a Python-embedded language + compiler for GPU kernels. You write from the perspective of **one program** — a block of work, analogous to a CUDA *block* — using NumPy-like ops on pointers; the compiler handles the within-block threading, warp assignment, memory coalescing, and shared-memory staging.

- **`tl.program_id(axis)`** returns this program's index in the launch grid. You launch `triton.cdiv(n, BLOCK)` programs; program `pid` owns the slice `[pid·BLOCK, (pid+1)·BLOCK)`.
- **Offsets** are *vectors*: `offs = pid*BLOCK + tl.arange(0, BLOCK)`. `tl.load(ptr + offs)` / `tl.store` move a whole block at once; the compiler coalesces the contiguous addresses.
- **Masks** guard ragged edges: `mask = offs < n` disables out-of-bounds lanes on load/store, with `other=` supplying a fill (`0` for sums, `-inf` for maxes) — one kernel handles any size, no remainder path.
- **FP32 accumulation**: a matmul sums `K` products; accumulating in FP16 lets rounding error compound across the K-loop. Use `tl.zeros(..., dtype=tl.float32)` and cast once at the end — the same reason tensor cores multiply low-precision but accumulate in FP32.

**When to use Triton:** for **fusing custom ops** that no library provides (attention variants, fused norm+activation, quantized matmuls) — you want more speed than composed PyTorch ops (which round-trip HBM between each) but not the cost of hand-writing CUDA C++. Not for re-implementing GEMM (cuBLAS wins) or for ops PyTorch already fuses well.

### Deep dive

- **`tl.constexpr` is compile-time.** `BLOCK_SIZE: tl.constexpr` is baked into the compiled kernel so it can size arrays and unroll loops; changing it recompiles. This is the knob `@triton.autotune` searches (over `BLOCK_*`, `num_warps`, `num_stages`) per problem shape.
- **You keep the CUDA mental models, lose the bookkeeping.** Occupancy, coalescing, and bank conflicts still explain *why* a Triton kernel is fast — but you express a block vector and let the compiler place warps, rather than writing per-thread code.
- **`tl.dot` targets tensor cores.** The tiled matmul loads `BLOCK_M×BLOCK_K` and `BLOCK_K×BLOCK_N` sub-tiles and `tl.dot`s them into the accumulator — the on-chip reuse (shared memory) is compiler-managed.
- **Fusion = fewer HBM round-trips.** The fused-softmax kernel loads a row once, does max/exp/sum on-chip, writes once — vs three PyTorch ops each round-tripping the intermediate through HBM. Same win FlashAttention scales up.
- **`tl.exp` is fast-approximate** (like CUDA `__expf`) — fine for softmax, worth knowing if you need bit-exact results.

### Code

The vector-add kernel — the whole model in eight lines:

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # block index in the grid
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # a VECTOR of indices
    mask = offs < n                              # guard the ragged last block
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# launch: grid = ceil(n / BLOCK) programs
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

### Interviewer follow-ups

- *"What does a Triton program correspond to in CUDA terms?"* → A block (CTA), not a thread. `tl.program_id` ≈ `blockIdx`; you operate on the block's whole data vector and never write per-thread code.
- *"Why is the `mask` needed?"* → The data size usually isn't a multiple of `BLOCK`, so the last program's offset vector overruns the array; the mask disables those lanes on load/store, avoiding OOB access without a separate remainder kernel.
- *"Your Triton matmul is slower than `torch.matmul` — is that a bug?"* → Usually not — cuBLAS is extremely tuned. Triton's payoff is fusing ops libraries don't provide, not beating GEMM. Also make sure you're timing the *second* call (the first JIT-compiles) and that the autotuner has run for the shape.
- *"How does Triton pick block size / warps?"* → It doesn't automatically — you list `triton.Config`s and `@triton.autotune` benchmarks them per `(M,N,K)` key, caching the winner. `constexpr` params are what make that search possible.
- *"Can you run Triton on CPU?"* → No production CPU backend — it needs a GPU (CUDA, or AMD ROCm/HIP) to compile and launch.

### Linked concepts

- Lesson: [Triton: Writing Your First GPU Kernels](../part3/triton-basics.md)
- Related: [CUDA execution model: warps, SIMT & occupancy](cuda-execution-model.md) (what the compiler places for you), [Memory coalescing, shared memory & bank conflicts](memory-coalescing.md) (why block loads are fast), [PagedAttention kernel](paged-attention-kernel.md) (a real kernel to read), [FlashAttention & IO-aware attention](flash-attention.md) (fusion at scale)
- Glossary: [Triton, SM / Warp / Occupancy](../glossary.md)
