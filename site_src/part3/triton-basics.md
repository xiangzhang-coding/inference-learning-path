# Triton: Writing Your First GPU Kernels

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB) · Triton (bundled with recent PyTorch)"
    All Triton API used here — `@triton.jit`, `tl.program_id`, `tl.arange`, `tl.load`/`tl.store` with `mask`/`other`, `tl.constexpr`, `triton.cdiv`, the grid-lambda launch, `tl.max`/`tl.sum`/`tl.exp`, `tl.dot`, `tl.zeros`, and `@triton.autotune` / `triton.Config` — is verified against the official Triton tutorials via Context7 (ADR-0004). Any throughput figures are **illustrative / order-of-magnitude references**; correctness is checked by comparing each kernel's output against the PyTorch reference (`torch.allclose`), which is exact up to floating-point.

---

## 1 · Intuition & why it matters

The [execution model](cuda-execution-model.md) and [memory access](memory-access.md) lessons gave you the *why* of fast kernels — warps, occupancy, coalescing. Triton is where you finally *write* one. The point (per ADR-0002) isn't to become a kernel author; it's the confidence that comes from having written a few: you'll read vLLM's Triton kernels as code you could have written, not hieroglyphics.

Here's what makes Triton the right rung between "call PyTorch ops" and "hand-write CUDA C++": **you write Python that operates on whole blocks of data, and the compiler handles the within-block threading, memory coalescing, and shared-memory staging for you.** In CUDA C++ you write code from the perspective of *one thread* and manually orchestrate 32-lane warps, shared-memory tiles, and bank-conflict-free layouts. In Triton you write from the perspective of *one program* (a block of work) using NumPy-like array ops on pointers; Triton lowers that to an efficient SIMT kernel. You keep the mental models from the last two lessons — they tell you *why* your Triton kernel is fast or slow — but you're freed from the per-thread bookkeeping. That trade is why Triton is how FlashAttention, many vLLM kernels, and most modern fused ops are actually written. → see the [Glossary](../glossary.md) for *Triton, SM / Warp / Occupancy, Coalescing*.

## 2 · Mental model

Triton's unit is the **program** (one instance of your kernel, like one CUDA *block*), identified by `tl.program_id`. You launch a **grid** of programs; each handles one tile of the output.

```text
CUDA C++  : you write ONE THREAD's view      Triton   : you write ONE PROGRAM's view (a block)
  idx = blockIdx.x*blockDim.x + threadIdx.x     pid    = tl.program_id(0)
  if (idx < n) out[idx] = x[idx] + y[idx];      offs   = pid*BLOCK + tl.arange(0, BLOCK)   # a VECTOR
  // you manage 32-lane warps, shared mem,       mask   = offs < n
  // coalescing, bank conflicts by hand          x = tl.load(x_ptr+offs, mask=mask)         # whole block
                                                 tl.store(out_ptr+offs, x+y, mask=mask)
                                                 # compiler picks warps, coalesces, stages SRAM

GRID of programs over the data:
   data:  [ block 0 ][ block 1 ][ block 2 ] ... [ block G-1 ]
   pid:       0          1          2      ...      G-1        G = triton.cdiv(n, BLOCK)
```

Three shapes to hold:

- **You think in blocks (vectors), not scalars.** `tl.arange(0, BLOCK)` makes a vector of offsets; `tl.load`/`tl.store` move a whole block at once. The compiler maps that block onto warps and coalesces the accesses — the coalescing you'd hand-craft in CUDA, Triton does from the contiguous offset pattern.
- **`mask` is how you handle ragged edges.** When the data size isn't a multiple of `BLOCK`, the last program's block runs past the end; `mask = offs < n` guards the out-of-bounds lanes on load and store (with `other=` supplying a fill value for masked loads).
- **`tl.constexpr` values are compile-time.** `BLOCK_SIZE: tl.constexpr` is baked into the compiled kernel (so it can size arrays and unroll loops); changing it recompiles. This is the hook `@triton.autotune` turns to search block sizes.

## 3 · Principle & math

### 3.1 The SPMD launch

Triton is **SPMD** (Single Program, Multiple Data): the same kernel body runs across a grid of programs, each keyed by `tl.program_id(axis)`. For a 1-D problem of `n` elements with block size `B`, you launch `G = ⌈n/B⌉` programs (`triton.cdiv(n, B)`); program `pid` owns elements `[pid·B, (pid+1)·B)`. There is no explicit thread loop — the vector width `B` *is* the parallelism a program expresses, and the compiler splits it across warps.

### 3.2 Pointers, offsets, and masks

A `torch.Tensor` argument arrives as a **pointer to its first element**. You compute a *vector* of addresses with `base_ptr + tl.arange(0, B)` (plus row/column strides for 2-D), then `tl.load(ptrs, mask, other)` fetches them. The mask $\text{offs} < n$ is the boundary guard; on a masked-out lane, load returns `other` (e.g. `0` for a sum, `-inf` for a max) and store is suppressed. Correct masking is what lets one kernel handle any size without a separate remainder path.

### 3.3 Reductions and the online-softmax echo

The **fused softmax** kernel loads a whole row into a block, then reduces along it: `tl.max(row, axis=0)` for stability, `tl.exp`, `tl.sum(..., axis=0)` for the normalizer — the entire row's softmax computed on-chip in one kernel, so the $O(\text{rows}\times\text{cols})$ intermediate never round-trips through HBM. That "load once, reduce on-chip, write once" is the same IO-aware move [FlashAttention](../part2/flash-attention.md) generalizes with online softmax. (Triton's `tl.exp` is fast-but-approximate, like CUDA's `__expf` — fine for softmax, worth knowing.)

### 3.4 `tl.dot` and the matmul accumulator

A tiled **matmul** program owns a `BLOCK_M × BLOCK_N` output tile. It walks the K dimension in steps of `BLOCK_K`, loading an $M\times K$ and a $K\times N$ sub-tile and multiplying them with `tl.dot(a, b)` (which targets the [tensor cores](../part0/gpu-hardware.md)), accumulating into a `float32` register tile — `acc += tl.dot(a, b)` — before casting and storing once. Accumulating in `float32` even for FP16 inputs is the standard precision guard. `@triton.autotune` then searches `BLOCK_M/N/K`, `num_warps`, and `num_stages` to find the fastest config for each problem shape — the tuning you'd otherwise do by hand.

## 4 · Complete runnable code + line-by-line

The three-kernel progression the issue names: **vector add → fused softmax → simple matmul**, each checked against PyTorch.

!!! gpu "GPU Lab — Triton requires a GPU"
    - **Min VRAM:** 2 GB (small test tensors; no model loaded)
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — matches the baseline
    - **Est. time / cost:** ~5 min (first run includes JIT compile) · ~¥0.5 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default). **Triton does not run on CPU** — it needs a GPU to compile and launch.
    - **Non-NVIDIA:** Triton has an AMD **ROCm/HIP** backend (`triton.runtime.driver.active.get_current_target().backend == "hip"`); the same kernels compile, but autotuned configs and `num_warps` (AMD wavefront = 64) differ. No TPU/Neuron backend.

**Kernel 1 — vector add** (the "hello world": grid, offsets, mask, load/store):

```python title="triton_vector_add.py"
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                     # which block am I? (1-D grid)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # a VECTOR of indices for this block
    mask = offsets < n_elements                     # guard the ragged last block
    x = tl.load(x_ptr + offsets, mask=mask)         # coalesced load of a whole block
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)   # write the block back, masked

def add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)   # G = ceil(n / BLOCK)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)            # launch the grid
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(98_432, device="cuda"); y = torch.rand(98_432, device="cuda")  # not a multiple of 1024
    out = add(x, y)
    print("max abs err vs torch:", (out - (x + y)).abs().max().item())            # ~0.0
```

**Line-by-line (kernel 1):** `tl.program_id(0)` is this program's index in the 1-D grid. `offsets` is a *vector* — `pid*BLOCK_SIZE` shifts to this block's slice, `tl.arange(0, BLOCK_SIZE)` spreads across it. `mask = offsets < n_elements` is essential because `98_432` isn't a multiple of `1024`, so the last program's block overruns the array; the mask suppresses those lanes on both load and store. `tl.load`/`tl.store` move the whole block — the compiler coalesces the contiguous addresses and assigns warps. The launcher's `grid` lambda computes the program count from the tunable `BLOCK_SIZE`; indexing `add_kernel[grid](...)` launches it. Expected: `max abs err ~ 0.0`.

**Kernel 2 — fused softmax** (one program per row; on-chip reduction):

```python title="triton_softmax.py"
import torch, triton, triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                          # one program handles one row
    cols = tl.arange(0, BLOCK_SIZE)                 # BLOCK_SIZE >= n_cols (power of two)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))  # pad with -inf
    x = x - tl.max(x, axis=0)                       # subtract row max for stability
    num = tl.exp(x)                                 # fast approximate exp (like __expf)
    y = num / tl.sum(num, axis=0)                   # normalize by the row sum
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    block = 1
    while block < n_cols:                           # next power of two >= n_cols
        block *= 2
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block)
    return out

if __name__ == "__main__":
    x = torch.randn(1823, 781, device="cuda")
    err = (softmax(x) - torch.softmax(x, axis=1)).abs().max().item()
    print("max abs err vs torch.softmax:", err)     # ~1e-6 (fast-exp approximation)
```

**Line-by-line (kernel 2):** the grid is `(n_rows,)` — one program per row. `BLOCK_SIZE` is the next power of two ≥ `n_cols`, so a whole row fits one block; `mask` + `other=-inf` pad the tail so the max/sum ignore it. `tl.max(x, axis=0)` and `tl.sum(..., axis=0)` are **on-chip reductions** over the block — the row is loaded from HBM once, softmaxed in SRAM/registers, written once. That's the fusion win: no separate max/exp/sum kernels, no HBM round-trips for the intermediates.

**Kernel 3 — simple tiled matmul** (2-D grid, `tl.dot`, `float32` accumulator, autotuned):

```python title="triton_matmul.py"
import torch, triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],                            # re-tune when the problem shape changes
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)                        # 2-D grid: this program owns C tile [pid_m, pid_n]
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)                         # accumulate in fp32
    for k in range(0, K, BLOCK_K):                                              # walk the K dimension
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)                                                     # tensor-core matmul
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)                            # cast + write once

def matmul(a, b):
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c

if __name__ == "__main__":
    a = torch.randn(512, 768, device="cuda", dtype=torch.float16)
    b = torch.randn(768, 256, device="cuda", dtype=torch.float16)
    err = (matmul(a, b).float() - (a.float() @ b.float())).abs().max().item()
    print("max abs err vs torch matmul:", err)      # small fp16 rounding, not algorithmic
```

**Line-by-line (kernel 3):** a **2-D grid** — `pid_m, pid_n` name the output tile this program computes. `offs_m[:, None]` / `offs_k[None, :]` broadcast 1-D offset vectors into a 2-D block of pointers (strides handle any layout). The K-loop loads a `BLOCK_M×BLOCK_K` and a `BLOCK_K×BLOCK_N` sub-tile and `tl.dot`s them into a **`float32` accumulator** (precision guard even though inputs are FP16), advancing the pointers by `BLOCK_K` each step. After the loop it casts to FP16 and stores once, masked at the M/N edges. `@triton.autotune` compiles each listed `triton.Config` and picks the fastest for the given `(M, N, K)` — the block-size/`num_warps`/`num_stages` search you'd otherwise do by hand.

## 5 · Lab — verify + peek at the tuning

Run all three `__main__` blocks above on a GPU: each prints a near-zero error against its PyTorch reference — proof the kernels are correct. Then observe Triton's compile-and-cache behavior and the autotuner:

```python title="triton_lab_notes.py"
# 1) First launch of each kernel is SLOW (JIT compile); subsequent launches hit the cache.
#    Time a second call, not the first, when comparing to torch.
# 2) The matmul autotuner benchmarks each Config on the FIRST call for a new (M,N,K) key,
#    then caches the winner. Change the shape -> it re-tunes.
# 3) Inspect the environment you're compiling for:
import triton
tgt = triton.runtime.driver.active.get_current_target()
print("backend:", tgt.backend)                      # 'cuda' on NVIDIA, 'hip' on AMD ROCm
```

**What to observe:** the first call to each kernel pays a one-time JIT compile; time the *second* call for a fair comparison against PyTorch. For matmul, the autotuner runs its config sweep once per new `(M, N, K)` and caches the winner — so a benchmark loop over a fixed shape sees the tuned kernel after the first iteration. The `backend` print confirms whether you're on CUDA or ROCm/HIP (the kernels are portable; the tuned configs are not). None of these kernels will beat cuBLAS/PyTorch on raw matmul — that's not the point; the point is you can now *read and modify* kernels of this shape.

## 6 · Common pitfalls / counter-intuitive points

- **Forgetting the `mask`.** If the data size isn't a multiple of `BLOCK_SIZE`, an unmasked kernel reads/writes out of bounds — garbage or a crash. The mask (`offs < n`) is not optional; `other=` supplies the neutral fill (`0` for sums, `-inf` for maxes).
- **Confusing the program with a thread.** `tl.program_id` is the *block* index (like CUDA `blockIdx`), not a thread. You never index individual threads in Triton — you operate on the whole block vector and let the compiler place warps.
- **Timing the first launch.** The first call JIT-compiles; it's dramatically slower than steady state. Always warm up, then time — the same discipline as the [CUDA graphs](../part2/kernel-fusion-cuda-graphs.md) lesson's benchmarking note.
- **Accumulating in the input dtype.** `tl.dot` into an FP16 accumulator loses precision fast; accumulate in `float32` (`tl.zeros(..., dtype=tl.float32)`) and cast at the end. This mirrors why matmul hardware accumulates in FP32.
- **Expecting to beat cuBLAS.** A teaching matmul won't out-run vendor libraries; Triton's value is fusing custom ops (like attention) that no library provides, not re-implementing GEMM.
- **`BLOCK_SIZE` must be `constexpr` and (for the softmax) a power of two ≥ the row.** It's a compile-time shape; a non-`constexpr` block size won't compile, and a too-small softmax block silently drops columns.
- **Triton needs a GPU.** There's no production CPU backend — a machine without CUDA/ROCm can't even compile these. Do the reading on any box; run the labs on the GPU.

## 7 · Interview links

- [Triton programming model](../interview/triton-programming.md) — the high-frequency question this lesson prepares you for: *what does a Triton program map to, how do `program_id`/offsets/masks work, why accumulate matmul in FP32, and when do you reach for Triton over PyTorch or CUDA C++?*

## 8 · Summary & further reading

**One line:** Triton lets you write GPU kernels as Python operating on whole blocks — you pick the grid and block, compute vector offsets, guard edges with masks, and reduce/`tl.dot` on-chip, while the compiler handles warps, coalescing, and SRAM staging; it's the productive middle rung (per ADR-0002) that makes vLLM's kernels readable and lightly editable.

Further reading:

- The **official Triton tutorials** — *Vector Addition*, *Fused Softmax*, *Matrix Multiplication* (the three kernels here, in depth) and *Fused Attention* (a Triton FlashAttention).
- The [FlashAttention](../part2/flash-attention.md) lesson — the online-softmax idea kernel-2's reduction echoes.
- The [CUDA execution model](cuda-execution-model.md) and [memory access](memory-access.md) lessons — the *why* behind block sizes, `num_warps`, and coalesced loads.
- Next: [Reading vLLM's PagedAttention kernel](paged-attention-kernel.md) — apply this reading skill to a real production kernel.

## 9 · Self-check

??? question "In the vector-add kernel, what does `tl.program_id(0)` return, and why is `mask = offsets < n_elements` necessary?"
    `tl.program_id(0)` returns this program instance's index in the 1-D launch grid — the *block* index (analogous to CUDA's `blockIdx.x`), not a thread index. Each program owns the slice `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)`. The mask is necessary because the total size usually isn't a multiple of `BLOCK_SIZE`, so the last program's block of offsets runs past the end of the array; `mask = offsets < n_elements` disables the out-of-bounds lanes on both `tl.load` and `tl.store`, preventing invalid memory access without needing a separate remainder kernel.

??? question "The fused softmax loads a row, does `tl.max`/`tl.exp`/`tl.sum`, and stores — where's the win over calling three PyTorch ops?"
    Fusion. The row is read from HBM **once** into a block, the max-subtract, exp, and sum-normalize all happen **on-chip** (SRAM/registers), and the result is written **once**. Three separate PyTorch ops would each round-trip the intermediate array through HBM (write exp output, read it back to sum, etc.) — for a memory-bound elementwise/reduction pattern, those extra HBM passes dominate. Keeping the whole row's softmax on-chip is the same IO-aware principle FlashAttention scales up with online softmax.

??? question "Why does the matmul kernel accumulate into a `float32` tensor even when the inputs and output are FP16?"
    To preserve precision. A matmul sums `K` products; accumulating those in FP16 lets rounding error build up across the K-loop and can visibly corrupt the result. Accumulating in `float32` (`tl.zeros(..., dtype=tl.float32)`) keeps the running sum accurate, and you cast to FP16 only once at the end before storing. This mirrors how tensor-core hardware itself multiplies in low precision but accumulates in FP32.
