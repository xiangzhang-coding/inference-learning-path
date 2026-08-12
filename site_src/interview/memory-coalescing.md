# Memory coalescing, shared memory & bank conflicts

!!! info "Baseline: **vLLM 0.26.0** · `torch.cuda` timing API + tensor-contiguity semantics verified via Context7 (ADR-0004)"

**Module:** Part 3 · GPU Programming (Triton)   ·   **Tests the lesson:** [Memory Access: Coalescing, Shared Memory, and Bank Conflicts](../part3/memory-access.md)

---

## Q: For a memory-bound kernel, how a warp touches memory matters more than how many warps you run. Explain memory coalescing, what uncoalesced access costs, what shared memory and bank conflicts are — and connect it to why FlashAttention is fast.

### Direct answer

Memory moves in fixed chunks (**32-byte sectors**, 128-byte lines). A warp's load is serviced by however many distinct transactions its 32 lane-addresses touch:

- **Coalesced**: lane $k$ reads word $k$ (contiguous) → 32×4 B = 128 B fall in one line → **1 transaction, ~100% useful**.
- **Uncoalesced**: a large stride scatters lanes into their own sectors → up to **32 transactions**, most of each wasted → efficiency down to ~1/8–1/32, effective bandwidth drops by the same factor. Same instruction, up to 32× the HBM traffic. The usual cause: reading a row-major array **down a column**.

**Shared memory** is a small, fast, *program-managed* SRAM scratchpad on each SM. Its job is **reuse**: load a tile from HBM once, read it many times cheaply — cutting HBM bytes by ~the reuse factor and raising [arithmetic intensity](arithmetic-intensity.md).

**Bank conflicts** are the shared-memory gotcha: shared memory has **32 banks** (word $w$ → bank $w \bmod 32$). If the 32 lanes hit 32 distinct banks it's conflict-free; if $n$ lanes hit different words in the *same* bank, those serialize ($n$-way conflict).

**FlashAttention** is fast for exactly this reason: it tiles Q/K/V into shared memory / registers and keeps the $S\times S$ scores on-chip, so it moves $O(S)$ HBM bytes instead of $O(S^2)$ — a pure reuse/coalescing win, same FLOPs.

### Deep dive

- **Coalescing is per-warp, per-instruction** — not a cache-over-time effect. It's whether *this* warp's 32 simultaneous addresses fall in few transactions. Fix: index the **fastest-varying** axis with `threadIdx.x` so consecutive lanes hit consecutive addresses.
- **Column access is the canonical bug.** Row-major $(r,c)$ at offset $r\cdot W+c$: walking columns (fixed $c$, varying $r$) puts lanes a full row apart — up to 32× traffic. Walking rows is contiguous.
- **Shared memory isn't free.** It's a win only if reuse amortizes the one-time HBM load *and* it's conflict-free. A single-use staging copy is pure overhead.
- **The padding trick.** Column access of a 32-wide shared tile sends all lanes to bank 0 (32-way conflict). Declaring the tile `[N][33]` makes a column walk addresses 33 apart; $33 \bmod 32 = 1$ → 32 distinct banks → conflict-free, for one wasted column.
- **Broadcast ≠ conflict.** All 32 lanes reading the *same* word is free (hardware broadcasts). Conflicts are lanes hitting *different* words in the *same* bank.

### Code

Both hazards as pure addressing arithmetic (no GPU): sectors moved per warp, and worst-case lanes per shared-memory bank.

```python
WARP, SECTOR_B, DTYPE_B, BANKS = 32, 32, 4, 32
def sectors(stride):                                 # distinct 32-B sectors a warp touches
    return len({((k * stride) * DTYPE_B) // SECTOR_B for k in range(WARP)})
def max_per_bank(stride):                            # worst bank's lane count (1 = conflict-free)
    c = {}
    for k in range(WARP):
        b = (k * stride) % BANKS; c[b] = c.get(b, 0) + 1
    return max(c.values())

for s in (1, 8, 32):                                 # coalescing: 4, 32, 32 sectors
    print(f"stride {s:>2}: {sectors(s):>2} sectors, {max_per_bank(s):>2}-way bank")
# stride  1:  4 sectors,  1-way bank   (ideal)
# stride  8: 32 sectors,  8-way ... -> uncoalesced + conflicted
# stride 32: 32 sectors, 32-way bank  (worst case)
```

### Interviewer follow-ups

- *"How do you make an access coalesced?"* → Ensure consecutive threads (lanes) read consecutive addresses — index the fastest-varying (innermost, unit-stride) dimension with `threadIdx.x`. For a row-major matrix, walk rows, not columns.
- *"When is shared memory *not* worth it?"* → When data is used once (no reuse to amortize the staging load) or when the access pattern is heavily bank-conflicting (serialized reads erase the benefit).
- *"A 32-way bank conflict on a shared tile — one-line fix?"* → Pad the inner dimension by one (`[N][33]`) to break the power-of-two periodicity so a column hits 32 distinct banks.
- *"Does coalescing matter for LLM decode?"* → Yes — decode is memory-bound (re-reads the KV cache each step), so effective bandwidth = coalescing quality directly caps throughput. It's why KV-cache layout and attention kernels obsess over contiguous, coalesced reads.
- *"You passed a transposed tensor into a custom kernel and it got slow — why?"* → A transposed view is non-contiguous; the kernel now reads strided (down columns), uncoalesced. Call `.contiguous()` first (if the reuse pays for the copy) or write the kernel to walk the contiguous axis.

### Linked concepts

- Lesson: [Memory Access: Coalescing, Shared Memory, and Bank Conflicts](../part3/memory-access.md)
- Related: [CUDA execution model: warps, SIMT & occupancy](cuda-execution-model.md) (the warps doing the accessing), [FlashAttention & IO-aware attention](flash-attention.md) (shared-memory tiling in action), [GPU memory hierarchy & roofline](gpu-memory-hierarchy.md) (the HBM/SRAM tiers), [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md) (reuse raises intensity)
- Glossary: [Coalescing / Shared memory / Bank conflict](../glossary.md)
