# Memory Access: Coalescing, Shared Memory, and Bank Conflicts

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB, Ada Lovelace, compute capability 8.9)"
    The `torch.cuda` timing API (`Event`, `elapsed_time`, `synchronize`) and `Tensor.contiguous` / `.t()` semantics are verified against PyTorch via Context7 (ADR-0004). The **32-byte sector / 128-byte line** transaction granularity and **32 shared-memory banks (4-byte stride)** are *architecture-documented* CUDA constants. All bandwidth figures are **illustrative / order-of-magnitude references** — the §4 sector/bank counts are exact arithmetic on the model (an addressing model, not a benchmark).

---

## 1 · Intuition & why it matters

The [execution model](cuda-execution-model.md) lesson said the GPU hides memory latency by keeping many warps busy. But for a **memory-bound** kernel — and LLM decode *is* memory-bound ([roofline](../part2/roofline-analysis.md): intensity ≈ 1) — the biggest lever is not how many warps you run, it's **how each warp touches memory**. Bandwidth is only "peak" if a warp's 32 lanes ask for bytes the hardware can fetch in one shot; ask badly and you move up to 32× the bytes for the same useful data.

Three levers, one theme — *stop moving bytes you don't need*:

- **Coalescing** — when a warp's 32 lanes read 32 *contiguous* addresses, the hardware folds them into a single wide memory transaction. Scatter those addresses and it issues many transactions, most of each one wasted. Same instruction, up to 32× the HBM traffic.
- **Shared memory** — a small, fast, *program-managed* [SRAM](../part0/gpu-hardware.md) scratchpad on each SM. Load a tile from HBM once, reuse it many times from shared memory: fewer HBM bytes for the same FLOPs → higher [arithmetic intensity](../interview/arithmetic-intensity.md). This is precisely the lever [FlashAttention](../part2/flash-attention.md) pulls to keep the $S\times S$ scores on-chip.
- **Bank conflicts** — shared memory is fast *only* if the 32 lanes hit 32 different banks. When several lanes collide on one bank, those accesses serialize — the shared-memory analogue of uncoalesced HBM.

Being able to say "this kernel is slow because its access is uncoalesced / it's bank-conflicting" is the core of reading vLLM's and Triton's memory-movement code. → see the [Glossary](../glossary.md) for *Coalescing / Shared memory / Bank conflict*.

## 2 · Mental model

What a *warp's* memory instruction becomes in hardware:

```text
COALESCED  — warp's 32 lanes read 32 contiguous floats (128 B, aligned)
  lane:  0  1  2  3 ... 31
  addr: [────────────────── one 128-byte line ──────────────────]   ⇒ 1 transaction, 100% useful

UNCOALESCED — lanes read with a big stride (e.g. one row apart)
  lane 0 ─► [line A]······  lane 1 ─► [line B]······  lane 2 ─► [line C]······
            (4 B used of 32) ...                                   ⇒ up to 32 transactions,
                                                                     ~1/8–1/32 useful

SHARED MEMORY — 32 banks, 4-byte words interleaved: bank = (word_index) mod 32
  conflict-free : lane k -> bank k                      ⇒ 1 shared-mem cycle
  2-way conflict: lane k -> bank (2k mod 32)            ⇒ 2 lanes per bank -> serialize x2
  broadcast     : all lanes -> same word                ⇒ free (hardware broadcasts)
```

Two shapes to hold:

- **The access pattern is a property of the *warp*, per instruction.** Coalescing isn't about caching over time; it's whether *this* warp's 32 simultaneous addresses fall in few transactions. The fix is almost always "make consecutive lanes touch consecutive addresses" — i.e. index the *fastest-varying* axis with `threadIdx.x`.
- **Shared memory trades HBM traffic for reuse.** It doesn't make a single load faster than a coalesced HBM load; it pays off when a tile is *reused* enough to amortize its one-time load — exactly why tiling + shared memory is the standard GEMM/attention pattern.

## 3 · Principle & math

### 3.1 Coalescing — transactions per warp

The memory system moves data in fixed chunks: **32-byte sectors**, grouped into 128-byte lines. A warp's load is serviced by however many distinct sectors its 32 lane-addresses touch. Define **efficiency** as the useful bytes over the moved bytes:

$$
\text{efficiency} = \frac{\text{bytes the lanes actually use}}{\text{bytes in the transactions the hardware moved}}
$$

- **Contiguous, aligned** (lane $k$ reads word $k$): 32 × 4 B = 128 B fall in one line → **1 line moved, efficiency ≈ 100%**.
- **Strided by $s$ words**: consecutive lanes land $4s$ bytes apart. For a large stride each lane lands in its *own* 32-byte sector, so the warp touches up to 32 sectors = 1024 B moved for 128 B used → **efficiency ≈ 1/8** (and only 4 B of each 32-B sector is used → down to **1/32** of a line). Effective bandwidth drops by the same factor.

The dominant real-world cause: reading a row-major matrix **down a column**. Row-major means element $(r,c)$ is at offset $r\cdot W + c$; if consecutive lanes take consecutive *rows* (fixed column), their addresses are $W$ words apart — a stride of a whole row → catastrophically uncoalesced. Walk *along a row* (consecutive columns) and it's contiguous. Same data, transposed access pattern, up to 32× the traffic.

### 3.2 Shared memory — reuse raises intensity

Consider a tile of data reused $R$ times. Without shared memory each use is an HBM read: $R$ HBM touches. With shared memory: **one** HBM read into the scratchpad, then $R$ cheap shared-memory reads. HBM bytes drop by ~$R$×, so on the [roofline](../part2/roofline-analysis.md) arithmetic intensity rises by ~$R$× — you climb toward the compute roof by shrinking the byte denominator, not by changing FLOPs. This is the same move [FlashAttention](../part2/flash-attention.md) makes (Q/K/V tiles loaded once, reused across the block) and the reason tiled GEMM beats naive GEMM.

### 3.3 Bank conflicts — the shared-memory gotcha

Shared memory is split into **32 banks**; successive 4-byte words map to successive banks, so word $w$ lives in bank $w \bmod 32$. A warp's shared-memory access is conflict-free when its 32 lanes hit 32 *distinct* banks (or all read the *same* word — the hardware **broadcasts** that for free). It conflicts when $n$ lanes target different words in the *same* bank: those $n$ accesses **serialize**, an $n$-way conflict costing ~$n$×.

The classic trigger is a power-of-two stride. Accessing a shared array with stride 2 (lane $k$ → word $2k$) maps lanes to banks $2k \bmod 32$, so lanes $k$ and $k+16$ collide → 2-way conflict. Stride 32 (a column of a 32-wide shared tile) sends *all* lanes to bank 0 → 32-way conflict, fully serialized. The textbook fix is **padding**: declare the tile one column wider (width 33 instead of 32) so a "column" walks addresses that are $33$ apart — $33 \bmod 32 = 1$, hitting all 32 banks, conflict-free. One wasted column of shared memory buys back a 32× serialization.

## 4 · Complete runnable code + line-by-line

This models both hazards with no GPU: (1) count the 32-byte **sectors** a warp's strided access touches (the coalescing tax), and (2) count the max lanes landing on one shared-memory **bank** (the conflict degree). Pure CPU, offline-runnable, deterministic.

```python title="coalescing_and_banks.py"
"""Model the two memory hazards SIMT charges for. Pure CPU, offline — this is the
addressing arithmetic the hardware does, not a timing benchmark."""

WARP, SECTOR_B, DTYPE_B, BANKS = 32, 32, 4, 32

def sectors_touched(stride_words, base_word=0):
    """One warp, lane k reads word (base + k*stride). Count distinct 32-byte sectors moved.
    Contiguous (stride 1) -> few sectors; large stride -> up to 32."""
    sectors = {((base_word + k * stride_words) * DTYPE_B) // SECTOR_B for k in range(WARP)}
    return len(sectors)

def max_lanes_per_bank(stride_words):
    """One warp, lane k accesses shared word k*stride. Return the worst bank's lane count
    (1 = conflict-free; n = n-way conflict serialized x n)."""
    counts = {}
    for k in range(WARP):
        bank = (k * stride_words) % BANKS
        counts[bank] = counts.get(bank, 0) + 1
    return max(counts.values())

if __name__ == "__main__":
    print("HBM coalescing — 32-byte sectors moved per warp (ideal = 4 for 128 B):")
    for s in (1, 2, 8, 32):
        sec = sectors_touched(s)
        print(f"  stride {s:>2} words: {sec:>2} sectors  ({4/sec*100:5.1f}% efficiency)")

    print("\nShared-memory bank conflicts — worst-case lanes on one bank (ideal = 1):")
    for s in (1, 2, 32, 33):
        n = max_lanes_per_bank(s)
        print(f"  stride {s:>2} words: {n:>2}-way  ({'conflict-free' if n == 1 else f'serialized x{n}'})")
```

**Line-by-line:**

- `sectors_touched` — turns each lane's word index into a byte address, floor-divides by the 32-byte `SECTOR_B` to get its sector id, and counts the *distinct* sectors in a Python `set`. Contiguous access (stride 1) packs 32 words into 4 sectors (128 B); a large stride scatters them into up to 32 sectors — the coalescing tax, quantified.
- `max_lanes_per_bank` — maps each lane's word to a bank with `word % 32` and returns the most crowded bank. `1` means every lane hit a different bank (conflict-free); `n` means $n$ lanes serialize on one bank.
- The `__main__` sweep shows both hazards *and* the fix: stride 32 is a 32-way bank conflict, but stride **33** (the padding trick) is back to conflict-free.

Expected output (addressing arithmetic, not a benchmark):

```text
HBM coalescing — 32-byte sectors moved per warp (ideal = 4 for 128 B):
  stride  1 words:  4 sectors  (100.0% efficiency)
  stride  2 words:  8 sectors  ( 50.0% efficiency)
  stride  8 words: 32 sectors  ( 12.5% efficiency)
  stride 32 words: 32 sectors  ( 12.5% efficiency)

Shared-memory bank conflicts — worst-case lanes on one bank (ideal = 1):
  stride  1 words:  1-way  (conflict-free)
  stride  2 words:  2-way  (serialized x2)
  stride 32 words: 32-way  (serialized x32)
  stride 33 words:  1-way  (conflict-free)
```

Both effects are pure addressing: nothing about the *values* changed, only which bytes the 32 lanes asked for. The stride-1 → stride-32 collapse (100% → 12.5% HBM efficiency; 1-way → 32-way bank serialization) and the stride-33 recovery are the whole reason kernels obsess over layout.

### Reading it in vLLM's source (v0.26.0)

vLLM bakes both levers of this lesson into its KV-cache write path. Two anchors:

- **The layout** — [`vllm/v1/attention/ops/paged_attn.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/ops/paged_attn.py), `PagedAttention.split_kv_cache`, computes `x = 16 // kv_cache.element_size()` and views the key cache as `[num_blocks, num_kv_heads, head_size // x, -1, x]`. That trailing `x` (= 8 for FP16) groups `head_size` into **16-byte chunks** — the alignment §6 warns about, chosen so each thread's load is a single aligned transaction. It's coalescing designed into the *data structure*, not just the access.
- **The kernel** — the write itself is `reshape_and_cache_kernel` in [`csrc/libtorch_stable/cache_kernels.cu`](https://github.com/vllm-project/vllm/blob/v0.26.0/csrc/libtorch_stable/cache_kernels.cu) (dispatched by the host `reshape_and_cache`, which `vllm/_custom_ops.py` exposes as `ops.reshape_and_cache`). It sets `constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;` — **8 FP16 elements = 16 bytes per vector store** — and moves the data through `vectorize_with_alignment<VEC_SIZE>(...)`. The flash variant `reshape_and_cache_flash_kernel` strides by warp lane (`vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32, ...)`): consecutive lanes write consecutive elements — the coalesced pattern of §3.1, in a production kernel.

You won't rewrite these (ADR-0002 — read + tune, don't hand-author), but you can now read `VEC_SIZE`, the `x` split, and the `lane, 32` stride and name them: *vectorized 16-byte aligned stores, laid out and indexed so the warp coalesces.*

## 5 · Lab — coalesced vs strided bandwidth, on real HBM

!!! gpu "GPU Lab"
    - **Min VRAM:** 4 GB (allocates a few hundred MB of float32 buffers; no model loaded)
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — matches the baseline
    - **Est. time / cost:** ~3 min · ~¥0.3 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** the coalescing *principle* holds on ROCm/others, but transaction granularity and shared-bank counts differ; effective-bandwidth ratios will vary.

You can't hand-place a warp from PyTorch, but you can feed a kernel a **contiguous** vs a **strided (transposed) view** of the same data and watch effective bandwidth collapse — the tensor-level shadow of uncoalesced access. Timed with verified CUDA events.

```python title="coalesced_vs_strided_bandwidth.py"
import torch
assert torch.cuda.is_available()

def bandwidth_gbps(read_tensor):
    """Copy read_tensor into a fresh contiguous output; report effective GB/s.
    A non-contiguous (transposed) source forces strided, uncoalesced reads."""
    nbytes = read_tensor.numel() * read_tensor.element_size()
    out = torch.empty(read_tensor.shape, device="cuda", dtype=read_tensor.dtype)
    for _ in range(3):                                  # warmup
        out.copy_(read_tensor)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        out.copy_(read_tensor)                          # reads + writes -> 2 * nbytes moved
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 50
    return 2 * nbytes / (ms / 1e3) / 1e9

n = 8192
x = torch.randn(n, n, device="cuda", dtype=torch.float32)   # 256 MiB, row-major
contiguous = x                                              # walk along rows -> coalesced
strided    = x.t()                                          # transposed VIEW -> reads down columns

print(f"contiguous copy : {bandwidth_gbps(contiguous):6.0f} GB/s  (coalesced reads)")
print(f"transposed copy : {bandwidth_gbps(strided):6.0f} GB/s  (strided, uncoalesced reads)")
print("(same 256 MiB of data; only the access pattern differs)")
```

**What to observe:** both copies move the identical 256 MiB, but the transposed source reads *down columns* of a row-major array — each warp's lanes land a full row apart, so the reads are uncoalesced and effective bandwidth drops to a fraction of the contiguous case (often several×; exact ratio is card- and size-dependent, illustrative). This is why PyTorch has `.contiguous()` and why passing a transposed view straight into a custom/Triton kernel can silently tank it. The same principle scaled up is what tiled + shared-memory kernels exist to fix.

## 6 · Common pitfalls / counter-intuitive points

- **Iterating the wrong axis.** The #1 real coalescing bug: indexing so consecutive threads step across the *slow* (row-stride) dimension of a row-major tensor. Make `threadIdx.x` index the **fastest-varying** axis (consecutive columns) so consecutive lanes hit consecutive addresses.
- **"Shared memory is always faster."** It's only a win if the loaded tile is *reused* enough to amortize its HBM load — and only if it's bank-conflict-free. A single-use staging copy through shared memory is pure overhead.
- **Power-of-two shared strides.** Column access of a 32-wide shared tile is a 32-way bank conflict. The `[N][33]` padding trick (declare one extra column) breaks the periodicity — cheap, and standard in transpose/GEMM kernels.
- **Misalignment breaks coalescing even at stride 1.** If the base address isn't aligned to the transaction size, a contiguous warp can straddle an extra sector/line. Alignment matters, not just contiguity.
- **Broadcast is not a conflict.** All 32 lanes reading the *same* shared word is free (hardware broadcasts). Conflicts are lanes hitting *different* words in the *same* bank — don't confuse the two.
- **Coalescing ≠ L2 caching.** Coalescing is about *one warp's* 32 addresses per instruction folding into few transactions. A strided kernel might still get L2 hits over time, but it has already paid the per-warp transaction tax — different mechanism, different fix.
- **`.contiguous()` copies.** Calling it fixes the access pattern for downstream kernels but costs a full pass over the data; don't sprinkle it blindly — know whether the reuse pays for the copy.
- **Coalescing isn't the only bandwidth lever — vector width is too.** On top of being coalesced, a warp can move more bytes per instruction with **128-bit (16-byte) vectorized loads/stores** (`float4`, `__ldg`) than with 32-bit scalar ones — fewer instructions issued for the same traffic. It's exactly what `reshape_and_cache` does (`VEC_SIZE = 8` FP16 elements = 16 bytes per store, via `vectorize_with_alignment`). Coalesced *and* wide beats coalesced *and* narrow — which is also why alignment (§6 above) matters: a 16-byte vector op needs a 16-byte-aligned address.

## 7 · Interview links

- [Memory coalescing, shared memory & bank conflicts](../interview/memory-coalescing.md) — the high-frequency question this lesson prepares you for: *what makes an access coalesced, what does uncoalesced cost, and what are shared memory and bank conflicts for?*

## 8 · Summary & further reading

**One line:** A warp's 32 lanes should touch 32 contiguous addresses so the hardware coalesces them into one wide transaction (scattered access moves up to 32× the bytes); shared memory is the program-managed SRAM scratchpad that trades one HBM load for many cheap reuses — fast only when its 32 lanes hit 32 distinct banks (else they serialize).

Further reading:

- *CUDA C++ Best Practices Guide* — "Coalesced Access to Global Memory" and "Shared Memory" (bank model, padding), first-party.
- The [FlashAttention](../part2/flash-attention.md) lesson — shared-memory tiling in action (keep the score tile on-chip, reuse it, never touch HBM).
- The [GPU Hardware Mental Model](../part0/gpu-hardware.md) lesson — the HBM/SRAM tiers and bandwidths this all rides on.
- The [Operator Roofline](../part2/roofline-analysis.md) lesson — why raising *effective* bandwidth and reuse moves a memory-bound kernel.

## 9 · Self-check

??? question "Why does reading a row-major matrix *down a column* destroy memory bandwidth, while reading *along a row* doesn't?"
    In a row-major layout, element $(r,c)$ sits at offset $r\cdot W + c$. Reading along a row means consecutive threads (lanes) read consecutive `c` → consecutive addresses → the warp's 32 accesses **coalesce** into ~one 128-byte transaction (≈100% efficient). Reading down a column means consecutive lanes read consecutive `r` at fixed `c` → addresses a full row ($W$ words) apart → each lane lands in its own 32-byte sector, so the warp issues up to 32 transactions and uses a small fraction of each. Same data, up to 32× the HBM traffic, so effective bandwidth collapses.

??? question "What is shared memory for, and when is using it *not* worth it?"
    Shared memory is a small, fast, program-managed SRAM scratchpad on each SM. Its purpose is **reuse**: load a tile from HBM once, then read it many times cheaply from shared memory, cutting HBM bytes by ~the reuse factor and raising arithmetic intensity (the FlashAttention / tiled-GEMM pattern). It's *not* worth it when the data is used only once (the staging load is pure overhead with no reuse to amortize) or when the access pattern causes heavy bank conflicts that serialize the shared reads.

??? question "You store a 32×32 tile in shared memory and each warp reads a *column* of it, and it's slow. What's happening, and what's the one-line fix?"
    Reading a column of a 32-wide shared tile means the 32 lanes access words 32 apart; since bank = word mod 32, **all 32 lanes map to the same bank** → a 32-way bank conflict that serializes the access ~32×. The fix is **padding**: declare the tile `[32][33]` (one extra column). Now a "column" walks addresses 33 apart, and $33 \bmod 32 = 1$, so the 32 lanes hit 32 distinct banks — conflict-free — at the cost of one unused column of shared memory.
