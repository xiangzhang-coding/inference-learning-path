# FlashAttention: the IO-aware Attention Kernel

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The vLLM attention-backend flag (`--attention-backend`) and PyTorch SDPA API are verified against vLLM 0.26.0 / PyTorch via Context7 (ADR-0004). Memory/latency figures are **illustrative / order-of-magnitude references**. The online-softmax equivalence in §4 is *exact* arithmetic (it reproduces the full softmax to machine precision).

---

## 1 · Intuition & why it matters

Here is the puzzle FlashAttention solves. In the [Operator Roofline](roofline-analysis.md) lesson we found that prefill attention *should* be compute-bound — its intensity grows like $7S$. Yet the textbook attention implementation is often **memory-bound and runs out of memory on long sequences**. Why the gap? Because the textbook version builds the full $S\times S$ score matrix in HBM, reads it back to softmax it, reads it *again* to multiply by $V$ — three round-trips over an $O(S^2)$ intermediate that never needed to exist.

FlashAttention is the fix, and it is not an approximation: it computes the **exact same output** with the **exact same FLOPs**, but it is *IO-aware*. It **tiles** Q, K, V into blocks and uses **online softmax** to produce the result in a single streaming pass, keeping the working set in [SRAM](../part0/gpu-hardware.md) and **never materializing the $S\times S$ matrix in HBM**. The payoffs: memory drops from $O(S^2)$ to $O(S)$ (so 32k-context prefill fits at all), HBM traffic collapses, and attention finally reaches the compute-bound potential the roofline promised. This is why every serious engine — vLLM included — uses a FlashAttention-family kernel by default. → see the [Glossary](../glossary.md) for *FlashAttention*, *HBM / SRAM*, *Kernel fusion*.

## 2 · Mental model

Two ways to compute the same attention, contrasted by what crosses the HBM line:

```text
NAIVE attention — three HBM round-trips over an S×S matrix
  Q,K ─► [ S = QKᵀ ] ──write──► HBM   (S×S, e.g. 4096² = 16.8M floats PER HEAD)
                                 │
         [ P = softmax(S) ] ◄─read──┘ ──write──► HBM   (another S×S)
                                                  │
         [ O = P·V ] ◄────────────────────read───┘     -> O(S²) memory, 3× the traffic

FLASH attention — one streaming pass, S×S is born and dies in SRAM
  for each Q tile (rows):
      init running (m=-inf, l=0, O=0)         # in SRAM/registers
      for each K,V tile (cols):               # stream over the sequence
          S_ij = Q_i · K_jᵀ                    # small tile, stays on-chip
          update m,l,O with ONLINE SOFTMAX     # rescale, accumulate
      write O_i once                          # -> O(S) memory, read Q,K,V once
```

Two shapes to hold:

- **The $S\times S$ matrix is an implementation artifact, not a requirement.** The *output* is only $S\times d$. Attention forces every query to see every key, but it does not force you to have all $S^2$ scores resident at once — you can consume each score the instant you compute it. FlashAttention exploits exactly that.
- **Same math, different memory schedule.** FlashAttention changes *when and where* bytes live, not *what* is computed. The FLOPs are identical; the HBM bytes go from $O(S^2)$ to $O(S\cdot d)$. That is a pure move on the [roofline](roofline-analysis.md): raise intensity by cutting the denominator (bytes), don't touch the numerator (FLOPs).

## 3 · Principle & math

### 3.1 The problem with the $S\times S$ intermediate

Standard attention for a query block computes $O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V$. Materialized, that is three tensors: scores $S=QK^\top\in\mathbb{R}^{S\times S}$, probabilities $P=\operatorname{softmax}(S)$, output $O=PV$. The $S\times S$ tensors are quadratic in sequence length — at $S=4096$, one head's score matrix is $4096^2 = 16.8$M elements (~34 MB in BF16), *per head, per layer*. Writing and re-reading them dominates both memory footprint and HBM traffic. This is the $\sim S^2 b$ byte term flagged in the [Operator Roofline](roofline-analysis.md) lesson that drags prefill attention off the compute roof.

### 3.2 Online softmax — the key trick

Softmax needs two reductions over the row: the **max** (for numerical stability) and the **sum of exponentials** (the normalizer). Naively both need the whole row at once. But both are *running* reductions you can maintain incrementally as scores stream in, rescaling the partial result whenever the running max moves. Processing scores in blocks, keep a running max $m$, running normalizer $\ell$, and running output accumulator $O$:

$$
m^{\text{new}} = \max(m,\ \tilde m), \qquad
\ell^{\text{new}} = e^{\,m - m^{\text{new}}}\,\ell + \sum_{j\in\text{block}} e^{\,s_j - m^{\text{new}}}
$$

$$
O^{\text{new}} = e^{\,m - m^{\text{new}}}\,O + \sum_{j\in\text{block}} e^{\,s_j - m^{\text{new}}}\,v_j
$$

where $\tilde m$ is the block's local max. The factor $e^{\,m-m^{\text{new}}}$ **rescales** everything accumulated so far to the new reference max — that correction is what makes the streaming result *exactly* equal to the one-shot softmax, not an approximation. After the last block, $O \mathbin{/}\ell$ is the final attention output.

### 3.3 Tiling → $O(S)$ memory and higher intensity

FlashAttention wraps that recurrence in two loops: an outer loop over **Q tiles** (rows of the output) and an inner loop over **K,V tiles** (the streamed sequence). Each tile is small enough to live in SRAM, so a score tile is computed, consumed by the online-softmax update, and discarded — it never touches HBM. HBM traffic becomes: read $Q,K,V$ once each and write $O$ once, i.e. $O(S\cdot d)$ instead of $O(S^2)$.

Back on the roofline: FLOPs are unchanged ($\approx 4n_qS^2d$ for prefill), but bytes drop from $\sim S^2b$ to $\sim S\,d\,b$, so arithmetic intensity rises by a factor $\sim S/d$ — pushing attention back onto the compute roof where the roofline said it belonged. (FlashAttention-2 further improves GPU work partitioning; FlashAttention-3 adds FP8 and Hopper-specific scheduling. Same IO-aware core.)

!!! note "Where this helps — and where it doesn't"
    The $O(S^2)\to O(S)$ win is a **prefill / long-context** story: prefill has $S$ query rows, so the score matrix is genuinely $S\times S$. In **decode**, one step has a single query ($S_q=1$), so scores are just a $1\times S$ vector — already $O(S)$. Decode's memory-bound wall is re-reading the **KV cache** (intensity $\approx 7$, from the [Operator Roofline](roofline-analysis.md) lesson), which FlashAttention doesn't change; decode-specific variants (FlashDecoding) instead parallelize *across* the KV length to raise occupancy.

## 4 · Complete runnable code + line-by-line

This proves online softmax (the flash inner loop) equals the one-shot softmax — **pure CPU, offline-runnable**, no GPU, no tensors. If the streaming result matches the full one to machine precision, the whole tiling scheme is exact.

```python title="online_softmax.py"
"""Online (tiled) softmax attention == full softmax attention (pure CPU, offline)."""
import math


def full_attention(q, K, V):
    """Textbook: materialize all scores, softmax, then weight V."""
    d = len(q)
    scores = [sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d) for k in K]
    m = max(scores)                                   # the S-wide max (needs whole row)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps)
    p = [e / Z for e in exps]                         # the full S×1 probability row
    return [sum(p[i] * V[i][j] for i in range(len(V))) for j in range(d)]


def online_attention(q, K, V, block=2):
    """FlashAttention's inner loop: stream K,V in tiles, rescale as the max moves."""
    d = len(q)
    m, l, acc = -math.inf, 0.0, [0.0] * d             # running max, normalizer, output
    for start in range(0, len(K), block):             # <- tile over the sequence
        for k, v in zip(K[start:start + block], V[start:start + block]):
            s = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)
            m_new = max(m, s)
            corr = math.exp(m - m_new) if m != -math.inf else 0.0   # rescale factor
            p = math.exp(s - m_new)
            l = l * corr + p                          # rescale old normalizer, add new
            acc = [acc[j] * corr + p * v[j] for j in range(d)]      # same for output
            m = m_new
    return [a / l for a in acc]                       # normalize once at the end


if __name__ == "__main__":
    q = [0.5, -0.3, 0.8, 0.1]
    K = [[0.2, 0.1, -0.4, 0.6], [0.9, -0.2, 0.3, 0.0], [-0.5, 0.4, 0.7, -0.1],
         [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, -0.8, 0.2], [-0.3, 0.5, 0.2, 0.9]]
    V = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]

    full = full_attention(q, K, V)
    online = online_attention(q, K, V, block=2)       # tile size doesn't change the result
    diff = max(abs(a - b) for a, b in zip(full, online))
    print("full   :", [round(x, 6) for x in full])
    print("online :", [round(x, 6) for x in online])
    print(f"max abs diff = {diff:.2e}   (tiled == full softmax)")
```

**Line-by-line:**

- `full_attention` — the naive path: it needs `m = max(scores)` over the **whole** row before it can exponentiate safely, which is exactly why it wants all $S$ scores resident.
- `online_attention` — the flash recurrence. `corr = exp(m - m_new)` is the rescaling factor from §3.2; it retroactively corrects the normalizer `l` and the output accumulator `acc` when a later block reveals a bigger max. Nothing wider than one `block` of scores ever exists at once.
- The `block` parameter is the tile size — change it (1, 2, 6) and the answer is identical; tiling is a memory schedule, not an approximation.
- `__main__` — runs both on the same fixed inputs and reports the max elementwise difference.

Expected output (exact arithmetic, not a benchmark):

```text
full   : [0.363083, 0.449897, 0.392487, 0.386499]
online : [0.363083, 0.449897, 0.392487, 0.386499]
max abs diff = 1.11e-16   (tiled == full softmax)
```

The difference is machine epsilon — floating-point noise, not algorithmic error. The streaming, tiled computation is the *same function* as the one-shot softmax. That equivalence is the entire license for FlashAttention to never build the $S\times S$ matrix.

## 5 · Lab — watch the $S^2$ memory vanish

!!! gpu "GPU Lab"
    - **Min VRAM:** 8 GB (allocates attention tensors only; no model loaded)
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — matches the baseline
    - **Est. time / cost:** ~5 min · ~¥0.5 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** `scaled_dot_product_attention` runs on ROCm/CPU too, but which fused backend it dispatches to (and whether flash is available) varies — check `torch.backends.cuda` on your platform.

PyTorch's `scaled_dot_product_attention` (SDPA) auto-dispatches to a FlashAttention kernel on a suitable GPU. Compare its peak memory against a naive materialized attention and watch the $O(S^2)$ term appear only in the naive path:

```python title="flash_vs_naive_memory.py"
import torch
import torch.nn.functional as F

assert torch.cuda.is_available()
dev, dt = "cuda", torch.bfloat16
B, H, D = 1, 28, 128                                   # Qwen2.5-7B attention shape

def peak_mb(fn, S):
    q = torch.randn(B, H, S, D, device=dev, dtype=dt)
    k, v = torch.randn_like(q), torch.randn_like(q)
    torch.cuda.reset_peak_memory_stats()
    fn(q, k, v)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def naive(q, k, v):                                    # materializes S×S scores in HBM
    scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)    # [B,H,S,S]  <- the O(S²) tensor
    return torch.softmax(scores, dim=-1) @ v

def flash(q, k, v):                                    # never materializes S×S
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

for S in (1024, 2048, 4096):
    print(f"S={S:>5}: naive {peak_mb(naive, S):8.1f} MB   flash {peak_mb(flash, S):7.1f} MB")
```

**What to observe:** naive peak memory grows **quadratically** — each doubling of $S$ roughly quadruples it (the $B\times H\times S\times S$ scores) — and will OOM well before 32k context. SDPA/flash grows **linearly** and stays small, because the scores never leave SRAM. That flat curve is why long-context inference is possible at all. In vLLM this kernel is the default attention backend on CUDA (`FLASH_ATTN`); you can pin it with `vllm serve … --attention-backend FLASH_ATTN` (verified 0.26.0).

## 6 · Common pitfalls / counter-intuitive points

- **"FlashAttention approximates attention."** It does not — it's bit-for-bit the same function (up to floating-point reordering), same FLOPs. §4 shows the equivalence. It's an *IO* optimization, not a *math* one.
- **Expecting a FLOP speedup.** The FLOPs are unchanged; the win is fewer HBM bytes (and $O(S)$ memory). On the roofline it *raises intensity* by shrinking the denominator — a bandwidth/memory win, which is why it helps most where attention was memory-bound or OOM-ing.
- **Assuming it rescues decode.** Its headline $O(S^2)\to O(S)$ win is a **prefill / long-context** effect. Decode's one-query step has no $S\times S$ matrix; decode stays memory-bound on KV-cache reads (see [Operator Roofline](roofline-analysis.md)). Different problem, different fix (FlashDecoding).
- **Forgetting the running max.** Streaming the sum of exponentials *without* tracking the running max overflows `exp()` on realistic scores. The `exp(m - m_new)` rescale isn't optional bookkeeping — it's what keeps the streaming softmax both stable and exact.
- **Head-dim / layout constraints.** Flash kernels support specific `head_dim`s, dtypes, and contiguous layouts; unsupported shapes silently fall back to a slower (materializing) path. If your "flash" attention isn't faster, check whether it actually dispatched to the flash backend.
- **It's the attention op only.** FlashAttention fuses the attention computation; the projection and FFN GEMMs are separate kernels (that's the [kernel fusion / CUDA graphs](kernel-fusion-cuda-graphs.md) lesson's territory).

## 7 · Interview links

- [FlashAttention & IO-aware attention](../interview/flash-attention.md) — the high-frequency question this lesson prepares you for: *why is FlashAttention faster if it does the same FLOPs; what does online softmax compute and why the running max; and where does it move attention on the roofline?*

## 8 · Summary & further reading

**One line:** FlashAttention computes exact attention with identical FLOPs but tiles Q/K/V and uses online softmax to keep the $S\times S$ scores in SRAM — turning $O(S^2)$ HBM traffic and memory into $O(S)$, raising arithmetic intensity, and making long-context prefill both fast and feasible.

Further reading:

- Dao et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — the origin; §3 there is the online-softmax derivation.
- Dao — *FlashAttention-2* (better work partitioning) and *FlashAttention-3* (FP8, Hopper) — the same IO-aware core, tuned.
- *FlashDecoding* — the decode-time variant that splits the KV length for occupancy.
- The [Operator Roofline](roofline-analysis.md) lesson — where the $S\times S$ byte term and prefill-attention intensity came from.

## 9 · Self-check

??? question "FlashAttention does the same FLOPs as naive attention. Why is it faster?"
    Because it moves **far fewer bytes through HBM**. Naive attention writes the $S\times S$ score matrix to HBM, reads it back to softmax, reads it again for $\cdot V$ — three round-trips over an $O(S^2)$ tensor. FlashAttention tiles Q/K/V and uses online softmax so each score tile is computed and consumed in SRAM, never written to HBM: traffic drops to $O(S\cdot d)$ and memory to $O(S)$. On the roofline it raises arithmetic intensity by shrinking the byte denominator, not by cutting FLOPs.

??? question "What does 'online softmax' compute, and why is the running max essential?"
    It computes the exact softmax as a **streaming reduction**: maintain a running max $m$, normalizer $\ell$, and output accumulator $O$, and when a new block reveals a larger max, rescale the accumulated $\ell$ and $O$ by $e^{\,m-m^{\text{new}}}$ before adding the block's contribution. The running max is essential for two reasons: it keeps $e^{(\cdot)}$ from overflowing (numerical stability), and the rescaling makes the streamed result *exactly* equal to the one-shot softmax rather than an approximation.

??? question "Does FlashAttention make single-stream *decode* faster? Why or why not?"
    Not fundamentally. Decode processes one query token per step, so attention scores are a $1\times S$ vector — there is no $O(S^2)$ matrix to avoid. Decode is memory-bound because each step re-reads the whole **KV cache** from HBM (intensity ≈ 7), which FlashAttention doesn't change. Its big win is prefill / long-context, where the score matrix is genuinely $S\times S$. Decode-specific kernels (FlashDecoding) instead split the KV length across SMs to raise occupancy.
