# Operator Roofline: Arithmetic Intensity of GEMM & Attention

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Hardware figures (peak FLOP/s, HBM bandwidth) are **illustrative / order-of-magnitude references** for a consumer RTX 4090. The intensity arithmetic below (FLOPs ÷ bytes) is *exact* — it depends only on tensor shapes and dtype, not on any benchmark.

---

## 1 · Intuition & why it matters

[Part 0's roofline lesson](../part0/gpu-hardware.md) handed you the model $\min(P, I\cdot B)$ and its ridge point $I^{*}=P/B$, and [Inference Flow](../part0/inference-flow.md) *asserted* the punchline: prefill $I\approx\text{thousands}$, decode $I\approx1$. Those were **whole-model averages** — the model treated as one $2N$-FLOP blob.

Open a profiler and that blob shatters into named kernels: `q_proj`, `k_proj`, the attention op, `gate_proj`, `down_proj`. Each has its **own** arithmetic intensity and its **own** spot on the roofline. The skill this lesson builds is deriving an operator's intensity **from its shapes alone** — so you can look at a trace and say "this matmul is memory-bound at batch 1 but compute-bound at batch 256," or "decode attention re-reads the whole KV cache for one token, so no amount of FLOPs helps it." Every optimization in Part 2 and beyond — [FlashAttention](../part0/gpu-hardware.md), kernel fusion, [continuous batching](../glossary.md) — is a *move on this per-operator roofline*, and you can't evaluate a move you can't compute. → see the [Glossary](../glossary.md) for *Roofline / Arithmetic Intensity*, *Memory-bound / Compute-bound*.

## 2 · Mental model

A decoder layer is two kinds of operator, and they sit on the roofline for opposite reasons:

```text
ONE DECODER LAYER, decomposed into operators
                                             intensity set by...
  x ──► [ q_proj ]  [ k_proj ]  [ v_proj ]   <- GEMM: weight W[K×N] reused
             │           │           │           across M tokens. I rises with M.
             └─────► [  ATTENTION  ] ◄──┘      <- NOT a weight matmul: Q·Kᵀ, ·V.
                          │                       "weights" = KV cache, which GROWS
                          ▼                       with context S. I set by GQA ratio.
                     [  o_proj  ]              <- GEMM
                          │
                     [ gate ][ up ]  ─► SwiGLU ─► [ down ]   <- GEMMs (the FLOP bulk)

  GEMM  archetype:  fixed weight bytes, FLOPs ∝ M (tokens in flight)  → batch to cross the ridge
  ATTN  archetype:  bytes ∝ KV size (∝ S), FLOPs ∝ S too             → the S cancels; regime is fixed
```

Now plot where those operators actually land on the roofline — the sloped **memory roof** ($I\cdot B$), the flat **compute roof** ($P$), and the ridge $I^{*}$ that divides them:

<svg viewBox="0 0 760 430" role="img" xmlns="http://www.w3.org/2000/svg" aria-label="Roofline plot (log-log): a sloped memory roof I·B meets a flat compute roof P at the ridge point I*≈165 FLOP/byte. Decode GEMM sits at I≈1 and decode attention at I≈7 — both on the sloped memory-bound roof, far left of the ridge; prefill sits on the flat compute-bound roof past the ridge." style="max-width:100%;height:auto;font-family:inherit">
  <title>Per-operator roofline (RTX 4090, illustrative)</title>
  <g stroke="currentColor" stroke-opacity="0.12">
    <line x1="220" y1="45" x2="220" y2="360"/><line x1="370" y1="45" x2="370" y2="360"/>
    <line x1="520" y1="45" x2="520" y2="360"/><line x1="670" y1="45" x2="670" y2="360"/>
    <line x1="70" y1="260" x2="700" y2="260"/><line x1="70" y1="160" x2="700" y2="160"/>
    <line x1="70" y1="60" x2="700" y2="60"/>
  </g>
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="70" y1="360" x2="700" y2="360"/><line x1="70" y1="360" x2="70" y2="45"/>
  </g>
  <g stroke="currentColor" stroke-width="2.5" fill="none">
    <line x1="70" y1="360" x2="403" y2="138"/><line x1="403" y1="138" x2="700" y2="138"/>
  </g>
  <line x1="403" y1="138" x2="403" y2="360" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" stroke-opacity="0.6"/>
  <g fill="currentColor">
    <circle cx="72" cy="358" r="4"/><circle cx="197" cy="275" r="4"/>
    <circle cx="403" cy="138" r="4.5"/><circle cx="530" cy="138" r="4"/>
  </g>
  <g fill="currentColor" font-size="12.5">
    <text x="98" y="352">GEMM decode · I≈1</text>
    <text x="210" y="272">attn decode · I≈7</text>
    <text x="300" y="120" text-anchor="end">ridge  I*=P/B ≈ 165</text>
    <text x="524" y="128" text-anchor="end">prefill · compute-bound</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-style="italic" opacity="0.7">
    <text x="120" y="205">memory-bound</text>
    <text x="545" y="175">compute-bound</text>
  </g>
  <g fill="currentColor" font-size="11" opacity="0.75" text-anchor="middle">
    <text x="70" y="378">1</text><text x="220" y="378">10</text><text x="370" y="378">100</text>
    <text x="520" y="378">1k</text><text x="670" y="378">10k</text>
  </g>
  <g fill="currentColor" font-size="11" opacity="0.75" text-anchor="end">
    <text x="62" y="364">1</text><text x="62" y="264">10</text><text x="62" y="164">100</text><text x="62" y="64">1k</text>
  </g>
  <text x="385" y="404" fill="currentColor" font-size="12.5" text-anchor="middle">arithmetic intensity  I  (FLOP/byte, log)</text>
  <text x="24" y="200" fill="currentColor" font-size="12.5" text-anchor="middle" transform="rotate(-90 24 200)">attainable  (TFLOP/s, log)</text>
</svg>

Two shapes to hold:

- **A GEMM's intensity is a dial you turn with the batch/token count $M$.** The weight matrix is read from HBM *once per step* no matter how many tokens ride along; pack more tokens ($M$) into that one read and each weight-byte does more FLOPs. That is the entire mechanical reason batching raises throughput — and it's a number you can compute.
- **Attention's intensity is (almost) fixed by the architecture, not the batch.** In decode you drag the whole KV cache across HBM to serve **one** query token; both the FLOPs and the bytes scale with context $S$, so $S$ *cancels* and the intensity lands on a constant set by the [GQA](../glossary.md) ratio. You can't batch a single stream's attention out of the memory-bound regime — you have to shrink the bytes or keep them in SRAM.

## 3 · Principle & math

### 3.1 The GEMM roofline

A linear layer computes $Y = XW$ with activations $X\in\mathbb{R}^{M\times K}$ and weights $W\in\mathbb{R}^{K\times N}$, where $M$ is the number of tokens processed together. Counting a multiply-add as 2 FLOPs, and $b$ bytes per element:

$$
\text{FLOPs} = 2MKN, \qquad
\text{bytes} = \underbrace{(MK}_{\text{read }X} + \underbrace{KN}_{\text{read }W} + \underbrace{MN)}_{\text{write }Y}\, b
$$

$$
I_{\text{gemm}}(M) = \frac{2MKN}{(MK + KN + MN)\,b}
$$

Read the two limits:

**Single-token decode, $M=1$** (and $1 \ll K,N$, so the $KN$ weight term dominates the denominator):

$$
I_{\text{gemm}}(1) \approx \frac{2KN}{KN\cdot b} = \frac{2}{b} = 1 \ \text{FLOP/byte (BF16)}
$$

You read $KN$ weights to do $2KN$ FLOPs on one token — intensity $\approx 2/b$, pinned to the **bandwidth roof** regardless of how big the matrix is. This is *why* single-stream decode is memory-bound: it isn't the model size, it's that a matmul with $M=1$ has nowhere near enough work per weight-byte.

**Batched / prefill, large $M$** (with $M \ll N$ still, so the fixed weight read $KN$ dominates):

$$
I_{\text{gemm}}(M) \approx \frac{2MKN}{KN\cdot b} = \frac{2M}{b}
$$

Intensity grows **linearly in $M$**. It crosses the ridge — becomes compute-bound — at

$$
M^{*} \approx \frac{I^{*} b}{2} = \frac{P b}{2B} \approx \frac{165 \times 2}{2} = 165 \ \text{tokens (4090, BF16)}
$$

so you need on the order of ~165 tokens in flight before a projection GEMM saturates the math units. (Activation traffic nudges the exact crossover a little higher; §4 computes it.) *This is the quantitative statement of "batching raises throughput" that [continuous batching](../glossary.md) cashes in.*

### 3.2 The attention roofline

Attention is **not** a fixed-weight GEMM: its operands are $Q$, $K$, $V$, and in decode the $K,V$ are the **KV cache** — whose size grows with context $S$. Per layer, one decode step has one query token attend to $S$ cached keys across $n_q$ query heads (each head_dim $d$), reading $K$ and $V$ for $n_{\text{kv}}$ KV heads:

$$
\text{FLOPs}_{\text{attn}} \approx \underbrace{2 n_q S d}_{QK^\top} + \underbrace{2 n_q S d}_{\text{scores}\cdot V} = 4 n_q S d, \qquad
\text{bytes}_{\text{attn}} \approx \underbrace{2 n_{\text{kv}} S d\, b}_{\text{read }K,V}
$$

$$
I_{\text{attn}}^{\text{decode}} = \frac{4 n_q S d}{2 n_{\text{kv}} S d\, b} = \frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}
$$

**The context length $S$ cancels.** Decode attention intensity is a *constant* set by the [GQA](../glossary.md) ratio $n_q/n_{\text{kv}}$. For `Qwen2.5-7B` ($n_q=28$, $n_{\text{kv}}=4$, BF16):

$$
I_{\text{attn}}^{\text{decode}} = \frac{2}{2}\cdot\frac{28}{4} = 7 \ \text{FLOP/byte}
$$

Two things fall out. First, GQA *raises* attention's intensity by exactly $n_q/n_{\text{kv}} = 7\times$ (each K/V byte is reused across 7 query heads) versus MHA's $I=2/b=1$ — but $7 \ll I^{*}\approx165$, so decode attention is **still firmly memory-bound**. (GQA's headline win is the $7\times$ *smaller* KV in bytes, from [Part 0](../part0/kv-cache.md); the intensity bump is a bonus.) Second, growing the context does **not** rescue it — a longer sequence reads proportionally more KV for proportionally more FLOPs.

**Prefill** processes $S$ query tokens against $S$ keys, reading each KV once and reusing it across all $S$ queries — the attention analog of large $M$:

$$
I_{\text{attn}}^{\text{prefill}} \approx \frac{4 n_q S^2 d}{2 n_{\text{kv}} S d\, b} = \frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}\cdot S = 7S \ \text{(Qwen, BF16)}
$$

Intensity grows **linearly in $S$**, crossing the ridge at $S \approx I^{*}b/(2\,n_q/n_{\text{kv}}) \approx 165/7 \approx 24$ tokens — so even a short prompt makes prefill attention compute-bound. (Caveat: this counts only the KV read. *Naive* attention materializes the $S\times S$ score matrix to HBM, adding $\sim S^2 b$ bytes and dragging intensity back down — which is exactly the trap [FlashAttention](../part0/gpu-hardware.md) avoids by keeping scores in SRAM. That's the next lesson's job.)

### 3.3 Reconciling with Part 0's "decode $I\approx1$"

Operator view: decode's projection/FFN GEMMs sit at $I\approx1$, attention at $I\approx7$. The whole-model **byte-weighted average** is still $\approx1$ because the ~14 GiB of weights dwarf the sub-GiB of KV read at moderate context — so the FFN's $I\approx1$ dominates the mean. The operator view doesn't contradict Part 0; it *refines* it, and it tells you which knob moves which kernel.

## 4 · Complete runnable code + line-by-line

This calculator turns shapes into intensities and regimes — **pure CPU, offline-runnable**, no GPU. It's the analysis you'd do *before* renting a card to decide what's worth optimizing.

```python title="operator_intensity.py"
"""Per-operator arithmetic intensity for GEMM & attention (pure CPU, offline)."""
RIDGE = 165.0   # 4090 ridge point I* = P/B ~= 165 FLOP/byte (illustrative)


def gemm_intensity(M: int, K: int, N: int, b: int = 2) -> float:
    flops = 2 * M * K * N                          # 2*M*K*N multiply-adds
    bytes_ = (M * K + K * N + M * N) * b           # read X, read W, write Y
    return flops / bytes_


def attn_intensity(n_q: int, n_kv: int, phase: str, S: int, b: int = 2) -> float:
    if phase == "decode":                          # 1 query token vs S cached KV
        flops = 4 * n_q * S                         # QK^T + scores*V  (per head_dim factor cancels)
        bytes_ = 2 * n_kv * S * b                   # read K and V
    else:                                          # prefill: S queries vs S keys
        flops = 4 * n_q * S * S
        bytes_ = 2 * n_kv * S * b                   # KV read once, reused across S queries
    return flops / bytes_                          # head_dim d cancels top and bottom


def regime(I: float) -> str:
    return "compute-bound" if I >= RIDGE else "memory-bound"


if __name__ == "__main__":
    # Verified Qwen2.5-7B-Instruct shapes: hidden 3584, heads 28, kv_heads 4, head_dim 128.
    H, N_Q, N_KV, D = 3584, 28, 4, 128
    OPS = {                                         # (K, N) of each linear layer
        "q_proj/o_proj": (H, N_Q * D),              # 3584 x 3584
        "k_proj/v_proj": (H, N_KV * D),             # 3584 x 512  (GQA: skinny)
        "gate/up_proj":  (H, 18944),                # 3584 x 18944
        "down_proj":     (18944, H),                # 18944 x 3584
    }

    print(f"ridge point I* = {RIDGE:.0f} FLOP/byte\n")
    for M in (1, 256):
        print(f"-- GEMMs at M={M} tokens in flight --")
        for name, (K, N) in OPS.items():
            I = gemm_intensity(M, K, N)
            print(f"  {name:<14} I = {I:8.1f} FLOP/byte  ({regime(I)})")
        print()

    print("-- attention (Qwen GQA 28/4) --")
    for S in (128, 512, 2048):
        Id = attn_intensity(N_Q, N_KV, "decode", S)
        Ip = attn_intensity(N_Q, N_KV, "prefill", S)
        print(f"  S={S:>5}: decode I={Id:5.1f} ({regime(Id)})  |  "
              f"prefill I={Ip:8.1f} ({regime(Ip)})")
```

**Line-by-line:**

- `gemm_intensity` — the §3.1 formula verbatim. The denominator's three terms are the read of $X$, the read of $W$, and the write of $Y$; at $M=1$ the $KN$ weight read swamps the other two, which is why the result pins to $\approx 1$.
- `attn_intensity` — decode reads the whole KV for **one** query token; prefill reuses that KV across **$S$** queries (the $S^2$ in the FLOPs). The `head_dim` $d$ appears in both numerator and denominator and cancels — intensity depends on the **GQA ratio**, not $d$.
- `regime` — a single comparison against the 4090 ridge (~165). Everything left of it is bandwidth-bound.
- `__main__` — plugs in the **verified** Qwen2.5-7B shapes and sweeps $M\in\{1,256\}$ for the GEMMs and $S\in\{128,512,2048\}$ for attention.

Expected output (exact arithmetic, not a benchmark):

```text
ridge point I* = 165 FLOP/byte

-- GEMMs at M=1 tokens in flight --
  q_proj/o_proj  I =      1.0 FLOP/byte  (memory-bound)
  k_proj/v_proj  I =      1.0 FLOP/byte  (memory-bound)
  gate/up_proj   I =      1.0 FLOP/byte  (memory-bound)
  down_proj      I =      1.0 FLOP/byte  (memory-bound)

-- GEMMs at M=256 tokens in flight --
  q_proj/o_proj  I =    224.0 FLOP/byte  (compute-bound)
  k_proj/v_proj  I =    162.9 FLOP/byte  (memory-bound)
  gate/up_proj   I =    236.0 FLOP/byte  (compute-bound)
  down_proj      I =    236.0 FLOP/byte  (compute-bound)

-- attention (Qwen GQA 28/4) --
  S=  128: decode I=  7.0 (memory-bound)  |  prefill I=   896.0 (compute-bound)
  S=  512: decode I=  7.0 (memory-bound)  |  prefill I=  3584.0 (compute-bound)
  S= 2048: decode I=  7.0 (memory-bound)  |  prefill I= 14336.0 (compute-bound)
```

Three lessons in one table: (1) at $M=1$ **every** GEMM is memory-bound at $I\approx1$; (2) by $M=256$ the fat FFN GEMMs are compute-bound but the *skinny* GQA projections (`k_proj/v_proj`, $N=512$) still lag at 162.9 — narrow matrices need more batch to cross; (3) decode attention is stuck at **7** no matter the context, while prefill attention rockets past the ridge — the same operator, opposite regimes.

### Reading it in vLLM's source (v0.26.0)

The two archetypes are two different code paths in vLLM, and finding them is the point of this read-along:

- **The GEMMs** are the linear layers in [`vllm/model_executor/layers/linear.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/linear.py) — `QKVParallelLinear` (the fused Q/K/V projection), `MergedColumnParallelLinear` (gate + up), and `RowParallelLinear` (`o_proj`, `down_proj`). Their intensity is the `gemm_intensity(M, K, N)` you just wrote; the batch dimension `M` is what a scheduler packs to push them right across the ridge.
- **The attention operator** is *not* one of those — it is dispatched through `AttentionBackendEnum` in [`vllm/v1/attention/backends/registry.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/backends/registry.py), whose `FLASH_ATTN` entry resolves to `vllm.v1.attention.backends.flash_attn.FlashAttentionImpl`. That's the op whose decode intensity is pinned at $2n_q/(n_{\text{kv}}b)$, context-independent.

Reading just those two files is enough to see why the engine treats "a batch of projection GEMMs" and "the attention kernel" as fundamentally different beasts on the roofline — the next lesson opens the attention one.

## 5 · Lab — see the GEMM roofline on your own card

!!! gpu "GPU Lab"
    - **Min VRAM:** any CUDA GPU (allocates a few 3584×3584 matrices; no model loaded)
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — matches the baseline
    - **Est. time / cost:** ~5 min · ~¥0.5 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** `torch.cuda.Event` timing is CUDA/ROCm; on AMD the same calls work with different achieved numbers, and CPU/TPU/Neuron have their own timers — check your platform's docs.

The $M$-lever is not theory — you can watch a single matmul climb the roofline as you raise $M$. Time `q_proj`-shaped GEMMs and turn wall-clock into achieved TFLOP/s:

```python title="gemm_roofline.py"
import torch

assert torch.cuda.is_available()
K = N = 3584                                  # Qwen q_proj shape
W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

def achieved_tflops(M: int, iters: int = 50) -> float:
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):                       # warmup (CUDA graphs, clocks)
        _ = X @ W
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        _ = X @ W
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1e3 / iters
    return (2 * M * K * N) / secs / 1e12       # achieved TFLOP/s

for M in (1, 16, 64, 256, 1024):
    print(f"M={M:>5}:  {achieved_tflops(M):6.1f} TFLOP/s achieved")
```

**What to observe:** at `M=1` you'll see a *tiny* fraction of the card's peak — the matmul is memory-bound, exactly the $I\approx1$ the calculator predicted. As $M$ climbs past ~150–200, achieved TFLOP/s flattens near the compute roof: the GEMM has crossed the ridge. You've drawn the left half of the roofline with one weight matrix. Stretch goal: time `x.copy_()` on a big tensor to get achieved GB/s (your real $B$), then confirm the `M=1` GEMM's TFLOP/s ≈ (that $B$) × 1 FLOP/byte.

## 6 · Common pitfalls / counter-intuitive points

- **Reading the whole-model average as an operator fact.** "Decode is $I\approx1$" is the byte-weighted mean; individual operators differ (attention sits at ~7). Optimize the operator, not the average.
- **Forgetting weights are read once per step, not once per token.** The GEMM denominator has $KN$ (the weight matrix) *not* $MKN$ — packing more tokens ($M$) is free on the weight-read side. Double-counting weights per token erases the entire batching win.
- **Assuming longer context makes decode attention worse *per byte*.** It doesn't — $S$ cancels; decode attention intensity is fixed at $2n_q/(n_{\text{kv}}b)$. Longer context costs more *total* bytes (bigger KV), not lower intensity.
- **Thinking GQA fixes the memory-bound regime.** GQA raises attention intensity $n_q/n_{\text{kv}}\times$ and shrinks KV bytes the same factor — huge for capacity — but $7 \ll 165$, so decode attention is *still* memory-bound. The regime is set by the ridge, not by GQA.
- **Skinny matrices cross the ridge late.** A GEMM with small $N$ (the GQA `k_proj`, $N=512$) needs a larger $M$ to become compute-bound than a fat FFN GEMM. "Batch 256 makes everything compute-bound" is false for the narrow projections.
- **Counting naive attention as compute-bound in prefill.** Only if scores stay in SRAM. Materializing the $S\times S$ score matrix to HBM adds $S^2 b$ bytes and can flip prefill attention back to memory-bound — the reason FlashAttention exists.

## 7 · Interview links

- [Arithmetic intensity of GEMM & attention](../interview/arithmetic-intensity.md) — the high-frequency question this lesson prepares you for: *derive the intensity of a decode matmul and of decode attention, explain why the latter is context-independent and what GQA does to it, and find the batch size at which a projection becomes compute-bound.*

## 8 · Summary & further reading

**One line:** a decoder layer is GEMMs (intensity $\propto M$, the batching dial → cross the ridge at $M^{*}\approx I^{*}b/2$) plus attention (decode intensity fixed at $2n_q/(n_{\text{kv}}b)$, context-independent and memory-bound; prefill intensity $\propto S$, compute-bound) — and computing each from shapes is how you decide what an optimization can and can't buy.

Further reading:

- Williams, Waterman, Patterson — *Roofline: An Insightful Visual Performance Model* — the origin of the per-operator roofline.
- Dao et al. — *FlashAttention* — why the $S\times S$ score matrix must stay in SRAM (the prefill-attention caveat above).
- The [GPU Hardware Mental Model](../part0/gpu-hardware.md) lesson — where $\min(P, I\cdot B)$ and the ridge point come from.
- The [Inference Flow](../part0/inference-flow.md) lesson — the whole-model average this lesson refines.

## 9 · Self-check

??? question "Derive the arithmetic intensity of a `q_proj` matmul (K=N=3584, BF16) at M=1 and at M=256. Which regime is each?"
    $I = 2MKN/((MK+KN+MN)b)$. At $M=1$: numerator $2\cdot3584^2$, denominator $\approx KN\cdot b = 3584^2\cdot2$, so $I\approx 2/b = 1$ FLOP/byte — **memory-bound**. At $M=256$: $I = 2\cdot256\cdot3584^2/((256\cdot3584 + 3584^2 + 256\cdot3584)\cdot2) = 224$ FLOP/byte — above the ~165 ridge, **compute-bound**. The batch dimension $M$ moved the same matmul across the ridge.

??? question "Why is decode attention's intensity independent of context length, and what does GQA change?"
    Both the FLOPs ($4n_qSd$) and the bytes read (the KV cache, $2n_{\text{kv}}Sd\,b$) scale with context $S$, so $S$ cancels: $I = \frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}$. GQA raises it by the group ratio $n_q/n_{\text{kv}}$ (Qwen: $7$, since each K/V is reused across 7 query heads) — but $7 \ll 165$, so decode attention stays **memory-bound**. GQA's bigger payoff is the $7\times$ smaller KV in bytes.

??? question "You profile decode and find the FFN `down_proj` at I≈1 but attention at I≈7, yet the whole model reports I≈1. No contradiction — why?"
    The whole-model intensity is a **byte-weighted average**. At moderate context the ~14 GiB of weights read every step dwarf the sub-GiB of KV read, so the FFN/projection GEMMs (which move almost all the bytes at $I\approx1$) dominate the mean. Attention's higher $7$ is real but rides on a small fraction of total bytes, so it barely moves the average. The operator view refines — doesn't overturn — the Part 0 average.
