# Arithmetic intensity of GEMM & attention

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 2 · Single-GPU Inference Performance   ·   **Tests the lesson:** [Operator Roofline: Arithmetic Intensity of GEMM & Attention](../part2/roofline-analysis.md)

---

## Q: Derive the arithmetic intensity of (a) a weight matmul in single-token decode and (b) the attention operator in decode. Explain why decode attention's intensity is independent of context length and what GQA does to it. Then: at what batch size does a projection GEMM become compute-bound on a 4090?

### Direct answer

**GEMM.** For $Y=XW$ with $X\in\mathbb{R}^{M\times K}$, $W\in\mathbb{R}^{K\times N}$, $b$ bytes/element: FLOPs $=2MKN$, bytes $=(MK+KN+MN)b$, so $I=\frac{2MKN}{(MK+KN+MN)b}$. At **$M=1$** the weight read $KN$ dominates the denominator, giving $I\approx\frac{2}{b}=1$ FLOP/byte (BF16) — **memory-bound**, regardless of matrix size. That's why single-stream decode is memory-bound: one token doesn't do enough work per weight-byte.

**Attention (decode).** One query token attends to $S$ cached tokens: FLOPs $\approx 4n_qSd$ ($QK^\top$ + scores·$V$), bytes $\approx 2n_{\text{kv}}Sd\,b$ (read K,V). So $I=\frac{4n_qSd}{2n_{\text{kv}}Sd\,b}=\frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}$. **The $S$ cancels** — both FLOPs and bytes scale with context — so intensity is a constant set by the GQA ratio. For Qwen2.5-7B ($n_q/n_{\text{kv}}=28/4=7$, BF16): $I=7$. GQA *raises* attention intensity $7\times$ (each K/V reused across 7 query heads) but $7\ll165$, so decode attention stays memory-bound; GQA's bigger win is the $7\times$ smaller KV in bytes.

**Batch to cross the ridge.** In the large-$M$ limit ($M\ll N$), $I\approx\frac{2M}{b}$, so it hits the ridge $I^{*}\approx165$ at $M^{*}\approx\frac{I^{*}b}{2}\approx165$ tokens (4090, BF16) — slightly higher once activation traffic is counted.

### Deep dive

- **Why the whole-model average is still ≈1.** The byte-weighted mean is dominated by the ~14 GiB of weights read every step; the sub-GiB KV read (attention, $I\approx7$) barely moves it. Part 0's "decode $I\approx1$" is that average — the operator view refines it.
- **Weights are read once per step, not per token.** The GEMM denominator has $KN$, not $MKN$; packing $M$ tokens into one weight read is the entire mechanical basis of continuous batching. Miscount this and the batching win disappears.
- **Skinny matrices cross late.** The GQA `k_proj`/`v_proj` ($N=512$) reach only $I\approx163$ at $M=256$ — still below the ridge — while fat FFN GEMMs ($N=18944$) are well past it. $M^{*}$ is per-operator.
- **Prefill attention flips.** With $S$ queries reusing the KV, $I\approx\frac{2}{b}\cdot\frac{n_q}{n_{\text{kv}}}\cdot S=7S$, compute-bound past $S\approx24$ — *unless* naive attention materializes the $S\times S$ scores to HBM, which adds $\sim S^2b$ bytes and drags it back (the FlashAttention motivation).

### Code

Shapes → intensity, no GPU:

```python
def gemm_I(M, K, N, b=2):
    return 2*M*K*N / ((M*K + K*N + M*N) * b)

def attn_decode_I(n_q, n_kv, b=2):
    return 2 * n_q / (n_kv * b)          # S cancels

print(round(gemm_I(1,    3584, 3584), 2))   # 1.0    -> memory-bound
print(round(gemm_I(256,  3584, 3584), 1))   # 224.0  -> compute-bound (> 165 ridge)
print(round(attn_decode_I(28, 4), 1))       # 7.0    -> memory-bound (GQA 28/4)
print(round(attn_decode_I(28, 28), 1))      # 1.0    -> hypothetical MHA
```

### Interviewer follow-ups

- *"Why doesn't a bigger GPU (more TFLOPS) speed up single-stream decode?"* → Decode sits at $I\approx1\ll I^{*}$, pinned to the bandwidth roof $I\cdot B$; more FLOPs raise $P$, which doesn't bind. You need more bandwidth, fewer bytes (quantization), or higher $I$ (batching).
- *"Where does FlashAttention fit on this roofline?"* → Prefill attention is compute-bound *only if* the $S\times S$ scores stay on-chip; FlashAttention tiles + uses online softmax to keep them in SRAM, avoiding the HBM round-trip that would otherwise lower intensity.
- *"Does GQA change the decode regime?"* → No — it raises attention $I$ by $n_q/n_{\text{kv}}$ (to 7 for Qwen) and shrinks KV bytes the same factor, but 7 is still far below the ridge. Regime is set by $I$ vs $I^{*}$, not by GQA.
- *"Your batch is 256 but the KV projections are still memory-bound — why?"* → They're skinny ($N=512$), so $I(256)\approx163<165$. Narrow GEMMs need more $M$ (or get fused with neighbors) to cross.

### Linked concepts

- Lesson: [Operator Roofline: Arithmetic Intensity of GEMM & Attention](../part2/roofline-analysis.md)
- Related: [GPU memory hierarchy & roofline](gpu-memory-hierarchy.md) (the roofline & ridge point this builds on), [Prefill vs decode](prefill-vs-decode.md)
- Glossary: [Roofline / Arithmetic Intensity, Memory-bound / Compute-bound, GQA, FlashAttention](../glossary.md)
