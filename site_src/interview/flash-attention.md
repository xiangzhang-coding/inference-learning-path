# FlashAttention & IO-aware attention

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 2 · Single-GPU Inference Performance   ·   **Tests the lesson:** [FlashAttention: the IO-aware Attention Kernel](../part2/flash-attention.md)

---

## Q: FlashAttention performs the same FLOPs as a naive attention implementation, yet it's faster and uses far less memory. Explain how. What does "online softmax" compute, why is the running max needed, and where does FlashAttention *not* help?

### Direct answer

FlashAttention is **IO-aware**: it computes the exact same output with the same FLOPs, but moves far fewer bytes through HBM. Naive attention materializes the $S\times S$ score matrix in HBM, reads it back to softmax, then reads it again to multiply by $V$ — three round-trips over an $O(S^2)$ intermediate. FlashAttention **tiles** Q, K, V into blocks and uses **online softmax** so each score tile is computed and consumed in SRAM and never written to HBM. Result: HBM traffic drops from $O(S^2)$ to $O(S\cdot d)$ and memory from $O(S^2)$ to $O(S)$. On the roofline it raises arithmetic intensity by shrinking the byte *denominator* (not the FLOP numerator), pushing attention back toward the compute roof.

**Online softmax** computes the exact softmax as a streaming reduction: maintain a running max $m$, normalizer $\ell$, and output accumulator $O$; when a new block reveals a larger max, rescale the accumulated $\ell$ and $O$ by $e^{\,m-m^{\text{new}}}$ before adding the block. The **running max is essential** for two reasons — it keeps $e^{(\cdot)}$ from overflowing (stability), and the rescaling makes the streamed result *exactly* equal to the one-shot softmax, not an approximation.

**Where it doesn't help:** single-stream **decode**. One decode step has a single query, so scores are a $1\times S$ vector — no $O(S^2)$ matrix to avoid. Decode stays memory-bound on **KV-cache reads**, which FlashAttention doesn't change.

### Deep dive

- **It's exact, not approximate.** The tiled/streaming computation is the same function as one-shot softmax (up to floating-point reordering). This is the license to skip building the $S\times S$ matrix.
- **The win scales with $S$.** The avoided term is $S^2$, so the longer the context, the bigger the memory and traffic savings — it's what makes 32k-context prefill feasible at all.
- **Fusion is the mechanism.** FlashAttention is a fused attention kernel: the $QK^\top$, softmax, and $\cdot\,V$ steps happen in one kernel with the intermediate never leaving SRAM. That's also fewer kernel launches (ties to [CUDA graphs / fusion](cuda-graphs-fusion.md)).
- **Variants.** FlashAttention-2 improves GPU work partitioning; FlashAttention-3 adds FP8/Hopper scheduling; FlashDecoding targets the decode phase by splitting the KV length across SMs for occupancy.

### Code

Online softmax equals full softmax — the proof that tiling is exact (pure CPU):

```python
import math
def online(q, K, V, block=2):
    d=len(q); m,l,acc=-math.inf,0.0,[0.0]*d
    for i in range(0,len(K),block):
        for k,v in zip(K[i:i+block],V[i:i+block]):
            s=sum(a*b for a,b in zip(q,k))/math.sqrt(d)
            mn=max(m,s); c=math.exp(m-mn) if m!=-math.inf else 0.0; p=math.exp(s-mn)
            l=l*c+p; acc=[acc[j]*c+p*v[j] for j in range(d)]; m=mn
    return [a/l for a in acc]     # == softmax(QKᵀ/√d)·V, to machine precision
```

### Interviewer follow-ups

- *"Same FLOPs — so where's the speedup coming from, precisely?"* → Fewer HBM bytes. Attention was memory/IO-bound on the $S\times S$ traffic; cutting bytes raises intensity toward the compute roof. It's a bandwidth win, not a FLOP win.
- *"Why can't you just stream the sum of exp without the max?"* → `exp(large score)` overflows to inf. The running max normalizes each exponent, and the $e^{m-m^{\text{new}}}$ rescale keeps the partial sums consistent as the max grows — that's what makes it both stable and exact.
- *"Does it speed up decode?"* → Not fundamentally: decode's scores are $1\times S$, already $O(S)$. Decode is bandwidth-bound on KV reads (intensity ≈ 7). FlashDecoding helps decode by a different mechanism (KV-length parallelism for occupancy).
- *"Why might your 'flash' attention not be faster?"* → It may have fallen back to a materializing path due to an unsupported head_dim/dtype/layout. Confirm the flash backend actually dispatched.

### Linked concepts

- Lesson: [FlashAttention: the IO-aware Attention Kernel](../part2/flash-attention.md)
- Related: [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md) (the $S\times S$ byte term & prefill-attention intensity), [CUDA graphs & kernel fusion](cuda-graphs-fusion.md)
- Glossary: [FlashAttention, HBM / SRAM, Kernel fusion, Roofline](../glossary.md)
