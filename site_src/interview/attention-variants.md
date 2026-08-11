# Attention variants: MHA / MQA / GQA

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 0 · Foundations   ·   **Tests the lesson:** [Transformer, the Infra View](../part0/transformer-infra.md)

---

## Q: What's the difference between MHA, MQA, and GQA? How does each change the KV cache and the throughput ceiling, and what's the quality trade-off?

### Direct answer

They differ only in **how many KV heads** a layer has, i.e. how many distinct Key/Value projections the query heads share:

- **MHA** (Multi-Head Attention): one K/V per query head — $n_{\text{kv}} = h$. Largest KV cache.
- **MQA** (Multi-Query Attention): a single K/V shared by all query heads — $n_{\text{kv}} = 1$. Smallest KV cache (an $h\times$ reduction), but the most aggressive and highest quality risk.
- **GQA** (Grouped-Query Attention): query heads share K/V in $g$ groups — $1 < n_{\text{kv}} = g < h$. The practical middle ground; MHA and MQA are its extremes.

The KV cache is $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$ bytes/token — **linear in $n_{\text{kv}}$**. Shrinking KV heads shrinks the cache proportionally, which lets more sequences fit in the leftover VRAM, which **raises the throughput ceiling** (decode is memory-bound, so more KV capacity ≈ more concurrency). Crucially it barely touches the **FLOP/parameter** budget — K/V projections are a small slice of a layer, and the FFN (the bulk) is untouched. Quality: MQA can measurably degrade; GQA with a modest group count (e.g. 4–8) recovers near-MHA quality, which is why nearly every modern model ships GQA.

### Deep dive

- **The Qwen2.5-7B example.** $h = 28$ query heads, $n_{\text{kv}} = 4$ KV heads → a $28/4 = 7\times$ smaller KV cache than the MHA equivalent: **56 KiB/token instead of 392 KiB/token**. On a 24 GB card that 7× is the difference between a handful and dozens of concurrent sequences.
- **Why it barely changes FLOPs.** Only the K and V projection matrices shrink (from $d\times d$ to $d\times n_{\text{kv}}d_h$). Q, O, and the entire FFN are unchanged — and the FFN alone is ~75% of params. So GQA is a **memory** optimization, not a compute one. → [Transformer, the Infra View](../part0/transformer-infra.md).
- **Why quality survives.** Query heads still each attend independently; they just look up shared K/V representations. With enough groups the model keeps most of MHA's expressive power. MQA's single group is where quality most often suffers, especially on long-context or retrieval-heavy tasks.
- **Interaction with everything downstream.** Smaller KV heads compound with `kv_cache_dtype=fp8` and PagedAttention — they all attack the same leftover-VRAM budget that caps concurrency.

### Code

The whole story is one ratio — KV heads → KV bytes:

```python
L, head_dim, b = 28, 128, 2          # Qwen2.5-7B layers, head_dim, BF16 bytes

def kib_per_token(n_kv): return 2 * L * n_kv * head_dim * b / 1024

for name, n_kv in [("MHA", 28), ("GQA-4 (Qwen)", 4), ("MQA", 1)]:
    print(f"{name:<14} n_kv={n_kv:>2} -> {kib_per_token(n_kv):6.0f} KiB/token")
# MHA            n_kv=28 ->    392 KiB/token
# GQA-4 (Qwen)   n_kv= 4 ->     56 KiB/token   (7x smaller)
# MQA            n_kv= 1 ->     14 KiB/token   (28x smaller)
```

### Interviewer follow-ups

- *"Does GQA speed up decode arithmetic?"* → not meaningfully; it shrinks the KV cache (bytes moved), which *is* the decode bottleneck, so it helps decode via **bandwidth/capacity**, not FLOPs.
- *"When would you still pick MHA?"* → tiny models or research where the KV cache isn't the constraint, or when maximum quality per parameter matters more than serving concurrency.
- *"How do you choose the group count?"* → empirically — enough groups to recover MHA-level quality on your eval, as few as possible to minimize KV. 4–8 is the common sweet spot.
- *"Where does MLA (DeepSeek) fit?"* → same goal (shrink KV) via a *low-rank latent* KV representation rather than fewer heads — a further point on the same "compress the KV column" axis.

### Linked concepts

- Lesson: [Transformer, the Infra View](../part0/transformer-infra.md)
- Related question: [KV cache & throughput ceiling](kv-cache.md) (why smaller KV → more concurrency)
- Glossary: [MHA / MQA / GQA, KV cache](../glossary.md)
