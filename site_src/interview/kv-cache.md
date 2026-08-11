# KV cache & throughput ceiling

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 0 · Foundations   ·   **Tests the lesson:** [KV Cache](../part0/kv-cache.md)

---

## Q: On a single 24 GB GPU serving a 7B model, why is it usually the KV cache — not the model weights or raw compute — that limits how many requests you can serve concurrently?

### Direct answer

Weights are a **fixed, one-time** memory cost; the KV cache is a **per-sequence** cost that grows with `batch × sequence_length`. Once weights are loaded, the *remaining* VRAM is a fixed budget, and each concurrent request eats into it in proportion to its context length. So concurrency is capped by "how many sequences' worth of KV cache fit in the leftover memory" — a **memory-bandwidth-and-capacity** problem, not a compute one. Decode is also memory-bound (each step re-reads the whole cache from HBM for tiny compute), so you hit the memory wall long before the FLOPs wall.

### Deep dive

- **The arithmetic.** KV bytes $= 2 \times L \times n_{\text{kv}} \times d_h \times b_{\text{dtype}} \times S \times B$. For `Qwen2.5-7B-Instruct` (28 layers, 4 KV heads, head_dim 128, BF16) that's **56 KiB/token** — so ~0.44 GiB for one 8192-token sequence, ~1.75 GiB at full 32k context.
- **Why not weights?** ~14 GiB (BF16) or ~5–6 GiB (AWQ 4-bit) — paid *once*. It doesn't grow with load.
- **Why not compute?** Decode does one token of math per step but re-reads the entire KV cache from HBM. Arithmetic intensity is low → bandwidth-bound. Adding FLOPs headroom doesn't help; adding KV capacity (or bandwidth) does.
- **What this implies for optimization.** Everything that raises throughput is really "fit/serve more KV": quantize weights (free VRAM for KV), `kv_cache_dtype=fp8` (halve KV bytes), PagedAttention (kill fragmentation), continuous batching (keep the KV budget full), prefix caching (avoid duplicate KV).

### Code

Reason about the ceiling before you rent the GPU:

```python
# per-token KV for Qwen2.5-7B-Instruct (BF16): 2 * 28 * 4 * 128 * 2 = 57344 bytes = 56 KiB
kib_per_token = 2 * 28 * 4 * 128 * 2 / 1024                 # 56.0

free_gib = 24 - 14                                          # weights ~14 GiB -> ~10 GiB for KV
seq_len = 8192
gib_per_seq = kib_per_token * seq_len / (1024 ** 2)         # ~0.44 GiB
max_concurrent = int(free_gib / gib_per_seq)                # ~22 (illustrative, ignores activations)
print(kib_per_token, round(gib_per_seq, 2), max_concurrent)
```

### Interviewer follow-ups

- *"How would you triple that concurrency number?"* → shrink weights (AWQ/GPTQ, frees VRAM), shrink KV (`kv_cache_dtype=fp8`, GQA already helps 7×), or cap `max_model_len` to the workload's real context.
- *"Where does PagedAttention fit?"* → it removes the *fragmentation* waste of contiguous KV allocation, so more of the leftover VRAM becomes usable KV — it raises effective capacity, not the raw formula.
- *"Why is `gpu_memory_utilization` risky to crank to 0.99?"* → activations spike under concurrent load; too tight a margin OOMs in production even if it booted fine.
- *"Prefill vs decode — which is memory-bound?"* → decode (memory-bound); prefill processes many tokens at once and is typically compute-bound.

### Linked concepts

- Lesson: [KV Cache](../part0/kv-cache.md)
- Glossary: [KV cache, GQA, Memory-bound, PagedAttention](../glossary.md)
