# KV Cache

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    All flags/APIs on this page are verified against vLLM 0.26.0 via Context7 (ADR-0004). Performance figures are **illustrative / order-of-magnitude references** — measure the real ones on your own AutoDL box. The KV-size arithmetic below is *exact* (it's just multiplication), not a benchmark.

---

## 1 · Intuition & why it matters

An LLM generates text one token at a time. To produce token *t*, attention needs the **Key** and **Value** vectors of *every* previous token. The naive approach recomputes all of them at every step — quadratic, wasteful work.

The **KV cache** is the fix: compute each token's K and V *once*, store them, and reuse them for all future steps. This turns per-step work from "re-attend to the whole history" into "attend using cached history + one new token."

Here's the catch, and why this is *the* lesson to internalize first: **the cache is not free — it lives in GPU memory and grows with every token and every concurrent request.** On a 24 GB card, it is usually the KV cache — *not* the model weights — that decides how many requests you can serve at once. Every serving optimization later in this path (PagedAttention, continuous batching, quantization, prefix caching) is, at heart, a way to **fit more useful KV cache into the same HBM**.

## 2 · Mental model

The one picture to hold: your 24 GB of VRAM is split between **weights** (paid once) and **KV cache** (paid per token, per concurrent sequence) — and it's the KV part that runs out first:

```text
24 GB VRAM, split two ways   (BF16 weights; illustrative)

  |<------------------------------ 24 GB total ------------------------------>|
  +----------------------+------+------+------+------+------+ ~ +------+-------+
  |     weights ~14 GB   | KV#1 | KV#2 | KV#3 | KV#4 | KV#5 | ~ | KV#N | free  |
  +----------------------+------+------+------+------+------+ ~ +------+-------+
   paid ONCE, fixed        \_____ KV cache: ~0.44 GiB per 8k-token sequence _____/
                            each grows +1 token / step;
                            N ~= 10 GB / 0.44 ~= 22 sequences  =  concurrency ceiling
```

Two consequences fall out immediately:

- **Decode is memory-bound.** Each step re-reads the *entire* KV cache from HBM but does a tiny amount of compute (one new token). The bottleneck is bandwidth, not FLOPs. → see the [Glossary](../glossary.md) entries for *Memory-bound* and *Roofline*.
- **Concurrency is a memory-budgeting problem.** Weights are a fixed cost paid once; KV cache is a per-sequence cost that scales with `batch × sequence_length`. Serving throughput is largely "how much KV cache fits" — every free byte the weights *don't* take is another slice of that KV region above.

## 3 · Principle & math

For a decoder-only Transformer, the KV cache stores one K tensor and one V tensor per layer, per KV head, per token. Its size is:

$$
\text{KV bytes} = \underbrace{2}_{K,\,V} \times L \times n_{\text{kv}} \times d_h \times b_{\text{dtype}} \times S \times B
$$

where $L$ = number of layers, $n_{\text{kv}}$ = number of **KV** heads (after grouping), $d_h$ = head dimension, $b_{\text{dtype}}$ = bytes per element (2 for BF16/FP16), $S$ = sequence length, $B$ = batch size.

Per single token ($S = B = 1$):

$$
\text{bytes/token} = 2\,L\,n_{\text{kv}}\,d_h\,b_{\text{dtype}}
$$

**Worked example — `Qwen2.5-7B-Instruct`** ($L=28$, $n_{\text{kv}}=4$, $d_h=128$, BF16 so $b=2$):

$$
2 \times 28 \times 4 \times 128 \times 2 = 57344 \text{ bytes} = 56 \text{ KiB/token}
$$

At its default 32 768-token context, a **single** sequence's KV cache is $56\,\text{KiB} \times 32768 \approx 1.75$ GiB. Run 8 such sequences concurrently and you've spent ~14 GiB on KV cache alone — on top of the weights.

**Why GQA matters.** Qwen2.5-7B uses $n_{\text{kv}}=4$ KV heads while having 28 *attention* heads. If it used plain MHA ($n_{\text{kv}}=28$), the cache would be $28/4 = 7\times$ larger — 392 KiB/token. That is [GQA](../glossary.md) buying you a 7× smaller KV cache, and it's why nearly every modern model uses it.

## 4 · Complete runnable code + line-by-line

This calculator is **offline-runnable** — pure CPU, no GPU, no network. It turns the formula above into something you can poke at.

```python title="kv_cache_size.py"
"""KV cache size calculator (pure CPU, offline-runnable)."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    num_layers: int       # transformer blocks (L)
    num_kv_heads: int     # KV heads AFTER grouping; == num_attention_heads for MHA
    head_dim: int         # dimension per head (d_h)
    dtype_bytes: int = 2  # BF16/FP16 = 2; FP8/INT8 = 1


def kv_bytes_per_token(cfg: ModelConfig) -> int:
    # leading 2 = one K tensor + one V tensor
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * cfg.dtype_bytes


def kv_bytes(cfg: ModelConfig, seq_len: int, batch: int = 1) -> int:
    return kv_bytes_per_token(cfg) * seq_len * batch


def gib(n: int) -> float:
    return n / (1024 ** 3)


if __name__ == "__main__":
    # Verified Qwen2.5-7B-Instruct config: 28 layers, 4 KV heads (GQA), head_dim 128.
    qwen = ModelConfig("Qwen2.5-7B-Instruct", num_layers=28, num_kv_heads=4, head_dim=128)

    per_tok = kv_bytes_per_token(qwen)
    print(f"{qwen.name}: {per_tok} bytes/token = {per_tok / 1024:.0f} KiB/token")
    for s in (2048, 8192, 32768):
        print(f"  seq_len={s:>6}: {gib(kv_bytes(qwen, s)):.2f} GiB (1 sequence)")

    # Same model, hypothetical plain MHA (num_kv_heads == num_attention_heads == 28):
    mha = ModelConfig("Qwen2.5-7B (hypothetical MHA)", num_layers=28, num_kv_heads=28, head_dim=128)
    ratio = kv_bytes_per_token(mha) / per_tok
    print(f"MHA would be {ratio:.0f}x larger: {kv_bytes_per_token(mha) / 1024:.0f} KiB/token")
```

**Line-by-line:**

- `ModelConfig` — the four architecture numbers that drive KV size, plus `dtype_bytes`. Note `num_kv_heads`, *not* attention heads: that distinction is the whole GQA story.
- `kv_bytes_per_token` — the formula from §3 with `S = B = 1`. The leading `2` is K **and** V; forgetting it is the most common factor-of-two bug.
- `kv_bytes` — scales per-token cost by `seq_len × batch`, exactly the $S \times B$ term.
- `gib` — bytes → GiB (`1024**3`), so the numbers are comparable to a card's advertised VRAM.
- `__main__` — plugs in the **verified** Qwen2.5-7B config, prints per-token and per-sequence sizes at three context lengths, then contrasts against a hypothetical MHA variant to expose the 7× GQA win.

Expected output (exact arithmetic, not a benchmark):

```text
Qwen2.5-7B-Instruct: 57344 bytes/token = 56 KiB/token
  seq_len=  2048: 0.11 GiB (1 sequence)
  seq_len=  8192: 0.44 GiB (1 sequence)
  seq_len= 32768: 1.75 GiB (1 sequence)
MHA would be 7x larger: 392 KiB/token
```

## 5 · Lab — see the KV cache govern capacity in vLLM

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~15 min · ~¥1 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** vLLM also runs on AMD ROCm and CPU builds; `kv_cache_dtype="fp8"` support and exact memory behavior vary by backend — check your platform's vLLM build notes.

The knobs that control the KV cache in vLLM 0.26.0 (all verified via Context7):

```python title="serve_kv.py"
from vllm import LLM, SamplingParams

# AWQ 4-bit weights (~5–6 GB) leave most of the 24 GB for KV cache.
# vLLM auto-detects AWQ from this checkpoint (quantization itself is Part 4).
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    max_model_len=8192,           # caps per-sequence KV length -> caps its KV cache
    gpu_memory_utilization=0.90,  # fraction of 24 GB vLLM may use (default is 0.92)
    kv_cache_dtype="fp8",         # ~halve KV bytes vs bf16 -> more concurrent seqs (illustrative)
    enable_prefix_caching=True,   # reuse KV of shared prefixes across requests
)

out = llm.generate(["Why is LLM decode memory-bound?"], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
```

Equivalent server CLI:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

**What to observe:** on startup vLLM prints the number of KV cache **blocks** it allocated. Lower `max_model_len` or `kv_cache_dtype fp8` → more blocks → more concurrent sequences. That block count *is* your concurrency ceiling, made concrete.

## 6 · Common pitfalls / counter-intuitive points

- **It's the KV cache, not the weights, that caps concurrency.** Weights are paid once; KV scales with `#sequences × length`. Doubling context roughly doubles KV, not compute.
- **The factor-of-2 bug.** K *and* V. Drop it and every estimate is half the truth.
- **Decode being memory-bound feels wrong** ("it's a huge model, surely it's compute!"). But each decode step re-reads the entire KV cache from HBM and does one token's worth of math — bandwidth is the wall.
- **`gpu_memory_utilization` is a double-edged knob.** Too high → OOM under real load (activations spike); too low → you leave KV capacity (throughput) on the table.
- **`kv_cache_dtype="fp8"` is not free accuracy-wise.** It halves KV bytes but can shift outputs — treat the quality delta as something to *measure*, not assume.
- **PagedAttention still has (small) waste.** vLLM allocates KV in fixed-size **blocks** (`--block-size` tokens); a sequence's final block is usually partly empty. This internal fragmentation is tiny compared to naive contiguous allocation — that's the point of paging.

## 7 · Interview links

- [KV cache & throughput ceiling](../interview/kv-cache.md) — the high-frequency question this lesson prepares you for: *why is the KV cache, not compute, usually the bottleneck?*

## 8 · Summary & further reading

**One line:** the KV cache trades recomputation for memory, and that memory — growing with tokens and concurrency — is the central constraint every serving optimization exists to relax.

Further reading:

- vLLM docs — [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/) and engine arguments (baseline v0.26.0).
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper).
- *GQA: Training Generalized Multi-Query Transformer Models* — why fewer KV heads.
- The [KV-cache math](../part2/kv-cache-math.md) lesson (Part 2) — the full sizing arithmetic (bytes/token → how many sequences fit).
- The [PagedAttention](../part5/paged-attention.md) lesson (Part 5) — how vLLM actually stores and manages this cache in fixed-size blocks.

## 9 · Self-check

??? question "A model has L=32 layers, n_kv=8 KV heads, head_dim=128, BF16. What's its per-token KV cache?"
    $2 \times 32 \times 8 \times 128 \times 2 = 131072$ bytes $= 128$ KiB/token. (Then multiply by sequence length and batch for the total.)

??? question "Why is the decode phase memory-bound rather than compute-bound?"
    Each decode step reads the *entire* KV cache from HBM but computes only one new token's worth of attention/FFN. The work-per-byte-moved is tiny, so the GPU stalls on memory bandwidth, not on arithmetic. Prefill, which processes many tokens at once, is the compute-bound counterpart.

??? question "You have 24 GB and ~14 GB of weights. Roughly how many 8192-token Qwen2.5-7B sequences fit in KV cache (BF16), ignoring activations?"
    Per sequence at 8192 tokens ≈ 0.44 GiB. Remaining ≈ 10 GiB / 0.44 ≈ **~22 sequences** (illustrative; real capacity is lower once activations and block padding are counted — and higher with `kv_cache_dtype fp8` or a quantized model that shrinks the weights).
