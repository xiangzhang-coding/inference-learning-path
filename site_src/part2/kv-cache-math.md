# KV Cache Memory Math: Sizing a Deployment

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Flags/log lines (`gpu_memory_utilization`, the startup "GPU KV cache size" report) are verified against vLLM 0.26.0 via Context7 (ADR-0004). VRAM sizes for weights/activations are **illustrative / order-of-magnitude references** — read the real ones off `nvidia-smi` and vLLM's own startup log. The budget arithmetic (a subtraction and a division) is *exact*.

---

## 1 · Intuition & why it matters

"How many users can one 4090 serve at 8k context?" is a capacity question you answer with **arithmetic, before renting the GPU**. [Part 0's KV cache lesson](../part0/kv-cache.md) gave you the per-token size $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$ and one back-of-envelope (~22 sequences). This lesson turns that single number into a **deployment plan**: assemble the *full* VRAM budget — weights + KV + activations + framework overhead — solve for the number you actually care about (max concurrency, or the max context you can promise), and know which knob buys how much.

This is the math behind every serving decision and half the [system-design interview](../interview/vram-capacity-planning.md): given a model, a card, and an SLO, how many concurrent requests fit, and what do you change to hit a target? Get it right and your box holds its latency promise; get it wrong and it OOMs at peak load with the queue full. → see the [Glossary](../glossary.md) for *KV cache*, *PagedAttention*, *SLO*.

## 2 · Mental model

VRAM is a stacked bar, and the KV cache is *whatever's left after the fixed costs*:

```text
  24 GB card  (gpu_memory_utilization caps the usable height, default 0.92)
  ┌──────────────────────────────────────────────┐  ── util · 24 GB ──┐
  │  CUDA context + framework      ~1–2 GB         │  fixed overhead    │
  ├──────────────────────────────────────────────┤                    │
  │  model weights   ~14 GB BF16 / ~5–6 GB AWQ     │  fixed, paid once  │  usable
  ├──────────────────────────────────────────────┤                    │
  │  activations / workspace   scales with batch   │  ~1 GB-ish         │
  ├──────────────────────────────────────────────┤                    │
  │  KV CACHE  ◄── everything left = your concurrency budget           │
  └──────────────────────────────────────────────┘  ───────────────────┘
        concurrency  =  (KV budget)  /  (KV per sequence)
                     =  (util·VRAM − weights − activations − overhead) / (κ · S)
```

Two shapes to hold:

- **Concurrency is a leftover, not a target you set directly.** You don't "set" how many sequences fit — you set weights (quantize or not), overhead margin (`gpu_memory_utilization`), and per-sequence length (`max_model_len`), and the KV budget — hence concurrency — is what remains. Every capacity lever works by *enlarging the leftover* or *shrinking the per-sequence cost*.
- **The biggest lever is usually the weights, not the KV dtype.** On a 24 GB card, BF16 weights (~14 GiB) eat most of the budget; quantizing them to 4-bit (~5–6 GiB) frees ~8 GiB straight into the KV budget — often a bigger concurrency win than halving KV bytes. Reach for weight quantization first, KV quantization second.

## 3 · Principle & math

From [Part 0](../part0/kv-cache.md), one sequence of length $S$ costs $\kappa S$ bytes of KV, where $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$ (for `Qwen2.5-7B`: $\kappa = 2\cdot28\cdot4\cdot128\cdot2 = 57{,}344$ bytes/token = 56 KiB). We don't re-derive it — we *spend* it.

The **KV budget** is the usable VRAM minus everything fixed:

$$
\text{KV}_{\text{budget}} = \underbrace{u \cdot V}_{\text{usable}} - \underbrace{W}_{\text{weights}} - \underbrace{A}_{\text{activations}} - \underbrace{O}_{\text{overhead}}
$$

where $u$ = `gpu_memory_utilization` (default 0.92), $V$ = card VRAM, $W$ = weight bytes, $A$ = activation/workspace, $O$ = CUDA-context + framework overhead. Then the two questions you actually ask:

$$
\boxed{N_{\text{seq}} = \left\lfloor \frac{\text{KV}_{\text{budget}}}{\kappa\,S} \right\rfloor}
\qquad
\boxed{S_{\max} = \left\lfloor \frac{\text{KV}_{\text{budget}}}{\kappa\,N} \right\rfloor}
$$

Max concurrency at a fixed context $S$, and the inverse — the longest context you can promise if you must serve $N$ concurrent streams.

**Worked plan — `Qwen2.5-7B` on a 24 GB 4090, $S = 8192$, $u=0.90$, overhead+activations $\approx 1.6$ GiB:**

| weights | KV dtype | $\kappa$ / token | KV per seq | KV budget | **max concurrent** |
|---|---|---|---|---|---|
| BF16 (~14.2 GiB) | BF16 | 56 KiB | 0.44 GiB | ~5.8 GiB | **~13** |
| AWQ 4-bit (~5.5 GiB) | BF16 | 56 KiB | 0.44 GiB | ~14.5 GiB | **~33** |
| AWQ 4-bit (~5.5 GiB) | FP8 | 28 KiB | 0.22 GiB | ~14.5 GiB | **~66** |

Read the escalation: quantizing **weights** roughly $2.5\times$ the concurrency (it frees the biggest block); then quantizing the **KV** doubles it again (each sequence is half the bytes). Two knobs take you from ~13 to ~66 concurrent 8k-context streams on the same card — *the entire serving-throughput story is visible in this one table.* (All figures illustrative; the exact numbers are the engine's to report — see the Lab.)

Note this is *more conservative* than Part 0's ~22: that back-of-envelope ignored activations/overhead and used the full 24 GiB. Subtracting real overhead and applying `gpu_memory_utilization` is what a production plan does.

## 4 · Complete runnable code + line-by-line

A capacity planner — **pure CPU, offline-runnable**, no GPU. It's exactly the arithmetic vLLM does internally to decide how many KV blocks to allocate.

```python title="vram_planner.py"
"""VRAM capacity planner: max concurrency & max context (pure CPU, offline)."""
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass
class Card:
    vram_gib: float = 24.0        # RTX 4090
    util: float = 0.90            # gpu_memory_utilization (vLLM default 0.92)
    overhead_gib: float = 1.6     # CUDA context + activations/workspace (illustrative)

    def kv_budget_gib(self, weight_gib: float) -> float:
        return self.util * self.vram_gib - weight_gib - self.overhead_gib


def kv_per_seq_gib(kappa_bytes: int, seq_len: int) -> float:
    return kappa_bytes * seq_len / GIB                 # kappa * S


def max_concurrency(card: Card, weight_gib: float, kappa_bytes: int, seq_len: int) -> int:
    return int(card.kv_budget_gib(weight_gib) / kv_per_seq_gib(kappa_bytes, seq_len))


def max_context(card: Card, weight_gib: float, kappa_bytes: int, n_seq: int) -> int:
    budget_bytes = card.kv_budget_gib(weight_gib) * GIB
    return int(budget_bytes / (kappa_bytes * n_seq))   # the inverse question


if __name__ == "__main__":
    card = Card()
    KAPPA_BF16, KAPPA_FP8 = 57344, 28672               # Qwen2.5-7B KV bytes/token (Part 0)
    W_BF16 = 7.615e9 * 2 / GIB                          # ~14.2 GiB dense weights
    W_AWQ = 5.5                                         # ~5.5 GiB 4-bit (illustrative, measured)
    S = 8192

    plans = [
        ("BF16 weights, BF16 KV", W_BF16, KAPPA_BF16),
        ("AWQ  weights, BF16 KV", W_AWQ,  KAPPA_BF16),
        ("AWQ  weights, FP8  KV", W_AWQ,  KAPPA_FP8),
    ]
    print(f"S={S} tokens, util={card.util}, overhead={card.overhead_gib} GiB\n")
    for label, w, kappa in plans:
        n = max_concurrency(card, w, kappa, S)
        budget = card.kv_budget_gib(w)
        print(f"{label}: KV budget {budget:5.1f} GiB -> ~{n:>3} concurrent seqs")

    # Inverse: to serve 64 concurrent streams on the best config, how long a context?
    s_max = max_context(card, W_AWQ, KAPPA_FP8, n_seq=64)
    print(f"\nAWQ + FP8 KV, target 64 concurrent -> max context ~= {s_max} tokens")
```

**Line-by-line:**

- `Card.kv_budget_gib` — the §3 budget: usable VRAM ($u\cdot V$) minus weights minus fixed overhead. This *is* the leftover the KV cache lives in.
- `kv_per_seq_gib` — $\kappa\cdot S$ from Part 0, in GiB. `kappa_bytes` already includes the ×2 for K and V and uses `n_kv` (GQA), so don't re-apply either.
- `max_concurrency` / `max_context` — the two boxed formulas: floor-divide the budget by per-sequence cost, or by per-token cost × target concurrency. Floors because a partial sequence doesn't fit.
- `__main__` — three plans escalating weight then KV quantization, plus the inverse question (fix concurrency, solve for context). `W_BF16` is computed from the param count; `W_AWQ` is an illustrative measured size (4-bit weights carry scales/zeros, so it's not exactly params/4).

Expected output (exact arithmetic, not a benchmark):

```text
S=8192 tokens, util=0.9, overhead=1.6 GiB

BF16 weights, BF16 KV: KV budget   5.8 GiB -> ~ 13 concurrent seqs
AWQ  weights, BF16 KV: KV budget  14.5 GiB -> ~ 33 concurrent seqs
AWQ  weights, FP8  KV: KV budget  14.5 GiB -> ~ 66 concurrent seqs

AWQ + FP8 KV, target 64 concurrent -> max context ~= 8484 tokens
```

The first three lines are the §3 table as code; the last line answers the inverse question a capacity planner actually gets asked ("we need 64 concurrent streams — how much context can we promise?"). Change one field — `util`, `overhead_gib`, `S` — and re-read the plan.

### Reading it in vLLM's source (v0.26.0)

The two startup log lines the Lab reads are printed by the code that does *this lesson's arithmetic* — a short read-along closes the loop:

- [`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_utils.py) computes the available KV bytes and logs `GPU KV cache size: … tokens` and `Maximum concurrency for … tokens per request: …x` — i.e. $\text{KV}_{\text{budget}}/\kappa$ and that divided by `max_model_len`, exactly your `max_concurrency()`.
- That budget then sizes the pool: `BlockPool` in [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/block_pool.py) allocates `num_gpu_blocks` fixed-size blocks, and `KVCacheManager` in [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py) (with a per-attention-type `SingleTypeKVCacheManager`) hands blocks to sequences. The block size defaults to `DEFAULT_BLOCK_SIZE = 16` tokens (`vllm.config.cache`) — and §6's "block padding" pitfall is exactly the last, partly-filled `KVCacheBlock`.

So when the Lab's reported number sits *just below* your planner's, you're watching `kv_cache_utils.py`'s honest budget (utilization + real overhead) against your ideal one — the same subtraction, done by the engine.

## 5 · Lab — check your plan against vLLM's own report

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB (loads `Qwen2.5-7B-Instruct-AWQ`)
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~10 min · ~¥1 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** the startup KV-cache report is backend-agnostic, but achievable sizes and `kv_cache_dtype="fp8"` support vary on ROCm/CPU — check your platform's vLLM build notes.

vLLM computes this exact budget at startup and **prints it**. Serve the model, read the log, and compare its number to your planner's:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8
```

On startup vLLM logs two verified lines (v0.26.0):

```text
GPU KV cache size: 524,288 tokens
Maximum concurrency for 8,192 tokens per request: 64.00x
```

**What to observe:** "GPU KV cache size" (in *tokens*) is exactly your $\text{KV}_{\text{budget}} / \kappa$, and "Maximum concurrency for 8,192 tokens" is that divided by `max_model_len` — vLLM's own version of `max_concurrency()`. Your planner derived ~66 for this config; the engine here reports ~64, and that small gap is the overhead/activation term you estimated versus what vLLM actually reserved — the engine's number sits *just below* the ideal-budget planner, never above. Now sweep: drop `--kv-cache-dtype fp8` (KV doubles → the token count roughly halves), or lower `--max-model-len` (more concurrency at less context). Each change moves the reported number exactly as the formula predicts — the budget math made observable. (Numbers above are illustrative; yours depend on the checkpoint and driver.)

## 6 · Common pitfalls / counter-intuitive points

- **Ignoring activations and overhead.** The naive "free VRAM ÷ KV per seq" (Part 0's ~22) overstates capacity. Real deployments lose ~1–2 GiB to CUDA context + activation/workspace *before* any KV — subtract it or you OOM under load.
- **Cranking `gpu_memory_utilization` to 0.99.** Activations spike with concurrent prefills; too thin a margin boots fine then OOMs at peak. The default 0.92 leaves headroom on purpose.
- **Optimizing KV dtype before weights.** On a 24 GB card the weights are the biggest block; quantizing them frees ~8 GiB — usually a bigger concurrency win than halving KV bytes. Do weights first.
- **Re-applying the ×2 or using `n_heads`.** $\kappa$ already has the factor of 2 (K *and* V) and uses `n_kv` (post-GQA), not attention heads. Double-counting either doubles/7×'s your estimate.
- **Treating `max_model_len` as free.** It caps per-sequence KV, so raising it for long context *directly* cuts concurrency ($N \propto 1/S$). It's a trade, not a default to max out.
- **Forgetting block padding.** [PagedAttention](../glossary.md) allocates KV in fixed-size blocks; each sequence's last block is partly empty, so real concurrency is a hair below the formula. Paging trades a little padding for the elimination of external fragmentation.

## 7 · Interview links

- [VRAM budget & max concurrency](../interview/vram-capacity-planning.md) — the high-frequency question this lesson prepares you for: *given Qwen2.5-7B on a 24 GB card at 8k context, walk the full VRAM budget, estimate max concurrency, and say how you'd hit a target of ~60 concurrent streams.*

## 8 · Summary & further reading

**One line:** concurrency is a *leftover* — $N_{\text{seq}} = \lfloor (u\cdot V - W - A - O)/(\kappa S)\rfloor$ — so capacity planning is subtract-the-fixed-costs then divide, and the biggest lever is quantizing the weights (frees the largest block) before quantizing the KV.

Further reading:

- vLLM docs — *Conserving Memory* / engine args (`gpu_memory_utilization`, `max_model_len`, `kv_cache_dtype`) and the startup KV-cache report, baseline v0.26.0.
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* — why blocks, and where the leftover actually goes.
- The [KV Cache](../part0/kv-cache.md) lesson — the per-token $\kappa$ this lesson spends.
- The sibling [Operator Roofline](roofline-analysis.md) lesson — why decode (which this KV feeds) is memory-bound in the first place.

## 9 · Self-check

??? question "A 24 GB card, util 0.90, ~1.6 GiB overhead, BF16 weights ~14.2 GiB, Qwen2.5-7B (κ=56 KiB/token), 8192-token context. How many concurrent sequences fit?"
    KV budget $= 0.90\times24 - 14.2 - 1.6 = 5.8$ GiB. Per sequence $= 56\,\text{KiB}\times8192 = 0.4375$ GiB. $N = \lfloor 5.8/0.4375\rfloor = $ **~13** (illustrative; ignores block padding). Quantizing the weights to AWQ (~5.5 GiB) raises the budget to ~14.5 GiB → ~33.

??? question "You need ~60 concurrent streams at 8k context on one 4090. What do you change, in what order?"
    (1) **Quantize the weights** (AWQ/GPTQ 4-bit) — frees ~8 GiB, the biggest single gain (~13 → ~33). (2) **Quantize the KV** (`kv_cache_dtype=fp8`) — halves per-sequence bytes (~33 → ~66), clearing 60. (3) If still short, **cap `max_model_len`** to the workload's real context (concurrency $\propto 1/S$). Weights first because they're the largest fixed block.

??? question "Why is the real max concurrency lower than `(24 GB − weights) / (κ·S)`?"
    That formula forgets three things: `gpu_memory_utilization` caps usable VRAM below the full 24 GB (default 0.92), CUDA context + activations/workspace consume ~1–2 GiB before any KV, and PagedAttention's fixed blocks leave each sequence's last block partly empty. The honest budget subtracts overhead and applies the utilization factor first — which is what vLLM's startup "GPU KV cache size" report reflects.
