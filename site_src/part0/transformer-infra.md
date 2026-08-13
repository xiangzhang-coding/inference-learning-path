# Transformer, the Infra View

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    All flags/APIs on this page are verified against vLLM 0.26.0 via Context7 (ADR-0004). The parameter and FLOP counts below are **exact arithmetic** derived from Qwen2.5-7B's published `config.json`; any throughput/latency implication is an **illustrative / order-of-magnitude reference**, not a benchmark.

---

## 1 · Intuition & why it matters

You already know *what* a Transformer computes. For inference infra, the useful question is different: **what does each part cost?** Read a decoder-only block not as "attention + MLP" but as a **bill of materials** with three columns:

- **Weight bytes** — VRAM paid **once** at load time. Fixed. Doesn't grow with load.
- **Prefill FLOPs** — the compute a token pays going through the weights. Sets how compute-bound prefill is.
- **KV cache bytes** — VRAM paid **per token, per concurrent sequence**. Grows with load. This is the one that decides concurrency (→ [KV Cache](kv-cache.md)).

Once you can map each structural choice — how many KV heads, how wide the FFN, tied or untied embeddings, dense or MoE — onto those three columns, model config numbers stop being trivia. They become the levers that predict VRAM, TTFT, and the throughput ceiling *before* you rent a GPU.

## 2 · Mental model

One decoder layer, each box tagged with its dominant cost:

```text
                       WEIGHTS   PREFILL FLOPs   KV CACHE
  x ─► RMSNorm         tiny      tiny            —
       │
       ├─► Q proj      medium    medium          —          }
       ├─► K proj      small     small           writes K    } attention:
       ├─► V proj      small     small           writes V    }  KV grows here!
       │   (RoPE on Q,K: no weights, cheap)                  }
       ├─► attention score·softmax··V   —   O(S²) at prefill  reads all K,V
       └─► O proj      medium    medium          —          }
       │
  x ─► RMSNorm         tiny      tiny            —
       └─► FFN (SwiGLU: gate, up, down)  BIG    BIG          —   <-- most params & FLOPs
```

Two budgets to keep separate:

- **Fixed budget (weights):** embedding + `L ×` (attention + FFN) + `lm_head`. Dominated by the **FFN**.
- **Per-token budget (KV):** only the **K and V projections** feed it, and only the **KV heads** (`n_kv`), not the query heads. This is the entire reason [GQA](../glossary.md) exists.

The counter-intuitive headline: **attention gets all the attention, but the FFN is where the parameters and FLOPs actually live.** Attention's distinctive cost is not FLOPs — it's the KV cache it forces you to store.

## 3 · Principle & math

Let $d$ = hidden size, $h$ = query heads, $n_{\text{kv}}$ = KV heads, $d_h$ = head dim (so $h\,d_h = d$), $d_{\text{ff}}$ = FFN intermediate size, $V$ = vocab, $L$ = layers.

**Attention projections** (Q, K, V, O), per layer — note K and V shrink with $n_{\text{kv}}$:

$$
P_{\text{attn}} = \underbrace{d\,(h\,d_h)}_{Q} + \underbrace{2\,d\,(n_{\text{kv}}\,d_h)}_{K,\,V} + \underbrace{(h\,d_h)\,d}_{O}
$$

**FFN** (Qwen uses SwiGLU → three matrices: gate, up, down), per layer:

$$
P_{\text{ffn}} = 3\,d\,d_{\text{ff}}
$$

**Embedding + LM head:** $V d$ each ($2Vd$ if untied, $Vd$ if tied).

**Prefill FLOPs** follow the standard rule of thumb — a matmul with $P$ weights costs $\approx 2P$ FLOPs per token (one multiply + one add per weight). So per-token forward FLOPs $\approx 2 \times (\text{non-embedding params})$. The **embedding is a lookup (gather), not a matmul → ~0 FLOPs**; the **lm_head is a real matmul → counts.** There's an *additional* attention-score term that scales as $O(S^2)$ over the sequence (the $QK^\top$ and $\cdot V$ matmuls); it's negligible at short context and grows at long context — that's a Part 1 Roofline topic (→ ticket #6).

**KV cache** reuses the KV Cache lesson's result exactly: $\kappa = 2\,L\,n_{\text{kv}}\,d_h\,b$ bytes/token. The knob here is $n_{\text{kv}}$:

- **MHA** (Multi-Head): $n_{\text{kv}} = h$ — one K/V per query head. Biggest cache.
- **MQA** (Multi-Query): $n_{\text{kv}} = 1$ — all query heads share one K/V. Smallest cache, most quality risk.
- **GQA** (Grouped-Query): $1 < n_{\text{kv}} < h$ — query heads share K/V in groups. The practical middle ground.

Qwen2.5-7B picks $h=28$, $n_{\text{kv}}=4$ → a $28/4 = 7\times$ smaller KV cache than MHA, with negligible quality loss. **GQA changes the KV column dramatically and the FLOP column barely at all** — that's why it's nearly universal.

**RoPE** injects position by rotating Q and K — **no weights, trivial FLOPs**, and its extrapolation behavior is what makes long context possible (→ Part 5B, ticket #16).

**MoE** replaces the one FFN with $E$ experts but routes each token to only $k$ of them. Parameters balloon (all $E$ experts live in VRAM) while **active FLOPs/token stay near a dense $k$-expert model**. So MoE trades *weight VRAM* for *cheap-per-token compute* — a different point on the cost map (→ Part 6, ticket #16/#17). Qwen2.5-7B is **dense**, so the counter below is dense.

## 4 · Complete runnable code + line-by-line

**Offline-runnable** — pure CPU, no GPU, no network. It turns every formula in §3 into a per-component table for the *verified* Qwen2.5-7B config.

```python title="param_flop_counter.py"
"""Per-component parameter & prefill-FLOP counter for a dense decoder-only LLM.

Pure CPU, offline-runnable. Counts the weight matrices that dominate VRAM and
FLOPs; RMSNorm params and attention biases (~0.1% for this model) are omitted
for clarity, so the total is a hair under the headline 7.62B.
"""
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    num_layers: int    # L
    hidden: int        # d
    num_heads: int     # h  (query heads)
    num_kv_heads: int  # n_kv  (<= h for GQA)
    head_dim: int      # d_h
    ffn_hidden: int    # d_ff (intermediate_size)
    vocab: int         # V
    tie_embeddings: bool = False


def attn_params(c: Config) -> int:
    q = c.hidden * c.num_heads * c.head_dim
    kv = 2 * c.hidden * c.num_kv_heads * c.head_dim   # K and V shrink with n_kv (GQA)
    o = c.num_heads * c.head_dim * c.hidden
    return q + kv + o


def ffn_params(c: Config) -> int:
    return 3 * c.hidden * c.ffn_hidden                # SwiGLU: gate, up, down


def report(c: Config) -> None:
    embed = c.vocab * c.hidden
    lm_head = 0 if c.tie_embeddings else c.vocab * c.hidden
    attn_all = attn_params(c) * c.num_layers
    ffn_all = ffn_params(c) * c.num_layers
    total = embed + attn_all + ffn_all + lm_head

    # Prefill FLOPs/token ~= 2 * (params a token flows through as a matmul).
    # The embedding is a gather (no matmul) -> ~0; the lm_head is a matmul -> counts.
    flop_bearing = attn_all + ffn_all + lm_head

    print(f"{c.name}")
    print(f"  {'component':<22}{'params':>16}{'share':>9}{'FLOP/token':>16}")
    rows = [
        ("embedding (lookup)", embed, "~0 (gather)"),
        ("all attention", attn_all, f"{2*attn_all/1e9:.2f} G"),
        ("all FFN", ffn_all, f"{2*ffn_all/1e9:.2f} G"),
        ("lm_head", lm_head, f"{2*lm_head/1e9:.2f} G"),
    ]
    for name, p, flop in rows:
        print(f"  {name:<22}{p:>16,}{p/total:>8.1%}{flop:>16}")
    print(f"  {'TOTAL':<22}{total:>16,}{1.0:>8.1%}{2*flop_bearing/1e9:>13.1f} G")


if __name__ == "__main__":
    # Verified against Qwen/Qwen2.5-7B-Instruct config.json (ADR-0004).
    qwen = Config("Qwen2.5-7B-Instruct", num_layers=28, hidden=3584, num_heads=28,
                  num_kv_heads=4, head_dim=128, ffn_hidden=18944, vocab=152064)
    report(qwen)
```

**Line-by-line:**

- `Config` — the seven numbers from `config.json` that drive every cost. `num_kv_heads` (4) is deliberately separate from `num_heads` (28): that gap *is* the GQA win.
- `attn_params` — Q and O are full-width ($d \times d$); **K and V scale with `num_kv_heads`**, so under GQA they're a fraction of Q's size. Sum of the four projections.
- `ffn_params` — three $d \times d_{\text{ff}}$ matrices for SwiGLU. With $d_{\text{ff}} = 18944 \approx 5.3d$, this dwarfs attention.
- `report` — separates the **fixed** budget (embedding, lm_head, and `L ×` each block) and prints params, share of total, and prefill FLOPs/token ($2 \times$ params, excluding the embedding gather).
- `__main__` — the **verified** dense Qwen2.5-7B config; no MoE, so experts don't enter.

Expected output (exact arithmetic, not a benchmark):

```text
Qwen2.5-7B-Instruct
  component                       params    share      FLOP/token
  embedding (lookup)         544,997,376    7.2%     ~0 (gather)
  all attention              822,083,584   10.8%          1.64 G
  all FFN                  5,703,204,864   74.9%         11.41 G
  lm_head                    544,997,376    7.2%          1.09 G
  TOTAL                    7,615,283,200  100.0%         14.1 G
```

The table makes the headline undeniable: **~75% of the parameters and ~81% of the per-token FLOPs are in the FFN.** Attention holds ~11% of params — its real cost lives in the KV *cache*, not here.

## 5 · Lab — confirm the counter against a live model

!!! gpu "GPU Lab"
    - **Min VRAM:** none for reading the config (CPU-only); 24 GB if you also load weights.
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — or run the config read on the **no-GPU** instance (free).
    - **Est. time / cost:** ~5 min · ~¥0 (config read is CPU-only) (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** reading a model config is pure Python — backend-independent.

You don't need a GPU to verify the architecture numbers — just read the config with `transformers` (the same config vLLM consumes):

```python title="inspect_config.py"
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print("layers      :", cfg.num_hidden_layers)      # 28
print("hidden      :", cfg.hidden_size)            # 3584
print("q heads     :", cfg.num_attention_heads)    # 28
print("kv heads    :", cfg.num_key_value_heads)    # 4   <-- GQA: 28/4 = 7x smaller KV
print("ffn hidden  :", cfg.intermediate_size)      # 18944
print("vocab       :", cfg.vocab_size)             # 152064
print("head_dim    :", cfg.hidden_size // cfg.num_attention_heads)   # 128
```

**What to observe:** plug these into `param_flop_counter.py` — the numbers match. Now try a thought experiment: set `num_kv_heads = 28` (hypothetical MHA) and re-run the KV formula from the [KV Cache](kv-cache.md) lesson — the KV cache jumps 7×, while this parameter/FLOP table barely moves. That contrast *is* the infra view.

## 6 · Common pitfalls / counter-intuitive points

- **FFN, not attention, dominates params and FLOPs.** Interview candidates reflexively say "attention is expensive." Attention's expensive *artifact* is the KV cache; its FLOPs are a minority of the layer.
- **GQA shrinks the KV cache, not the compute.** It changes `n_kv`, which is in the KV formula but almost absent from the FLOP total. Don't expect GQA to speed up prefill much.
- **KV size scales with `n_kv × head_dim`, not `num_heads`.** Using query-head count in the KV formula is the classic factor-of-7 error for this model.
- **MoE total params ≠ active params.** A "57B MoE" may activate only ~14B/token. VRAM tracks total (all experts resident); compute tracks active. Two different columns.
- **Tied vs untied embeddings.** Qwen2.5-7B is *untied* — embedding and lm_head are separate matrices (~0.55B each). Tying them saves ~0.55B params; assuming the wrong one skews your VRAM estimate.
- **"2 × params" is a per-token prefill estimate.** It excludes the $O(S^2)$ attention-score FLOPs, which matter only at long context (Part 1).

## 7 · Interview links

- [Attention variants: MHA / MQA / GQA](../interview/attention-variants.md) — the high-frequency question this lesson prepares you for: *how do the attention variants change the KV cache and the throughput ceiling, and what's the quality trade-off?*
- Related: [KV cache & throughput ceiling](../interview/kv-cache.md) — the memory-budget consequence of the KV column above.

## 8 · Summary & further reading

**One line:** read a Transformer block as three cost columns — fixed weight VRAM (mostly FFN), per-token prefill FLOPs (mostly FFN), and per-token KV cache (only K/V, only `n_kv` heads) — and every architecture choice becomes a predictable move on that map.

Further reading:

- *Qwen2.5 Technical Report* — the config numbers used above.
- *GQA: Training Generalized Multi-Query Transformer Models* — why fewer KV heads barely hurt quality.
- *RoFormer* (RoPE) — rotary position embedding and its extrapolation.
- The [KV Cache](kv-cache.md) lesson — the per-token column in depth.
- The [FlashAttention](../part2/flash-attention.md) lesson (Part 2) — how the attention column is actually computed IO-efficiently.
- The [long-context inference](../part6/long-context-inference.md) lesson (Part 6) — where RoPE's extrapolation and the KV column meet at scale.

## 9 · Self-check

??? question "In Qwen2.5-7B, which single component holds most of the parameters, and roughly what fraction?"
    The **FFN** (SwiGLU gate/up/down), at ~75% of all parameters (~5.7B of 7.6B). Attention projections are only ~11%. The intuition "attention is the big cost" is about the KV *cache*, not the weights.

??? question "You switch a model from GQA (n_kv=4) to MHA (n_kv=28). Which cost columns move, and by how much?"
    The **KV cache** grows 7× (28/4) — the K and V projections and the per-token KV bytes both scale with `n_kv`. The **FLOP/token** total barely moves (K,V projections are a small slice of the layer, and the FFN — the bulk — is untouched). So MHA mainly costs you *concurrency*, not compute.

??? question "Why does a Mixture-of-Experts model raise VRAM far more than it raises per-token compute?"
    All $E$ experts' weights must be resident in VRAM (fixed budget balloons), but each token is routed to only $k \ll E$ experts, so the FLOPs it actually incurs are close to a dense $k$-expert FFN. Total params and active params are different columns — MoE buys cheap-per-token compute at the price of weight VRAM.
