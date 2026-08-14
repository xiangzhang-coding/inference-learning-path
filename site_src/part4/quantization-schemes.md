# Quantization Choices: Granularity, Symmetry, What to Quantize, and PTQ vs QAT

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    vLLM's quantization naming used here — the `WxAy` scheme (e.g. `W4A16`, `W8A8`), `LLM(..., quantization="fp8")` / `vllm serve --quantization fp8`, and that dynamic FP8 quantizes Linear weights per-tensor while scaling activations dynamically per-tensor — is verified against vLLM 0.26.0 via Context7 (ADR-0004). The granularity/symmetry demos in §4 are **pure-Python math**, offline; error bounds (`≤ step/2`) are exact. Any accuracy/size figures are **illustrative / order-of-magnitude references**. Concrete method families (GPTQ/AWQ/SmoothQuant/…) and the hands-on `Qwen2.5-7B` INT4 run are **#11**.

---

## 1 · Intuition & why it matters

The [previous lesson](quantization-basics.md) gave you the one map — $\hat{x}=\text{scale}\cdot(q-z)$, error $\le \text{scale}/2$ — and the one enemy: a wide range (outliers) inflates the step for everyone sharing a scale. Everything real about quantization is the set of *engineering choices* that keep that step small at low bit-widths. There are four, and an interviewer will expect you to place any method (AWQ, GPTQ, FP8, SmoothQuant) on them:

1. **Granularity** — how many values share one `(scale, zero_point)`? Per-tensor (one for all), per-channel (one per row), or per-group (one per ~128).
2. **Symmetry** — symmetric (no zero-point) or asymmetric (with one)?
3. **What to quantize** — weights only, or weights *and* activations? (vLLM writes this as `W4A16` vs `W8A8`.)
4. **How to get there** — PTQ (quantize a trained model) or QAT (train with quantization simulated)?

Each is a slider on the same trade-off — **accuracy ↔ size/speed ↔ complexity**. The reason these exist is the outlier problem: finer granularity and the right symmetry shrink the effective range each scale must cover, so you can drop to INT4 without wrecking quality. → see the [Glossary](../glossary.md) for *per-tensor/channel/group, PTQ/QAT, weight-only vs weight+activation*.

## 2 · Mental model

The four choices, and where the common methods land (a language-neutral structured comparison, so ASCII, per ADR-0005):

```text
GRANULARITY  (coarse ─────────────────► fine;  finer = smaller range/scale, more scale-storage)
   per-tensor            per-channel (per row)          per-group (e.g. every 128)
   1 scale / matrix      1 scale / output channel       1 scale / 128 weights
   outlier ruins all     isolates outlier channels      isolates outlier regions  ← INT4 sweet spot

SYMMETRY        symmetric (z = 0, no offset, fast matmul)   |  asymmetric (z ≠ 0, fits skewed data)
                good for zero-centred weights               |  good for post-ReLU / skewed activations

WHAT TO QUANTIZE   W4A16 / W8A16  (weight-only)   |   W8A8 (weight + activation)
                   dequant→FP16 matmul; MEMORY win |   INT8 tensor cores; COMPUTE win, activation outliers HARD

HOW               PTQ (post-training, ± calibration data)   |   QAT (simulate quant while training)
                  cheap, no retrain — inference default      |   best accuracy, expensive — training-side
```

Zooming in on **granularity** — the highest-leverage choice — as a spatial tiling of a weight matrix (finer = each scale covers a smaller range):

```text
  per-tensor              per-channel (per row)         per-group (block of ~128)
  ┌───────────┐           ┌───────────┐                 ┌─────┬─────┐
  │ s s s s s │           │ a a a a a │  scale a         │ p p │ q q │  scales p,q on row 0
  │ s s s s s │           │ b b b b b │  scale b         │ r r │ t t │  scales r,t on row 1
  │ s s s s s │           │ c c c c c │  scale c         │ u u │ v v │  scales u,v on row 2
  └───────────┘           └───────────┘                 └─────┴─────┘
  1 scale / matrix        1 scale / output row           1 scale / block   ← INT4 sweet spot
  outlier ruins all       isolates outlier rows          isolates outlier regions
```

Two shapes to hold:

- **Finer granularity buys accuracy with a little storage.** Splitting one scale into many shrinks the range each must cover, so a channel-local or group-local outlier no longer coarsens the whole tensor. The cost is more stored scales (higher effective bits) and slightly more complex kernels — a real but usually cheap trade.
- **"What to quantize" decides whether you win memory or compute.** Weight-only (`W4A16`) cuts HBM traffic → faster memory-bound *decode*, and it's easy because weights are static and well-behaved. Weight+activation (`W8A8`) also cuts *compute* via INT8 tensor cores, but activations are dynamic and outlier-prone, so it needs extra tricks (the method families in #11).

## 3 · Principle & the four choices

### 3.1 Granularity: per-tensor → per-channel → per-group

Recall error $\le (h-\ell)/(2(2^b-1))$: it's driven by the *range each scale covers*. Sharing one scale across a whole weight matrix (**per-tensor**) means a single outlier channel sets a coarse step for all channels. **Per-channel** (one scale per output row) isolates that outlier to its own row, so well-behaved channels get a fine step. **Per-group** (one scale per contiguous block of ~128 weights) goes finer still — the standard for INT4 LLM weights. The cost is storage: per-group with $g=128$ adds ~$16/128\approx0.125$ effective bits/weight — cheap for the accuracy it buys. §4 shows a clean channel getting a ~24× finer step under per-channel than per-tensor.

### 3.2 Symmetric vs asymmetric

**Symmetric** fixes $z$ at the grid centre (no zero-point offset), so the dequantized matmul has no cross-term — simpler and faster on hardware. It fits data centred on $0$, like most **weights**. **Asymmetric** keeps a zero-point, spending bits to fit a skewed range — the right choice for **activations** that are one-sided (e.g. post-ReLU, all $\ge 0$), where symmetric would waste half its levels on a negative range that never occurs. §4 shows symmetric being ~2× coarser than asymmetric on all-positive data. Rule of thumb: **symmetric for weights, asymmetric where the distribution is skewed.**

### 3.3 What to quantize: weight-only vs weight+activation

vLLM names schemes `WxAy` = $x$-bit weights, $y$-bit activations:

- **Weight-only** (`W4A16`, `W8A16`): quantize weights, keep activations in FP16. Weights are **dequantized to FP16 on-chip** and the matmul runs in FP16 — so the win is **HBM bandwidth** (fewer weight bytes → faster memory-bound *decode*), not FLOPs. Easy and popular because weights are static and near-symmetric; this is the AWQ/GPTQ INT4 regime (#11).
- **Weight+activation** (`W8A8`): quantize both, so the matmul runs on **INT8 tensor cores** — a genuine **compute** speedup on top of the memory win, valuable for compute-bound *prefill* and large batches. The catch: activations are computed at runtime and have large outliers, so naive `W8A8` loses accuracy — hence SmoothQuant (migrate outliers weight-ward) and friends (#11). vLLM's dynamic **FP8** path is a middle ground: it quantizes Linear weights per-tensor and scales activations *dynamically per-tensor* each forward pass (no calibration needed), trading some latency benefit for accuracy.
- **KV-cache quantization** is a separate axis — quantize the stored [KV cache](../part0/kv-cache.md) to fit more sequences (helps the [VRAM budget](../interview/vram-capacity-planning.md)); it's orthogonal to weight/activation quant.

### 3.4 PTQ vs QAT

**PTQ (Post-Training Quantization)** takes an already-trained model and quantizes it — optionally using a small **calibration** set to pick good ranges/scales (percentile clipping, outlier handling). No retraining, minutes-to-hours, and it's what inference infra uses (GPTQ/AWQ/FP8 are all PTQ). **QAT (Quantization-Aware Training)** simulates quantization *during* training so the model learns weights robust to it — best accuracy at very low bits, but it needs the training pipeline, data, and compute. For serving, the issue's guidance holds: **inference focuses on PTQ**; reach for QAT only when PTQ can't hold accuracy at the bit-width you need.

## 4 · Complete runnable code + line-by-line

Two demos, pure-Python and offline: per-tensor vs per-channel on a matrix with an outlier channel, and symmetric vs asymmetric on skewed data.

```python title="granularity_and_symmetry.py"
"""Granularity and symmetry: two knobs that shrink the range each scale covers.
Pure Python, offline. Error bound per value is step/2 (see the previous lesson)."""

def affine_step(xs, bits):
    """Asymmetric step size = range / (2^b - 1)."""
    return (max(xs) - min(xs)) / ((1 << bits) - 1)

def symmetric_step(xs, bits):
    """Symmetric step maps [-amax, amax] onto a signed grid of 2^(b-1)-1 levels each side."""
    amax = max(abs(v) for v in xs)
    return amax / ((1 << (bits - 1)) - 1)

if __name__ == "__main__":
    W = [[0.1, -0.2, 0.15, -0.05],      # clean channel
         [-0.3, 0.25, -0.1, 0.2],       # clean channel
         [8.0, -0.1, 0.05, 0.2]]        # one channel with a big outlier (8.0)

    # --- granularity: one scale for all vs one per channel (row) ---
    flat = [x for row in W for x in row]
    pt = affine_step(flat, bits=4)                       # per-tensor: global range sets the step
    pcs = [affine_step(row, bits=4) for row in W]        # per-channel: each row its own step
    print(f"per-tensor  INT4 step: {pt:.4f}  (one scale for the whole matrix; the 8.0 outlier sets it)")
    print("per-channel INT4 step: " + "  ".join(f"row{i} {s:.4f}" for i, s in enumerate(pcs)))
    print(f"  clean row0 error bound: per-tensor <= {pt/2:.4f}  vs  per-channel <= {pcs[0]/2:.4f}  "
          f"(~{round((pt/2)/(pcs[0]/2))}x better)")

    # --- symmetry: skewed, all-positive data (like post-activation values) ---
    act = [0.0, 0.1, 0.4, 0.8, 2.0]
    asym, sym = affine_step(act, bits=4), symmetric_step(act, bits=4)
    print(f"\nskewed data {act}:")
    print(f"  asymmetric INT4 step {asym:.4f} (bound {asym/2:.4f})  vs  symmetric {sym:.4f} (bound {sym/2:.4f})")
    print(f"  -> symmetric wastes ~half the grid on unused negatives (~{round(sym/asym)}x coarser)")
```

**Line-by-line:**

- `affine_step` / `symmetric_step` — the step sizes from the previous lesson: asymmetric spans the full `[min, max]`; symmetric spans `[-amax, amax]` on a signed grid, so if the data is one-sided it wastes the levels below $0$.
- **Granularity block** — `pt` is one step for the whole matrix (the `8.0` outlier sets it); `pcs` gives each row its own. The clean `row0`'s error bound collapses ~24× going per-tensor → per-channel, because its scale no longer has to span the outlier's range.
- **Symmetry block** — on all-positive `act`, symmetric's step is ~2× the asymmetric one, since half its grid covers negatives that never occur. Asymmetric spends those levels where the data actually is.

Expected output (exact arithmetic, not a benchmark):

```text
per-tensor  INT4 step: 0.5533  (one scale for the whole matrix; the 8.0 outlier sets it)
per-channel INT4 step: row0 0.0233  row1 0.0367  row2 0.5400
  clean row0 error bound: per-tensor <= 0.2767  vs  per-channel <= 0.0117  (~24x better)

skewed data [0.0, 0.1, 0.4, 0.8, 2.0]:
  asymmetric INT4 step 0.1333 (bound 0.0667)  vs  symmetric 0.2857 (bound 0.1429)
  -> symmetric wastes ~half the grid on unused negatives (~2x coarser)
```

Read it off: per-channel gives the clean rows a ~24× finer step while confining the `8.0` outlier to its *own* row's scale — the whole reason per-channel/group is standard for weights. And asymmetric is ~2× finer on skewed data — the reason activations often want a zero-point. Both are the same principle: **shrink the range each scale must cover.**

## 5 · Lab — pick a scheme, and connect it to a real flag

The numeric lab is the sweep above (try per-group by slicing each row into blocks of, say, 2 and giving each its own step — error drops again). The *applied* half is recognizing these choices in vLLM's naming, which needs no GPU to read:

```python title="scheme_names.py"
# vLLM expresses "what to quantize" as WxAy, and picks methods via `quantization=`.
# (Names/flags verified against vLLM 0.26.0; running them is #11's hands-on lesson.)
schemes = {
    "W4A16": "4-bit weights, 16-bit activations  -> weight-only; memory/decode win (AWQ/GPTQ INT4)",
    "W8A8":  "8-bit weights, 8-bit activations   -> INT8 tensor cores; compute win, needs outlier handling",
    "fp8":   "dynamic FP8: Linear weights per-tensor, activations scaled per-tensor per-forward (no calibration)",
}
for name, note in schemes.items():
    print(f"{name:6} {note}")
# In vLLM:  LLM(model, quantization="fp8")   or   vllm serve <model> --quantization fp8
```

**What to observe:** the `WxAy` name tells you immediately whether a method is a memory play (`W4A16`, activations untouched) or a compute play (`W8A8`, both quantized). Map any method you meet in #11 onto the four choices — its bit-widths (what to quantize), its granularity, its symmetry, and that it's PTQ — and you can reason about its accuracy/speed profile without memorizing it. Actually *running* `--quantization` on `Qwen2.5-7B` is the [#11 hands-on lesson](index.md).

## 6 · Common pitfalls / counter-intuitive points

- **Per-tensor on weights with outlier channels.** One outlier row sets a coarse step for the whole matrix. Use per-channel or per-group — the standard for INT4 weights — to isolate outliers. This is the single biggest accuracy lever at low bits.
- **Symmetric on skewed activations.** Symmetric wastes half the grid on a range the data never visits (e.g. negatives for post-ReLU). Use asymmetric where the distribution is one-sided; symmetric is for zero-centred weights.
- **Treating `W8A8` as "just more quantization than `W4A16`".** They optimize different things: `W4A16` is a *memory/decode* win (activations stay FP16); `W8A8` is a *compute* win (INT8 tensor cores) but must tame activation outliers. More quantized ≠ strictly better — it's a different trade.
- **Reaching for QAT to deploy.** QAT needs the training pipeline and is rarely worth it for serving; PTQ (± calibration) is the inference default. Use QAT only when PTQ can't hold quality at your target bits.
- **Assuming finer granularity is free.** More scales = higher effective bits and sometimes slower kernels. Per-group ~128 is the usual sweet spot, not per-element.
- **Forgetting activations are the hard part.** Weights are static and near-symmetric (easy); activations are dynamic and outlier-heavy (hard). That asymmetry is why weight-only quant is so popular and why activation quant needs SmoothQuant-style tricks (#11).
- **Static vs dynamic activation scales — a hidden axis.** Beyond symmetry, an activation scale can be computed *statically* (once, from calibration) or *dynamically* (recomputed each forward pass). Because activations swing hard with the input, dynamic scaling usually holds accuracy better — vLLM's FP8 path scales activations *dynamically per-tensor* each forward (no calibration), and some `W8A8` pipelines go finer with *per-token* scales. Static is simpler/faster but assumes a stable distribution, so off-distribution traffic bites. Not universal — static wins where kernel simplicity or fixed scales matter.

## 7 · Interview links

- [Quantization schemes: granularity, symmetry, what to quantize, PTQ vs QAT](../interview/quantization-schemes.md) — the high-frequency question this lesson prepares you for: *per-tensor vs per-channel vs per-group, symmetric vs asymmetric, weight-only vs weight+activation (W4A16 vs W8A8), and why inference uses PTQ.*

## 8 · Summary & further reading

**One line:** Four choices tame the outlier problem and place every quantization method — granularity (per-tensor→channel→group shrinks the range each scale covers), symmetry (symmetric for zero-centred weights, asymmetric for skewed activations), what to quantize (weight-only `W4A16` = memory/decode win, `W8A8` = compute win but activation outliers are hard), and how (PTQ for inference, QAT only when PTQ can't hold quality).

Further reading:

- The [Quantization Basics](quantization-basics.md) lesson — the affine map and error bound these choices tune.
- Next (#11): the method families — **GPTQ, AWQ, SmoothQuant, FP8, LLM.int8()** — as concrete points in this design space, plus the hands-on `Qwen2.5-7B` → INT4 run in vLLM.
- *SmoothQuant* (Xiao et al.) — migrating activation outliers into weights to make `W8A8` work; the clearest motivation for the "activations are the hard part" pitfall.

## 9 · Self-check

??? question "Weights in a matrix have one channel with much larger magnitude than the rest. Why does per-tensor INT4 quantization hurt the other channels, and how does per-channel fix it?"
    The quantization error is bounded by half a step, and the step is `range / (2^b − 1)` — set by the **range the scale must cover**. With per-tensor quantization, one scale spans the *whole* matrix, so the outlier channel's large magnitude stretches the range and forces a coarse step on *every* channel, including the small well-behaved ones — their values get rounded away. Per-channel gives each output row its own scale, so a clean row's step is set only by *its* (small) range — often ~10–30× finer — while the outlier is confined to its own row's coarse scale. Per-group (~128 weights per scale) refines this further and is standard for INT4 weights.

??? question "What's the difference between `W4A16` and `W8A8`, and which speeds up decode vs prefill?"
    `W4A16` = 4-bit weights, 16-bit (FP16) activations — **weight-only**: weights are dequantized to FP16 on-chip and the matmul stays FP16, so the win is **HBM bandwidth** (fewer weight bytes). That accelerates **memory-bound decode**. `W8A8` = 8-bit weights *and* 8-bit activations — the matmul runs on **INT8 tensor cores**, a genuine **compute** speedup that also helps **compute-bound prefill** and large batches. The trade: `W8A8` must handle activation outliers (they're dynamic and heavy-tailed), so it needs methods like SmoothQuant, whereas `W4A16` is easier because weights are static and near-symmetric.

??? question "Why does inference-side quantization overwhelmingly use PTQ rather than QAT? When would you still reach for QAT?"
    PTQ (Post-Training Quantization) takes an already-trained model and quantizes it, optionally with a small calibration set to choose good ranges — no retraining, cheap (minutes to hours), and it slots into a serving pipeline. That matches inference infra's constraints, and the dominant methods (GPTQ, AWQ, FP8) are all PTQ. QAT (Quantization-Aware Training) simulates quantization during training so the model learns quant-robust weights — it gives the best accuracy at very low bit-widths but requires the full training pipeline, data, and compute. You reach for QAT only when PTQ can't hold acceptable quality at the bit-width you need (e.g. aggressive sub-4-bit), which is uncommon for the INT4/INT8 regimes serving typically targets.
