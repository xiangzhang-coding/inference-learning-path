# Why Quantization Speeds Up Inference: the Affine Map & the Precision Trade-off

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The vLLM quantization surface referenced here (`LLM(..., quantization="fp8")`, `vllm serve --quantization fp8`) is verified against vLLM 0.26.0 via Context7 (ADR-0004). The affine quantization map in §3–§4 is **math, not a library call** — the runnable code is pure-Python and offline. Memory/speedup figures are **illustrative / order-of-magnitude references**; the quantization-error bounds (`≤ step/2`) are exact arithmetic on the model. Concrete method families (GPTQ/AWQ/…) and a hands-on `Qwen2.5-7B` INT4 run are the **next lesson's** territory.

---

## 1 · Intuition & why it matters

You already know from the [number formats](../part0/number-formats.md) lesson what FP16/INT8/INT4 *are*, and from the [operator roofline](../part2/roofline-analysis.md) that **decode is memory-bound**: each step's time is dominated by reading bytes from HBM, not by doing math. Put those together and quantization's payoff is immediate. The bytes a decode step reads are mostly the **model weights** — a 7B model in FP16 is ~14 GB streamed through HBM every step. Store those weights in INT4 instead (0.5 byte/param) and you read ~3.5 GB: **~4× less traffic, so a memory-bound decode step runs up to ~4× faster** (illustrative). That is the entire throughput lever.

The one idea to anchor everything: **quantization trades precision for bandwidth.** You represent each weight with fewer bits, which shrinks HBM traffic (speed, on a memory-bound workload) at the cost of representing it less exactly (accuracy). Because decode is bandwidth-starved, those saved bytes convert almost directly into speed — which is why low-bit weights are close to free performance, and why every serious deployment quantizes. The whole game is then: **how few bits can you use before the precision loss hurts quality?** — the question this Part answers. → see the [Glossary](../glossary.md) for *Quantization, PTQ/QAT, per-tensor/channel/group*.

!!! note "Weight-only quantization speeds up *memory*, not *compute*"
    A crucial nuance: the popular INT4 weight-only path (AWQ/GPTQ, vLLM's `W4A16`) stores weights in 4 bits but **dequantizes them back to FP16 on-chip** before the matmul — the arithmetic is still FP16. So the win is **fewer HBM bytes** (which is exactly what memory-bound decode needs), *not* fewer FLOPs. Speeding up compute needs quantized *activations* too (INT8 `W8A8` on INT8 tensor cores) — the harder path in the [next lesson](quantization-schemes.md).

## 2 · Mental model

Quantization maps a continuous range onto a small integer grid, and back:

```text
FP16 weights (continuous)        INT4 grid (16 levels)          dequantized (back to FP16)
  -0.9 ─────────── 3.0            0  1  2 ... 15                 x̂ = scale·(q − zero)
   real values on a line    ──►   ▏──▏──▏── ... ──▏     ──►      lands on the nearest grid point
                                  └ step = scale ┘                 error ≤ scale/2

STORE: the int q (4 bits)  +  one (scale, zero_point) per group   ← the only floats kept
READ back: x̂ = scale·(q − zero_point)      ← "dequantize", done on-chip before the matmul
```

Three shapes to hold:

- **A quantized value is an integer index into an evenly-spaced grid.** `scale` is the spacing (the real-world size of one step); `zero_point` is which integer maps to real 0. Store the small integers plus a handful of `(scale, zero_point)` floats — that's the compression.
- **The step size is set by the range and the bit count:** $\text{scale} = \text{range}/(2^b-1)$. More bits → finer grid → smaller error. A wide range (one big outlier) → coarse grid → everyone suffers. This single relation drives the whole Part.
- **Dequantization is cheap and happens at use.** Weight-only quant keeps the *stored* form small; it reconstructs FP16 just before the matmul. The saving is in what crosses the HBM line, which is precisely decode's bottleneck.

## 3 · Principle & math

### 3.1 The affine (uniform) quantization map

Pick a target of $b$ bits, giving integer levels $[0, q_{\max}]$ with $q_{\max}=2^b-1$. For real values spanning $[\ell, h]$, **asymmetric** (affine) quantization is:

$$
\text{scale} = \frac{h-\ell}{q_{\max}}, \qquad z = \operatorname{round}\!\left(\frac{-\ell}{\text{scale}}\right)
$$

$$
q = \operatorname{clamp}\!\big(\operatorname{round}(x/\text{scale}) + z,\ 0,\ q_{\max}\big), \qquad \hat{x} = \text{scale}\cdot(q - z)
$$

$z$ (the zero-point) is the integer that represents real $0$, so $0$ is quantized exactly — important because zeros are everywhere (padding, ReLU outputs, pruned weights). **Symmetric** quantization drops the offset ($z$ fixed at the grid centre / $0$ for a signed grid) and sets $\text{scale}=\max|x|/q_{\max}^{\text{signed}}$: simpler and no offset term in the matmul, at the cost of wasting levels if the data isn't centred on $0$ (the [next lesson](quantization-schemes.md)).

### 3.2 The precision cost: error is bounded by half a step

Rounding to the nearest grid point means the reconstruction error per value is at most half the step:

$$
|x - \hat{x}| \le \frac{\text{scale}}{2} = \frac{h-\ell}{2\,(2^b-1)}
$$

Two consequences you'll use constantly:

- **Each bit halves the error.** Going $b \to b+1$ roughly doubles $q_{\max}$, halving `scale` and the error bound. INT8 vs INT4 is ~16× finer steps.
- **Outliers are poison.** The error scales with the *range* $h-\ell$. A single large-magnitude weight stretches the range, inflating `scale` for *every* value sharing that scale. This is why granularity (per-channel/group) and outlier-aware methods exist — the whole subject of the next lesson and the method families in #11.

### 3.3 Why fewer bits → more throughput (and the effective bit count)

Memory moved for the weights scales with bits-per-weight, and decode is memory-bound, so throughput scales roughly inversely with the bit width:

$$
\text{weight bytes} = N_{\text{params}}\times \frac{\text{bits}}{8}, \qquad
\text{decode speedup} \approx \frac{\text{FP16 bytes}}{\text{quantized bytes}} = \frac{16}{\text{bits}}\ \text{(illustrative, memory-bound)}
$$

So INT4 ≈ 4× less weight traffic than FP16. One honesty check: the stored `(scale, zero_point)` floats add overhead, so the *effective* bits-per-weight is a bit above the nominal — e.g. INT4 with a per-group scale every 128 weights adds ~$16/128=0.125$ bits, giving ~4.1 effective bits, not 4. Small, but it's why "4-bit" models are a little larger than $N/2$ bytes.

## 4 · Complete runnable code + line-by-line

Affine quantize/dequantize, pure-Python and offline. It shows the step size and error bound for INT8 vs INT4, how an outlier inflates both, and a concrete round-trip.

```python title="affine_quantization.py"
"""Affine quantization: trade bits for bandwidth, at a bounded cost in precision.
Pure Python, offline — the affine map is universal math, not a library call."""

def quantize_dequantize(xs, bits):
    """Asymmetric affine quant -> dequant. Returns (reconstructed_values, scale)."""
    qmax = (1 << bits) - 1                       # 2^b - 1 levels: 255 (INT8), 15 (INT4)
    lo, hi = min(xs), max(xs)
    scale = (hi - lo) / qmax                     # real-world size of one grid step
    zero = round(-lo / scale)                    # the integer that maps to real 0
    out = []
    for x in xs:
        q = round(x / scale) + zero              # to the nearest grid index...
        q = min(max(q, 0), qmax)                 # ...clamped into [0, qmax]
        out.append(scale * (q - zero))           # dequantize: back to a real value
    return out, scale

def report(label, xs):
    lo, hi = min(xs), max(xs)
    print(f"range [{lo:.2f}, {hi:.2f}] (width {hi - lo:.2f}) — {label}")
    for bits in (8, 4):
        _, scale = quantize_dequantize(xs, bits)
        print(f"  INT{bits}: scale {scale:.4f}, max error <= {scale / 2:.4f}, "
              f"{16 / bits:.1f}x smaller than FP16")

if __name__ == "__main__":
    w_outlier = [-0.9, -0.2, 0.1, 0.5, 3.0]      # one big weight (3.0) stretches the range
    w_clean   = [-0.9, -0.2, 0.1, 0.5, 0.9]      # same, but no outlier
    report("one outlier at 3.0 stretches the range", w_outlier)
    report("no outlier", w_clean)

    recon, scale = quantize_dequantize(w_outlier, bits=4)
    err = max(abs(a - b) for a, b in zip(w_outlier, recon))
    print(f"\nINT4 round-trip of {w_outlier}:")
    print(f"  reconstructed: {[round(v, 2) for v in recon]}")
    print(f"  max abs error: {err:.2f}   (<= step/2 = {scale / 2:.2f}; the outlier forced a coarse step on all)")
```

**Line-by-line:**

- `quantize_dequantize` — the §3.1 map made literal. `qmax` is the number of levels; `scale` divides the range across them; `zero` is the integer for real $0$. The loop rounds each value to the nearest grid index, clamps it into range, and reconstructs $\hat{x}=\text{scale}\cdot(q-z)$. It returns the dequantized values so you can see the precision loss directly.
- `report` — prints, per bit-width, the step `scale`, the exact error bound `scale/2`, and the compression vs FP16 (`16/bits`). Run on a weight vector with and without an outlier.
- `__main__` — the two vectors differ only in whether a `3.0` is present; watch how it changes the INT4 step. The final round-trip shows actual reconstructed values and the realized max error under the bound.

Expected output (exact arithmetic, not a benchmark):

```text
range [-0.90, 3.00] (width 3.90) — one outlier at 3.0 stretches the range
  INT8: scale 0.0153, max error <= 0.0076, 2.0x smaller than FP16
  INT4: scale 0.2600, max error <= 0.1300, 4.0x smaller than FP16
range [-0.90, 0.90] (width 1.80) — no outlier
  INT8: scale 0.0071, max error <= 0.0035, 2.0x smaller than FP16
  INT4: scale 0.1200, max error <= 0.0600, 4.0x smaller than FP16

INT4 round-trip of [-0.9, -0.2, 0.1, 0.5, 3.0]:
  reconstructed: [-0.78, -0.26, 0.0, 0.52, 3.12]
  max abs error: 0.12   (<= step/2 = 0.13; the outlier forced a coarse step on all)
```

Read it off: INT4 vs INT8 is a coarser step (fewer bits) but 4× vs 2× smaller than FP16 — the bandwidth win. And the single `3.0` more than **doubles** the INT4 error (0.13 vs 0.06 bound) for *every* weight, because they all share one range-driven scale. That's the outlier problem, quantified — and the reason the next lesson reaches for finer granularity.

## 5 · Lab — error vs bit-width, and the outlier tax

This lab needs no GPU — it's the pure-Python model above, swept over bit-widths. (The GPU comes in the [next lesson's](quantization-schemes.md) hands-on quantization and #11.)

```python title="quant_error_sweep.py"
from affine_quantization import quantize_dequantize   # from §4

weights = [-0.9, -0.2, 0.1, 0.5, 0.9]                  # a "clean" channel
print("bits  step(scale)  error_bound  compression")
for bits in (8, 6, 4, 3, 2):
    _, scale = quantize_dequantize(weights, bits)
    print(f"  {bits}     {scale:7.4f}     {scale/2:7.4f}      {16/bits:.1f}x")
```

**What to observe:** halving the bits roughly doubles both the step and the error bound, while the compression rises as `16/bits`. The sweep makes the trade concrete — INT8 is nearly lossless here, INT4 is usually the sweet spot for LLM weights (big memory win, tolerable error), and INT2/INT3 start to bite. Re-run with a `3.0` appended to `weights` and every row's error roughly doubles: the outlier tax again. This is the whole quantization design space in ten lines — the rest of the Part is *how* to keep the error small at low bits (granularity, outlier handling, better methods).

## 6 · Common pitfalls / counter-intuitive points

- **"INT4 weights make the math 4× faster."** No — weight-only INT4 dequantizes to FP16 before the matmul, so FLOPs are unchanged. The 4× is **memory traffic**, which is what speeds up *memory-bound decode*. Compute speedups need quantized activations (INT8 tensor cores), a different and harder path.
- **Ignoring outliers.** Error scales with the *range*; one large-magnitude value inflates `scale` for everything sharing it. Naive per-tensor quantization of weights/activations with outliers is the #1 accuracy killer — and the reason for per-channel/group and outlier-aware methods.
- **Forgetting the scale/zero-point overhead.** "4-bit" isn't exactly 0.5 byte/param — the stored `(scale, zero_point)` per group add a little. Finer granularity ⇒ more scales ⇒ higher effective bits. There's a granularity-vs-size trade even before accuracy.
- **Confusing quantization with low-precision training.** This is *post-hoc* compression of a trained model for inference (PTQ), not training in low precision. Different goal, different failure modes (next lesson).
- **Assuming more bits is always safer/needed.** On memory-bound decode, extra bits are pure cost (more traffic, slower) for accuracy you may not need. The engineering question is the *minimum* bits that hold quality, not the maximum.
- **Symmetric everywhere.** Symmetric quant wastes half the grid on skewed data (e.g. all-positive activations). When to use which is the next lesson.

## 7 · Interview links

- [Quantization: why it speeds up inference & the precision trade-off](../interview/quantization-basics.md) — the high-frequency question this lesson prepares you for: *why does quantization raise throughput, does it speed up compute or memory, what's the affine map, and what bounds the error?*

## 8 · Summary & further reading

**One line:** Quantization maps weights onto a small integer grid via an affine map ($\hat{x}=\text{scale}\cdot(q-z)$, error $\le \text{scale}/2$), trading precision for far fewer HBM bytes — which, because decode is memory-bound, converts almost directly into throughput (INT4 ≈ 4× less weight traffic than FP16); the design challenge is minimizing the precision loss at low bit-widths.

Further reading:

- The [Number Formats](../part0/number-formats.md) lesson — what FP16/INT8/INT4 are and their range-vs-resolution trade, the input to this Part.
- The [Operator Roofline](../part2/roofline-analysis.md) lesson — why decode is memory-bound, hence why fewer weight bytes ≈ more speed.
- Next: [Quantization Choices](quantization-schemes.md) — granularity, symmetry, what to quantize, and PTQ vs QAT; then #11's method families (GPTQ/AWQ/…) and the hands-on `Qwen2.5-7B` INT4 run.
- *LLM.int8()* (Dettmers et al.) — the outlier problem in activations, made concrete.

## 9 · Self-check

??? question "Weight-only INT4 quantization gives a big decode speedup. Is the arithmetic faster? Explain what actually speeds up."
    The arithmetic is **not** faster — weight-only INT4 stores weights in 4 bits but dequantizes them back to FP16 on-chip before the matmul, so the FLOPs are identical to FP16. What speeds up is **HBM traffic**: reading ~4× fewer weight bytes per decode step. Because decode is *memory-bound* (its time is dominated by reading weights and KV from HBM, not by compute), cutting the bytes ~4× makes the step up to ~4× faster. To speed up the *compute* you'd also have to quantize activations and use INT8 tensor cores (W8A8), which is harder because activations have outliers.

??? question "Write the affine dequantization formula and state what bounds the quantization error. Why do outliers make quantization worse?"
    Dequantization is $\hat{x} = \text{scale}\cdot(q - z)$, where $q$ is the stored integer, `scale` is the step size, and $z$ (zero-point) is the integer mapping to real $0$. Rounding to the nearest grid point bounds the error at half a step: $|x-\hat{x}| \le \text{scale}/2 = (h-\ell)/(2(2^b-1))$. Outliers hurt because the error scales with the **range** $h-\ell$: a single large-magnitude value stretches the range, which inflates `scale` — and therefore the error — for *every* value sharing that scale, even the small well-behaved ones.

??? question "You quantize a 7B model's weights from FP16 to INT4. Estimate the weight-memory reduction, and name one reason the real size is a bit larger than the naive estimate."
    FP16 is 2 bytes/param → ~14 GB for 7B weights; INT4 is 0.5 byte/param → ~3.5 GB, a **~4× reduction** ($16/4$). On memory-bound decode that translates to roughly up to a 4× reduction in weight traffic (illustrative). The real quantized model is a little larger than 3.5 GB because you must also store the `(scale, zero_point)` values — one set per quantization group — so the *effective* bits-per-weight is slightly above 4 (e.g. a per-128 group scale adds ~0.125 bits/weight).
