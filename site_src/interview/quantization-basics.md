# Quantization: why it speeds up inference & the precision trade-off

!!! info "Baseline: **vLLM 0.26.0** · quantization surface verified via Context7 (ADR-0004)"

**Module:** Part 4 · Quantization   ·   **Tests the lesson:** [Why Quantization Speeds Up Inference: the Affine Map & the Precision Trade-off](../part4/quantization-basics.md)

---

## Q: Why does quantization raise LLM inference throughput? Does it speed up compute or memory? Explain the affine map and what bounds the error.

### Direct answer

Quantization raises throughput because **decode is memory-bound** — a step's time is dominated by reading bytes from HBM, mostly the model weights. Storing weights in fewer bits shrinks that traffic: FP16 (2 B/param) → INT4 (0.5 B/param) is ~4× fewer weight bytes, so a memory-bound decode step runs up to ~4× faster.

**Compute or memory?** For the common **weight-only** path (`W4A16`, AWQ/GPTQ), it's a **memory** win only: weights are dequantized back to FP16 on-chip and the matmul stays FP16, so FLOPs are unchanged. Speeding up *compute* requires quantizing **activations** too (`W8A8` on INT8 tensor cores).

**The affine map:** with $q_{\max}=2^b-1$ over a range $[\ell,h]$,

$$\text{scale}=\frac{h-\ell}{q_{\max}},\quad z=\operatorname{round}(-\ell/\text{scale}),\quad \hat{x}=\text{scale}\cdot(q-z)$$

$z$ (zero-point) is the integer for real $0$ (asymmetric); symmetric drops it.

**Error bound:** rounding to the nearest grid point gives $|x-\hat{x}|\le \text{scale}/2=(h-\ell)/(2(2^b-1))$. Each extra bit halves it; a wider range (outliers) inflates it for everything sharing the scale.

### Deep dive

- **Why "memory-bound ⇒ bits ≈ speed".** On the [roofline](arithmetic-intensity.md), decode sits far left (intensity ≈ 1); its time ∝ bytes moved. Weight bytes ∝ bits/param, so decode speedup ≈ `16/bits` vs FP16 (illustrative). Prefill (compute-bound) benefits less from weight-only quant.
- **Effective bits > nominal.** You also store `(scale, zero_point)` per group; a per-128 group adds ~0.125 bits/weight, so "INT4" is ~4.1 effective bits and the file is a bit over `N/2` bytes.
- **Outliers are the enemy.** Error ∝ range. One large weight stretches the range and coarsens the step for all values under that scale — the reason for per-channel/group granularity and outlier-aware methods.
- **Zeros quantize exactly (asymmetric).** Because $z$ maps to real $0$, padding/ReLU/pruned zeros are represented without error — one reason asymmetric suits skewed, zero-heavy data.

### Code

The affine map + error bound, pure Python:

```python
def quantize_dequantize(xs, bits):
    qmax = (1 << bits) - 1
    lo, hi = min(xs), max(xs)
    scale = (hi - lo) / qmax
    zero = round(-lo / scale)
    out = [scale * (min(max(round(x / scale) + zero, 0), qmax) - zero) for x in xs]
    return out, scale

w = [-0.9, -0.2, 0.1, 0.5, 3.0]          # the 3.0 outlier stretches the range
_, s4 = quantize_dequantize(w, 4)
print(f"INT4 step {s4:.2f}, error <= {s4/2:.2f}")   # step 0.26, error <= 0.13
```

The `3.0` forces a 0.26 step (0.13 error bound) on every weight — drop it and the step halves.

### Interviewer follow-ups

- *"Does INT4 make the GEMM 4× faster?"* → No — weight-only INT4 dequantizes to FP16, so the GEMM's FLOPs are unchanged. The 4× is HBM traffic, which is what memory-bound decode is limited by. Compute speedup needs INT8 activations + tensor cores.
- *"Why does quantization help decode more than prefill?"* → Decode is memory-bound (reads weights+KV, tiny compute); prefill is compute-bound. Weight-only quant cuts bytes, so it helps the memory-bound phase most.
- *"What sets the quantization error?"* → `scale/2`, and `scale = range/(2^b−1)`. So error is driven by bit-width (each bit halves it) and range (outliers inflate it).
- *"Is a 4-bit model exactly 1/4 the size of FP16?"* → Slightly larger — you also store per-group scales/zero-points, so effective bits are a bit above 4.
- *"What's the point of the zero-point?"* → It lets real 0 map to an exact integer and fits skewed (asymmetric) ranges; symmetric quant sets it to the grid centre and skips the offset for a simpler matmul.

### Linked concepts

- Lesson: [Why Quantization Speeds Up Inference](../part4/quantization-basics.md)
- Related: [Quantization schemes: granularity, symmetry, PTQ vs QAT](quantization-schemes.md) (how to keep error small at low bits), [Number formats & precision](number-formats.md) (what FP16/INT8/INT4 are), [Arithmetic intensity of GEMM & attention](arithmetic-intensity.md) (why decode is memory-bound)
- Glossary: [Quantization, PTQ/QAT, per-tensor/channel/group](../glossary.md)
