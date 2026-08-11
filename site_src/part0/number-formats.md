# Number Formats: FP16 · BF16 · FP8 · INT8 · INT4

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The `--dtype` and `--kv-cache-dtype` flags on this page are verified against vLLM 0.26.0 via Context7 (ADR-0004). The bit-layout facts and the range/precision arithmetic are *exact*; any speedup figure is an **illustrative / order-of-magnitude reference**. This lesson covers the *formats*; the quantization *methods* (GPTQ/AWQ/SmoothQuant) are [Part 4](../part4/index.md).

---

## 1 · Intuition & why it matters

Every weight, activation, and KV-cache entry in an LLM is ultimately a small bundle of **bits**. How *many* bits, and how they're *split up*, decides three things at once: how much VRAM the model eats, how many bytes cross the HBM line each step, and how much numerical error you inject. Because inference is [memory-bound](gpu-hardware.md), that middle one is decisive — **halving the bits per number roughly halves the bytes moved, which roughly doubles decode throughput.** Number format *is* the throughput lever; quantization is just the art of pulling it without breaking the model.

So before touching a single quantization method, you need a crisp picture of the formats themselves: the venerable **FP32/FP16/BF16**, the aggressive **FP8** (two flavors), and the integer **INT8/INT4** that weight quantization lives in. The whole game is a trade between **range** (how big/small a number you can represent) and **precision** (how finely you can distinguish nearby numbers) — and floats and integers make that trade in fundamentally different ways. Get this lesson right and [Part 4](../part4/index.md) becomes "which method preserves accuracy," not "wait, what's e4m3?" → see the [Glossary](../glossary.md) for *FP8 / INT8 / INT4*, *Per-tensor / per-channel / per-group*, *KV-cache quantization*.

## 2 · Mental model

A floating-point number is **sign · mantissa · 2^exponent** — the exponent buys *range*, the mantissa buys *precision*. An integer format is **one shared scale · a small integer** — all the range lives in the scale, all the values share it.

```text
FLOATS  (per-value exponent: range and precision both baked into each number)
  bit layout            S = sign,  E = exponent (range),  M = mantissa (precision)
  FP32   S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM   1+8+23   range ~1e38   the reference
  FP16   S EEEEE MMMMMMMMMM                   1+5+10   range ~6e4    precise, SMALL range -> overflow risk
  BF16   S EEEEEEEE MMMMMMM                   1+8+7    range ~1e38   FP32's range, COARSE precision
  FP8e4m3  S EEEE MMM                         1+4+3    range ~448    FP8 for weights/activations
  FP8e5m2  S EEEEE MM                         1+5+2    range ~6e4    FP8 with more range, less precision

INTEGERS  (one shared scale s for a whole tensor/channel/group)
  INT8    [ -128 .. 127 ]   real value  r ≈ s·(q - z)      8 bits/number
  INT4    [   -8 ..   7 ]   real value  r ≈ s·(q - z)      4 bits/number  <- weight-only quant
                             q = stored integer, s = scale, z = zero-point
```

Two shapes to hold:

- **Range vs precision is a see-saw at fixed bit-width.** FP16 and BF16 are *both* 16 bits, but FP16 spends 5 bits on exponent / 10 on mantissa (precise but overflows past ~65504), while BF16 spends 8/7 — the *same exponent as FP32*, so it never overflows where FP32 wouldn't, at the cost of a coarser mantissa. That single reallocation is why BF16 won for deep learning: model tensors span a huge dynamic range, and an overflow to `inf` is fatal while a little rounding noise is survivable.
- **Floats carry range per-value; integers share one scale.** An INT8 tensor is 256 evenly-spaced levels between `−s·128` and `s·127`. That's brutally efficient (8 bits, no exponent) *if* the values are well-scaled and not dominated by a few outliers — which is exactly the tension quantization methods in [Part 4](../part4/index.md) exist to manage (per-channel/per-group scales, outlier handling).

## 3 · Principle & math

A floating-point value with sign $s\in\{0,1\}$, $E$ exponent bits (bias $2^{E-1}-1$), and $M$ mantissa bits, for a normal number, is:

$$
x = (-1)^{s}\,\Bigl(1 + \tfrac{m}{2^{M}}\Bigr)\, 2^{\,e - \text{bias}}, \qquad 0 \le m < 2^{M}
$$

The **exponent width sets the range** — roughly the largest magnitude is $\sim 2^{2^{E-1}}$ — and the **mantissa width sets the precision** — the relative gap between adjacent representable numbers is $\approx 2^{-M}$. Compare 16-bit formats:

- **FP16** ($E=5, M=10$): max ≈ $65504$, relative step $\approx 2^{-10}\approx 0.001$. Precise, but anything past 65504 becomes `inf`.
- **BF16** ($E=8, M=7$): max ≈ $3.39\times10^{38}$ (FP32's range), relative step $\approx 2^{-7}\approx 0.008$. Eight times coarser than FP16, but essentially un-overflowable — the right trade when tensors span many orders of magnitude.

**FP8** pushes this to 8 bits with two standardized splits: **E4M3** (max ≈ $448$, more precision) for weights/activations, and **E5M2** (max ≈ $57344$, more range) for gradients/where range matters. Both need a **scaling factor** to map real tensors into their tiny range — which is why vLLM's FP8 KV cache warns it "may cause accuracy drop without a proper scaling factor."

**Integer** formats drop the per-value exponent entirely. A real value $r$ is reconstructed from a stored integer $q$, a shared **scale** $s$, and a **zero-point** $z$:

$$
r \approx s\,(q - z), \qquad s = \frac{\max|w|}{2^{\,b-1}-1}\ \text{(symmetric, } b\text{-bit)}
$$

The quantization error is bounded by half a step, $|r - \hat r| \le s/2$, so a *smaller* scale (tighter value range, or finer granularity — [per-tensor vs per-channel vs per-group](../glossary.md)) means less error. INT8 gives $2^8=256$ levels; INT4 only $2^4=16$, which is why 4-bit is almost always **weight-only** (weights tolerate it; activations, with their outliers, usually don't).

Finally the memory story, which is the whole point: **bytes per number $= \text{bits}/8$**. Weights of $N$ params cost $N\cdot\text{bits}/8$ bytes; the [KV cache](kv-cache.md) shrinks by the same ratio when stored in FP8. Going BF16 → INT4 is a **4× cut** in both VRAM footprint and bytes-moved-per-step — and since decode is bandwidth-bound, that ratio flows almost directly into throughput.

## 4 · Complete runnable code + line-by-line

This is **offline-runnable** — pure CPU, needs only `numpy`. It prints the format reference table, demonstrates the BF16-vs-FP16 range/precision see-saw on real values, and does an INT8 quant round-trip so you can *see* the error bound $s/2$.

```python title="number_formats.py"
"""Number-format explorer: layout table, float range/precision, INT8 round-trip (CPU)."""
import numpy as np

# --- Part 1: the reference table (bit widths + standard maxima are exact facts) ---
FORMATS = [
    # name,      bits, sign, exp, mant, ~max            note
    ("FP32",     32,   1,    8,   23,   "3.40e+38",     "reference"),
    ("FP16",     16,   1,    5,   10,   "6.55e+04",     "precise, small range"),
    ("BF16",     16,   1,    8,    7,   "3.39e+38",     "FP32 range, coarse"),
    ("FP8 E4M3",  8,   1,    4,    3,   "4.48e+02",     "FP8 weights/act"),
    ("FP8 E5M2",  8,   1,    5,    2,   "5.73e+04",     "FP8 more range"),
    ("INT8",      8,   0,    0,    0,   "s * 127",      "integer + scale"),
    ("INT4",      4,   0,    0,    0,   "s * 7",        "integer + scale"),
]
print(f"{'format':9} {'bits':>4} {'S':>2} {'E':>2} {'M':>3} {'~max':>10}   note")
for name, bits, s, e, m, mx, note in FORMATS:
    e_s = str(e) if e else "-"
    m_s = str(m) if m else "-"
    print(f"{name:9} {bits:>4} {s:>2} {e_s:>2} {m_s:>3} {mx:>10}   {note}")

# --- Part 2: BF16 vs FP16 — range and precision see-saw ---
def to_bf16(x):
    """Truncating BF16 (round-toward-zero). Real HW rounds-to-nearest;
    the point here is the FIELD WIDTHS, not the last bit."""
    u = np.float32(x).view(np.uint32)
    u = (u >> 16) << 16                       # drop the low 16 mantissa bits
    return u.view(np.float32)

print("\nvalue 1/3 (precision test):")
print(f"  fp32={float(np.float32(1/3)):.6g}  "
      f"fp16={float(np.float16(1/3)):.6g}  bf16={float(to_bf16(1/3)):.6g}")
print("value 1e5 (range test):")
print(f"  fp32={float(np.float32(1e5)):.6g}  "
      f"fp16={float(np.float16(1e5)):.6g}  bf16={float(to_bf16(1e5)):.6g}")

# --- Part 3: INT8 symmetric quantization round-trip ---
w = np.array([-2.5, -0.3, 0.0, 0.8, 3.1, 12.0], dtype=np.float32)
scale = np.abs(w).max() / 127                 # per-tensor symmetric scale s
q = np.round(w / scale).astype(np.int8)       # stored integers in [-128, 127]
deq = q.astype(np.float32) * scale            # reconstruct r ≈ s*q  (z = 0)
print(f"\nINT8 scale s = {scale:.5f}   (error bound s/2 = {scale/2:.5f})")
print(f"  q   = {q.tolist()}")
print(f"  max abs error = {np.abs(w - deq).max():.5f}")
```

**Line-by-line:**

- **Part 1** — the reference table. Bit widths and the standard maxima (FP16 65504, FP8-E4M3 448, FP8-E5M2 57344) are fixed facts, printed as data so there's no arithmetic to get wrong. Read the `S/E/M` columns as "where the 16 (or 8) bits go."
- `to_bf16` — emulates BF16 by *truncating* FP32's low 16 mantissa bits (BF16 is literally "FP32 with 16 fewer mantissa bits"). Real hardware rounds to nearest; truncation is close enough to show the width effect.
- **Part 2** — two probes. `1/3` tests **precision**: FP16's 10-bit mantissa lands closer than BF16's 7-bit. `1e5` tests **range**: it's fine in FP32 and BF16 but **overflows FP16 to `inf`** (past 65504). One number wins each test — that's the see-saw.
- **Part 3** — symmetric INT8: `scale = max|w| / 127` maps the biggest magnitude to ±127; `round(w/scale)` stores integers; `q*scale` reconstructs. The reported max error stays under `s/2`, exactly the bound from §3.

Expected output (exact arithmetic, not a benchmark):

```text
format    bits  S  E   M       ~max   note
FP32        32  1  8  23   3.40e+38   reference
FP16        16  1  5  10   6.55e+04   precise, small range
BF16        16  1  8   7   3.39e+38   FP32 range, coarse
FP8 E4M3     8  1  4   3   4.48e+02   FP8 weights/act
FP8 E5M2     8  1  5   2   5.73e+04   FP8 more range
INT8         8  0  -   -     s * 127   integer + scale
INT4         4  0  -   -       s * 7   integer + scale

value 1/3 (precision test):
  fp32=0.333333  fp16=0.333252  bf16=0.332031
value 1e5 (range test):
  fp32=100000  fp16=inf  bf16=99840

INT8 scale s = 0.09449   (error bound s/2 = 0.04724)
  q   = [-26, -3, 0, 8, 33, 127]
  max abs error = 0.04409
```

`fp16=inf` on the range test is the whole reason training and inference default to BF16, and `max abs error = 0.04409 < s/2` is the quantization error bound made concrete.

## 5 · Lab — flip the format in vLLM and watch VRAM move

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB (loads `Qwen2.5-7B-Instruct`)
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~15 min · ~¥1 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** FP8 KV cache support is hardware-gated — per vLLM, CUDA 11.8+ supports `fp8`/`fp8_e5m2` and ROCm supports `fp8` (= `fp8_e4m3`); other backends differ. Check your platform's vLLM build.

Two verified `vllm serve` flags let you feel the range/precision trade on real weights and the real KV cache. Set the **compute dtype** (weights/activations) with `--dtype`:

```bash
# BF16 (default for this model): FP32's range, half the bytes of FP32
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 8192
```

Independently, store the **KV cache** in FP8 with `--kv-cache-dtype` (verified literal: `auto`, `fp8` = `fp8_e4m3`, `fp8_e5m2`):

```bash
# KV cache in FP8 e4m3: ~2x smaller KV cache -> more concurrent sequences fit in 24 GB
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8_e4m3 --max-model-len 8192
```

**What to observe:** with FP8 KV cache, the per-token KV footprint (the [KV Cache](kv-cache.md) $\kappa$) roughly halves, so vLLM can hold more sequences' KV in the same 24 GB — a direct throughput win on the exact bottleneck Part 0A identified. Watch the startup log's reported KV-cache blocks grow, and heed vLLM's own warning that FP8 KV cache "may cause accuracy drop without a proper scaling factor." **Weight** quantization to INT4/INT8 (AWQ/GPTQ) is a *method*, not just a dtype flip — that's [Part 4](../part4/index.md); here you've felt the dtype lever, next you'll learn to pull it safely.

## 6 · Common pitfalls / counter-intuitive points

- **"Fewer bits is always faster and basically free."** Faster, often; free, no. Below ~8 bits, and especially for *activations* (which have outliers), naive quantization falls off an accuracy cliff — the reason [Part 4](../part4/index.md) methods (per-group scales, SmoothQuant, outlier handling) exist.
- **Confusing BF16 and FP16.** Same size, opposite trade: BF16 = FP32's range, coarse precision; FP16 = fine precision, small range (overflows past 65504). Deep learning picks BF16 because overflow is fatal and rounding noise isn't.
- **FP8 without a scale is a footgun.** E4M3 maxes out at ~448; feed it un-scaled tensors and everything saturates. FP8 always rides with a scaling factor — the "proper scaling factor" vLLM warns about.
- **INT4 for activations.** INT4 weight-*only* is common and works; INT4 *activations* almost never survive because outliers blow the 16-level budget. Know which tensor you're quantizing.
- **`--dtype` (compute) ≠ `--kv-cache-dtype` (storage).** They're separate knobs: you can run BF16 weights with an FP8 KV cache. `--dtype auto` just picks the model's declared dtype.
- **"Bit-width sets accuracy."** Granularity matters as much: per-group INT4 can beat per-tensor INT8 on some tensors, because a tighter scale means a smaller error bound $s/2$.

## 7 · Interview links

- [Number formats & precision](../interview/number-formats.md) — the high-frequency question this lesson prepares you for: *compare FP16/BF16/FP8/INT8/INT4 by bit layout and the range-vs-precision trade, say why BF16 beat FP16 for DL, and explain why low-bit formats speed up memory-bound decode.*

## 8 · Summary & further reading

**One line:** a number format trades **range** (exponent bits, or the scale) against **precision** (mantissa bits, or the integer width) at a fixed bit budget — BF16 keeps FP32's range for safety, FP8/INT8/INT4 shrink the bytes to buy the memory-bound decode throughput that quantization ([Part 4](../part4/index.md)) then works to preserve.

Further reading:

- Kalamkar et al. — *A Study of BFLOAT16 for Deep Learning Training* — why the 8-bit exponent won.
- OCP / NVIDIA-Arm-Intel — *FP8 Formats for Deep Learning* — the E4M3 / E5M2 spec and their maxima.
- vLLM docs — *Engine arguments* (`--dtype`) and the KV-cache dtype config, baseline v0.26.0.
- The [GPU Hardware](gpu-hardware.md) lesson — why fewer bytes/number maps almost directly to decode throughput.

## 9 · Self-check

??? question "FP16 and BF16 are both 16 bits. What exactly is traded between them, and why does deep learning prefer BF16?"
    They reallocate the same 16 bits: FP16 = 5 exponent / 10 mantissa (fine precision, but max ≈ 65504 → overflows to `inf` beyond that); BF16 = 8 exponent / 7 mantissa (same *range* as FP32, ≈ $3.4\times10^{38}$, but ~8× coarser precision). DL prefers BF16 because model tensors span a huge dynamic range — an overflow to `inf` corrupts the whole computation, while a bit of extra rounding noise is tolerable. Range safety beats precision here.

??? question "Why does quantizing to a lower-bit format speed up LLM *decode* specifically, and roughly by how much for BF16 → INT4?"
    Decode is [memory-bound](gpu-hardware.md): each step's time is set by bytes moved across HBM (weights + KV cache), not FLOPs. Bytes/number $= \text{bits}/8$, so BF16 (16-bit) → INT4 (4-bit) is a **4× cut** in bytes moved per step, which — for a bandwidth-bound workload — flows almost directly into ~4× the decode throughput (illustrative; real gains are lower due to overhead and dequant). The FLOP count barely matters because the GPU was idle waiting on memory anyway.

??? question "What's the difference between E4M3 and E5M2, and why does INT4 usually apply to weights only, not activations?"
    E4M3 (4 exp / 3 mant, max ≈ 448) has more precision; E5M2 (5 exp / 2 mant, max ≈ 57344) has more range — you pick E5M2 when dynamic range matters, E4M3 when precision does, and both need a scaling factor. INT4 gives only 16 levels, so it's fine for **weights** (smooth, bounded distributions) but usually breaks on **activations**, whose per-token *outliers* need range that 16 levels can't cover without huge error — handled by weight-only INT4 or outlier-aware methods in [Part 4](../part4/index.md).
