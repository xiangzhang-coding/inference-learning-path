# Number formats & precision

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 0 · Foundations   ·   **Tests the lesson:** [Number Formats: FP16 · BF16 · FP8 · INT8 · INT4](../part0/number-formats.md)

---

## Q: Compare FP16, BF16, FP8, INT8, and INT4 by bit layout and the range-vs-precision trade. Why did BF16 beat FP16 for deep learning? What are E4M3 vs E5M2? And why do low-bit formats speed up decode specifically?

### Direct answer

Every format splits a bit budget between **range** (exponent bits for floats, or the shared scale for ints) and **precision** (mantissa bits, or the integer width):

- **FP16** (1+5+10): fine precision, but max ≈ 65504 → overflows to `inf`.
- **BF16** (1+8+7): FP32's exponent, so FP32's range (≈ $3.4\times10^{38}$), ~8× coarser precision.
- **FP8 E4M3** (1+4+3, max ≈ 448) and **E5M2** (1+5+2, max ≈ 57344): 8-bit floats needing a scaling factor.
- **INT8** / **INT4**: integers reconstructed as $r \approx s(q-z)$; 256 vs 16 levels.

**BF16 beat FP16** because model tensors span a huge dynamic range — an overflow to `inf` corrupts everything, while extra rounding noise is survivable, so keeping FP32's range (BF16) matters more than FP16's extra mantissa bits. **E4M3 vs E5M2**: E4M3 trades range for precision (weights/activations), E5M2 trades precision for range (where dynamic range matters); both need a scale. **Low-bit speeds up decode** because decode is [memory-bound](../part0/gpu-hardware.md) — time is set by bytes moved (bits/8 per number), not FLOPs — so BF16→INT4 is a ~4× cut in bytes/step, flowing almost directly into ~4× decode throughput.

### Deep dive

- **The float formula.** $x = (-1)^s(1+m/2^M)\,2^{e-\text{bias}}$: exponent width sets range ($\sim2^{2^{E-1}}$), mantissa width sets relative precision ($\approx 2^{-M}$). FP16 and BF16 are both 16 bits but reallocate 5/10 vs 8/7 — the whole difference.
- **Integer quant error is bounded.** With symmetric scale $s=\max|w|/(2^{b-1}-1)$, error $\le s/2$. A *smaller* scale (tighter range, or finer [granularity](../glossary.md) — per-tensor → per-channel → per-group) means less error — which is why per-group INT4 can rival per-tensor INT8.
- **Why INT4 is weight-only.** 16 levels suit weights (smooth, bounded) but break on activations, whose per-token *outliers* need range 16 levels can't cover without huge error — hence weight-only INT4 or outlier-aware methods ([Part 4](../part4/index.md)).
- **The memory chain.** bytes/param = bits/8; weights of $N$ params and the [KV cache](../part0/kv-cache.md) both shrink by the same ratio. On a bandwidth-bound phase, fewer bytes ≈ proportionally more throughput — the reason quantization is *the* decode lever.

### Code

The range-vs-precision see-saw on real values (needs `numpy`):

```python
import numpy as np
# 1/3 tests precision; 1e5 tests range
print(float(np.float16(1/3)), float(np.float16(1e5)))   # 0.33325...  inf   <- precise, overflows
# BF16 via truncation (top 16 bits of fp32): FP32's range, coarse mantissa
u = (np.float32(1e5).view(np.uint32) >> 16) << 16
print(float(u.view(np.float32)))                        # 99840.0            <- has the range
```

### Interviewer follow-ups

- *"Model runs BF16 but you set `--kv-cache-dtype fp8`. Legal? Why do it?"* → Yes — compute dtype (`--dtype`) and KV-cache storage dtype (`--kv-cache-dtype`) are independent knobs. FP8 KV cache ~halves the per-token KV footprint, fitting more concurrent sequences in the same VRAM. Caveat: vLLM warns it "may cause accuracy drop without a proper scaling factor."
- *"Why can't you just use INT4 for everything?"* → Activations have outliers that blow 16 levels; naive low-bit activations fall off an accuracy cliff. Weight-only INT4, per-group scales, or outlier handling (SmoothQuant) are needed — [Part 4](../part4/index.md).
- *"FP16 or BF16 for a model with large activation magnitudes?"* → BF16 — FP16 would overflow past 65504. Range safety beats precision when magnitudes are large.
- *"Does halving bits always halve latency?"* → Only for the memory-bound part, and minus overhead (dequant, non-tensor ops). Decode benefits most (bandwidth-bound); a compute-bound prefill benefits less directly.

### Linked concepts

- Lesson: [Number Formats: FP16 · BF16 · FP8 · INT8 · INT4](../part0/number-formats.md)
- Related lesson: [GPU Hardware Mental Model](../part0/gpu-hardware.md) (why fewer bytes → more decode throughput)
- Glossary: [FP8 / INT8 / INT4, Per-tensor / per-channel / per-group, KV-cache quantization](../glossary.md)
