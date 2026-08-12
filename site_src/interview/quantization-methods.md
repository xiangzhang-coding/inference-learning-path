# Quantization methods: GPTQ vs AWQ vs SmoothQuant vs FP8 vs LLM.int8()

!!! info "Baseline: **vLLM 0.26.0** · tooling/flags verified via Context7 (ADR-0004)"

**Module:** Part 4 · Quantization   ·   **Tests the lesson:** [Quantization Method Families](../part4/quantization-methods.md)

---

## Q: Compare the main quantization methods — GPTQ, AWQ, SmoothQuant, FP8, LLM.int8(). Place each on the design-space axes, name its trick, and pick one for a given bottleneck.

### Direct answer

Each method is a point in the [design space](../part4/quantization-schemes.md) (bits, granularity, what-to-quantize) plus one anti-outlier trick:

| Method | Scheme | Trick | Primary win |
|---|---|---|---|
| **GPTQ** | W4A16, per-group | layer-wise **error correction** (Hessian-based) | memory / **decode** |
| **AWQ** | W4A16, per-group | scale **salient** weight channels (activation-aware) | memory / **decode** |
| **SmoothQuant** | W8A8 | **migrate** activation outliers → weights | compute / **prefill+batch** |
| **FP8 (E4M3)** | W8A8 | float format = more **dynamic range** (often no calibration) | compute+memory (**Hopper/Ada**) |
| **LLM.int8()** | W8A8 | keep **outlier dims in FP16**, rest INT8 | memory (**accuracy-safe** INT8) |

**Selection:** decode-bound serving (the common case) → **AWQ or GPTQ** INT4 (weight-only, big memory/decode win, easy). Compute-bound prefill / large batch → **SmoothQuant or FP8** (INT8/FP8 tensor cores). Hopper/Ada hardware → **FP8** (best modern default). Accuracy-critical INT8 → **LLM.int8()**. Orthogonally, **FP8 KV cache** for long context / more sequences.

### Deep dive

- **Weight-only (W4A16) vs weight+activation (W8A8) is the primary split.** AWQ/GPTQ leave activations FP16 → memory/decode win, easy (weights are static, near-symmetric). SmoothQuant/FP8/LLM.int8() quantize activations → compute win but must tame activation **outliers** — which is the trick each contributes.
- **GPTQ vs AWQ.** Both are INT4 weight-only, per-group, PTQ with calibration; GPTQ corrects rounding error layer-by-layer, AWQ protects the weights tied to large activations. In practice comparable quality; both are standard for INT4.
- **Why FP8 tolerates outliers.** An exponent gives FP8 far more dynamic range than INT8 at 8 bits, so it represents outliers without a per-channel migration — hence often no calibration (dynamic per-tensor scaling). Needs FP8 tensor cores (Hopper/Ada).
- **Tooling.** llm-compressor is the current path (`GPTQModifier(scheme="W4A16"/"W8A8")`, `SmoothQuantModifier`); **AutoAWQ is deprecated** into it. vLLM **auto-detects** a prebuilt checkpoint's method from config.

### Code

The methods as a design-space table (pure Python):

```python
METHODS = {  # name: (W-bits, A-bits, primary_win)
    "GPTQ": (4, 16, "memory / decode"), "AWQ": (4, 16, "memory / decode"),
    "SmoothQuant": (8, 8, "compute / prefill+batch"),
    "FP8 (E4M3)": (8, 8, "compute + memory (Hopper/Ada)"),
    "LLM.int8()": (8, 8, "memory (accuracy-safe INT8)"),
}
decode = [n for n, (w, a, win) in METHODS.items() if "decode" in win]
print(decode)   # ['GPTQ', 'AWQ']  <- weight-only INT4, the decode play
```

### Interviewer follow-ups

- *"Decode-bound serving on a 4090 — which method?"* → INT4 weight-only, **AWQ or GPTQ** (`W4A16`): activations stay FP16, weight bytes drop ~4×, decode speeds up; easy and well-supported.
- *"Why can't you just INT8 the activations naively?"* → Activations have large per-channel outliers a single INT8 scale can't hold. SmoothQuant migrates them into weights; LLM.int8() keeps outlier dims in FP16; FP8 uses an exponent for range. That's the whole reason `W8A8` methods exist.
- *"When FP8 over INT4?"* → On Hopper/Ada when you also want a *compute* speedup (prefill/large batch) and near-lossless quality with no calibration — FP8 is `W8A8` on FP8 tensor cores. INT4 is the portable memory/decode play on any Ampere+.
- *"What does KV-cache quant add?"* → Orthogonal memory: FP8 KV (`kv_cache_dtype="fp8"`) frees VRAM for longer context / more sequences; stack it on INT4 weights.
- *"Which tool, and is AutoAWQ still it?"* → llm-compressor (recipes = schemes). AutoAWQ is deprecated into llm-compressor; use recipes or a prebuilt checkpoint.

### Linked concepts

- Lesson: [Quantization Method Families](../part4/quantization-methods.md)
- Related: [Quantization schemes (the axes)](quantization-schemes.md), [Quantization basics (the outlier problem)](quantization-basics.md), [Quantizing & serving in practice](quantization-serving.md), [VRAM budget](vram-capacity-planning.md) (what KV-cache quant reclaims)
- Glossary: [GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()](../glossary.md)
