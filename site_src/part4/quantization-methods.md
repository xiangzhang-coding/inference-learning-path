# Quantization Method Families: GPTQ, AWQ, SmoothQuant, FP8, LLM.int8(), KV-cache

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The tooling named here — **llm-compressor** (`oneshot` + `GPTQModifier(scheme="W4A16"/"W8A8")`, `SmoothQuantModifier`), vLLM's auto-detection of pre-quantized checkpoints, and `kv_cache_dtype="fp8"` — is verified against vLLM 0.26.0 via Context7 (ADR-0004). **AutoAWQ is deprecated**; its functionality moved into llm-compressor (verified). The method→design-space table in §4 is a **classification, not a computation** (pure-Python, offline). Accuracy/speed figures are **illustrative / order-of-magnitude references**; the hands-on run is the [next lesson](quantization-lab.md).

---

## 1 · Intuition & why it matters

The [schemes lesson](quantization-schemes.md) gave you four axes — granularity, symmetry, what-to-quantize (`W4A16` vs `W8A8`), and PTQ vs QAT. A named method (GPTQ, AWQ, SmoothQuant, FP8, LLM.int8()) is just **a specific point in that space, plus one clever trick for keeping accuracy at low bits**. Once you see them that way, you don't memorize six methods — you place each on the axes you already know and reason about its accuracy/speed profile. That's exactly the interview skill: "walk me through AWQ vs GPTQ vs SmoothQuant" is a placement question, not a trivia question.

The one framing to carry: **every method is fighting the same enemy from the [basics lesson](quantization-basics.md) — the outlier that inflates the step.** They differ only in *how*. GPTQ corrects the rounding error layer-by-layer; AWQ protects the weights that matter most to the output; SmoothQuant moves activation outliers into the weights where they're easier to quantize; FP8 uses a format with more dynamic range; LLM.int8() keeps the few outlier dimensions in FP16. Same problem, five tricks — plus **KV-cache quantization**, which applies the whole idea to a different tensor (the [KV cache](../part0/kv-cache.md)) instead of the weights. → see the [Glossary](../glossary.md) for the method names.

## 2 · Mental model

The six families, placed on the axes you already know:

```text
                 W-bits  A-bits  granularity        the trick                       primary win
  GPTQ            4 (8)   16      per-group          layer-wise error correction     memory / decode
  AWQ             4       16      per-group          scale by activation salience    memory / decode
  SmoothQuant     8       8       per-tensor/chan    migrate act. outliers → weights compute / prefill+batch
  FP8 (E4M3)      8       8       per-tensor         float format = more range       compute+memory (Hopper/Ada)
  LLM.int8()      8       8       per-chan+FP16 outl keep outlier dims in FP16        memory (accuracy-safe INT8)
  KV-cache FP8    —       —       (the KV tensor)    quantize K/V, not weights        memory (longer ctx / more seqs)

  weight-only (W4A16): AWQ, GPTQ  ── decode/memory play, activations stay FP16, easy
  weight+activation (W8A8): SmoothQuant, FP8, LLM.int8() ── compute play, must tame activation outliers
```

Two shapes to hold:

- **Weight-only vs weight+activation splits the field.** AWQ/GPTQ are `W4A16` — a memory/decode win, easy because weights are static and near-symmetric. SmoothQuant/FP8/LLM.int8() are `W8A8` — a compute win (INT8/FP8 tensor cores) that must handle *activation* outliers, which is the hard part each solves differently.
- **KV-cache quantization is a separate lever.** It's orthogonal to weight/activation quant — you can stack FP8 KV cache on top of an INT4-weight model. It buys memory for *longer context / more concurrent sequences* (the [VRAM budget](../interview/vram-capacity-planning.md)), not weight-read speed.

## 3 · Principle — the six families

### 3.1 GPTQ — layer-wise error correction (weight-only)

GPTQ quantizes weights one layer at a time, and after rounding each column it **updates the remaining columns to compensate** for the error introduced (a second-order / Hessian-based correction using a small calibration set). The result is INT4 weights (`W4A16`, per-group) that track the FP16 output far better than naive rounding. Cost: calibration + the error-correction pass (minutes). It's a memory/decode play — activations stay FP16.

### 3.2 AWQ — activation-aware weight protection (weight-only)

AWQ's insight: not all weights matter equally — the ones multiplying large-magnitude activation channels dominate the output. It measures activation magnitudes on a calibration set and **scales those salient weight channels** before quantizing so they survive INT4 with less error (folding the inverse scale into the next op). Also `W4A16`, per-group, PTQ. In practice AWQ and GPTQ are the two dominant INT4 weight-only methods; both are excellent for decode-bound serving.

### 3.3 SmoothQuant — migrate outliers to make W8A8 work

The blocker for `W8A8` is that **activations have big per-channel outliers** (weights don't). SmoothQuant applies a per-channel scaling that **shifts the "difficulty" from activations into weights** — divide activations by a factor $s$ and multiply the corresponding weights by $s$ (mathematically identity) — so both become quantization-friendly. Now INT8 activations work, the matmul runs on INT8 tensor cores, and you get a *compute* speedup for prefill/large batches. PTQ, per-tensor/channel.

### 3.4 FP8 — a float format with more dynamic range

FP8 (usually **E4M3**: 4 exponent, 3 mantissa bits) is `W8A8` in a *floating-point* 8-bit format. Because it has an exponent, it covers a much wider dynamic range than INT8 at the same bit count — so it tolerates outliers better and often needs **no calibration** (activations scaled dynamically per-tensor each forward pass, as in vLLM's `--quantization fp8`). It runs on FP8 tensor cores (Hopper/Ada), giving a compute *and* memory win. The modern default when the hardware supports it.

### 3.5 LLM.int8() — mixed-precision outlier decomposition

LLM.int8() (the bitsandbytes method) makes INT8 accuracy-safe by **splitting the matmul**: the few activation dimensions with outliers are kept in **FP16** and computed separately, while the vast majority run in INT8; the two partial results are summed. So it's `W8A8`-ish but with an FP16 escape hatch for the ~0.1% of dimensions that would otherwise wreck the quantization. Prioritizes accuracy over raw speed.

### 3.6 KV-cache quantization — the other tensor

Everything above quantizes *weights* (and sometimes activations). **KV-cache quantization** instead compresses the stored [KV cache](../part0/kv-cache.md) — e.g. FP8 K/V via vLLM's `kv_cache_dtype="fp8"` (no calibration; scales default to 1.0). It doesn't speed up the weight read; it **frees VRAM**, which buys longer context or more concurrent sequences. Orthogonal — stack it on top of weight quantization.

## 4 · Complete runnable code + line-by-line

A pure-Python placement of each method on the [schemes-lesson](quantization-schemes.md) axes, plus a selector — offline, no GPU. It turns "six methods to memorize" into "one table you can regenerate."

```python title="method_families.py"
"""Place each quantization method on the design-space axes (from the schemes lesson).
Pure Python, offline — a classification, not a computation."""

# name: (weight_bits, act_bits, granularity, calibration, primary_win)
METHODS = {
    "GPTQ":        (4, 16, "per-group",                 "yes (Hessian-based)",    "memory / decode"),
    "AWQ":         (4, 16, "per-group",                 "yes (activation-aware)", "memory / decode"),
    "SmoothQuant": (8,  8, "per-tensor/channel",        "yes (migrate outliers)", "compute / prefill+batch"),
    "FP8 (E4M3)":  (8,  8, "per-tensor",                "no (dynamic act.)",      "compute + memory (Hopper/Ada)"),
    "LLM.int8()":  (8,  8, "per-channel + FP16 outliers","no (runtime)",          "memory (accuracy-safe INT8)"),
}

def recommend(goal):
    """Which methods target a given goal (substring of the primary win)?"""
    return [name for name, (wb, ab, *_rest, win) in METHODS.items() if goal in win]

if __name__ == "__main__":
    for name, (wb, ab, gran, calib, win) in METHODS.items():
        print(f"{name}: W{wb}A{ab}, {gran}, calibration={calib}, win={win}")
    print()
    print("for 'decode':", recommend("decode"))     # weight-only INT4 methods
    print("for 'compute':", recommend("compute"))   # weight+activation methods
```

**Line-by-line:**

- `METHODS` — the six families as `(weight_bits, act_bits, granularity, calibration, primary_win)`. Reading a row *is* placing the method: `W4A16` (AWQ/GPTQ) = weight-only/decode; `W8A8` (SmoothQuant/FP8/LLM.int8()) = weight+activation/compute.
- `recommend(goal)` — filters by the `primary_win` field; `wb, ab, *_rest, win` unpacks the tuple, ignoring the middle. It's how you'd answer "what should I use to speed up decode?" from the framework rather than memory.
- `__main__` — prints the table, then the two canonical selections (decode → weight-only; compute → weight+activation).

Expected output (a classification table, not a benchmark):

```text
GPTQ: W4A16, per-group, calibration=yes (Hessian-based), win=memory / decode
AWQ: W4A16, per-group, calibration=yes (activation-aware), win=memory / decode
SmoothQuant: W8A8, per-tensor/channel, calibration=yes (migrate outliers), win=compute / prefill+batch
FP8 (E4M3): W8A8, per-tensor, calibration=no (dynamic act.), win=compute + memory (Hopper/Ada)
LLM.int8(): W8A8, per-channel + FP16 outliers, calibration=no (runtime), win=memory (accuracy-safe INT8)

for 'decode': ['GPTQ', 'AWQ']
for 'compute': ['SmoothQuant', 'FP8 (E4M3)']
```

The selector's answers fall straight out of the axes: **decode/memory → the `W4A16` weight-only methods** (AWQ, GPTQ); **compute → the `W8A8` methods** (SmoothQuant, FP8). You never needed to memorize which is which — the design space told you.

## 5 · Lab — a method is a recipe

You don't hand-implement these — you hand a **recipe** to a tool. The modern, vLLM-endorsed tool is **llm-compressor** (AutoAWQ is deprecated into it). Reading these verified recipes shows how a method name becomes a config; *running* them is the [next lesson](quantization-lab.md), so this is a no-GPU reading lab:

```python title="recipes.py"
# llm-compressor recipes — each is a method placed on the axes, as code.
# (API verified against the vLLM 0.26.0 quantization docs; run it in the next lesson.)
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# INT4 weight-only (AWQ/GPTQ territory): W4A16, per-group, never touch lm_head.
w4a16 = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# INT8 weight+activation: SmoothQuant tames activation outliers, then GPTQ does W8A8.
w8a8 = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
# FP8 (no calibration) is even simpler — vLLM can do it in-flight: quantization="fp8".
print("W4A16 scheme:", w4a16.scheme)         # -> W4A16   (weight-only, the decode play)
print("W8A8 stages :", [type(m).__name__ for m in w8a8])  # SmoothQuant then GPTQ
```

**What to observe:** the `scheme="W4A16"` / `"W8A8"` strings are literally the axes from the last lesson — the tool speaks the same language. `ignore=["lm_head"]` is the universal "don't quantize the output projection" rule (it's tiny and precision-sensitive). The `W8A8` recipe is *two* stages — SmoothQuant first (migrate outliers) then GPTQ (round) — because activation quantization needs the outlier fix; the `W4A16` recipe is one stage because weight-only doesn't. That structural difference *is* §3.3. In the [next lesson](quantization-lab.md) you'll hand a recipe like `w4a16` to `oneshot(...)`, save the checkpoint, serve it in vLLM, and measure what it cost you.

## 6 · Common pitfalls / counter-intuitive points

- **Treating the methods as a ranking.** There's no "best" — AWQ/GPTQ win for decode-bound INT4 serving; SmoothQuant/FP8 win when you need compute (prefill/large batch); LLM.int8() when INT8 accuracy is paramount. Match the method to the bottleneck.
- **Using AutoAWQ for new work.** It's deprecated; its functionality lives in llm-compressor now. Reach for llm-compressor recipes (or a prebuilt checkpoint), not the old AutoAWQ path.
- **Forgetting FP8 needs the hardware.** FP8 tensor-core acceleration is Hopper/Ada (compute capability ≥ 8.9); on older GPUs FP8 may be emulated or unsupported. INT4 weight-only (AWQ/GPTQ) is the portable memory play.
- **Confusing KV-cache quant with weight quant.** They target different tensors and different wins: weight quant speeds the weight read (decode); KV-cache quant frees VRAM (longer context / more sequences). Stack them.
- **Skipping calibration domain.** GPTQ/AWQ/SmoothQuant calibrate on a small dataset — if it's wildly off-distribution from your traffic, the chosen scales fit the wrong ranges. Use representative prompts.
- **Quantizing `lm_head`.** The output projection is small and precision-sensitive; every recipe here `ignore`s it. Quantizing it is a classic accuracy own-goal.

## 7 · Interview links

- [Quantization methods: GPTQ vs AWQ vs SmoothQuant vs FP8 vs LLM.int8()](../interview/quantization-methods.md) — the high-frequency question this lesson prepares you for: *place each method on the axes, say which trick it uses, and pick one for a given bottleneck.*

## 8 · Summary & further reading

**One line:** Each quantization method is a point in the design space plus one anti-outlier trick — GPTQ (layer-wise error correction), AWQ (protect salient weights) are `W4A16` decode/memory plays; SmoothQuant (migrate outliers), FP8 (float range), LLM.int8() (FP16 outlier dims) are `W8A8` compute plays; and KV-cache FP8 is an orthogonal memory lever — so you place a method rather than memorize it, and hand its recipe to llm-compressor.

Further reading:

- The [Quantization Basics](quantization-basics.md) and [Schemes](quantization-schemes.md) lessons — the outlier problem and the four axes every method is built on.
- *GPTQ*, *AWQ*, *SmoothQuant*, *LLM.int8()* — the original papers; each is one row of §4 in depth.
- llm-compressor docs (the recommended tool) and vLLM's quantization guide — the recipes and supported formats.
- Next: [Hands-On — Quantize Qwen2.5-7B to INT4](quantization-lab.md) — turn a recipe into a served model and measure quality vs throughput.

## 9 · Self-check

??? question "Place AWQ and SmoothQuant on the design-space axes. Which speeds up decode, which speeds up prefill, and why?"
    **AWQ** is `W4A16` — weight-only INT4, per-group, PTQ; its trick is scaling *salient* weight channels (those multiplying large activations) so they survive INT4. Because activations stay FP16 and weights are just dequantized before an FP16 matmul, its win is **HBM bandwidth** → it speeds up **memory-bound decode**. **SmoothQuant** is `W8A8` — it migrates activation outliers into weights so INT8 *activations* become quantizable, letting the matmul run on **INT8 tensor cores**. That's a **compute** win → it helps **compute-bound prefill and large batches**. Different axis (what-to-quantize), different bottleneck.

??? question "Why does `W8A8` need a trick like SmoothQuant (or LLM.int8()'s FP16 split) while `W4A16` doesn't?"
    Because `W8A8` quantizes **activations**, and activations have large per-channel **outliers** that a single INT8 scale can't represent without huge error (the outlier inflates the step for the whole channel). SmoothQuant fixes this by scaling activations down and weights up (an identity transform) so both are quantization-friendly; LLM.int8() instead keeps the ~0.1% outlier dimensions in FP16 and only quantizes the rest. `W4A16` leaves activations in FP16 entirely, so it never faces the activation-outlier problem — only weights (which are static and near-symmetric) get quantized, and per-group granularity handles their milder outliers.

??? question "When would you add FP8 KV-cache quantization, and what does it buy that INT4 weight quantization doesn't?"
    Add FP8 KV-cache quant (`kv_cache_dtype="fp8"`) when you're **memory-bound on the KV cache** — long contexts or many concurrent sequences — because it halves KV bytes vs FP16 and frees VRAM for more sequences / longer context (raising the [concurrency ceiling](../interview/vram-capacity-planning.md)). It's *orthogonal* to weight quantization: INT4 weights cut the **weight** read (speeding decode's weight traffic), while FP8 KV cuts the **KV** footprint (capacity, not weight speed). You stack them — INT4 weights + FP8 KV — to win on both weight bandwidth and KV capacity at once.
