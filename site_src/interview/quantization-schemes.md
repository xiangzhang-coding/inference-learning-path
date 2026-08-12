# Quantization schemes: granularity, symmetry, what to quantize, PTQ vs QAT

!!! info "Baseline: **vLLM 0.26.0** · `WxAy` naming & `quantization=` flag verified via Context7 (ADR-0004)"

**Module:** Part 4 · Quantization   ·   **Tests the lesson:** [Quantization Choices: Granularity, Symmetry, What to Quantize, and PTQ vs QAT](../part4/quantization-schemes.md)

---

## Q: Walk through the quantization design choices. Per-tensor vs per-channel vs per-group; symmetric vs asymmetric; weight-only vs weight+activation (W4A16 vs W8A8); and why inference uses PTQ.

### Direct answer

Four choices, all shrinking the range each scale must cover (error $\le \text{scale}/2$, $\text{scale}=\text{range}/(2^b-1)$):

- **Granularity** — per-tensor (one scale/matrix) → per-channel (one/row) → per-group (one/~128). Finer isolates outliers to a smaller region, so clean weights get a fine step; cost is more stored scales (higher effective bits). Per-group is standard for INT4 weights.
- **Symmetry** — symmetric ($z=0$, no offset, faster matmul) suits zero-centred **weights**; asymmetric ($z\ne0$) suits skewed **activations** (e.g. all-positive post-ReLU), where symmetric wastes half the grid.
- **What to quantize** (`WxAy`) — **`W4A16`** = 4-bit weights, FP16 activations → weight-only, **memory/decode** win (dequant→FP16 matmul; AWQ/GPTQ). **`W8A8`** = both 8-bit → INT8 tensor cores, **compute** win (helps prefill/large batch) but activation outliers make it hard (SmoothQuant).
- **How** — **PTQ** quantizes a trained model (± calibration), cheap, the inference default; **QAT** simulates quant during training for best low-bit accuracy but needs the training pipeline. Inference focuses on PTQ.

### Deep dive

- **Why finer granularity works.** Error ∝ the range one scale covers. Per-tensor lets one outlier channel coarsen the whole matrix; per-channel/group confines it, giving clean channels a ~10–30× finer step for ~0.1 extra bits/weight (per-group 128).
- **Weights easy, activations hard.** Weights are static and near-symmetric → weight-only quant is easy and popular. Activations are dynamic and heavy-tailed → `W8A8` needs outlier migration (SmoothQuant) or dynamic per-tensor scaling (vLLM's FP8 path scales activations per-forward, no calibration).
- **KV-cache quant is a separate axis** — quantize the stored KV to fit more sequences (helps the [VRAM budget](vram-capacity-planning.md)); orthogonal to weight/activation quant.
- **Calibration.** PTQ often runs a small calibration set to pick ranges (percentile clipping beats naive min/max when outliers exist) — cheap, no gradient updates.

### Code

Granularity's effect on the step size (pure Python):

```python
def step(xs, bits):                                  # asymmetric step = range / (2^b-1)
    return (max(xs) - min(xs)) / ((1 << bits) - 1)

W = [[0.1, -0.2, 0.15, -0.05], [-0.3, 0.25, -0.1, 0.2], [8.0, -0.1, 0.05, 0.2]]
flat = [x for row in W for x in row]
print(f"per-tensor step  {step(flat, 4):.4f}")               # 0.5533 (the 8.0 sets it for all)
print(f"per-channel row0 {step(W[0], 4):.4f}")               # 0.0233 (~24x finer for the clean row)
```

Per-channel gives the clean row a ~24× finer step by not sharing a scale with the `8.0` outlier.

### Interviewer follow-ups

- *"Where would you place AWQ / GPTQ / SmoothQuant / FP8 on these axes?"* → AWQ/GPTQ: weight-only INT4 (`W4A16`), per-group, PTQ. SmoothQuant: `W8A8` (weight+activation INT8) that migrates activation outliers into weights so per-tensor works, PTQ. FP8: often `W8A8`-style with FP8 formats; vLLM's dynamic FP8 quantizes weights per-tensor and scales activations per-tensor per-forward. (Details in the methods lesson.)
- *"Why per-group and not per-element?"* → Per-element would store a scale per weight — no compression. Per-group (~128) captures local outliers for ~0.1 extra bits/weight; it's the accuracy/size sweet spot.
- *"Symmetric or asymmetric for weights? For activations?"* → Symmetric for weights (roughly zero-centred; simpler matmul). Asymmetric for skewed activations (e.g. post-ReLU) so you don't waste half the grid.
- *"Why is `W8A8` harder than `W4A16` even though it's 'less aggressive' on weights?"* → Because it quantizes **activations**, which are dynamic and outlier-prone; naive INT8 activations lose accuracy, needing outlier handling. `W4A16` leaves activations in FP16, sidestepping that.
- *"When QAT over PTQ?"* → Only when PTQ can't hold quality at the target bit-width (e.g. aggressive sub-4-bit); QAT costs a training run.

### Linked concepts

- Lesson: [Quantization Choices: Granularity, Symmetry, What to Quantize, PTQ vs QAT](../part4/quantization-schemes.md)
- Related: [Quantization: why it speeds up inference](quantization-basics.md) (the affine map/error these tune), [VRAM budget & max concurrency](vram-capacity-planning.md) (what KV-cache quant reclaims), [Number formats & precision](number-formats.md) (FP16/INT8/INT4/FP8)
- Glossary: [per-tensor/channel/group, PTQ/QAT, weight-only vs weight+activation](../glossary.md)
