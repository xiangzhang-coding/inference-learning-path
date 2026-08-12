# Long-context inference: positions, sinks & the KV wall

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 6 · Advanced Inference Topics   ·   **Tests the lesson:** [Long-Context Inference: RoPE Scaling, Attention Sink & the KV Wall](../part6/long-context-inference.md)

---

## Q: Why do models break past their training length, how does RoPE scaling fix it, what is the attention sink, and why is the KV cache — not compute — the long-context ceiling?

### Direct answer

Long-context inference is **two orthogonal problems**: **coherence** (can the model make sense at position 128K?) and **capacity** (does the [KV cache](../part0/kv-cache.md) fit, and can others still run?).

**Why it breaks:** [RoPE](../glossary.md) encodes position as a rotation angle $m\theta_i$ ($\theta_i = \text{base}^{-2i/d}$). A model trained to $L_\text{train}$ has only seen angles up to $L_\text{train}\cdot\theta_i$; past that, low-frequency pairs produce **out-of-distribution** rotations and attention degrades. It's not a hard cutoff — it's angles leaving the trained manifold.

**RoPE scaling** rescales position back into the trained range: **Position Interpolation** (scale positions by $s=L_\text{train}/L_\text{target}$), **NTK-aware** (raise the base), **YaRN** (per-frequency + attention-temperature — the strong default). In vLLM 0.26.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":4.0,...}}'` + `--max-model-len` (**`--rope-scaling` is deprecated**).

**Attention sink:** softmax must sum to 1, so the model dumps surplus attention onto the first few tokens — they're "sinks." Drop them (naive sliding window) and quality collapses; keep **sinks + recent window** for bounded-memory streaming (StreamingLLM).

**Why KV is the ceiling:** KV is **linear in length** — one 128K request can need GBs, crushing [concurrency](../part5/continuous-batching.md) long before FLOPs matter.

### Deep dive

- **The capacity arithmetic.** $\text{KV bytes} = 2\,L\,H_\text{kv}\,d_\text{head}\,\text{seq\_len}\,b$; only `seq_len` scales, linearly. On a 16 GB KV budget a 7B model fits ~73 requests at 4K but only ~2 at 128K.
- **Capacity levers:** **fp8 KV** (`kv_cache_dtype="fp8"`, ~2× requests, "may cause accuracy drop without a proper scaling factor" → `calculate_kv_scales`), **GQA** (fewer $H_\text{kv}$), **sliding window** (cap effective length), **[chunked prefill](../part5/scheduler-chunked-prefill-pd.md)** (don't let a 128K prefill freeze decode).
- **Coherence ≠ capacity.** YaRN makes the model *understand* position 128K but costs nothing back on memory; fp8 makes it *fit* but doesn't help comprehension. You need both.
- **Advertised ≠ effective.** Passing needle-in-haystack ≠ multi-hop reasoning across the whole window.

### Code

The capacity wall as arithmetic (Qwen2.5-7B shapes, GQA):

```python
LAYERS, KV_HEADS, HEAD_DIM, KV_GB = 28, 4, 128, 16
per_tok = lambda b: 2*LAYERS*KV_HEADS*HEAD_DIM*b          # KV bytes/token, all layers
conc    = lambda n, b: (KV_GB*1024**3)//(per_tok(b)*n)    # max concurrent seqs of length n
for n in [4096, 32768, 131072]:
    print(n, f"{per_tok(2)*n/1024**3:.2f} GB/req", "conc", conc(n,2), "→ fp8", conc(n,1))
# 4096 0.22 GB/req conc 73 → fp8 146 | 131072 7.00 GB/req conc 2 → fp8 4
```

### Interviewer follow-ups

- *"Is long context a compute or memory problem?"* → **Memory.** KV is linear in length and caps concurrency; FLOPs are secondary. Size the KV budget first.
- *"`--max-model-len 128000` alone — enough?"* → No. Without a RoPE scaling override the model runs into untrained positions → garbage. Need the scaling config *and* the length.
- *"Why not just keep the last N tokens?"* → Attention sinks. The first tokens absorb surplus softmax mass; evicting them collapses quality. Keep sinks + window.
- *"How do you fit more long requests?"* → fp8 KV (~2×), GQA, sliding window, chunked prefill. Not raising `max_num_seqs` — the block pool binds, not the seq cap.
- *"Is `--rope-scaling` how you configure YaRN in 0.26.0?"* → No — deprecated. Use `--hf-overrides` with `rope_parameters` (`rope_type:"yarn"`).

### Linked concepts

- Lesson: [Long-Context Inference](../part6/long-context-inference.md)
- Related: [KV cache & throughput ceiling](kv-cache.md) & [Attention variants: MHA/MQA/GQA](attention-variants.md) (the KV term GQA shrinks), [PagedAttention: block manager & fragmentation](kv-cache-block-manager.md) (the block pool long seqs starve), [Chunked prefill & PD disaggregation](chunked-prefill-pd.md) (slicing the giant prefill), [Quantization methods: GPTQ/AWQ/SmoothQuant/FP8](quantization-methods.md) (fp8 KV)
- Glossary: [Long-context inference, RoPE](../glossary.md)
