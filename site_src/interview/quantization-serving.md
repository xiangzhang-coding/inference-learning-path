# Quantizing & serving in practice: quantize → serve → validate

!!! info "Baseline: **vLLM 0.26.0** · tooling/flags verified via Context7 (ADR-0004)"

**Module:** Part 4 · Quantization   ·   **Tests the lesson:** [Hands-On: Quantize Qwen2.5-7B to INT4, Serve in vLLM, Compare Quality & Throughput](../part4/quantization-lab.md)

---

## Q: You're asked to ship an INT4 version of Qwen2.5-7B on a single 4090. Walk through quantizing it, serving it, and proving it didn't lose quality. What exactly do you measure, and with what settings?

### Direct answer

Four steps:

1. **Get the INT4 checkpoint** — either a prebuilt one (`Qwen/Qwen2.5-7B-Instruct-AWQ`) or self-quantize with **llm-compressor**: `oneshot` + `GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])` on a small representative calibration set, `save_pretrained(save_compressed=True)`. (This is a one-time offline cost, do it in 无卡 mode.)
2. **Serve** in vLLM by pointing at the checkpoint — vLLM **auto-detects** the method from config (no `--quantization` flag), using the INT4 Marlin kernel.
3. **Validate quality** — A/B the [small eval set](../eval/small.md) on FP16 vs INT4 with **identical greedy settings** (`temperature=0.0`, fixed `seed`) via `LLM.chat` (applies the chat template), and diff **per-category** accuracy.
4. **Measure speed/memory** — `vllm bench throughput` on both, compare **output tokens/s** (decode-heavy shape); note the freed VRAM (~15 GB → ~4–5 GB weights) and optionally add `--kv-cache-dtype fp8` + raise concurrency.

**Deliverable:** a before→after table (VRAM, output tokens/s, per-category accuracy).

### Deep dive

- **Quantization fails quietly.** The model stays fluent while getting more answers wrong, so "looks fine" is worthless — you need a fixed eval and a *number*. Greedy + seed makes FP16 and INT4 directly comparable (the only variable is the weights).
- **Quantize once, serve many.** The quantization pass is a one-time offline cost; the checkpoint is reused for every request. Never quantize per-request.
- **The win is decode, not prefill.** INT4 weight-only cuts the memory-bound weight read → decode throughput. Benchmark a decode-heavy shape; prefill (compute-bound) barely moves because weights are dequantized to FP16 for the matmul.
- **Spend the freed VRAM.** ~10 GB freed → raise `--max-num-seqs` / `--gpu-memory-utilization` (more KV cache → more concurrency), or stack FP8 KV cache. The concurrency gain is opt-in.

### Code

A/B quality with identical greedy settings (reuses the eval scorer):

```python
from vllm import LLM, SamplingParams
from score import load_items, summarize          # from the small eval set

items = load_items("small_eval.jsonl")
convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # greedy + seed => comparable

for tag, model in {"fp16": "Qwen/Qwen2.5-7B-Instruct",
                   "int4": "Qwen/Qwen2.5-7B-Instruct-AWQ"}.items():
    llm = LLM(model=model, max_model_len=4096)   # vLLM auto-detects INT4
    outs = [o.outputs[0].text for o in llm.chat(convos, sp)]
    print(tag, "acc=", round(summarize(items, outs)["accuracy"], 3)); del llm
# illustrative:  fp16 acc= 0.95   int4 acc= 0.90
```

### Interviewer follow-ups

- *"Why greedy with a fixed seed?"* → So a re-run can't differ by chance; any accuracy delta is attributable to the quantization, not sampling noise. It's the only fair A/B.
- *"Why `chat` not `generate`?"* → `chat` applies the Instruct chat template; `generate` feeds a raw prompt, so an Instruct model scores terribly for a reason unrelated to quantization.
- *"Accuracy held overall but `format` dropped from 1.0 to 0.4 — what now?"* → A category collapse means INT4 hurt that capability; back off — INT8 / `W8A8`, a different method, finer granularity, or better calibration. Per-category is why you don't just look at the overall number.
- *"vLLM run — do you pass `--quantization`?"* → Not for a prebuilt/compressed checkpoint; vLLM auto-detects from config. The flag is for in-flight quant (`fp8`, `bitsandbytes`).
- *"How do you make the throughput gain show up?"* → Benchmark a decode-heavy shape and actually use the freed VRAM (higher concurrency / `--max-num-seqs`), else you've bought memory you're not spending.

### Linked concepts

- Lesson: [Hands-On: Quantize Qwen2.5-7B to INT4](../part4/quantization-lab.md)
- Related: [Quantization methods (what to reach for)](quantization-methods.md), [Number formats & precision](number-formats.md), [VRAM budget & max concurrency](vram-capacity-planning.md) (spending the freed memory), [Latency vs throughput metrics](latency-throughput-metrics.md) (what to measure)
- Glossary: [Quantization, PTQ/QAT, KV-cache quantization](../glossary.md)
