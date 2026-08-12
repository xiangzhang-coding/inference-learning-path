# The Tuning Knobs: Sweeping the Throughput/Latency Curve

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Every knob named here is verified against vLLM 0.26.0 via Context7 (ADR-0004): `gpu_memory_utilization` (default **0.92**), `max_num_seqs` (**128**), `max_num_batched_tokens` (**2048**, auto-tuned), `enable_chunked_prefill` (**True**), `enable_prefix_caching` (**on in V1**), `quantization`, `kv_cache_dtype="fp8"`, `enforce_eager` (disables CUDA graphs), `tensor_parallel_size`, `max_model_len`. The §4 map is a **direction map, not a benchmark** — it states which *way* each knob pushes the curve, never a magnitude. The §5 sweep produces numbers, but **they're yours**: per ADR-0004 the author doesn't execute; every figure is an **illustrative / order-of-magnitude reference** you measure on your own box against the [eval set](../eval/index.md).

---

## 1 · Intuition & why it matters

You've met every mechanism ([batching](continuous-batching.md), [paging](paged-attention.md), [chunked prefill](scheduler-chunked-prefill-pd.md), [prefix caching](prefix-caching.md), [spec decoding](speculative-decoding.md)) and the [map](vllm-architecture-map.md) of where they live. This is the payoff lesson: **the knobs that expose those mechanisms, and how to turn them for a real SLO.** "You're serving Qwen on a 4090 and TTFT is too high — what do you change?" is the question all of Part 5 was building toward, and the answer is never one knob — it's knowing which knob moves which end of the throughput↔latency curve, and *measuring* the move.

The one mental shift: **there is no universally 'fast' config — only a config tuned to a target.** Every knob trades one thing for another (throughput for latency, VRAM for quality, TTFT for ITL). So the professional workflow isn't "set the magic values"; it's **sweep**: fix an [eval set](../eval/index.md), change *one* knob, measure the (quality, throughput, latency) triple, and keep the change only if the trade is worth it. This lesson gives you the direction of each knob (so you sweep the *right* one) and the harness to measure the magnitude (because the magnitude is always machine-specific). → see the [Glossary](../glossary.md) for the metric vocabulary (TTFT, TPOT/ITL, throughput, goodput).

## 2 · Mental model

One curve, two ends, and which knob pushes which way:

```text
        THROUGHPUT  ◄─────────────────────────────────────►  LATENCY
        (tokens/s, many concurrent)          (low TTFT / ITL, few concurrent)

  push toward THROUGHPUT →            push toward LATENCY →
    gpu_memory_utilization ↑            max_num_batched_tokens ↓ (smoother ITL)
    max_num_seqs ↑                      enforce_eager = False (keep CUDA graphs)
    max_num_batched_tokens ↑            speculative decoding (single-stream)
    quantization (INT4) → more KV       fewer concurrent requests
    kv_cache_dtype fp8 → more KV        (all "capacity" knobs also cut queueing latency)

  FREE wins (help both, cost ~nothing):
    enable_prefix_caching (on shared prefixes)   quantization (frees VRAM AND speeds decode)

  the SWEEP loop (how you actually find the setting):
    fix eval set → change ONE knob → measure (quality, throughput, latency) → keep if trade worth it
```

Three shapes to hold:

- **Knobs live on one curve; you pick a point, not a "best."** Pushing toward throughput (bigger batch, more KV capacity) usually costs per-request latency at saturation; pushing toward latency (smaller token budget, spec decoding) usually costs aggregate throughput. Name your SLO first, then turn the knob that moves that end.
- **Capacity knobs are the master lever, and some are near-free.** Anything that fits *more KV* — `gpu_memory_utilization ↑`, [quantization](../part4/index.md), [FP8 KV cache](../part4/quantization-methods.md) — raises the concurrency ceiling, which lifts throughput *and* cuts queueing latency. Quantization and prefix caching are the closest to free lunches (they help without a symmetric cost, modulo a little quality).
- **You never tune blind — you sweep against an eval set.** The [measurement loop](../eval/index.md) is the method: baseline → change one knob → re-measure the (quality, throughput, latency) triple → keep or revert. One knob at a time, fixed sampling. The numbers are always yours.

## 3 · Principle — the knobs, grouped by what they move

### 3.1 Capacity knobs (raise the concurrency ceiling → throughput)

These all enlarge the [KV-cache budget](paged-attention.md), fitting a bigger [continuous batch](continuous-batching.md):

- **`gpu_memory_utilization`** (0.92) — fraction of VRAM for the engine; ↑ → more KV blocks → bigger batch. Too high → OOM at startup.
- **`quantization`** (INT4/AWQ) — shrinks weights → frees budget for KV *and* speeds memory-bound decode. Costs a little quality (measure it).
- **`kv_cache_dtype="fp8"`** — halves KV bytes → ~2× KV capacity → more sequences. Costs a little KV precision.
- **`max_model_len`** — caps per-sequence KV; lowering it lets more (shorter-context) sequences fit.

### 3.2 Batch-shape knobs (throughput vs latency balance)

- **`max_num_seqs`** (128) — running-set width ceiling; ↑ → more concurrency, but per-request latency rises once you saturate compute.
- **`max_num_batched_tokens`** (2048, auto) — the [chunked-prefill](scheduler-chunked-prefill-pd.md) dial: ↑ → better TTFT & throughput but worse ITL; ↓ → smoother ITL. The docs suggest >8192 for throughput on small models / big GPUs.
- **`enable_chunked_prefill`** (True) — lets prefill share a step with decode; keeps ITL smooth under long prompts.

### 3.3 Reuse & latency knobs

- **`enable_prefix_caching`** (on) — skips prefill for [shared prefixes](prefix-caching.md); near-free throughput + TTFT on hit-heavy traffic.
- **speculative decoding** ([`speculative_config`](speculative-decoding.md)) — cuts single-stream TPOT at low batch; fades/backfires at high batch.
- **`enforce_eager`** (False) — keep it False to retain [CUDA graphs](../part2/kernel-fusion-cuda-graphs.md) (lower decode latency); set True only to save VRAM/startup at a decode-speed cost.

### 3.4 Scale knob

- **`tensor_parallel_size`** ([TP](../part2/index.md)) — split the model across GPUs; fits bigger models / adds headroom and cuts compute latency, at the cost of cross-GPU communication. Multi-GPU (A100 territory per ADR-0001); on one 4090 it's 1.

### 3.5 The method: sweep, don't guess

No table of "recommended values" survives contact with your model, hardware, and traffic. The durable skill is the **sweep**: pick the knob that moves your target end (§2), vary it across a few values, and measure the (quality, throughput, latency) triple against a fixed [eval set](../eval/index.md) — changing *one* knob at a time with fixed sampling (`temperature=0`, fixed `seed`). Keep the setting whose trade you can defend.

## 4 · Complete runnable code + line-by-line

A pure-Python **direction map**: each knob, which way it pushes throughput and latency, what it trades, and a recommender for a stated goal. It encodes §3 so you sweep the *right* knob — offline, no GPU, and deliberately **no fabricated magnitudes** (those are what the §5 sweep measures).

```python title="tuning_knobs_map.py"
"""Each vLLM knob and which way it pushes the throughput/latency curve.
Pure Python, offline — a direction map, not measured magnitudes (those are yours to measure)."""

# knob: (throughput, latency, what it trades, default)
KNOBS = {
    "gpu_memory_utilization ↑": ("↑ more KV blocks",   "≈ (risk OOM)",   "VRAM headroom",            "0.92"),
    "max_num_seqs ↑":           ("↑ wider batch",      "↑ at saturation","batch width",              "128"),
    "max_num_batched_tokens ↑": ("↑ + better TTFT",    "↑ ITL",          "TTFT vs ITL",              "2048"),
    "quantization INT4/AWQ":    ("↑ frees VRAM",       "↓ per-token",    "some output quality",      "off"),
    "kv_cache_dtype fp8":       ("↑ ~2x KV capacity",  "≈",              "some KV precision",        "auto"),
    "enable_prefix_caching":    ("↑ on shared prefix", "↓ TTFT on hits", "~nothing (V1 default on)", "on"),
    "enforce_eager=True":       ("↓ no CUDA graphs",   "↑ decode",       "saves VRAM/startup",       "off"),
    "tensor_parallel_size ↑":   ("↑ bigger models fit","↓ (+comm cost)", "multi-GPU + comm",         "1"),
}

def recommend(goal):
    """Knobs that push a goal ('throughput' or 'latency') the good way (ignoring their trade)."""
    out = []
    for knob, (thru, lat, *_rest) in KNOBS.items():
        if goal == "throughput" and thru.startswith("↑"): out.append(knob)
        if goal == "latency"    and lat.startswith("↓"):  out.append(knob)
    return out

if __name__ == "__main__":
    print(f"{'knob':<26}{'throughput':<21}{'latency':<17}trades")
    for knob, (thru, lat, trade, _d) in KNOBS.items():
        print(f"{knob:<26}{thru:<21}{lat:<17}{trade}")
    print("\nfor throughput:", recommend("throughput"))
    print("for latency   :", recommend("latency"))
```

**Line-by-line:**

- `KNOBS` — each knob as `(throughput-direction, latency-direction, trade, default)`. The values are **directions** (↑/↓/≈), not numbers — because the *direction* is a property of the mechanism (verifiable reasoning), while the *magnitude* is a property of your box (must be measured). This is the honest half of tuning.
- `recommend(goal)` — filters knobs by which end they push the good way. Note it *ignores the trade* — it tells you the candidates to sweep, not the answer; the trade (and the eval-set measurement) decides which you actually keep.
- `__main__` — prints the table, then the throughput-pushers and the latency-pushers. Some knobs (quantization, prefix caching, TP) appear on *both* lists — those are the closest to free wins.

Expected output (a direction map, not a benchmark):

```text
knob                      throughput           latency          trades
gpu_memory_utilization ↑  ↑ more KV blocks     ≈ (risk OOM)     VRAM headroom
max_num_seqs ↑            ↑ wider batch        ↑ at saturation  batch width
max_num_batched_tokens ↑  ↑ + better TTFT      ↑ ITL            TTFT vs ITL
quantization INT4/AWQ     ↑ frees VRAM         ↓ per-token      some output quality
kv_cache_dtype fp8        ↑ ~2x KV capacity    ≈                some KV precision
enable_prefix_caching     ↑ on shared prefix   ↓ TTFT on hits   ~nothing (V1 default on)
enforce_eager=True        ↓ no CUDA graphs     ↑ decode         saves VRAM/startup
tensor_parallel_size ↑    ↑ bigger models fit  ↓ (+comm cost)   multi-GPU + comm

for throughput: ['gpu_memory_utilization ↑', 'max_num_seqs ↑', 'max_num_batched_tokens ↑', 'quantization INT4/AWQ', 'kv_cache_dtype fp8', 'enable_prefix_caching', 'tensor_parallel_size ↑']
for latency   : ['quantization INT4/AWQ', 'enable_prefix_caching', 'tensor_parallel_size ↑']
```

Read the `latency` list carefully: **quantization, prefix caching, and TP appear on both lists** — they raise throughput *and* cut latency, which is why they're the first things to reach for. Everything else is a genuine trade you must measure. The map tells you *which* knob to sweep; only the sweep tells you *how far*.

## 5 · Lab — run a real sweep against the eval set

!!! gpu "GPU Lab (single-card sweep, runnable)"
    - **Min VRAM:** none to read the map; ~16 GB to run the sweep with `Qwen2.5-7B-Instruct` (AWQ)
    - **Suggested AutoDL card:** RTX 4090 (24 GB); `tensor_parallel_size` sweeps need ≥2 GPUs (A100, ADR-0001)
    - **Est. time / cost:** reading ~20 min (free, no-card mode) · a small sweep ~20–40 min · a few ¥ (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** the sweep logic is backend-agnostic; `kv_cache_dtype` / CUDA-graph support and startup time vary by backend.

The sweep reuses the [eval-set measurement loop](../eval/index.md): fix inputs, change one knob, record the triple. This driver is the honest core — it *orchestrates* runs and prints deltas; the numbers come from your GPU, not this page.

```python title="knob_sweep.py"
# API verified against vLLM 0.26.0 (LLM, SamplingParams). Run on a GPU; numbers are YOURS.
import time
from vllm import LLM, SamplingParams
from score import load_items, summarize      # from the eval-set small-set page

items = load_items("small_eval.jsonl")        # the fixed inputs (Eval Sets, ticket #3)
convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # fixed sampling -> comparable

def measure(**engine_kwargs):
    """One sweep point: build the engine with these knobs, run the eval, return (quality, tok/s)."""
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=4096, **engine_kwargs)
    t0 = time.perf_counter()
    outs = llm.chat(convos, sp)                # chat() applies the template (see Eval Sets)
    dt = time.perf_counter() - t0
    quality = summarize(items, [o.outputs[0].text for o in outs])["accuracy"]
    tok_s = sum(len(o.outputs[0].token_ids) for o in outs) / dt
    return quality, tok_s

# Sweep ONE knob (gpu_memory_utilization) — change nothing else between points.
for gmu in (0.80, 0.90, 0.94):
    q, tps = measure(gpu_memory_utilization=gmu)
    print(f"gpu_memory_utilization={gmu}: accuracy={q:.2%}  throughput={tps:.0f} tok/s (illustrative)")
```

**What to observe / do:**

1. **One knob, three points.** Sweeping `gpu_memory_utilization` upward should fit more KV blocks (watch the startup "# GPU blocks" line rise) and raise throughput — until it OOMs. Accuracy should be *flat* (this knob doesn't touch quality); if it isn't, something else changed.
2. **Swap the knob.** Replace the loop with `quantization` on/off, or `enable_prefix_caching` with a shared-prefix workload, or `max_num_batched_tokens` in (2048, 8192) while watching TTFT vs ITL. Each reproduces one row of §4's map — as *your* numbers.
3. **Keep the discipline.** One knob per sweep, `temperature=0` + fixed `seed`, same eval set before and after — exactly the [eval-set loop](../eval/index.md). A quality delta you can't attribute to one knob is a wasted experiment.

## 6 · Common pitfalls / counter-intuitive points

- **Changing several knobs at once.** Then you can't attribute the result — quality dropped, but was it the INT4 or the higher `gpu_memory_utilization`? Sweep **one** knob at a time; it's slower but it's the only way to learn your curve.
- **Chasing a blog's 'optimal' values.** Someone's `max_num_batched_tokens=16384` was tuned for *their* model/GPU/traffic. Copying magnitudes skips the measurement that makes them meaningful. Copy the *method* (sweep), not the numbers.
- **`gpu_memory_utilization=1.0`.** Leaves no headroom for activation spikes / allocator fragmentation → OOM. Push up in small steps and watch.
- **Optimizing throughput when you're latency-bound (or vice-versa).** Raising `max_num_seqs` boosts throughput but *worsens* per-request latency at saturation — the wrong move if users complain about slow responses. Name the SLO first.
- **Forgetting quality in the triple.** Throughput and latency aren't the whole story — a knob that speeds things up but tanks accuracy (aggressive quant, tiny `max_model_len` truncating prompts) is a regression. Always measure the (quality, throughput, latency) *triple* against the eval set.
- **Non-deterministic sweeps.** `temperature>0` makes re-runs differ by chance, so a "regression" may be noise. Fix sampling (`temperature=0`, `seed`) for every comparison.
- **Leaving `enforce_eager=True` in production.** It's a debugging/memory-saving flag; it disables CUDA graphs and *raises* decode latency. Don't ship it unless you truly need the VRAM.

## 7 · Interview links

- [Tuning knobs: which one for which SLO](../interview/tuning-knobs.md) — the high-frequency question this lesson prepares you for: *given a TTFT / throughput / OOM problem, name the knob, its direction on the curve, and its trade — and describe the sweep you'd run.*

## 8 · Summary & further reading

**One line:** There's no universally fast config — only one tuned to an SLO, so the durable skill is the **sweep**: know which end of the throughput↔latency curve each knob moves (capacity knobs like `gpu_memory_utilization`/quantization/FP8-KV raise the concurrency ceiling; `max_num_batched_tokens` trades TTFT for ITL; `enforce_eager` and spec decoding touch decode latency; TP scales across GPUs), then fix an eval set, change **one** knob, and measure the (quality, throughput, latency) triple — keeping the change only when the trade is worth it, with every number measured on your own box.

Further reading:

- vLLM `docs/configuration/optimization.md` — the official knob reference and tuning guidance (chunked prefill, `max_num_batched_tokens`).
- The [Eval Sets](../eval/index.md) — the measurement loop and harness this sweep reuses; the [Capstone](../capstone/index.md) is one big before→after sweep.
- Every prior Part 5 lesson — each knob exposes a mechanism: [batching](continuous-batching.md), [paging](paged-attention.md), [chunked prefill](scheduler-chunked-prefill-pd.md), [prefix caching](prefix-caching.md), [spec decoding](speculative-decoding.md); and the [architecture map](vllm-architecture-map.md) says which box each turns.
- [Part 4 Quantization](../part4/index.md) — the biggest capacity knob (weights → KV budget) and the [FP8 KV cache](../part4/quantization-methods.md).

## 9 · Self-check

??? question "TTFT is too high on a single 4090 serving Qwen2.5-7B. Name the knobs you'd consider, their direction, and the sweep you'd run."
    First, TTFT is dominated by prefill. Candidates: (1) **`max_num_batched_tokens ↑`** — lets more of a prompt's prefill run per step → lower TTFT (trade: worse ITL for running streams). (2) **`enable_prefix_caching`** — if prompts share a prefix (system prompt / few-shot), hits skip that prefill entirely → big TTFT drop, ~free. (3) **capacity knobs** (`gpu_memory_utilization ↑`, quantization, FP8 KV) — if TTFT is high because requests are *queued* waiting for KV room (admission), more capacity cuts queueing latency. The **sweep**: fix the [eval set](../eval/index.md) and a fixed-sampling config, change **one** of these at a time across a few values, and record (accuracy, TTFT, throughput). Keep the setting whose TTFT gain is worth its ITL/quality cost — measured on your box, not assumed.

??? question "Which knobs help *both* throughput and latency, and why are they special?"
    **Quantization** (INT4/AWQ), **prefix caching**, and — where available — **FP8 KV cache** and **TP**. They're special because they don't sit on the throughput↔latency *trade*: quantization frees VRAM (→ bigger batch → throughput) *and* shrinks the weight read (→ faster memory-bound decode → latency); prefix caching skips redundant prefill (→ higher throughput *and* lower TTFT on hits); FP8 KV raises capacity (→ throughput) at negligible latency cost; TP splits compute across GPUs (→ lower latency) while adding headroom (→ throughput). They cost something else instead — a little output/KV quality, or communication/hardware — but not the *other* end of the curve. That's why they're the first knobs to reach for, and pure trade-off knobs (`max_num_seqs`, `max_num_batched_tokens`) come after.

??? question "Why is 'set these optimal values' bad advice, and what's the correct workflow?"
    Because the optimum is a function of your **model, hardware, and traffic** — a `max_num_batched_tokens` or `gpu_memory_utilization` that's ideal on someone's A100 with long prompts can be wrong on your 4090 with short ones, and can silently cost quality. Magnitudes don't transfer; only *directions* do. The correct workflow is the **sweep against a fixed eval set**: establish a baseline (quality, throughput, latency) with fixed sampling (`temperature=0`, `seed`), change **one** knob across a few values, re-measure the triple, and keep the change only if its trade is defensible for your SLO — then move to the next knob. You copy the *method*, never the numbers; every figure is measured on your own setup ([ADR-0004](../eval/index.md): the author states none as fact).
