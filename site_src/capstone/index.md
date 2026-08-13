# Capstone: Max Out Qwen2.5-7B Throughput on One 4090 — the Before → After Report

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Every flag this Capstone assembles is verified against vLLM 0.26.0 via Context7 (ADR-0004): the prebuilt AWQ checkpoint (`Qwen/Qwen2.5-7B-Instruct-AWQ`, quantization **auto-detected**, no `--quantization` flag), `--kv-cache-dtype fp8`, `--gpu-memory-utilization` (default **0.92**), `--max-num-seqs` (**128**), `--max-num-batched-tokens` (**2048**, auto-tuned), `--enable-chunked-prefill` (**True**), `--enable-prefix-caching` (resolved from model config), `--max-model-len`, `--tensor-parallel-size` (**1** on one card), plus **`vllm bench throughput`** and **`vllm bench serve`**. **The author does not run any of this** (ADR-0004): every VRAM / tokens-per-second / req/s / accuracy number below is an **illustrative / order-of-magnitude reference**. The whole point of the Capstone is that the real before→after numbers are the ones **you** measure on your own AutoDL box.

---

## 1 · Intuition & why it matters

Everything in Parts 0–8 was a knob. The Capstone is where you turn them — in order, one at a time — on a single 4090, and produce the one artifact that actually gets you the job: a **before → after report** that says *what you changed, how much it moved, and how you know it didn't break the model.*

The trap beginners fall into is "I made it fast." That's not an engineering claim — it's a vibe. The professional claim is a **table with a baseline**: FP16 did *X* output tokens/s at *Y* accuracy; after AWQ + FP8 KV + capacity tuning it does *3.2X* at *Y − 0.02* accuracy, at a p99 TTFT still under the SLO. Every number attributable to exactly one change. That table is what an interviewer means by "walk me through an optimization you did," and it's what this project builds.

The one discipline to internalize — it's the same [sweep loop](../part5/tuning-knobs-sweep.md) from Part 5, now run end-to-end: **baseline first, change ONE thing, measure the (quality, throughput, latency) triple, attribute the delta, keep or revert.** The mechanisms are already yours (quantization, PagedAttention, continuous batching, prefix caching, the knobs); the Capstone is the *method* that turns them into a defensible result. → see the [Glossary](../glossary.md) for the metric vocabulary.

## 2 · Mental model

Two things to hold: an **ordered ladder** of optimizations, and a **measurement spine** that runs at every rung.

```text
  THE LADDER (climb in this order — biggest, safest levers first)     THE SPINE (run at EVERY rung)
                                                                     ┌───────────────────────────┐
  rung 0  BASELINE           FP16 Qwen2.5-7B, stock config           │ 1. quality  (eval A/B,     │
             │               ── measure everything here ──           │    greedy + seed, per-cat) │
             ▼                                                        │ 2. throughput (output      │
  rung 1  QUANTIZE           AWQ INT4 weights (auto-detected)         │    tok/s, vllm bench)      │
             │               frees ~10 GB, speeds memory-bound decode │ 3. latency/knee (p99 TTFT/ │
             ▼                                                        │    TPOT, bench serve sweep)│
  rung 2  FP8 KV CACHE       --kv-cache-dtype fp8 → ~2× KV capacity   └───────────────────────────┘
             │                                                                    │
             ▼                                                          for each rung:
  rung 3  SPEND THE VRAM     --gpu-memory-utilization ↑, --max-num-seqs ↑         change ONE thing
             │               (freed VRAM → bigger continuous batch)               ▼
             ▼                                                          measure the TRIPLE
  rung 4  PREFIX CACHING     --enable-prefix-caching (if shared prefix)           ▼
             │                                                          attribute the delta
             ▼                                                                    ▼
  rung 5  BATCH-SHAPE TUNE   --max-num-batched-tokens (TTFT ↔ ITL)      keep it? (quality GATE)
             │                                                                    ▼
             ▼                                                          revert if it fails the gate
  rung 6  (multi-GPU: TP)    OUT of single-4090 scope — noted, not done here (Part 7 / A100)
             │
             ▼
        FINAL REPORT         baseline row → one row per kept rung → totals
```

Three shapes to keep:

- **Order matters, and it's not arbitrary.** [Quantization](../part4/quantization-lab.md) goes first because it's the biggest, most-free lever — it lifts *both* gates (frees VRAM for more [KV cache](../part2/kv-cache-math.md) *and* speeds [memory-bound decode](../part2/roofline-analysis.md)). You climb from "helps both ends, cheap" down to "genuine trade you must tune," so each rung stands on the freed resources of the one below it. Spending VRAM on batch width (rung 3) only makes sense *after* quantization freed it (rungs 1–2).
- **The spine is the same at every rung — that's what makes deltas attributable.** Same [eval set](../eval/small.md), same fixed sampling (`temperature=0`, fixed `seed`), same benchmark shape. Change the eval or the shape between rungs and your before→after table is comparing apples to oranges.
- **Quality is a gate, not a column you fill in and forget.** A rung that raises throughput but fails the eval A/B (or collapses one category) is **reverted or backed off** — that decision, made out loud, is the most senior thing in the whole report. "I tried FP8 KV, math accuracy dropped 8 points, so I kept BF16 KV" is a *better* answer than a bigger throughput number.

## 3 · Principle — the method

The Capstone teaches no new mechanism; it composes the ones you have with a rigorous method. Five rules.

### 3.1 Baseline first — a delta needs a "before"

You cannot report an improvement without the number you improved *from*. Rung 0 is the un-optimized model measured on all three axes: FP16 `Qwen2.5-7B-Instruct`, stock config, on the [eval set](../eval/small.md) and `vllm bench`. Everything after is a delta against this row. Skipping the baseline is the single most common way a "3× speedup" turns out to be unfalsifiable.

### 3.2 The three axes, measured the same way every time

- **Quality** — the A/B from the [quantization lab](../part4/quantization-lab.md): run the [small eval set](../eval/small.md) with `temperature=0.0` + fixed `seed` via `LLM.chat`, compare **per-category** accuracy. Greedy + seed is what makes a re-run comparable; per-category is what tells you *what* a rung broke.
- **Throughput** — **output tokens/s** from `vllm bench throughput` (or inline timing), on a fixed decode-heavy shape. Output tok/s is the decode number the whole stack is trying to move.
- **Latency / knee** — **p99 TTFT and TPOT** at your SLO, found by sweeping `vllm bench serve --request-rate` upward (the [load-testing lesson](../part8/load-testing-knee.md)). The knee — the highest offered load still meeting the SLO — is the honest capacity number, not raw throughput.

### 3.3 One change at a time, and attribute it

Change exactly one knob per rung. If you flip AWQ *and* raise `gpu-memory-utilization` in the same step and throughput doubles, you've learned nothing about which one did it — and if quality dropped, you can't tell which to blame. This is slower and it is the only way the report means anything.

### 3.4 The quality gate

At every rung, the eval A/B is a **pass/fail gate**, not decoration. A small overall accuracy drop is expected and fine; a *category collapse* (e.g. `math` 1.0 → 0.3, or the `bilingual` items tanking — important for a Chinese-capable base model) is a **revert** signal. When a rung fails the gate you back off: more bits, keep BF16 KV, different calibration — and you **write down that you did.** The report's credibility comes from its reverts as much as its wins.

### 3.5 Budget discipline — most of this is free

The ¥500 budget (ADR-0001) is generous *if* you don't burn GPU time on things that don't need a GPU. Do all downloads and any self-quantization in AutoDL **无卡 mode** (no-GPU, ~free). Open a GPU only to measure — the eval A/B and the `vllm bench` runs. A full before→after report is a few tens of minutes of GPU time, single digits of ¥ — the budget is for iterating, not for a single run.

## 4 · Complete runnable code + line-by-line

The deliverable is a report, so the code is a **report generator**: measure one config's (quality, throughput) on the fixed eval set, then drive the ladder and emit a Markdown table. It reuses `score.py` (`load_items`/`summarize`) from the [small eval set](../eval/small.md) — don't re-invent the scorer. **Runnable by you on a 4090; not executed by the author** — numbers in comments are illustrative.

**Step 1 — measure one rung** (quality A/B + decode throughput for a single config):

```python title="capstone_stage.py"
"""Measure ONE config on the fixed eval set: per-category quality + decode throughput.
Reuses load_items/summarize from the small eval set (#3). API verified vs vLLM 0.26.0.
Author does not execute (ADR-0004); returned numbers are yours."""
import time
from vllm import LLM, SamplingParams
from score import load_items, summarize          # from the small eval set page

ITEMS = load_items("small_eval.jsonl")            # the FIXED inputs — never change between rungs
CONVOS = [[{"role": "user", "content": it["prompt"]}] for it in ITEMS]
SP = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # greedy + seed => comparable rungs

def measure(model: str, **engine_kwargs) -> dict:
    """Build the engine with these knobs, run the eval, return one report row."""
    llm = LLM(model=model, max_model_len=4096, **engine_kwargs)   # vLLM auto-detects AWQ from config
    t0 = time.perf_counter()
    outs = llm.chat(CONVOS, SP)                   # chat() applies the Instruct template
    dt = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outs]
    report = summarize(ITEMS, texts)              # overall + per-category accuracy
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    row = {
        "accuracy": round(report["accuracy"], 3),
        "by_category": report["by_category"],
        "throughput_tok_s": round(out_tokens / dt, 1),   # coarse decode tok/s on this small set
    }
    del llm                                        # free VRAM before the next rung (two 7B won't co-reside)
    return row
```

**Line-by-line (step 1):** `ITEMS`/`CONVOS`/`SP` are module-level so **every rung shares identical inputs and sampling** — the precondition for attributable deltas (§3.2). `measure` builds an engine with the rung's `engine_kwargs`, runs `LLM.chat` (template applied — `generate` would feed a malformed prompt and tank quality for the wrong reason), and returns a row with overall accuracy, the per-category breakdown (the *gate*, §3.4), and a coarse output-tok/s. `del llm` frees VRAM so the next rung's `LLM(...)` doesn't OOM. This small-set throughput is a quick relative signal; the *authoritative* throughput/knee numbers come from `vllm bench` (step 3).

**Step 2 — drive the ladder and emit the report table:**

```python title="capstone_report.py"
"""Run the optimization ladder and print a before->after Markdown table.
Each rung changes ONE thing vs the row above it (§3.3). Numbers are illustrative."""
from capstone_stage import measure

FP16 = "Qwen/Qwen2.5-7B-Instruct"
AWQ  = "Qwen/Qwen2.5-7B-Instruct-AWQ"     # prebuilt INT4; download in 无卡 mode

# The ladder: (label, model, engine_kwargs) — each row changes exactly ONE lever vs the previous.
LADDER = [
    ("0 · baseline (FP16)",     FP16, {}),
    ("1 · AWQ INT4 weights",    AWQ,  {}),
    ("2 · + FP8 KV cache",      AWQ,  {"kv_cache_dtype": "fp8"}),
    ("3 · + spend freed VRAM",  AWQ,  {"kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.94,
                                       "max_num_seqs": 256}),
    ("4 · + prefix caching",    AWQ,  {"kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.94,
                                       "max_num_seqs": 256, "enable_prefix_caching": True}),
]

rows = []
baseline_tps = None
for label, model, kw in LADDER:
    r = measure(model, **kw)
    if baseline_tps is None:
        baseline_tps = r["throughput_tok_s"]
    speedup = r["throughput_tok_s"] / baseline_tps
    rows.append((label, r["accuracy"], r["throughput_tok_s"], speedup))

print("| stage | eval acc | output tok/s | speedup vs baseline |")
print("|---|---|---|---|")
for label, acc, tps, sp in rows:
    print(f"| {label} | {acc:.3f} | {tps:.0f} | {sp:.2f}× |")
# illustrative output (YOURS WILL DIFFER):
#   | 0 · baseline (FP16)    | 0.95 |  620 | 1.00× |
#   | 1 · AWQ INT4 weights   | 0.90 | 1180 | 1.90× |   <- decode win + freed VRAM
#   | 2 · + FP8 KV cache     | 0.90 | 1210 | 1.95× |   <- more KV room; check per-category quality!
#   | 3 · + spend freed VRAM | 0.90 | 1820 | 2.94× |   <- bigger batch fills the freed memory
#   | 4 · + prefix caching   | 0.90 | 2050 | 3.31× |   <- only if the workload shares a prefix
```

**Line-by-line (step 2):** `LADDER` is the §2 ladder as data — read down the `engine_kwargs` and each row adds **exactly one** key vs the one above (§3.3), so a row's delta is attributable to that key. The loop measures each rung, computes speedup against the baseline row, and prints a Markdown table you paste straight into the report. The **quality column is the gate**: watch it across rows — if a rung drops accuracy sharply or collapses a category (inspect `r["by_category"]`), that rung gets reverted (§3.4), not shipped. Prefix caching (rung 4) only helps if your traffic actually shares a prefix (system prompt / few-shot); on unique prompts it's a no-op — measure, don't assume. Note the offline ladder stops at rung 4: rungs 0–4 are **capacity/throughput** levers you can read off this offline batch, whereas **rung 5 (batch-shape tuning, `max_num_batched_tokens`) is a TTFT↔ITL *serving* knob** — its effect shows in the `vllm bench serve` knee sweep (step 3), not in offline decode tok/s, so you tune it there against your p99 and fill the rung-5 row of the report from *that* measurement.

**Step 3 — the authoritative throughput & knee** (shell; `pip install vllm[bench]`):

```bash title="bench.sh"
# Decode throughput, baseline vs fully-tuned. Same shape both runs => comparable. (vllm bench throughput)
vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct       --num-prompts 200 --input-len 256 --output-len 256
vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct-AWQ   --num-prompts 200 --input-len 256 --output-len 256 \
    --kv-cache-dtype fp8
# Output line: "Throughput: X requests/s, Y total tokens/s, Z output tokens/s" — compare Z across runs.

# The KNEE at your SLO: serve the tuned config, then sweep the arrival rate (see the load-testing lesson).
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --max-num-seqs 256 --enable-prefix-caching &
#   then: python sweep_knee.py   (from the load-testing lesson — steps --request-rate, reads p99 TTFT/goodput)
```

**Line-by-line (step 3):** `vllm bench throughput` is the authoritative decode number (a real batched workload, not the tiny eval set) — run it on the baseline and the tuned config with the **same** `--input-len`/`--output-len`/`--num-prompts` and compare **output tokens/s**. Use a decode-heavy shape (long `--output-len`); a prefill-heavy shape understates the AWQ win because weight-only quant doesn't cut prefill FLOPs. For the *latency* half, serve the tuned config and run the [knee sweep](../part8/load-testing-knee.md) (`sweep_knee.py`) to get the p99 TTFT/TPOT and the SLO-limited req/s — the capacity number you actually report.

## 5 · Lab — produce your before → after report

!!! gpu "Capstone Lab (single-GPU, budget-bounded)"
    - **Min VRAM:** 24 GB. FP16 `Qwen2.5-7B` (~15 GB weights + KV) is the tightest rung; every optimized rung fits with room to spare.
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — the ADR-0001 baseline. **No multi-GPU** (rung 6 / TP is out of scope; noted for [Part 7](../part7/nccl-and-launching-tp-pp.md)).
    - **Est. time / cost:** downloads + any self-quantization in **无卡 mode** (~free); GPU time ≈ **30–60 min** for the full ladder A/B + both benches + a short knee sweep · **≈ ¥3–8** of GPU time (**illustrative**; well within ¥500, which is meant to cover *iterating* on this, not one pass).
    - **Platform:** NVIDIA CUDA (default). AWQ Marlin + FP8 tensor-core paths need Ampere+/Ada — the 4090 (Ada) is fine.
    - **Non-NVIDIA:** the scorer and the report generator are pure Python and run anywhere; the `LLM(...)` / `vllm bench` steps need a supported vLLM backend, and FP8 KV needs a backend that implements it.

**Run order:** (1) in 无卡 mode, download `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-7B-Instruct-AWQ` and copy `small_eval.jsonl` + `score.py` from the [eval set](../eval/small.md); (2) open the GPU, run `capstone_report.py` (the ladder A/B) and read the quality column at each rung; (3) run `bench.sh` for the authoritative decode throughput and a short knee sweep; (4) fill in the template below; (5) **power off.**

Copy this and fill in **your** measured numbers — blanks are yours, example values are illustrative:

```markdown title="before_after_report.md"
# Before → After: Qwen2.5-7B on one RTX 4090

SLO: p99 TTFT ≤ ____ ms, p99 TPOT ≤ ____ ms   ·   Workload: ____-in / ____-out, prefix shared? ____
Baseline vLLM 0.26.0 · greedy (temperature=0, seed=0) · eval = small set (20 items)

| stage (one change each)   | eval acc | per-cat regressions | output tok/s | speedup | knee req/s @ SLO | kept? |
|---------------------------|---------:|---------------------|-------------:|--------:|-----------------:|:-----:|
| 0 · baseline (FP16)       |   ____   | —                   |     ____     |  1.00×  |       ____        |  n/a  |
| 1 · AWQ INT4 weights      |   ____   | ____                |     ____     |  ____   |       ____        | ____  |
| 2 · + FP8 KV cache        |   ____   | ____                |     ____     |  ____   |       ____        | ____  |
| 3 · + spend freed VRAM    |   ____   | ____                |     ____     |  ____   |       ____        | ____  |
| 4 · + prefix caching      |   ____   | ____                |     ____     |  ____   |       ____        | ____  |
| 5 · batch-shape tune      |   ____   | ____                |     ____     |  ____   |       ____        | ____  |

## What I kept and why
- Rung __ : kept — <throughput gain> at <quality cost>, worth it for <SLO>.
- Rung __ : REVERTED — <what regressed> (e.g. FP8 KV dropped math 8 pts); backed off to <fallback>.

## Headline (defensible, one line)
Starting from FP16 (___ tok/s, acc ___), the kept stack reached ___ tok/s (___× ) at acc ___,
with p99 TTFT ___ ms / p99 TPOT ___ ms at ___ req/s — measured on my 4090, spend ≈ ¥___.
```

The headline line is the interview answer: a baseline, a multiple, an accuracy delta, a latency budget, and a cost — every number yours, every rung attributable.

## 6 · Common pitfalls / counter-intuitive points

- **No baseline row.** A "3× speedup" against nothing is unfalsifiable. Rung 0 (FP16, stock) is non-negotiable — it's the denominator of every delta.
- **Changing several knobs per rung.** Flip AWQ *and* `gpu-memory-utilization` together and you can't attribute the result — and if quality dropped, you can't tell which lever to blame. One change per rung, always (§3.3).
- **Skipping the quality gate.** Quantization and FP8 KV degrade *silently* — fluent, wrong-er output. "Looks fine" is not a signal; the [eval A/B](../part4/quantization-lab.md) with greedy + seed is. A category collapse (watch `math`, `format`, and the `bilingual` items) means revert.
- **The OOM cascade from stacking capacity knobs.** Rungs 2–3 free and then *spend* VRAM; push `gpu-memory-utilization` to 1.0 or `max-num-seqs` too high and you OOM at startup or under a burst. Raise in small steps and watch the "# GPU blocks" line.
- **Reporting throughput with no SLO.** "2050 tok/s" is meaningless without "at p99 TTFT ≤ X." The honest capacity number is **goodput at the knee** ([load-testing](../part8/load-testing-knee.md)), not the saturation figure from `--request-rate inf`.
- **Benchmarking a cold server / too-short run.** Cold CUDA graphs and warm-up transients aren't steady state. Warm the server and use enough prompts before you trust a number.
- **Chasing someone's magic values.** A blog's `max-num-batched-tokens=16384` was tuned for *their* model/GPU/traffic. Copy the *method* (the ladder + the sweep), never the magnitudes — those are a property of your box.
- **Trying TP on one card.** `--tensor-parallel-size > 1` needs ≥2 GPUs; on a single 4090 it's 1. Multi-GPU is [Part 7](../part7/nccl-and-launching-tp-pp.md) (A100 territory, ADR-0001) and explicitly out of the single-4090 Capstone scope — note it as future work, don't fake it.
- **Declaring victory on a favorable shape.** AWQ's decode win shows on long outputs; a prefill-heavy shape hides it. Fix and report the input/output split, or the number doesn't transfer.

## 7 · Interview links

- [System design: sizing & designing an inference service](../interview/system-design.md) — the long-form question the Capstone rehearses end-to-end: *given a model, hardware, an SLO and a peak QPS, do the napkin math, design the service, and defend every trade-off* — with several complete worked designs. The Capstone is the hands-on half; that page is the whiteboard half.
- [Capacity Planning: From One GPU's Throughput to a Fleet](../part8/capacity-planning.md) — turns your measured per-instance knee into a GPU count.
- [Tuning knobs: which one for which SLO](../interview/tuning-knobs.md) — the per-knob direction/trade the ladder applies.

## 8 · Summary & further reading

**One line:** The Capstone is one big **before → after sweep** — climb an ordered ladder (quantize → FP8 KV → spend the freed VRAM on batch width → prefix caching → batch-shape tuning; multi-GPU/TP out of single-4090 scope) changing **one** lever per rung, run the same spine at every rung (quality A/B with greedy+seed on the fixed [eval set](../eval/small.md), output tok/s via `vllm bench throughput`, and the p99/knee via a `vllm bench serve` sweep), gate every rung on quality (revert a category collapse and *say so*), and produce a report whose headline is a defensible line — baseline → multiple → accuracy delta → latency budget → cost — with every number measured on your own 4090 inside ¥500.

Further reading:

- [Part 4 · Quantize Qwen2.5-7B to INT4](../part4/quantization-lab.md) — rung 1, and the quality-A/B discipline the whole spine reuses; the [FP8 KV cache](../part4/quantization-methods.md) is rung 2.
- [Part 5 · Tuning knobs sweep](../part5/tuning-knobs-sweep.md) — the sweep this project runs end-to-end, and the [continuous batching](../part5/continuous-batching.md) / [PagedAttention](../part5/paged-attention.md) / [prefix caching](../part5/prefix-caching.md) mechanisms each rung exposes.
- [Part 8 · Load-testing the knee](../part8/load-testing-knee.md) and [SLO-driven tuning](../part8/slo-driven-tuning.md) — the latency half of the spine and the "tune vs. scale" decision.
- [Part 8 · Capacity planning](../part8/capacity-planning.md) — from your measured knee to a fleet.
- [The Eval Sets](../eval/index.md) — the measurement loop and harness (`score.py`, [small](../eval/small.md) / [large](../eval/large.md)) every rung leans on.

## 9 · Self-check

??? question "Your teammate shows a benchmark: 'the AWQ + FP8 KV + big-batch config does 2050 output tok/s.' What three things do you ask before believing it's a win?"
    (1) **Against what baseline?** 2050 tok/s means nothing without the FP16 stock number it improved from and the *same benchmark shape* for both — otherwise it's unfalsifiable (§3.1). (2) **At what quality?** Quantization and FP8 KV degrade silently; ask for the [eval A/B](../part4/quantization-lab.md) with greedy + seed and the *per-category* breakdown — a 2050-tok/s config that collapsed `math` or the `bilingual` items is a regression, not a win (§3.4). (3) **At what SLO?** Raw throughput ignores latency; ask for the p99 TTFT/TPOT and the [knee](../part8/load-testing-knee.md) (goodput at the SLO), because a saturated server hits big throughput while every user waits. And a fourth: **was it one change or several?** If AWQ, FP8 KV, and the batch knobs were flipped together, the number can't be attributed — you don't know which lever earned it or which one would break if reverted.

??? question "Why does the ladder put quantization first and 'spend the freed VRAM' later, rather than the other way round?"
    Because the rungs have a **dependency order**, not a free choice. Quantization is first because it's the biggest, most-free lever: it lifts *both* gates at once — AWQ frees ~10 GB of weight VRAM (more room for [KV cache](../part2/kv-cache-math.md)) *and* shrinks the [memory-bound decode](../part2/roofline-analysis.md) weight read (faster tokens) — at only a small, measurable quality cost. "Spend the freed VRAM" (raise `gpu-memory-utilization` / `max-num-seqs` for a bigger [continuous batch](../part5/continuous-batching.md)) is a *consumer* of the resource quantization *produced*: doing it first, on FP16, either OOMs or fits far fewer sequences. So you climb from "helps both ends, cheap, and frees a resource" down to "trade-off knob that spends that resource." Each rung stands on the one below it — which is also why you measure after each: the gain of rung 3 only exists because rungs 1–2 made room for it.

??? question "In the interview you present this report and mention you *reverted* FP8 KV cache because it dropped math accuracy. Is admitting a reverted optimization a weakness? What does it demonstrate?"
    It's the opposite of a weakness — it's the most senior signal in the report. It demonstrates that you (1) **measured quality, not just speed** — you'd never have caught a silent, category-specific degradation otherwise; (2) **treated quality as a gate**, so you optimized toward a *shippable* config rather than a bigger-but-broken number; and (3) **could attribute the regression to one lever**, which is only possible because you changed one thing at a time. A candidate who reports only monotonic wins either got lucky or didn't look — real optimization work is full of levers that don't pay off, and knowing *which* to drop, with a number to justify it, is the skill. The headline "3.3× at −0.02 accuracy, having rejected FP8 KV for an 8-point math drop" is far stronger than a naked "3.5×."
