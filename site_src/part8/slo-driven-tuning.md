# SLO-Driven Tuning: From Metrics to a Tuning Loop

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090"
    Verified against vLLM 0.26.0 via Context7 (ADR-0004): the knobs are the engine flags you've met — **`--max-num-seqs`**, **`--max-num-batched-tokens`** (the chunked-prefill dial), **`--gpu-memory-utilization`**, **`--max-model-len`**, **`--enable-prefix-caching`**, speculative decoding (`--speculative-config`), and [quantization](../part4/quantization-methods.md). You measure against the SLO with **`vllm bench serve`** (percentiles via `--percentile-metrics "ttft,tpot,itl,e2el"`; it validates **goodput** against SLOs) and read the binding constraint from **`/metrics`** (`vllm:num_requests_waiting`, `gpu_cache_usage_perc`, the prefill/decode histograms). All numbers here are **illustrative / order-of-magnitude references** — the winning config is workload-specific; measure your own.

---

## 1 · Intuition & why it matters

You can [measure the knee](load-testing-knee.md) and [read the metrics](observability-profiling.md). Now the on-call/system-design question: **given a latency target, how do you tune the engine to serve the most traffic that still meets it?** Turning knobs at random is fiddling. Tuning is a *loop*, and it starts from a number the business gives you, not from a knob.

That number is the **SLO** — e.g. "p99 TTFT ≤ 300 ms and p99 TPOT ≤ 50 ms at 20 req/s." Everything follows from it:

1. **The SLO defines success, and the metric is goodput.** Raw throughput is vanity; **goodput** — requests/s that meet *all* the SLO targets — is the score. A config that does 1500 tok/s but blows p99 TTFT has *zero* goodput against a TTFT SLO.
2. **You tune the *binding* constraint, not a random knob.** At any moment one thing limits you: prefill is too slow, decode is too slow, the KV pool is full, or the queue is deep. The metrics tell you which. Turning a decode knob when the **queue** is the problem does nothing — that's a capacity problem you fix by [adding replicas](routing-autoscaling.md), not by tuning.

So: define the SLO, find the binding constraint from metrics, turn the *one* knob that moves it, re-measure goodput, repeat. This lesson is that loop made concrete. → see the [Glossary](../glossary.md) for *SLO, Goodput, Knee*.

## 2 · Mental model

The SLO carves a **target box** on the latency/throughput plane. Tuning = push goodput as high as possible *inside* the box, by relaxing whichever wall is currently binding.

```text
   THE SLO IS A BOX; TUNING MOVES YOU INSIDE IT
                                          the loop:
   p99 TTFT                               ┌───────────────────────────────────────────┐
     ▲   ✗ over TTFT SLO                  │ 1. DEFINE SLO   p99 TTFT≤300ms, TPOT≤50ms   │
     │ ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ ← TTFT limit       │ 2. MEASURE      vllm bench serve → goodput  │
     │ ░░░░░░░░░░░░ │                      │ 3. DIAGNOSE     which constraint binds?     │
     │ ░ TARGET  ░ │  maximize goodput     │      queue?  prefill?  decode?  KV pool?    │
     │ ░  BOX    ░ │  in here              │ 4. TURN ONE KNOB that moves THAT constraint │
     │ ░░░░░░░░░░░░ │                      │ 5. RE-MEASURE goodput → keep if better      │
     └─────────────┴──────────▶ QPS       │    repeat until goodput stops improving     │
                    ✗ over QPS = queue     └───────────────────────────────────────────┘

   DIAGNOSE → KNOB:
     queue deep (num_requests_waiting↑)  → NOT a tuning problem: add replicas / route (Part 8 routing)
     prefill slow (request_prefill_time↑, TTFT↑) → --max-num-batched-tokens (chunked prefill), prefix caching
     decode slow  (request_decode_time↑, TPOT↑)  → --max-num-seqs, quantization, speculative decoding
     KV full (gpu_cache_usage_perc→1.0)          → --gpu-memory-utilization↑, --max-model-len↓, KV quant
```

Three shapes to keep:

- **The SLO comes first; goodput is the score.** Without a target you can't say a config is "better" — faster on one axis is always slower on another. The SLO turns a multi-objective mess into one number: goodput inside the box.
- **One knob moves one wall.** Most knobs trade TTFT against throughput (the `--max-num-batched-tokens` chunked-prefill dial is the cleanest example). Turn **one at a time** and re-measure, or you can't attribute the change.
- **Not every problem is a tuning problem.** A deep queue means you're past the [knee](load-testing-knee.md) — no engine knob raises the ceiling much; you need *more instances*. Diagnosing that correctly stops you from tuning for hours against a capacity wall.

## 3 · Principle

### 3.1 Start from the SLO

Write the target explicitly: latency percentiles (**p99 TTFT**, **p99 TPOT/ITL**), a **throughput/QPS** you must sustain, and sometimes a max cost. Goodput is then well-defined: the request rate at which *all* targets hold. Every tuning decision is judged by whether it raises goodput — nothing else.

### 3.2 Diagnose the binding constraint

Read `/metrics` (previous lesson) to find what's actually limiting you *right now*:

- **Queue-bound** — `num_requests_waiting` is deep and rising. You're past the knee; **this is not a tuning problem**. Add replicas / autoscale ([routing lesson](routing-autoscaling.md)).
- **Prefill-bound** — `request_prefill_time_seconds` and TTFT high; long prompts stalling the batch. Tune prefill.
- **Decode-bound** — `request_decode_time_seconds` and TPOT high; the running batch is bandwidth-limited. Tune decode.
- **KV-bound** — `gpu_cache_usage_perc` near 1.0, preemptions happening. Free or grow KV.

### 3.3 The knob for the constraint

Each knob moves one wall (these are the [Part 5 tuning knobs](../part5/tuning-knobs-sweep.md), now framed by which constraint they relieve):

| Binding constraint | Knob | Effect / trade |
|---|---|---|
| Prefill / TTFT | **`--max-num-batched-tokens`** (chunked prefill) | smaller chunks → lower TTFT (decode interleaves sooner) but a bit less prefill throughput; the cleanest TTFT↔throughput dial |
| Prefill (shared prompts) | **`--enable-prefix-caching`** | reuse shared-prefix KV → skips repeated prefill → lower TTFT for RAG/chat |
| Decode / TPOT | **`--max-num-seqs`** | wider batch → more throughput, but past a point longer TPOT and KV pressure |
| Decode / TPOT | **[quantization](../part4/quantization-methods.md)** (W4A16, FP8) | less weight bandwidth per token → faster decode; watch quality |
| Decode / TPOT | **speculative decoding** (`--speculative-config`) | draft-and-verify → fewer forward passes when memory-bound; backfires under high load |
| KV pool | **`--gpu-memory-utilization`** ↑ | bigger KV block pool → more concurrency headroom (if VRAM spare) |
| KV pool | **`--max-model-len`** ↓ | caps per-request KV → more concurrent sequences fit |

### 3.4 The loop

**Define SLO → measure goodput (`vllm bench serve`) → diagnose the binding constraint (`/metrics`) → turn the one matching knob → re-measure.** Keep a config only if it *raises goodput while still meeting the SLO*. Stop when goodput plateaus — you've hit the hardware's real limit for this workload, and further gains need different hardware or more replicas. Crucially, tune against a workload that matches production (input/output length mix): the winning config for 512-in/128-out is not the winner for 4k-in/1k-out.

## 4 · Complete runnable code + line-by-line

An SLO-gated tuning loop: sweep one knob, keep the value that maximizes goodput *subject to* the SLO.

```python title="slo_tune.py"
"""SLO-driven tuning loop for ONE knob (here: --max-num-seqs).
For each candidate value: (re)start the server, run vllm bench serve, and score it
by GOODPUT — throughput that also meets the p99 SLOs. Keep the best config.
Read-only logic; the runs need a GPU + server. Numbers illustrative."""
import json, subprocess, time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SLO = {"p99_ttft_ms": 300, "p99_tpot_ms": 50}    # THE SLO — success is defined here, not by a knob
CANDIDATES = [64, 128, 256, 384]                 # sweep ONE knob at a time
TEST_QPS = 20                                    # the load the SLO must hold at

def measure(max_num_seqs):
    server = subprocess.Popen(["vllm", "serve", MODEL,   # restart with this knob value
        "--max-num-seqs", str(max_num_seqs), "--gpu-memory-utilization", "0.90"])
    wait_until_ready("http://localhost:8000/health")     # poll /health (200) before load
    subprocess.run(["vllm", "bench", "serve", "--backend", "vllm", "--model", MODEL,
        "--endpoint", "/v1/completions", "--dataset-name", "random",
        "--random-input-len", "512", "--random-output-len", "128",
        "--num-prompts", "500", "--request-rate", str(TEST_QPS),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result", "--result-filename", "r.json"], check=True)   # --result-filename illustrative
    r = json.load(open("r.json"))
    server.terminate(); time.sleep(5)
    meets_slo = r["p99_ttft_ms"] <= SLO["p99_ttft_ms"] and r["p99_tpot_ms"] <= SLO["p99_tpot_ms"]
    goodput = r["request_throughput"] if meets_slo else 0.0     # goodput = throughput ONLY if SLO holds
    return goodput, r["p99_ttft_ms"], r["p99_tpot_ms"], meets_slo

best = (0.0, None)
for v in CANDIDATES:                              # one knob, several values
    goodput, ttft, tpot, ok = measure(v)
    print(f"max_num_seqs={v:>3} | p99 TTFT {ttft:6.1f} | p99 TPOT {tpot:5.1f} | "
          f"goodput {goodput:5.2f} req/s | {'MEETS SLO' if ok else 'VIOLATES SLO'}")
    if goodput > best[0]:
        best = (goodput, v)                       # keep the config with the highest SLO-passing goodput
print(f"\nBest --max-num-seqs = {best[1]} at {best[0]:.2f} req/s goodput (illustrative — measure your own)")
```

**Line-by-line:**

- **`SLO = {...}`** — the loop's north star. Success is "meets these p99 targets"; a config that misses them scores **zero goodput**, no matter how fast its mean is.
- **`CANDIDATES` (one knob)** — sweep `--max-num-seqs` alone. Changing several knobs per run makes the result unattributable (§6); isolate one.
- **`wait_until_ready(/health)`** — the [server lesson](openai-server.md)'s liveness gate: benchmark only after the engine is up, or you measure cold-start.
- **`--percentile-metrics "ttft,tpot,itl,e2el"`** — pull the **tail**; the SLO is on p99, so the median would mislead.
- **`goodput = throughput if meets_slo else 0.0`** — the whole discipline in one line: throughput *only counts* when the SLO holds. This is what stops you from "optimizing" into a faster-but-SLO-violating config.
- **`best = max goodput`** — keep the value that serves the most SLO-compliant traffic. When the sweep plateaus, you've found this workload's ceiling on this hardware — more gains need [replicas](routing-autoscaling.md) or different hardware, not more tuning.
- **`--result-filename` / JSON keys** — illustrative (as in the [knee lesson](load-testing-knee.md)); confirm the exact result-file flag and field names for your version.

## 5 · Lab — tune to an SLO on a 4090

!!! gpu "GPU Lab (single-GPU)"
    - **Min VRAM:** `Qwen2.5-7B-Instruct` on a **24 GB RTX 4090** (BF16, or INT4 for more KV headroom). The loop restarts the server per candidate, so budget a few minutes each.
    - **Suggested AutoDL card:** single **RTX 4090 (24 GB)** (ADR-0001).
    - **Est. time / cost:** ~30–50 min for a 4-point single-knob sweep · **~¥2–6** (illustrative). Keep `--num-prompts` modest; the *shape* of the goodput curve is the deliverable.
    - **Platform:** NVIDIA CUDA (default). **Non-NVIDIA:** the loop is HTTP + a benchmark client — hardware-agnostic; only the winning values differ per backend.

Steps:

1. **Write the SLO down.** Pick p99 TTFT / TPOT targets and a test QPS that reflect your use case. Everything is judged against these.
2. **Diagnose first.** Run one baseline, read `/metrics`: is the queue deep (→ this is a routing/capacity problem, stop tuning), prefill-bound, decode-bound, or KV-bound? Tune the matching knob.
3. **Sweep one knob.** Run `slo_tune.py` (start with `--max-num-seqs`, or `--max-num-batched-tokens` if prefill-bound). Watch goodput rise then plateau; note where p99 crosses the SLO.
4. **Confirm, don't stack.** Take the winning value, then tune a *second* knob from that baseline — one at a time. Stop when goodput plateaus. **Power off.**

## 6 · Common pitfalls / counter-intuitive points

- **Tuning without an SLO.** "Make it faster" has no answer — faster TTFT usually costs throughput and vice-versa. Without a target box you can't call any config better. Write the SLO first.
- **Optimizing raw throughput.** A config that maxes tok/s but violates p99 TTFT has **zero goodput** against a latency SLO. Score by goodput, not throughput.
- **Turning several knobs at once.** Change `--max-num-seqs` and `--max-num-batched-tokens` together and you can't tell which helped (or that they cancelled). One knob per run, re-measure, then move on.
- **Tuning the wrong constraint.** If `num_requests_waiting` is deep, the bottleneck is **capacity**, not the engine config — no decode knob fixes a queue. Add [replicas](routing-autoscaling.md). Diagnose before you tune.
- **Tuning on the wrong workload.** The winning config for short prompts loses for long ones (prefill- vs decode-heavy saturate different resources). Tune against a prompt/length mix that matches production.
- **Chasing past the plateau.** When goodput stops improving, you've hit the hardware limit for this workload. Further knob-twiddling is noise; the real lever is different hardware or more instances.
- **Ignoring quality when quantizing / speculating.** [Quantization](../part4/quantization-methods.md) and speculative decoding raise decode goodput but can move quality; validate on your [eval set](../eval/index.md), not just the latency numbers.

## 7 · Interview links

- [SLO-driven tuning: goodput, the binding constraint & the loop](../interview/slo-driven-tuning.md) — the high-frequency question this lesson prepares you for: *why you start from the SLO and score by goodput, how you read the binding constraint (queue / prefill / decode / KV) from metrics, which knob relieves which, and why one-knob-at-a-time against a production-like workload is the only honest loop.*

## 8 · Summary & further reading

**One line:** Tuning is a loop anchored on the **SLO**: define the latency/throughput target, score every config by **goodput** (throughput that *meets* the SLO — measured with `vllm bench serve`), read the **binding constraint** from `/metrics` (queue → add replicas, not tune; prefill → `--max-num-batched-tokens` / prefix caching; decode → `--max-num-seqs` / quantization / speculative decoding; KV → `--gpu-memory-utilization` / `--max-model-len`), turn the **one** knob that relieves it, re-measure, and stop when goodput plateaus — always against a production-like workload.

Further reading:

- The [tuning-knobs sweep](../part5/tuning-knobs-sweep.md) — the mechanics of each knob and its throughput/latency curve.
- The [load-testing lesson](load-testing-knee.md) — the knee and goodput this loop optimizes against.
- The [observability lesson](observability-profiling.md) — the metrics that reveal the binding constraint.
- vLLM `docs/configuration/optimization.md` — `max_num_batched_tokens` and the chunked-prefill TTFT/throughput trade.

## 9 · Self-check

??? question "Config A does 1500 tok/s at p99 TTFT 900 ms; config B does 1100 tok/s at p99 TTFT 250 ms. Your SLO is p99 TTFT ≤ 300 ms. Which is better, and what's the general principle?"
    **B.** Against a p99 TTFT ≤ 300 ms SLO, config A **violates** the target, so its *goodput* is **0** no matter how high its raw throughput — every request is too slow to count. Config B meets the SLO, so its 1100 tok/s is real, usable goodput. The principle: **score by goodput, not raw throughput.** Throughput without a latency budget is vanity; the only number that matters is how much traffic you serve *within* the SLO. A always looks better on a throughput chart and is useless in production under this SLO.

??? question "You spend an afternoon tuning `--max-num-seqs`, quantization, and speculative decoding, but p99 barely moves. Then you notice `vllm:num_requests_waiting` is deep the whole time. What went wrong?"
    You tuned the wrong constraint. A deep, persistent **`num_requests_waiting`** means requests are **queuing** — you're past the [knee](load-testing-knee.md), so the bottleneck is **capacity**, not engine configuration. Decode/KV knobs move throughput and TPOT *within* one instance's ceiling, but none of them raises the ceiling enough to drain a standing queue caused by offered load exceeding max completion rate. The fix is **more instances** (scale out) plus [prefix-aware routing / autoscaling](routing-autoscaling.md), not another knob. Lesson: **diagnose the binding constraint from metrics first** — if the queue is the wall, stop tuning and add capacity.

??? question "Why tune one knob at a time, and why re-measure against a workload that mirrors production rather than a generic benchmark?"
    **One knob at a time** because tuning knobs interact and trade off (most move TTFT against throughput): change two at once and you can't attribute the delta — a gain from one can be masked or cancelled by the other, and you'll "learn" the wrong lesson. Isolate, re-measure goodput, keep or revert, then move to the next. **Production-like workload** because the binding constraint depends on the prompt/output length mix: a prefill-heavy workload (long inputs, short outputs) is limited by different resources than a decode-heavy one (short inputs, long outputs), so the winning `--max-num-batched-tokens` / `--max-num-seqs` differ. Tuning on 512-in/128-out and deploying for 4k-in/1k-out ships a config optimized for the wrong bottleneck. Match the benchmark's input/output distribution (and cache-hit pattern) to production, or the goodput you measured won't be the goodput you get.
