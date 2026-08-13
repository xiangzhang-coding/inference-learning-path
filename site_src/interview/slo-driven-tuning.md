# SLO-driven tuning: goodput, the binding constraint & the loop

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [SLO-Driven Tuning: From Metrics to a Tuning Loop](../part8/slo-driven-tuning.md)

---

## Q: You're told "make it faster." How do you turn that into a disciplined tuning loop — what do you optimize, how do you find what to tune, which knob for which constraint, and what's the discipline?

### Direct answer

**Optimize goodput against an SLO, not raw throughput.** First write the SLO (e.g. p99 TTFT ≤ 300 ms, p99 TPOT ≤ 50 ms at 20 req/s). **Goodput** = requests/s meeting *all* targets; a config that maxes tok/s but blows p99 scores **zero**.

**The loop:** define SLO → **measure** goodput (`vllm bench serve`, p99 metrics) → **diagnose** the binding constraint from `/metrics` → turn **one** knob that relieves it → re-measure → keep if goodput rose → stop at the plateau.

**Constraint → knob:**

- **Queue-bound** (`num_requests_waiting` deep) → **not a tuning problem**: add replicas / [route](routing-autoscaling.md).
- **Prefill / TTFT** → `--max-num-batched-tokens` (chunked-prefill dial), `--enable-prefix-caching`.
- **Decode / TPOT** → `--max-num-seqs`, [quantization](../part4/quantization-methods.md), speculative decoding.
- **KV-bound** (`gpu_cache_usage_perc`→1.0) → `--gpu-memory-utilization`↑, `--max-model-len`↓, KV quant.

**Discipline:** one knob at a time, re-measure, against a **production-like** workload.

### Deep dive

- **Why goodput.** Throughput and latency trade off; only goodput collapses the multi-objective into one score (throughput *inside* the SLO box).
- **Diagnose before tuning.** The metrics say which wall binds. Tuning decode when the *queue* is the wall does nothing — that's capacity, not config.
- **One knob at a time.** Knobs interact and mostly trade TTFT↔throughput; changing two makes the delta unattributable.
- **Workload-specific.** Prefill-heavy (long in, short out) and decode-heavy (short in, long out) bind different resources → different winning configs. Tune on your real mix.
- **The plateau.** When goodput stops rising you've hit the hardware limit for this workload — more gains need different hardware or more instances, not more knob-twiddling.

### Code

```python
SLO = {"p99_ttft_ms": 300, "p99_tpot_ms": 50}       # success is defined HERE, not by a knob
# for each candidate value of ONE knob: restart server, run vllm bench serve at target QPS,
r = json.load(open("r.json"))
meets = r["p99_ttft_ms"] <= SLO["p99_ttft_ms"] and r["p99_tpot_ms"] <= SLO["p99_tpot_ms"]
goodput = r["request_throughput"] if meets else 0.0  # throughput counts ONLY if the SLO holds
# keep the value with the highest SLO-passing goodput; stop when it plateaus.
```

### Interviewer follow-ups

- *"A: 1500 tok/s @ p99 TTFT 900 ms. B: 1100 @ 250 ms. SLO ≤ 300 ms — which?"* → B. A violates the SLO → goodput 0. Score by goodput, not throughput.
- *"You tuned for hours, p99 barely moved, and the queue was deep the whole time?"* → Wrong constraint: a deep queue = capacity, add replicas; no decode knob drains it.
- *"Why one knob at a time?"* → Knobs interact/trade off; two-at-once is unattributable.
- *"Which knob lowers TTFT without much throughput loss?"* → smaller `--max-num-batched-tokens` (chunked prefill lets decode interleave sooner); prefix caching for shared prompts.
- *"When do you stop tuning?"* → When goodput plateaus — hardware limit for this workload; scale out or change hardware.
- *"Gotcha when quantizing to raise decode goodput?"* → validate quality on your eval set; latency win can cost accuracy.

### Linked concepts

- Lesson: [SLO-Driven Tuning: From Metrics to a Tuning Loop](../part8/slo-driven-tuning.md)
- Related: [Tuning knobs: which one for which SLO](tuning-knobs.md) (each knob's curve), [Load-testing & the concurrency knee](load-testing-knee.md) (goodput & the knee), [Observability & profiling](observability-profiling.md) (metrics that reveal the constraint)
- Glossary: [SLO, Goodput, Knee](../glossary.md)
