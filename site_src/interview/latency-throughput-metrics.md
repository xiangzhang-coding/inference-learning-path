# Latency vs throughput metrics

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 0 · Foundations   ·   **Tests the lesson:** [Inference Performance Metrics](../part0/metrics.md)

---

## Q: Define TTFT, TPOT/ITL, throughput, and goodput. How would you measure each? Why does increasing batch size raise throughput but hurt TTFT/TPOT, and what does goodput add over raw throughput?

### Direct answer

Four numbers, all at once:

- **TTFT** (Time To First Token) = time from request arrival to the first output token; dominated by **prefill**.
- **TPOT** (Time Per Output Token) = mean time per *subsequent* token = $(t_{\text{last}}-t_{\text{first}})/(N-1)$; dominated by **decode**. **ITL** (Inter-Token Latency) is the *per-gap* version — report ITL for jitter, TPOT for the average.
- **Throughput** = tokens (or requests) processed per second across the whole system.
- **Goodput** = the throughput of only those requests that met their latency **SLO** (e.g. TTFT ≤ 0.5 s *and* TPOT ≤ 50 ms). Always ≤ throughput.

**Measure** them client-side from per-request timestamps (arrival + each token's arrival time), or with vLLM's built-in harness `vllm bench serve` (reports TTFT/TPOT/ITL/throughput at percentiles), or by scraping the server's Prometheus `/metrics` (`vllm:time_to_first_token_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, `vllm:generation_tokens_total`). Always quote **p50/p90/p99**, never just the mean.

**Bigger batch** raises throughput because decode's weight reads amortize across more sequences (the point of continuous batching) — but each request now shares the GPU with more work, so its TTFT and TPOT rise. There's no single "faster." **Goodput** is what catches this: a batch-size sweep raises throughput monotonically but goodput *peaks then falls*, and that peak is the real operating point — raw throughput past it is tokens nobody received on time.

### Deep dive

- **The two latencies map to the two phases.** TTFT is a prefill metric (digest the whole prompt before any output), TPOT a decode metric (one memory-bound step per token). A request's end-to-end latency ≈ $\text{TTFT} + (N-1)\cdot\text{TPOT}$. Users feel them differently: slow TTFT = frozen screen; slow TPOT = choppy stream.
- **Little's Law ties it together.** $L = \lambda W$ (concurrency = arrival rate × latency). To serve $\lambda$ at latency $W$ you need $L$ requests resident (your batch/KV budget); pushing $\lambda$ while GPU-bound forces $W$ up; the **knee** of the throughput-vs-latency curve is where $W$ climbs faster than $\lambda$.
- **Throughput vs goodput, precisely.** goodput $=\frac{\sum_r N_r\,\mathbb{1}(\text{SLO}_r)}{W}$ — same window, numerator restricted to SLO-satisfying requests. The gap between them is the throughput you're "earning" by breaking promises.
- **Tails, not means.** SLOs are written against p99. A great mean with a p99 that violates the SLO means your unluckiest 1% are consistently failed — invisible in the average.

### Code

Everything derives from per-request timestamps — pure CPU:

```python
arrival, tok = 0.20, [0.90, 1.00, 1.10]     # one request: sent at .20, 3 tokens
ttft = tok[0] - arrival                       # 0.70  (prefill wait)
tpot = (tok[-1] - tok[0]) / (len(tok) - 1)    # 0.10  (mean inter-token gap)
e2e  = tok[-1] - arrival                      # 0.90  = ttft + (N-1)*tpot
met  = ttft <= 0.5 and tpot <= 0.05           # False -> counts for throughput, NOT goodput
print(round(ttft,2), round(tpot,2), round(e2e,2), met)   # 0.7 0.1 0.9 False
```

### Interviewer follow-ups

- *"Throughput rose 20% after doubling the batch but users say it's slower — reconcile."* → Both true: aggregate throughput up (weights amortized across the batch), per-request TTFT/TPOT up (each shares the GPU). If the latency rise crossed the SLO, goodput likely *fell* even as throughput rose — the real regression.
- *"Client-measured TTFT is higher than the server's histogram. Why?"* → Client TTFT = server compute + queueing + network RTT. Compare against `vllm:time_to_first_token_seconds` to separate "model slow" from "link slow."
- *"Why p99 over mean?"* → SLOs are tail promises; the mean hides the 1% that consistently violate.
- *"How does continuous batching change the ITL you observe?"* → It makes inter-token gaps *uneven* — a step that admits new requests is heavier — so a good mean TPOT can still feel choppy. Look at ITL percentiles.
- *"You must not forget one thing before trusting any of these numbers."* → Warmup: the first request pays CUDA-graph capture + cache warmup; discard it or your p99 is really "the first request."

### Linked concepts

- Lesson: [Inference Performance Metrics](../part0/metrics.md)
- Related lesson: [Inference Flow: Prefill & Decode](../part0/inference-flow.md) (why TTFT↔prefill, TPOT↔decode)
- Glossary: [TTFT, TPOT / ITL, Throughput, Goodput, SLO, Knee](../glossary.md)
