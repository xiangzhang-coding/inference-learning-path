# Inference Performance Metrics

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    All CLI/flags/metric names on this page are verified against vLLM 0.26.0 via Context7 (ADR-0004). Latency/throughput figures are **illustrative / order-of-magnitude references** — measure the real ones with `vllm bench serve` on your own AutoDL box. The metric arithmetic below (differences, ratios, percentiles) is *exact*.

---

## 1 · Intuition & why it matters

"Is it fast?" is the wrong question. A serving system is fast in **four different ways at once**, and they trade against each other, so you cannot tune anything until you can name and measure each one:

- How long until the user sees *anything*? → **TTFT** (Time To First Token)
- Once streaming, how snappy is each subsequent word? → **TPOT / ITL** (Time Per Output Token / Inter-Token Latency)
- How many tokens can the *whole box* push per second across all users? → **throughput**
- How many of those tokens actually arrived *within the latency promise* we made? → **goodput**

The trap is optimizing one and silently wrecking another. Cram more requests into a batch and aggregate **throughput** soars — but every request's **TTFT** and **TPOT** get worse, because each now shares the GPU. A system can report glorious throughput while quietly violating its latency SLO on 30% of requests; **goodput** is the metric that refuses to let you hide that. This lesson gives you the precise definitions, the measurement recipe, and the one law (Little's) that ties latency, concurrency, and throughput together. → see the [Glossary](../glossary.md) for *TTFT*, *TPOT / ITL*, *Throughput*, *Goodput*, *SLO*.

## 2 · Mental model

One request, as a timeline — every metric is a segment of it:

```text
SINGLE REQUEST (streaming)
  t_arrival        t_first                                   t_last
     |                |        |       |       |       |        |
     |<---- TTFT ---->| tok#1  tok#2   tok#3   tok#4   tok#5    |
     |                |<-ITL->|<-ITL->|<-ITL->|<-ITL->|         |
     |                                                          |
     |<---------------------- e2e latency --------------------->|

  TTFT = t_first - t_arrival                 (dominated by PREFILL)
  ITL  = gap between consecutive tokens       (each decode step)
  TPOT = mean ITL = (t_last - t_first)/(N-1)  (dominated by DECODE)
  e2e  = TTFT + (N-1)*TPOT

MANY REQUESTS (the system view)
  throughput = (all output tokens) / wall-clock          <- raw tokens/s
  goodput    = (tokens/reqs that MET the SLO) / wall-clock <- honest tokens/s
                     e.g. SLO = "TTFT <= 0.5s AND TPOT <= 50ms"
```

Two shapes to hold:

- **Per-request latency is two numbers, not one.** TTFT (the wait) and TPOT (the cadence) come from the two phases of [inference flow](inference-flow.md) — prefill sets TTFT, decode sets TPOT — and users feel them differently: a slow TTFT is a frozen screen; a slow TPOT is choppy streaming.
- **System throughput and per-request latency pull in opposite directions.** Bigger batches amortize weight reads across more requests (higher throughput, the whole point of [continuous batching](../glossary.md)) but each request waits behind more work (worse TTFT/TPOT). Goodput is where that tension gets scored honestly.

## 3 · Principle & math

Let a request arrive at $t_0$, emit its first token at $t_1$, its last at $t_e$, producing $N$ output tokens with per-gap latencies $\ell_1,\dots,\ell_{N-1}$.

$$
\text{TTFT} = t_1 - t_0, \qquad
\text{ITL}_i = \ell_i, \qquad
\text{TPOT} = \frac{t_e - t_1}{N-1} = \frac{1}{N-1}\sum_{i} \ell_i
$$

$$
\text{e2e latency} = \text{TTFT} + (N-1)\cdot\text{TPOT} = t_e - t_0
$$

TPOT is just the **mean** ITL; report ITL when you care about *jitter* (batching makes gaps uneven), TPOT when you care about the *average* cadence.

Across a set of $R$ requests over a wall-clock window $W$:

$$
\text{output throughput} = \frac{\sum_r N_r}{W}\ \text{(tok/s)}, \qquad
\text{request throughput} = \frac{R}{W}\ \text{(req/s)}
$$

**Goodput** restricts the numerator to requests that satisfied the [SLO](../glossary.md). With an indicator $\mathbb{1}(\text{SLO}_r)$ that is 1 when request $r$ met *every* latency target:

$$
\text{goodput} = \frac{\sum_r N_r \cdot \mathbb{1}(\text{SLO}_r)}{W}
$$

Goodput $\le$ throughput always; the gap is the throughput you're "earning" by breaking promises. A batch-size sweep typically *raises* throughput monotonically but *peaks* goodput and then falls — that peak is the operating point you actually want.

**Little's Law** ties it all together. For a system in steady state with average concurrency $L$ (requests in flight), arrival rate $\lambda$, and mean latency $W$:

$$
L = \lambda \cdot W
$$

Read three ways: to serve arrival rate $\lambda$ at latency $W$ you need $L=\lambda W$ requests resident (sets your batch/KV budget); pushing $\lambda$ up while GPU-bound forces $W$ up (latency degrades under load); and the [knee](../glossary.md) of the throughput-vs-latency curve is where $W$ starts climbing faster than $\lambda$. Percentiles matter here — quote **p50, p90, p99**, never just the mean, because tail latency is what violates SLOs.

## 4 · Complete runnable code + line-by-line

This computes every metric above from a **fixed set of per-request timestamp logs** — pure CPU, offline-runnable, deterministic. It's exactly what a client-side harness (or `vllm bench serve`) does internally.

```python title="metrics.py"
"""Compute TTFT / TPOT / ITL / throughput / goodput from request logs (pure CPU)."""
from dataclasses import dataclass
from statistics import mean


@dataclass
class RequestLog:
    arrival: float             # t0: when the request was sent
    token_times: list[float]   # absolute times of each emitted output token


def ttft(r: RequestLog) -> float:
    return r.token_times[0] - r.arrival                       # first token wait

def tpot(r: RequestLog) -> float:
    n = len(r.token_times)
    return (r.token_times[-1] - r.token_times[0]) / (n - 1)   # mean inter-token gap

def e2e(r: RequestLog) -> float:
    return r.token_times[-1] - r.arrival                      # total wall time

def meets_slo(r: RequestLog, max_ttft: float, max_tpot: float) -> bool:
    return ttft(r) <= max_ttft and tpot(r) <= max_tpot        # ALL targets must hold


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = round((p / 100) * (len(s) - 1))                       # nearest-rank on 0..n-1
    return s[k]


if __name__ == "__main__":
    # Three synthetic requests (times in seconds). C has a slow TTFT on purpose.
    reqs = [
        RequestLog(0.00, [0.20, 0.25, 0.30, 0.35, 0.40]),           # A: TTFT .20, TPOT .05
        RequestLog(0.10, [0.60, 0.64, 0.68, 0.72, 0.76, 0.80]),     # B: TTFT .50, TPOT .04
        RequestLog(0.20, [0.90, 1.00, 1.10]),                       # C: TTFT .70, TPOT .10
    ]
    MAX_TTFT, MAX_TPOT = 0.50, 0.05                              # the SLO

    ttfts = [ttft(r) for r in reqs]
    tpots = [tpot(r) for r in reqs]
    print(f"TTFT  p50={percentile(ttfts,50):.2f}s  p99={percentile(ttfts,99):.2f}s  mean={mean(ttfts):.3f}s")
    print(f"TPOT  p50={percentile(tpots,50):.2f}s  mean={mean(tpots):.4f}s")

    wall = max(r.token_times[-1] for r in reqs) - min(r.arrival for r in reqs)
    total_out = sum(len(r.token_times) for r in reqs)
    good_out = sum(len(r.token_times) for r in reqs if meets_slo(r, MAX_TTFT, MAX_TPOT))
    print(f"throughput = {total_out}/{wall:.2f}s = {total_out/wall:.2f} tok/s")
    print(f"goodput    = {good_out}/{wall:.2f}s = {good_out/wall:.2f} tok/s "
          f"({good_out}/{total_out} tokens met SLO)")
```

**Line-by-line:**

- `RequestLog` — the raw material of every serving benchmark: when you sent the request and the timestamp of each token that came back. Everything else is derived.
- `ttft` / `tpot` / `e2e` — direct transcriptions of the §3 formulas. Note `tpot` divides by `n-1` (there are $N-1$ *gaps* between $N$ tokens), a classic off-by-one to get right.
- `meets_slo` — the AND is the point: a request is "good" only if it met **every** target. One violated dimension disqualifies it.
- `percentile` — nearest-rank on a 0-indexed sorted list; p99 of 3 points is the max, which is *why* tail metrics need volume to be meaningful.
- `__main__` — three requests; C's 0.70 s TTFT busts the 0.50 s SLO, so its 3 tokens count toward throughput but **not** goodput.

Expected output (exact arithmetic, not a benchmark):

```text
TTFT  p50=0.50s  p99=0.70s  mean=0.467s
TPOT  p50=0.05s  mean=0.0633s
throughput = 14/1.10s = 12.73 tok/s
goodput    = 11/1.10s = 10.00 tok/s (11/14 tokens met SLO)
```

Throughput says "12.7 tok/s"; goodput says "only 10.0 of that was delivered on-promise." The 2.7 tok/s gap is exactly request C — high on the throughput board, invisible on goodput. That gap is what a load test hunts for.

## 5 · Lab — measure it for real with vLLM

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB (loads `Qwen2.5-7B-Instruct-AWQ`)
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~20 min · ~¥1–2 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** `vllm bench serve` is backend-agnostic (it's an HTTP client); the *server* it hits must be a working vLLM build for your platform (ROCm/CPU numbers differ).

vLLM ships the exact harness from §4 as a CLI. First serve the model, then benchmark it, then read the server's own metrics — three verified surfaces.

Serve the model (Prometheus metrics are on by default at `/metrics`):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192 --enable-per-request-metrics
```

Drive load and get percentiled TTFT/TPOT/ITL + throughput (`vllm bench` needs `pip install vllm[bench]`):

```bash
vllm bench serve \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host localhost --port 8000 \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 200 --max-concurrency 16
```

Read the server's aggregated histograms directly (verified metric names, `vllm:` prefix):

```bash
curl http://localhost:8000/metrics | grep -E \
  'vllm:time_to_first_token_seconds|vllm:request_prefill_time_seconds|vllm:request_decode_time_seconds|vllm:generation_tokens_total'
```

**What to observe:** re-run `vllm bench serve` with `--max-concurrency` at 1, 8, 32, 64 and watch the trade play out — **output-token throughput climbs** while **p99 TTFT and TPOT climb too**. Plot throughput (x) against p99 TTFT (y) and you've drawn the latency-throughput curve by hand; its [knee](../glossary.md) is your goodput-maximizing operating point. (vLLM even automates this: `vllm bench sweep serve_workload` runs the sweep and `vllm bench sweep plot` draws the curve.)

## 6 · Common pitfalls / counter-intuitive points

- **Reporting the mean, hiding the tail.** Mean latency looks great while p99 violates the SLO for your unluckiest 1% of users. Always quote p50/p90/p99; SLOs are written against tails.
- **Confusing throughput with goodput.** A batch-size sweep raises throughput monotonically but goodput *peaks then falls*. Optimizing raw throughput past the goodput peak buys tokens nobody received on time.
- **Bigger batch, better *and* worse.** Larger batches raise aggregate throughput but raise per-request TTFT/TPOT. There is no single "faster" — state which metric you mean.
- **Measuring client-side includes the network.** Client-observed TTFT = server compute + queueing + network RTT. Compare against the server's `vllm:` histograms to separate "the model is slow" from "the link is slow."
- **Forgetting warmup.** The first request pays CUDA-graph capture and weight/cache warmup. Discard warmup requests or your p99 is really "the first request."
- **TPOT hides ITL jitter.** Continuous batching makes inter-token gaps uneven (a step that admits new requests is heavier). A good average TPOT can still feel choppy — look at ITL percentiles, not just the mean.

## 7 · Interview links

- [Latency vs throughput metrics](../interview/latency-throughput-metrics.md) — the high-frequency question this lesson prepares you for: *define TTFT/TPOT/ITL/throughput/goodput, say how you'd measure each, and explain why batch size trades TTFT for throughput and what goodput adds over throughput.*

## 8 · Summary & further reading

**One line:** serving speed is four coupled numbers — TTFT (prefill), TPOT/ITL (decode), throughput (the box), goodput (throughput that kept its SLO promise) — measured with percentiles and tied together by Little's Law $L=\lambda W$; you can't tune what you can't name.

Further reading:

- vLLM docs — *Benchmarking* (`vllm bench serve` / `sweep`) and *v1 Metrics* (the `vllm:` Prometheus surface), baseline v0.26.0.
- Zhong et al. — *DistServe* — where "goodput under SLO" is made the primary objective and prefill/decode are disaggregated to hit it.
- The [Inference Flow](inference-flow.md) lesson — why TTFT is a prefill metric and TPOT a decode metric.

## 9 · Self-check

??? question "Define TTFT and TPOT, tie each to a phase of inference, and give the formula for a request's end-to-end latency."
    TTFT = time from request arrival to the first output token, dominated by **prefill** (digesting the whole prompt). TPOT = mean time per subsequent output token = $(t_{\text{last}}-t_{\text{first}})/(N-1)$, dominated by **decode** (one memory-bound step per token). End-to-end latency $= \text{TTFT} + (N-1)\cdot\text{TPOT}$.

??? question "Throughput went up 20% after you doubled the batch size, but users complain it got slower. Reconcile this."
    Both are true. Doubling the batch amortizes weight reads across more requests, so **aggregate throughput** (tokens/s across the box) rises — but each request now shares the GPU with more work, so its **TTFT and TPOT** (what a single user feels) rise. "Faster" was never one number. If the latency rise pushed requests past the SLO, **goodput** likely fell even as throughput rose — that's the real regression.

??? question "Why prefer goodput over throughput as a tuning objective, and why quote p99 rather than mean latency?"
    Throughput counts *all* tokens including those delivered too late to matter; **goodput** counts only tokens whose request met the latency SLO, so maximizing it optimizes *useful* work and reveals the batch-size peak beyond which extra throughput is illusory. **p99** (not mean) because SLOs are promises about the *tail* — a great mean with a p99 that violates the SLO means 1% of users are consistently failed, which the mean conceals.
