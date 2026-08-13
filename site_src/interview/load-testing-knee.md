# Load-testing & the concurrency knee (Little's Law)

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [Load-Testing to Find the Concurrency Knee](../part8/load-testing-knee.md)

---

## Q: What is the concurrency "knee," why does the latency curve bend there, how does Little's Law explain the runaway past it, what's the difference between open-loop and closed-loop load, and which metric do you actually report?

### Direct answer

The **knee** is the offered load at which a single instance's running batch fills and requests start to **queue** (`vllm:num_requests_waiting` climbs off zero). **Below** it: adding load raises throughput, latency barely moves (spare batch width). **Above** it: the GPU is saturated so throughput **flattens**, while latency **runs away** because every new request waits behind a growing backlog.

**Little's Law** ($L = \lambda W$, steady state: avg requests in system = arrival rate × time in system) explains it. Below the knee $W$ is ~constant, so $L$ grows linearly with $\lambda$. At the knee the arrival rate $\lambda$ reaches the max completion rate $\mu$; push $\lambda > \mu$ and there's **no steady state** — $L$ and $W$ grow without bound. The knee is exactly $\lambda \approx \mu$.

**Open-loop** (fixed **arrival rate**, `--request-rate λ`, Poisson-spaced) models real traffic and *can* overload → reveals the knee. **Closed-loop** (fixed **concurrency**, `--max-concurrency N`) self-limits (new request only on completion) → never shows runaway. The default `--request-rate inf` is a saturation test (max throughput, ignores latency).

**Report goodput** — requests/s meeting **all** SLOs (read on **p99**) — not raw throughput. Find the knee by **sweeping** `--request-rate` up and taking the last rate that meets the SLO.

### Deep dive

- **The queue is the knee.** One gauge, `vllm:num_requests_waiting`, going positive *is* the knee, live. It's the direct read of "arrivals exceed completions."
- **Throughput past the knee is a lie.** The plateau is real (GPU 100% busy) but useless — every user is queued seconds deep. That's why capacity must be stated *with* a latency SLO.
- **The SLO defines the ceiling.** Identical hardware, different SLOs ("p99 TTFT < 200 ms" vs "< 2 s") → different knees. Ship **goodput at your SLO**.
- **Tail, not median.** A great median with a terrible p99 fails 1% of users badly; the SLO is written on the tail (`--percentile-metrics "ttft,tpot,itl,e2el"`).
- **Workload shape is part of the answer.** 512-in/128-out (decode-heavy) and 4k-in/1k-out (prefill-heavy) saturate different resources → different knees. Fix and report the lengths/dataset.

### Code

```bash
# Open-loop run at a fixed arrival rate (NOT 'inf'); Poisson-spaced arrivals
vllm bench serve --backend vllm --model qwen2.5-7b --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 500 --request-rate 8 \
  --percentile-metrics "ttft,tpot,itl,e2el" --save-result
# sweep --request-rate 2,4,8,16,32 → knee = last rate with p99 TTFT ≤ SLO and goodput still rising.
# watch the queue at the knee:  curl -s localhost:8000/metrics | grep num_requests_waiting
```

### Interviewer follow-ups

- *"You report `--request-rate inf` throughput as capacity — problem?"* → Saturation test: max throughput, ignores latency; every user is queued. Capacity = knee at the SLO, from finite-rate sweeps.
- *"Why does latency go vertical past the knee?"* → Little's Law: $\lambda > \mu$ → no steady state → queue $L$ and wait $W$ grow unbounded; throughput can't exceed $\mu$.
- *"Open-loop vs closed-loop — when each?"* → Open-loop `--request-rate` for the real ceiling/knee (can overload); closed-loop `--max-concurrency` to characterize a fixed client pool (self-limits, no runaway).
- *"Median TTFT is great — ship it?"* → No — SLO is on p99; a bad tail fails users. Read percentiles.
- *"One number to summarize capacity?"* → Goodput at the SLO, plus the workload shape (input/output lengths) it was measured at.
- *"Benchmark latency looks noisy over localhost."* → Use `127.0.0.1` (vLLM's tooling note) to avoid IPv6 resolution stalls; warm up first; use enough prompts for steady state.

### Linked concepts

- Lesson: [Load-Testing to Find the Concurrency Knee](../part8/load-testing-knee.md)
- Related: [Latency vs throughput metrics](latency-throughput-metrics.md) (TTFT/TPOT/ITL/goodput defined), [Tuning knobs: which one for which SLO](tuning-knobs.md) (the knobs that *move* the knee), [Routing, autoscaling & KV-aware routing](routing-autoscaling.md) (what you do once you hit the knee)
- Glossary: [Knee, SLO, Goodput](../glossary.md)
