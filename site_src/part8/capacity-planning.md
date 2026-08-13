# Capacity Planning: From One GPU's Throughput to a Fleet

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Verified against vLLM 0.26.0 via Context7 (ADR-0004): you measure per-instance capacity with **`vllm bench serve`**, which prints **`Request throughput (req/s)`**, **`Output token throughput (tok/s)`**, **`Total token throughput (tok/s)`**, and Mean/Median/**P99** TTFT/TPOT/ITL; you scale out with **`--tensor-parallel-size`** (`-tp`, default 1) and **`--pipeline-parallel-size`** (`-pp`, default 1); the VRAM knobs (`--gpu-memory-utilization`, `--max-model-len`, `--quantization`, `--kv-cache-dtype`) are the ones you met in [KV Cache Memory Math](../part2/kv-cache-math.md). Every latency/throughput number here is an **illustrative / order-of-magnitude reference** — the arithmetic (a division and a ceiling) is exact; the *inputs* are yours to measure.

---

## 1 · Intuition & why it matters

Every production and system-design interview eventually opens with the same napkin question: **"design an inference service for X QPS at p99 latency Y."** Before you draw a single box, you have to answer *how much hardware does that even take?* — and the honest answer is arithmetic, done before you rent a GPU.

[Part 2's KV Cache Memory Math](../part2/kv-cache-math.md) answered half of it: **how many concurrent sequences fit** on one card (a *memory* question — concurrency is a leftover of VRAM). This lesson answers the other half and joins them into a fleet:

1. **How fast is one instance?** Not "throughput" as one number — that's meaningless without a latency budget — but *requests/s at your SLO*, which is the [knee](load-testing-knee.md) you already know how to measure.
2. **Is the latency SLO even feasible?** Decode is [memory-bound](../part2/roofline-analysis.md), so a single stream has a hard **TPOT floor** set by HBM bandwidth. If your SLO is below the floor, no amount of scaling helps — you need a smaller/quantized model or faster silicon.
3. **How many instances, and how many GPUs?** Peak QPS divided by per-instance goodput, with headroom to stay left of the knee — times the [TP degree](../glossary.md) per instance.

Get this right and you walk into the design question with a defensible GPU count and a bill. Get it wrong and you either over-provision (burn budget) or under-provision (queue explodes at peak). → see the [Glossary](../glossary.md) for *SLO, Knee, Goodput, TP degree*.

## 2 · Mental model

One instance has **two independent ceilings**; its real capacity is the *lower* of the two. Then the fleet is peak load divided by that.

```text
  ONE INSTANCE = min(what FITS, how FAST)          THE FLEET = load ÷ per-instance goodput

  ┌─ MEMORY gate (Part 2) ────────────┐            peak QPS  λ
  │  N_seq = (u·V − W − A − O) / (κ·S) │                │
  │  "how many streams FIT"            │                ▼
  └────────────────────────────────────┘        ┌──────────────┐   r_inst = req/s @ SLO
                    │  min                        │  ÷ (ρ·r_inst)│   (the KNEE, measured)
  ┌─ SPEED gate (roofline) ───────────┐          └──────────────┘   ρ = headroom < 1
  │  TPOT_floor = bytes/token ÷ BW    │                │
  │  aggregate tok/s rises with batch │                ▼
  │  "how FAST it drains the batch"    │        N_inst = ⌈ λ / (ρ·r_inst) ⌉
  └────────────────────────────────────┘                │
                    │                                    ▼
                    ▼                             N_GPU = N_inst × TP
        r_inst = (output tok/s) / (mean output tokens per request)
```

Three shapes to hold:

- **An instance's capacity is `min(fits, fast)`.** The [memory gate](../part2/kv-cache-math.md) says how many sequences fit; the speed gate says how fast the GPU drains that batch. Sizing on one alone lies: a config that *fits* 66 streams but only *decodes* fast enough for 30 at your TPOT is a 30-stream instance. The binding one is your real capacity.
- **Latency and throughput are different questions with different floors.** The single-stream **TPOT floor** (bandwidth) decides *whether the SLO is reachable at all*; the **aggregate throughput** (batched, at the knee) decides *how much traffic one box carries*. Continuous batching decouples them — that's why aggregate tok/s ≫ single-stream tok/s.
- **The fleet is a ceiling, sized at peak with headroom.** You divide *peak* QPS (not mean) by per-instance goodput, and you keep a utilization headroom $\rho<1$ so normal load sits *left of the knee* — leaving room for bursts, a failed replica, and a rolling deploy. Sizing at 100% of the knee means the first traffic spike queues.

## 3 · Principle & math

### 3.1 The speed gate: decode's TPOT floor

Decode generates one token per step and is **[memory-bound](../part2/roofline-analysis.md)**: each step must read the model weights (and the active KV) from HBM. The fastest a *single* stream can emit a token is therefore bounded by bandwidth, not FLOPs:

$$
\text{TPOT}_{\text{floor}} \;\approx\; \frac{W + \kappa S}{\beta_{\text{eff}}}
\qquad
\beta_{\text{eff}} = \eta\,\beta_{\text{peak}}
$$

where $W$ = weight bytes read per step, $\kappa S$ = this stream's KV bytes (from [Part 0](../part0/kv-cache.md), usually $\ll W$ at short context), $\beta_{\text{peak}}$ = HBM peak bandwidth (RTX 4090 ≈ 1008 GB/s), and $\eta$ = achieved efficiency (~0.6–0.8 illustrative). For `Qwen2.5-7B` BF16 ($W\approx15.2$ GB) at short context:

$$
\text{TPOT}_{\text{floor}} \approx \frac{15.2}{0.7\times1008} \approx 21.5\ \text{ms} \;\Rightarrow\; \sim46\ \text{single-stream tok/s.}
$$

**This is the feasibility check.** If the SLO demands p99 TPOT ≤ 15 ms for a 7B BF16 model on a 4090, it is *physically impossible* on one stream — you must quantize weights (shrinks $W$, e.g. AWQ 4-bit → $W\approx5.5$ GB → floor ~7.8 ms) or move to faster hardware. No batching or routing fixes a sub-floor latency SLO.

### 3.2 The throughput gate: aggregate tok/s and req/s

[Continuous batching](../part5/continuous-batching.md) amortizes that one weight read across **every** stream in the batch, so aggregate output throughput climbs with batch size until the GPU saturates at the **[knee](load-testing-knee.md)**. You don't estimate the peak — you *measure* it with `vllm bench serve` and read **`Output token throughput (tok/s)`** at the offered load where p99 still meets the SLO. Turn that into requests/s:

$$
r_{\text{inst}} \;=\; \frac{T_{\text{out}}}{\bar{o}}
$$

where $T_{\text{out}}$ = per-instance output tok/s at the knee (illustrative: ~2000 tok/s for `Qwen2.5-7B-AWQ` on a 4090) and $\bar{o}$ = mean output tokens per request. At $\bar{o}=256$: $r_{\text{inst}}\approx 2000/256 \approx 7.8$ req/s — the instance's honest capacity *at that SLO and length mix*.

### 3.3 The fleet formula

Size for **peak** load $\lambda_{\text{peak}}$ with a utilization headroom $\rho$ (keep normal load left of the knee — e.g. $\rho=0.7$):

$$
\boxed{\;N_{\text{inst}} = \left\lceil \frac{\lambda_{\text{peak}}}{\rho\,r_{\text{inst}}} \right\rceil\;}
\qquad
\boxed{\;N_{\text{GPU}} = N_{\text{inst}} \times \text{TP}\;}
$$

TP (tensor-parallel degree, `--tensor-parallel-size`) is >1 only when the model *doesn't fit* one GPU or you need TP to hit the TPOT floor; for a 7B on a 4090, TP = 1. **Worked napkin** — peak 50 QPS, $\bar{o}=256$, $r_{\text{inst}}\approx7.8$, $\rho=0.7$:

$$
N_{\text{inst}} = \left\lceil \frac{50}{0.7\times7.8} \right\rceil = \lceil 9.2 \rceil = 10 \ \text{instances} = 10\ \text{GPUs (TP=1)}.
$$

Note what each number does: quantizing weights lifts *both* gates (frees VRAM → bigger batch fits, and shrinks $W$ → higher $T_{\text{out}}$ and lower TPOT floor), which is why it's the first lever in [Part 2](../part2/kv-cache-math.md) *and* here. Halving $\bar{o}$ (shorter outputs) doubles $r_{\text{inst}}$ and halves the fleet. All figures illustrative — measure $T_{\text{out}}$ and $\bar{o}$ on your workload.

## 4 · Complete runnable code + line-by-line

A fleet planner — **pure CPU, offline-runnable**, no GPU. It does the feasibility check (§3.1), the req/s conversion (§3.2), and the fleet sizing (§3.3) in one pass.

```python title="fleet_planner.py"
"""Fleet capacity planner: TPOT-floor feasibility + instances + GPUs (pure CPU, offline).
Inputs are yours to MEASURE (bandwidth efficiency, knee throughput, output length);
the arithmetic is exact. All defaults are illustrative / order-of-magnitude."""
from dataclasses import dataclass
from math import ceil

@dataclass
class Plan:
    weight_gb: float = 15.2       # Qwen2.5-7B BF16 weights read per decode step (~5.5 if AWQ 4-bit)
    hbm_gbps: float = 1008.0      # RTX 4090 peak HBM bandwidth (GB/s)
    hbm_eff: float = 0.70         # achieved fraction of peak (illustrative; measure yours)
    out_tok_s_at_knee: float = 2000.0   # per-instance OUTPUT tok/s at the SLO knee (MEASURE via vllm bench serve)
    mean_output_tokens: float = 256.0   # mean generated tokens per request (from your traffic)
    tp: int = 1                   # --tensor-parallel-size (1 if the model fits one GPU)

    def tpot_floor_ms(self) -> float:                       # §3.1 single-stream latency floor
        return self.weight_gb / (self.hbm_eff * self.hbm_gbps) * 1000.0

    def req_per_s(self) -> float:                           # §3.2 knee throughput → requests/s
        return self.out_tok_s_at_knee / self.mean_output_tokens

    def fleet(self, peak_qps: float, headroom: float = 0.70) -> tuple[int, int]:
        r = headroom * self.req_per_s()                     # usable req/s, kept LEFT of the knee
        n_inst = ceil(peak_qps / r)                         # §3.3 ceiling — you can't buy a fraction of a box
        return n_inst, n_inst * self.tp                     # instances, then GPUs = instances × TP

def feasible(plan: Plan, slo_tpot_ms: float) -> bool:
    return plan.tpot_floor_ms() <= slo_tpot_ms              # below the floor => impossible on this model/HW

if __name__ == "__main__":
    peak_qps, slo_tpot = 50.0, 50.0                         # the SLO: 50 QPS peak, p99 TPOT <= 50 ms
    for label, w in [("BF16 weights", 15.2), ("AWQ 4-bit weights", 5.5)]:
        p = Plan(weight_gb=w)
        n_inst, n_gpu = p.fleet(peak_qps)
        ok = "OK" if feasible(p, slo_tpot) else "INFEASIBLE (below TPOT floor)"
        print(f"{label:>18}: TPOT floor {p.tpot_floor_ms():4.1f} ms [{ok}] | "
              f"{p.req_per_s():4.1f} req/s/inst | fleet {n_inst} inst = {n_gpu} GPU")
```

**Line-by-line:**

- **`tpot_floor_ms`** — §3.1: weight bytes ÷ effective bandwidth. This is the *fastest* one stream can decode; the SLO's p99 TPOT must sit *above* it or the design is dead on arrival. KV is omitted (≪ weights at short context); add $\kappa S$ for long-context planning.
- **`req_per_s`** — §3.2: the measured knee **output** throughput divided by mean output length. `out_tok_s_at_knee` is the one input you cannot derive on paper — it comes from `vllm bench serve` (the [load-testing lesson](load-testing-knee.md)).
- **`fleet`** — §3.3: apply headroom (`0.70` → normal load sits at 70% of the knee), ceiling-divide peak QPS, then multiply by TP to get GPUs. The **ceiling** matters: 9.2 instances means you buy 10.
- **`feasible`** — the gate that saves you a wasted design: an SLO below the TPOT floor can't be met by adding boxes; report it and change the model/hardware.
- **`__main__`** — runs the same SLO against BF16 and AWQ weights so the effect of the biggest lever is visible in one table.

Expected output (exact arithmetic, not a benchmark):

```text
      BF16 weights: TPOT floor 21.5 ms [OK] |  7.8 req/s/inst | fleet 10 inst = 10 GPU
 AWQ 4-bit weights: TPOT floor  7.8 ms [OK] |  7.8 req/s/inst | fleet 10 inst = 10 GPU
```

Both meet the 50 ms TPOT SLO here (the floor only *fails* at very tight SLOs), and both size to 10 GPUs at the *same* illustrative knee throughput — but in reality AWQ's freed VRAM raises `out_tok_s_at_knee` (bigger batch fits), so re-measure it and the AWQ fleet shrinks. Change `slo_tpot` to `15` and watch BF16 flip to **INFEASIBLE** while AWQ still passes — the feasibility check earning its keep.

## 5 · Lab — measure the one input you can't guess

!!! gpu "GPU Lab (single-GPU)"
    - **Min VRAM:** `Qwen2.5-7B-Instruct` (or `-AWQ`) on a **24 GB RTX 4090**. The planner is CPU-only; the *one* GPU step is measuring the knee throughput.
    - **Suggested AutoDL card:** single **RTX 4090 (24 GB)** (ADR-0001).
    - **Est. time / cost:** ~15–25 min for a short rate sweep · **~¥1–3** (illustrative). You need one number: output tok/s at the SLO knee.
    - **Platform:** NVIDIA CUDA (default). **Non-NVIDIA:** the arithmetic is hardware-agnostic; only the measured knee and the HBM bandwidth differ (ROCm/TPU/Neuron have their own $\beta_{\text{peak}}$).

Steps:

1. **Serve and sweep.** Start the [OpenAI server](openai-server.md), then sweep offered load with `vllm bench serve --request-rate` upward (the [knee lesson](load-testing-knee.md)). At each rate read `Output token throughput (tok/s)` and `P99 TPOT`.
2. **Find the knee at your SLO.** The last rate where p99 still meets the SLO is your knee; its output tok/s is `out_tok_s_at_knee`.
3. **Feed the planner.** Plug that number and your real `mean_output_tokens` into `fleet_planner.py`. Read the instance and GPU count.
4. **Sanity-check the floor.** Confirm the reported P99 TPOT sits above the computed `tpot_floor_ms` — it always should; if a *single*-stream TPOT is below your floor estimate, your `hbm_eff` guess was too low. **Power off.**

## 6 · Common pitfalls / counter-intuitive points

- **Quoting throughput as one number.** "The 4090 does 2000 tok/s" is meaningless without the SLO and the length mix — at a tighter p99 the same box does far less. Always size on **goodput at the SLO** (the knee), never peak throughput.
- **Sizing at 100% of the knee.** Provision so normal load sits at ~60–80% of the knee. At 100% there's no room for a burst, a failed replica, or a rolling deploy — the queue explodes on the first spike. That's the $\rho$ headroom.
- **Using mean QPS instead of peak.** Traffic is bursty; a fleet sized for the daily mean queues every busy hour. Size for the peak you must hold (plus headroom), then let [autoscaling](routing-autoscaling.md) shed cost off-peak.
- **Assuming 2× GPUs = 2× throughput.** Tensor parallelism adds an all-reduce per layer ([Part 7](../part7/nccl-and-launching-tp-pp.md)); TP=2 is <2× the tok/s of two independent TP=1 replicas. Use TP to *fit* a model or *hit the TPOT floor*, not as a throughput multiplier — for a 7B, TP=1 replicas scale more efficiently.
- **Estimating the speed gate but forgetting the memory gate.** A batch that decodes fast on paper still has to *fit* in the [KV budget](../part2/kv-cache-math.md). Real per-instance capacity is `min(fits, fast)`; check both.
- **Ignoring prefill.** These estimates are decode-centric. A prefill-heavy workload (long prompts, short answers) is TTFT/compute-bound, not decode-bound — its knee is set by prefill, and [chunked prefill](../part5/scheduler-chunked-prefill-pd.md) / prefix caching move it. Size against your real input/output split.
- **Point-estimating variable output length.** $\bar{o}$ is a mean; a heavy tail of long generations eats KV and lowers effective $r_{\text{inst}}$. Plan with the distribution (or a p90 length), not just the average.

## 7 · Interview links

- [System design: sizing & designing an inference service](../interview/system-design.md) — the high-frequency **long-form** questions this lesson prepares you for: *given a model, hardware, an SLO and a peak QPS, do the napkin (feasibility → per-instance goodput → fleet), then design the whole service — routing, autoscaling, KV-aware caching, quantization, multi-tenancy — and defend the trade-offs.* Contains several complete worked designs.

## 8 · Summary & further reading

**One line:** capacity planning is two gates and a division — a single instance's capacity is $\min(\text{what fits}, \text{how fast})$ (VRAM concurrency from [Part 2](../part2/kv-cache-math.md); decode's bandwidth-set TPOT floor and its measured knee throughput here), requests/s = output-tok/s ÷ mean output length, and the fleet is $N_{\text{inst}}=\lceil \lambda_{\text{peak}}/(\rho\,r_{\text{inst}})\rceil$, $N_{\text{GPU}}=N_{\text{inst}}\times\text{TP}$ — always sized at peak with headroom, and gated first by whether the SLO even clears the TPOT floor.

Further reading:

- The [KV Cache Memory Math](../part2/kv-cache-math.md) lesson — the *memory* gate (concurrency as a VRAM leftover) this lesson pairs with the *speed* gate.
- The [load-testing lesson](load-testing-knee.md) — how to *measure* the per-instance knee throughput ($T_{\text{out}}$, $r_{\text{inst}}$) that the planner consumes.
- The [roofline lesson](../part2/roofline-analysis.md) — why decode is memory-bound, the premise behind the TPOT floor.
- The [routing & autoscaling lesson](routing-autoscaling.md) — how the fleet you sized here is actually run (router, prefix-aware routing, queue-based autoscaling).
- vLLM `docs/serving/parallelism_scaling.md` — single-GPU → TP (within a node) → TP×PP (across nodes), the rule behind $N_{\text{GPU}}=N_{\text{inst}}\times\text{TP}$.

## 9 · Self-check

??? question "Your SLO is p99 TPOT ≤ 15 ms for Qwen2.5-7B on a single RTX 4090. Is it feasible, and what do you check first?"
    Check the **TPOT floor** before anything else. Decode is memory-bound, so a single stream can't emit tokens faster than $W/\beta_{\text{eff}}$. For BF16 weights ($W\approx15.2$ GB) at $\eta\approx0.7$, $\beta_{\text{peak}}\approx1008$ GB/s: floor $\approx 15.2/(0.7\times1008)\approx21.5$ ms — **above** the 15 ms SLO, so BF16 is **infeasible** no matter how you scale (batching and routing don't lower single-stream latency). The fix is to shrink $W$: **AWQ 4-bit** drops it to ~5.5 GB → floor ~7.8 ms, which clears 15 ms. Only *after* the floor passes do you size throughput and the fleet.

??? question "One 4090 sustains ~2000 output tok/s at your SLO, mean output is 256 tokens, and peak traffic is 50 req/s. How many GPUs, and why not fewer?"
    Per-instance capacity $r_{\text{inst}} = 2000/256 \approx 7.8$ req/s **at the knee**. Sizing at 100% of the knee is unsafe (no burst/failure/deploy room), so apply headroom $\rho\approx0.7$: usable $\approx5.5$ req/s. Fleet $N_{\text{inst}}=\lceil 50/5.5\rceil = 10$ instances $= 10$ GPUs (TP=1, since 7B fits one card). Fewer (say 9) would put normal load *at* the knee and queue on the first spike — you'd violate p99 exactly when traffic is highest. The headroom and the ceiling are what make the estimate a *safe* number, not just a possible one.

??? question "Marketing wants to cut the GPU bill in half without touching the SLO. Give two levers and why each works."
    **(1) Quantize the weights (AWQ/GPTQ 4-bit).** It lifts *both* gates: frees ~8 GiB of VRAM so a bigger batch fits (memory gate), and shrinks $W$ so aggregate tok/s rises and the TPOT floor drops (speed gate) — higher $r_{\text{inst}}$ per instance means fewer instances for the same QPS. **(2) Shorten outputs.** $r_{\text{inst}}=T_{\text{out}}/\bar{o}$, so halving mean output length (via `max_tokens` caps, better prompts, or stop sequences) doubles requests/s per instance and halves the fleet — and it's free. Secondary levers: **prefix caching** if prompts share a system preamble (skips repeated prefill → higher effective throughput), and **autoscaling** to shed off-peak instances so you pay for the mean, not the peak, over a day. Each is a knob on $r_{\text{inst}}$ or on when you pay for capacity — the two terms in the fleet formula.
