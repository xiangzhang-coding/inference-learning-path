# GPU Hardware Mental Model

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Hardware figures on this page (peak FLOP/s, HBM bandwidth, SM count) are **illustrative / order-of-magnitude references** for a consumer RTX 4090 — read the exact numbers off your own card's spec sheet or `nvidia-smi`. The roofline arithmetic (a `min` and a division) is *exact*, not a benchmark.

---

## 1 · Intuition & why it matters

A datacenter GPU markets itself with one giant number: **TFLOPS**, its peak arithmetic rate. That number is almost irrelevant to LLM decode. The [Inference Flow](inference-flow.md) lesson showed decode runs at an arithmetic intensity of ≈ 1 FLOP/byte — meaning for every byte it drags out of memory it does about one multiply-add. A GPU that can do *hundreds* of FLOPs per byte of bandwidth is then, by construction, sitting almost idle: it finishes the math instantly and spends the rest of every step **waiting on memory**.

So the single most useful thing you can carry into every later chapter is a **hardware mental model built around bandwidth, not FLOPs**. Once you can picture where the bytes live (registers → SRAM → HBM), how fast each tier moves them, and how the roofline turns "intensity" into "attainable throughput," every optimization in this path stops being a trick to memorize and becomes a *move on a diagram you can draw*: FlashAttention keeps bytes in SRAM; quantization shrinks the bytes; continuous batching raises the intensity so the FLOPs finally get used. → see the [Glossary](../glossary.md) for *SM / Warp / Occupancy*, *HBM / SRAM*, *Roofline*.

## 2 · Mental model

Hold two pictures in your head: the **memory pyramid** (where bytes live and how fast they move) and the **execution fabric** (who does the math).

```text
                    THE MEMORY PYRAMID  (per RTX 4090, illustrative)
                    ┌───────────────────────────────┐
   fastest, tiniest │  Registers      ~KB/SM    ~10s of TB/s │  on-chip
        ▲           ├───────────────────────────────┤
        │           │  SRAM: L1 / shared mem  ~100 KB/SM  ~TB/s │  on-chip  <- FlashAttention lives here
        │           ├───────────────────────────────┤
        │           │  L2 cache       ~72 MB    ~few TB/s     │  on-chip
        │           ├───────────────────────────────┤
   slowest, biggest │  HBM/GDDR6X     24 GB     ~1 TB/s       │  OFF-chip <- weights + KV cache live here
                    └───────────────────────────────┘
                     Every decode step pulls weights + KV across this last, slow line.

                    THE EXECUTION FABRIC
   GPU = 128 SMs (Streaming Multiprocessors), each runs many WARPS (32 threads, lockstep).
   An SM hides memory latency by SWAPPING warps: while warp A waits on HBM, warp B computes.
   OCCUPANCY = how many warps are resident to swap between. High occupancy hides latency;
   it does NOT raise the bandwidth ceiling.
```

Two shapes to internalize:

- **The bandwidth cliff.** Moving between tiers is not a gentle slope — HBM is roughly an *order of magnitude* slower than on-chip SRAM. An algorithm that re-reads data from HBM when it could have kept it in SRAM pays that cliff on every access. That single fact is the whole motivation for IO-aware kernels.
- **Latency hiding ≠ bandwidth.** SMs are superb at hiding *latency* (the wait for the first byte) by juggling warps. They cannot manufacture *bandwidth* (bytes per second). When decode is bandwidth-starved, adding warps/occupancy buys you nothing — the pipe is already full of the wrong thing: waiting.

## 3 · Principle & math — the roofline

The [roofline model](../glossary.md) turns the two hardware numbers — peak compute $P$ (FLOP/s) and memory bandwidth $B$ (byte/s) — plus a workload's [arithmetic intensity](inference-flow.md) $I$ (FLOP/byte) into the **attainable throughput**:

$$
\text{attainable}(I) \;=\; \min\bigl(\,P,\; I \cdot B\,\bigr)
$$

Read it as two regimes joined at a corner. When $I$ is small, $I\cdot B < P$ and you're pinned to the **bandwidth roof** $I\cdot B$ — *memory-bound*. When $I$ is large, the $P$ term wins and you're on the flat **compute roof** — *compute-bound*. The crossover is the **ridge point**:

$$
I^{*} \;=\; \frac{P}{B}
$$

Any workload with $I < I^{*}$ is memory-bound on this machine; any with $I > I^{*}$ is compute-bound. Plug in illustrative RTX 4090 numbers — $P \approx 165\ \text{TFLOP/s}$ (BF16, dense) and $B \approx 1\ \text{TB/s}$ (GDDR6X):

$$
I^{*} \;\approx\; \frac{165 \times 10^{12}}{1 \times 10^{12}} \;\approx\; 165\ \text{FLOP/byte}
$$

Now overlay the workload. Decode's intensity is ≈ 1 FLOP/byte — **two orders of magnitude below the ridge** — so decode attains ≈ $1 \cdot B$ = ~1 TFLOP/s, using **< 1%** of the 165 TFLOP/s the card can do. Prefill's intensity climbs into the thousands, clears the ridge, and lands on the flat compute roof. *Same GPU, same weights, opposite regimes* — exactly the asymmetry [Inference Flow](inference-flow.md) predicted, now read off the hardware's own roofline.

This is why the marketing TFLOPS number lies about decode: you only reach it at $I \ge I^{*}$, and decode lives nowhere near there. The levers that actually help decode all **move you rightward or lift the bandwidth roof**: quantization (fewer bytes → higher $I$ *and* fewer bytes to move), batching (reuse weights across a batch → higher $I$), and IO-aware kernels (avoid HBM round-trips → effectively more usable $B$).

## 4 · Complete runnable code + line-by-line

This roofline calculator is **offline-runnable** — pure CPU, no GPU, no network. It turns $\min(P, I\cdot B)$ into numbers so "decode wastes the GPU" becomes arithmetic you can poke at.

```python title="roofline.py"
"""Roofline calculator: attainable throughput vs arithmetic intensity (pure CPU)."""
from dataclasses import dataclass


@dataclass
class GPU:
    name: str
    peak_flops: float   # P, FLOP/s  (BF16 dense; illustrative)
    bandwidth: float    # B, byte/s  (HBM; illustrative)

    @property
    def ridge_point(self) -> float:
        return self.peak_flops / self.bandwidth        # I* = P / B  (FLOP/byte)


def attainable(gpu: GPU, intensity: float) -> float:
    return min(gpu.peak_flops, intensity * gpu.bandwidth)   # the roofline: min(P, I*B)


if __name__ == "__main__":
    # Illustrative RTX 4090: ~165 TFLOP/s BF16 dense, ~1 TB/s GDDR6X.
    gpu = GPU("RTX 4090", peak_flops=165e12, bandwidth=1.0e12)
    print(f"{gpu.name}: ridge point I* = {gpu.ridge_point:.0f} FLOP/byte\n")

    # Sweep intensities that bracket real LLM phases:
    #   decode ~1, some fused ops ~10, the ridge itself, prefill ~1000+.
    for I in (1, 10, gpu.ridge_point, 1000):
        got = attainable(gpu, I)
        regime = "memory-bound" if I < gpu.ridge_point else "compute-bound"
        util = got / gpu.peak_flops
        print(f"I={I:7.0f} FLOP/byte -> {got/1e12:6.1f} TFLOP/s "
              f"({util:6.1%} of peak, {regime})")
```

**Line-by-line:**

- `GPU` — a machine is *two* numbers for roofline purposes: peak compute `peak_flops` ($P$) and memory `bandwidth` ($B$). Everything else (SM count, clocks) rolls up into $P$.
- `ridge_point` — $I^{*} = P/B$, the intensity where the sloped bandwidth roof meets the flat compute roof. Left of it = memory-bound; right = compute-bound.
- `attainable` — the roofline itself: `min(P, I*B)`. Below the ridge the `I*B` term wins; above it, `P` caps you.
- `__main__` — sweeps four intensities that bracket real phases. Watch decode ($I=1$) attain a rounding-error fraction of peak while $I=1000$ (prefill-like) saturates it.

Expected output (exact arithmetic, not a benchmark):

```text
RTX 4090: ridge point I* = 165 FLOP/byte

I=      1 FLOP/byte ->    1.0 TFLOP/s (  0.6% of peak, memory-bound)
I=     10 FLOP/byte ->   10.0 TFLOP/s (  6.1% of peak, memory-bound)
I=    165 FLOP/byte ->  165.0 TFLOP/s (100.0% of peak, compute-bound)
I=   1000 FLOP/byte ->  165.0 TFLOP/s (100.0% of peak, compute-bound)
```

Decode at $I=1$ uses **0.6%** of the card. That 0.6% is not a bug in your code — it is the roofline telling you the workload, not the GPU, is the limit. Everything in Parts 4–7 is a fight to raise that fraction.

## 5 · Lab — read your card's real roofline inputs

!!! gpu "GPU Lab"
    - **Min VRAM:** any CUDA GPU (this lab only *reads* device properties; no model is loaded)
    - **Suggested AutoDL card:** RTX 4090 (24 GB) — matches the baseline
    - **Est. time / cost:** ~5 min · ~¥0.5 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** `torch.cuda` properties are NVIDIA/ROCm-specific; on AMD ROCm the same PyTorch call works but field names/units can differ, and CPU/TPU/Neuron backends expose their own tooling — check your platform's docs.

Don't trust a slide's TFLOPS. Read the two roofline inputs your card actually reports, then feed them into `roofline.py` above:

```python title="device_roofline.py"
import torch

props = torch.cuda.get_device_properties(0)
print("name:            ", props.name)
print("SM count:        ", props.multi_processor_count)   # SMs available to swap warps across
print("total VRAM (GB): ", round(props.total_memory / 1024**3, 1))

# Peak FLOP/s and HBM bandwidth are NOT exposed by torch; take them from the
# vendor spec sheet (illustrative 4090: ~165 TFLOP/s BF16 dense, ~1 TB/s) and
# plug BOTH into roofline.py to get *your* ridge point.
```

**What to observe:** `multi_processor_count` is the width of the execution fabric (128 on a 4090); `total_memory` is your KV-cache + weights budget (the [KV Cache](kv-cache.md) ceiling). PyTorch deliberately does *not* hand you peak FLOP/s or bandwidth — those are the vendor's numbers, and the whole point of this lesson is that the *bandwidth* one, not the FLOP/s one, governs decode. A stretch exercise: time a large `torch.mm` (compute-bound) versus a big `x.copy_()` (bandwidth-bound) and confirm the copy gets nowhere near peak FLOP/s — the roofline in the wild.

## 6 · Common pitfalls / counter-intuitive points

- **"Higher TFLOPS = faster inference."** Only above the ridge point. Decode lives at $I \approx 1 \ll I^{*}$, so a card with 2× the FLOPS but the *same* bandwidth decodes at essentially the same speed. Buy bandwidth for decode, not FLOPs.
- **The peak TFLOPS on the box often assumes sparsity or FP8.** The headline "≈330 TFLOP/s" for a 4090 is the 2:4-sparse figure; dense BF16 is roughly half. Always ask "dense or sparse? what dtype?" before quoting a peak.
- **Occupancy is not throughput.** Maxing out resident warps hides *latency*; it cannot exceed the *bandwidth* roof. A memory-bound kernel at 100% occupancy is still bandwidth-bound.
- **Achievable bandwidth < peak bandwidth.** Real kernels hit maybe 70–85% of the spec-sheet HBM number. The roofline's $B$ is a ceiling, not a promise.
- **HBM vs SRAM is the whole game.** "The GPU is slow" almost always means "I'm re-reading HBM." FlashAttention is famous precisely because it keeps the attention working set in SRAM instead of round-tripping through HBM.

## 7 · Interview links

- [GPU memory hierarchy & roofline](../interview/gpu-memory-hierarchy.md) — the high-frequency question this lesson prepares you for: *walk the memory tiers and SM/warp model, then use the roofline and its ridge point to explain why LLM decode is memory-bound.*

## 8 · Summary & further reading

**One line:** a GPU is a bandwidth machine wearing a compute machine's spec sheet — the roofline $\min(P, I\cdot B)$ and its ridge point $I^{*}=P/B$ tell you that LLM decode ($I\approx1$) is pinned to the bandwidth roof, so every decode optimization is a move to raise intensity or dodge HBM.

Further reading:

- Williams, Waterman, Patterson — *Roofline: An Insightful Visual Performance Model* (the origin of the model).
- Dao et al. — *FlashAttention* — the canonical "keep it in SRAM" IO-aware kernel.
- Your GPU's vendor whitepaper — for the *dense* peak FLOP/s and HBM bandwidth to feed the roofline.
- The [Inference Flow](inference-flow.md) lesson — where decode's $I\approx1$ intensity came from.

## 9 · Self-check

??? question "State the roofline equation and its ridge point, and say in one sentence why decode is memory-bound on a 4090."
    Attainable throughput $= \min(P,\ I\cdot B)$ where $P$ = peak FLOP/s, $B$ = bandwidth, $I$ = arithmetic intensity; the ridge point is $I^{*}=P/B$ (≈165 FLOP/byte for a 4090 at ~165 TFLOP/s and ~1 TB/s). Decode's $I\approx1$ is ~two orders of magnitude below $I^{*}$, so it sits on the sloped bandwidth roof at ~1 TFLOP/s (< 1% of peak) — memory-bound.

??? question "A vendor doubles a GPU's TFLOPS but keeps HBM bandwidth the same. Does single-stream decode get faster? Does prefill?"
    Single-stream decode: essentially **no** — it's below the ridge, pinned to $I\cdot B$, and $B$ is unchanged. Prefill: **yes** — it's compute-bound (above the ridge), so it rides the higher $P$ roof. This is why "decode wants bandwidth, prefill wants FLOPs" and why the two phases benefit from different hardware.

??? question "Why does raising a memory-bound kernel's occupancy from 50% to 100% often not speed it up?"
    Occupancy governs *latency hiding* — how many warps an SM can swap between while waiting on memory. Once there are enough warps to keep the memory pipe saturated, more warps add no bandwidth; the kernel is already limited by bytes/second ($B$), not by idle SMs. You'd need higher $B$ (better hardware), fewer bytes (quantization), or higher $I$ (batching, fusion) — not more occupancy.
