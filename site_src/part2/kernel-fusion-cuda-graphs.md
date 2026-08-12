# Kernel Fusion & CUDA Graphs: Killing Decode Launch Overhead

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    vLLM flags (`enforce_eager`, `compilation_config` / `cudagraph_mode`) and the PyTorch `torch.cuda.graph` API are verified against vLLM 0.26.0 / PyTorch via Context7 (ADR-0004). Kernel counts, launch overheads, and throughput figures are **illustrative / order-of-magnitude references** — measure your own. The launch-overhead arithmetic in §4 is *exact* (it's the model's own definition).

---

## 1 · Intuition & why it matters

You've spent Part 2 proving decode is [memory-bound](roofline-analysis.md): each step moves a lot of bytes for little compute. There's a *second*, sneakier tax on decode that has nothing to do with bandwidth — **kernel launch overhead**. A single decode step doesn't run one kernel; it runs *hundreds* — an RMSNorm, a QKV projection, the attention op, an output projection, a gate/up projection, a SiLU, a down projection, residual adds, twice per layer, across ~28 layers. Every one of those is a separate GPU kernel, and every launch costs the **CPU** a few microseconds to dispatch.

Because decode kernels are small (batch 1, one token), the GPU finishes each almost instantly and then **sits idle waiting for the CPU to launch the next one**. Add up a few microseconds × hundreds of kernels and you get a fixed per-step tax that can be 15–30% of the step — and it gets *relatively worse* the more you optimize the compute (quantize the weights, and the GPU work shrinks while the launch overhead stays put). Two techniques kill it: **kernel fusion** merges many ops into one kernel (fewer launches, fewer HBM round-trips), and **CUDA graphs** record the entire kernel sequence once and replay it with a *single* launch. This is the last single-GPU lever in Part 2, and it's why vLLM captures CUDA graphs by default. → see the [Glossary](../glossary.md) for *CUDA graphs*, *Kernel fusion*.

## 2 · Mental model

The problem is a CPU↔GPU ping-pong; both fixes change who does the launching:

```text
EAGER — CPU launches each kernel; GPU idles between tiny kernels
  CPU: [launch k1]   [launch k2]   [launch k3]  ...  (~5 µs submit each)
  GPU:    [k1]▪▪gap▪▪  [k2]▪▪gap▪▪  [k3]▪▪gap▪▪       <- GPU waits on the CPU
          └ tiny memory-bound kernel; done before the next launch arrives

KERNEL FUSION — merge ops so there are fewer, bigger kernels
  CPU: [launch fused]          GPU: [ norm+proj+bias+act fused ]   <- fewer launches,
                                                                       fewer HBM trips

CUDA GRAPH — record the whole sequence once, replay with ONE launch
  capture (once):  record k1..kN into a graph
  every step:      CPU: [g.replay()]   GPU: [k1][k2][k3]...[kN]   <- back-to-back,
                                                                     no per-kernel gaps
```

Two shapes to hold:

- **A launch is CPU work; the GPU can starve on it.** When a kernel's GPU time is smaller than the CPU's launch+dispatch latency, the GPU finishes and waits. Decode's kernels are exactly that small, so the step becomes *launch-bound* on top of memory-bound. Prefill kernels are big (many tokens), so their launch overhead is a rounding error — this tax is **decode-specific**.
- **Fusion shrinks the count; graphs amortize the launches.** Fusion attacks $N$ (and the intermediate HBM traffic between fused ops); CUDA graphs attack the per-launch cost by replaying $N$ kernels with one CPU submit. They compose: vLLM fuses what it can *and* wraps the result in a graph.

## 3 · Principle & math

### 3.1 The launch-overhead model

Model a decode step as $N$ kernels with per-launch CPU overhead $\tau$, plus the actual GPU compute time $T_{\text{compute}}$ (which for memory-bound decode is essentially [bytes ÷ bandwidth](roofline-analysis.md)). In **eager** mode the tiny kernels can't hide the launches, so they add up:

$$
T_{\text{eager}} \approx T_{\text{compute}} + N\,\tau
$$

A **CUDA graph** replays all $N$ kernels with a single CPU submit, so the $N\tau$ term collapses:

$$
T_{\text{graph}} \approx T_{\text{compute}} + \tau, \qquad
\text{speedup} = \frac{T_{\text{compute}} + N\tau}{T_{\text{compute}} + \tau}
$$

The speedup is large exactly when $N\tau$ is comparable to $T_{\text{compute}}$ — i.e. when compute is *small*. That's the counter-intuitive tie-back to the roofline: the more you shrink the GPU work (quantize weights, small batch), the *bigger* the relative launch tax, and the more CUDA graphs buy you. A 7B at $\sim430$ kernels, $\tau\approx5\,\mu s$ gives $N\tau\approx2.15$ ms/step — trivial next to prefill, decisive for decode.

### 3.2 Why fusion helps twice

A fused kernel does two things at once. It **cuts the launch count** ($N$ down), and — for elementwise/reduction chains — it **avoids HBM round-trips**: instead of kernel A writing its output to HBM and kernel B reading it back, the fused kernel keeps the intermediate in registers/SRAM. That's a bandwidth win on top of the launch win, and it's why fusing the many small ops *around* the big GEMMs (norms, bias, activation, residual) matters even though the GEMM itself is already one kernel.

### 3.3 What CUDA graphs require — and why `enforce_eager` exists

A CUDA graph records **specific kernels operating on specific memory addresses**. Replay reuses those captured pointers, so: (1) shapes must be **static** — vLLM captures a graph per batch-size bucket and *pads* the running batch up to the nearest captured size; (2) inputs must be copied into the same static buffers each step; (3) you must **warm up** before capturing (so allocations/autotuning settle). Dynamic control flow or novel shapes fall back to eager. Capturing graphs also costs **memory** (the captured buffers), which is why vLLM exposes `enforce_eager=True` to disable them — trading decode throughput for VRAM and flexibility. (The same VRAM you could instead spend on [KV cache](kv-cache-math.md).)

## 4 · Complete runnable code + line-by-line

This turns the launch-overhead model into numbers — **pure CPU, offline-runnable**, no GPU. It shows *why* quantized models benefit more from CUDA graphs.

```python title="launch_overhead.py"
"""Decode launch-overhead model: eager vs CUDA-graph step time (pure CPU, offline)."""
from dataclasses import dataclass


@dataclass
class DecodeStep:
    weight_gib: float                 # bytes pulled per decode step (weights dominate)
    n_layers: int = 28                # Qwen2.5-7B
    kernels_per_layer: int = 15       # norms, projections, attn, act, adds (illustrative)
    launch_us: float = 5.0            # CPU-side per-kernel launch overhead (illustrative)
    bandwidth_bps: float = 1.0e12     # ~1 TB/s HBM (illustrative 4090)

    @property
    def n_kernels(self) -> int:
        return self.n_layers * self.kernels_per_layer + 10        # + embed / final / lm_head

    @property
    def compute_ms(self) -> float:
        return self.weight_gib * 1024**3 / self.bandwidth_bps * 1e3   # bytes / bandwidth

    @property
    def launch_ms(self) -> float:
        return self.n_kernels * self.launch_us / 1e3              # N * tau

    def eager_ms(self) -> float:
        return self.compute_ms + self.launch_ms                   # T_compute + N*tau

    def graph_ms(self) -> float:
        return self.compute_ms + self.launch_us / 1e3             # T_compute + one submit


if __name__ == "__main__":
    for label, w in (("AWQ  weights (~5.5 GiB)", 5.5), ("BF16 weights (~14.2 GiB)", 14.2)):
        s = DecodeStep(weight_gib=w)
        e, g = s.eager_ms(), s.graph_ms()
        print(f"{label}: {s.n_kernels} kernels | compute {s.compute_ms:5.1f} ms | "
              f"launch {s.launch_ms:4.2f} ms")
        print(f"   eager {e:5.1f} ms -> {1000/e:5.1f} tok/s | "
              f"graph {g:5.1f} ms -> {1000/g:5.1f} tok/s | speedup {e/g:.2f}x")
```

**Line-by-line:**

- `n_kernels` — a decode step's kernel count: ~15 per layer × 28 layers, plus embedding/final-norm/lm_head. Hundreds of launches, each tiny.
- `compute_ms` — the memory-bound decode compute: bytes pulled (dominated by weights) ÷ bandwidth. This is the roofline result from earlier in Part 2 — decode is bandwidth-bound, so weight bytes set the floor.
- `launch_ms` — $N\tau$, the fixed CPU-side tax that eager mode pays every step.
- `eager_ms` vs `graph_ms` — the model from §3.1: eager pays all $N$ launches; the graph pays one. The gap is pure overhead a graph reclaims.
- `__main__` — runs it for AWQ vs BF16 weights so you can see the quantized model's *bigger* relative win.

Expected output (exact arithmetic, not a benchmark):

```text
AWQ  weights (~5.5 GiB): 430 kernels | compute   5.9 ms | launch 2.15 ms
   eager   8.1 ms -> 124.1 tok/s | graph   5.9 ms -> 169.2 tok/s | speedup 1.36x
BF16 weights (~14.2 GiB): 430 kernels | compute  15.2 ms | launch 2.15 ms
   eager  17.4 ms ->  57.5 tok/s | graph  15.3 ms ->  65.6 tok/s | speedup 1.14x
```

Same 2.15 ms launch tax in both — but it's **36%** on top of the AWQ model's small 5.9 ms compute versus **14%** on the BF16 model's 15.2 ms. Quantize the weights to go faster, and CUDA graphs matter *more*, not less: you shrank the compute, so the fixed launch overhead looms larger. That's why vLLM captures graphs by default and why it's most impactful on the quantized, small-batch decode you'll run on a 4090.

## 5 · Lab — measure the launch tax, then toggle it in vLLM

!!! gpu "GPU Lab"
    - **Min VRAM:** 8 GB for Part A (tiny tensors); 24 GB for Part B (loads the model)
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~10 min · ~¥1 of GPU time (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** `torch.cuda.graph` is CUDA-only; ROCm has HIP-graph equivalents, and vLLM's `enforce_eager` toggle works on any backend but the CUDA-graph capture path is NVIDIA/ROCm-specific.

**Part A — the launch tax in raw PyTorch.** Launch many tiny kernels in a loop, then replay them as one captured graph:

```python title="cuda_graph_demo.py"
import torch

assert torch.cuda.is_available()
x = torch.zeros(1024, device="cuda")

def many_tiny_ops(x):                 # stand-in for a decode step's hundreds of small kernels
    for _ in range(200):
        x = x + 1.0                    # each is a separate, tiny, launch-bound kernel
    return x

def timed(fn, iters=100):
    for _ in range(10): fn()           # warmup
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); [fn() for _ in range(iters)]; e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms/iter

# Eager: 200 launches every iteration
eager = timed(lambda: many_tiny_ops(x.clone()))

# CUDA graph: capture the 200 ops once, replay with a single submit
static = x.clone()
g = torch.cuda.CUDAGraph()
many_tiny_ops(static)                  # warm up the exact workload before capture
with torch.cuda.graph(g):
    static_out = many_tiny_ops(static)
graph = timed(g.replay)

print(f"eager: {eager*1e3:6.1f} µs/iter   graph: {graph*1e3:6.1f} µs/iter "
      f"-> {eager/graph:.1f}x fewer launch stalls")
```

**Part B — the same lever, end to end in vLLM.** Serve with graphs on (default) vs off and compare decode throughput:

```bash
# CUDA graphs ON (default) — vLLM captures graphs per batch bucket at startup
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192

# CUDA graphs OFF — eager mode (saves VRAM, pays the launch tax every step)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192 --enforce-eager
```

**What to observe:** Part A's graph replay is many-fold faster per iteration — that gap is the launch overhead the model in §4 predicts. In Part B, `--enforce-eager` should *lower* decode throughput (tok/s) while *freeing* some VRAM (no captured graphs) — the exact trade from §3.3. For finer control than the on/off switch, vLLM exposes `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'` (verified 0.26.0). (Numbers are illustrative; measure on your box.)

## 6 · Common pitfalls / counter-intuitive points

- **Expecting CUDA graphs to help prefill.** They don't, much — prefill kernels are big and compute-bound, so launch overhead is negligible ($N\tau \ll T_{\text{compute}}$). This is a **decode** optimization, for the same reason decode is the memory-bound phase: the per-kernel GPU work is tiny.
- **Quantized ⇒ graphs matter *less*? Backwards.** Shrinking compute (AWQ, small batch) makes the fixed launch tax a *bigger* fraction of the step — graphs help *more*. §4 shows 36% vs 14%.
- **Forgetting the static-shape constraint.** Graphs capture fixed shapes and memory addresses; vLLM pads batches up to captured buckets. Novel shapes or data-dependent control flow fall back to eager — if throughput doesn't improve, check whether capture actually happened.
- **Skipping warmup before capture.** Capturing a cold workload records allocator/autotune artifacts and can crash or mis-capture. Always run the exact workload a few times first (the demo does).
- **`enforce_eager` is free memory, not free speed.** It saves the VRAM the captured graphs would use (spendable on [KV cache](kv-cache-math.md)) but pays the launch tax every decode step. It's a trade — reach for it when you're VRAM-starved or debugging, not by default.
- **Fusion ≠ graphs.** Fusion reduces the *number* of kernels (and intermediate HBM traffic); graphs reduce the *launch cost* of whatever kernels remain. `torch.compile` does fusion automatically; vLLM does both. Confusing them leads to "I fused, why didn't launches drop to one?" — fusion lowers $N$, it doesn't collapse launches to a single submit.

## 7 · Interview links

- [CUDA graphs & kernel fusion](../interview/cuda-graphs-fusion.md) — the high-frequency question this lesson prepares you for: *why is decode launch-bound but prefill isn't; what do CUDA graphs remove and what do they require; and why do quantized models benefit more?*

## 8 · Summary & further reading

**One line:** decode fires hundreds of tiny kernels per token, so CPU launch overhead ($N\tau$) becomes a fixed per-step tax that grows *relatively* as you shrink compute — kernel fusion cuts the kernel count (and intermediate HBM traffic) while CUDA graphs replay the whole sequence with one submit, which is why both are decode-phase, quantization-friendly throughput wins.

Further reading:

- PyTorch docs — *Accelerating PyTorch with CUDA Graphs* and the `torch.cuda.graph` / `make_graphed_callables` API notes.
- vLLM docs — *CUDA graphs* design doc and `compilation_config` / `cudagraph_mode` (baseline v0.26.0); `enforce_eager` in *Conserving Memory*.
- The [Operator Roofline](roofline-analysis.md) lesson — why decode's per-kernel GPU work is tiny (and thus launch-bound) in the first place.
- The [FlashAttention](flash-attention.md) lesson — the other Part 2 kernel-level win, fusing the attention op itself.

## 9 · Self-check

??? question "Why does kernel launch overhead hurt decode but barely touch prefill?"
    Launch overhead is a fixed CPU-side cost ($\tau$) per kernel. Its impact depends on how it compares to each kernel's GPU time. **Decode** kernels are tiny (batch 1, one token, memory-bound), so the GPU finishes before the CPU launches the next — the step becomes launch-bound, and $N\tau$ (hundreds of kernels × µs) is a real fraction of the step. **Prefill** kernels process many tokens at once (big, compute-bound), so $N\tau \ll T_{\text{compute}}$ and the launches hide in the shadow of real work.

??? question "What do CUDA graphs remove, and what do they require in return?"
    They remove the **per-kernel CPU launch/dispatch overhead**: a captured graph replays all $N$ kernels with a single submit, collapsing $N\tau$ to $\approx\tau$. In return they require **static shapes and fixed memory addresses** (so vLLM captures a graph per batch-size bucket and pads to it), inputs copied into the same static buffers each step, and a **warmup** before capture. They also cost VRAM for the captured buffers — which is why `enforce_eager=True` exists to disable them.

??? question "You quantize a model's weights and decode compute drops from 15 ms to 6 ms per step. Do CUDA graphs now matter more or less?"
    **More.** The launch tax $N\tau$ (say ~2 ms) is fixed, so it goes from ~14% of a 15 ms step to ~36% of a 6 ms step — the relative overhead grew when you shrank the compute. Optimizing the GPU work makes the fixed launch overhead loom larger, so CUDA graphs (and fusion) deliver a bigger relative speedup on quantized, small-batch decode — exactly the regime you run on a single 4090.
