# Parallelism: TP/PP/DP/EP & when to use each

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 7 · Multi-GPU & Distributed   ·   **Tests the lesson:** [Why Parallelize, and How: Tensor / Pipeline / Data / Expert Parallelism](../part7/parallelism-strategies.md)

---

## Q: Why parallelize at all, what does each of TP/PP/DP/EP split and cost to communicate, why does TP stay within a node while PP crosses them, and how do you pick a strategy?

### Direct answer

You parallelize for **exactly two reasons**, and naming which one matters:

1. **Capacity — it won't fit.** Weights + [KV cache](../part0/kv-cache.md) + activations exceed one GPU (a 70B model is ~130 GB FP16, ~33 GB INT4 — both over a 24 GB card). You must **split the model**.
2. **Throughput — it's too slow.** The model fits but one GPU can't hit your QPS/SLO. You **replicate** it.

Four cuts, each with a different axis and communication tax:

- **Tensor Parallel (TP)** — split each layer's matrices (Megatron column/row split), **two all-reduces per layer, every token**. Bandwidth-hungry → **NVLink → within one node**. `tensor_parallel_size`. Degree is a power of 2 (splits attention heads). Also cuts latency.
- **Pipeline Parallel (PP)** — split the *layers* into stages; only a **point-to-point activation handoff** per boundary → **survives across nodes**; cost is the **pipeline bubble** $\frac{p-1}{m+p-1}$, hidden by many microbatches. `pipeline_parallel_size`.
- **Data Parallel (DP)** — *replicate* the whole model, split requests. Zero per-GPU memory savings → pure **throughput** lever, **requires the model to fit**. `--data-parallel-size`.
- **Expert Parallel (EP)** — **MoE only**: split experts across GPUs, **all-to-all** token routing. `--enable-expert-parallel`, experimental in 0.26.0, size = `tensor_parallel_size × data_parallel_size`.

**Decision tree:** fits on 1 GPU → single GPU (+DP for throughput); exceeds 1 GPU but fits a node → **TP** (≤ GPUs/node); exceeds a node → **TP within node + PP across nodes** (TP = GPUs/node, PP = nodes); MoE → add **EP**.

### Deep dive

- **TP mechanism.** $Y=XW$ splits $W=[W_1|\dots|W_N]$ by columns (GPU $k$ gets $Y_k=XW_k$); the next linear is row-split and consumes the slices, producing a partial sum an **all-reduce** completes — one after attention, one after the FFN = **2 per layer**. Traffic $\approx \text{tokens}\times d_\text{hidden}\times b$ per all-reduce, on *every* token including every decode step → small, frequent, latency-sensitive → needs NVLink, capped at a node.
- **Why the topology split.** TP's per-token all-reduce is bandwidth-bound → NVLink → within a node. PP only ships activations point-to-point at stage boundaries → tolerant of slow/high-latency links → across nodes. Hence **TP = GPUs/node, PP = nodes**.
- **PP bubble.** $p$ stages, $m$ microbatches → idle fraction $\frac{p-1}{m+p-1}$; feed $m \gg p$ to amortize.
- **DP ≠ memory.** Every replica is a full copy; it never helps "won't fit," only "too slow."
- **EP is auto-sized.** You set TP and DP; EP falls out as their product. Only meaningful for MoE.
- **They compose.** Real large-scale serving = TP (intra-node) × PP (inter-node) × DP (replicas) × EP (MoE), on orthogonal axes.

### Code

The fit-or-parallelize decision plus TP's per-token all-reduce tax (Qwen/Llama shapes; a memory/communication model):

```python
GPU, KV = 24, 6; AVAIL = GPU - KV                        # one 24GB card; ~6GB for KV+act (illustrative)
wgb  = lambda p_b, b: p_b*1e9*b/1024**3                  # weight GB: params(B) x bytes/param
def min_tp(p_b, b):                                      # smallest power-of-2 TP so a shard fits
    need, tp = wgb(p_b, b), 1
    while need/tp > AVAIL: tp *= 2                        # power of 2 -> heads split evenly
    return tp
ar_kb = lambda h, L, b=2: 2*L*h*b/1024                   # 2 all-reduces/layer, ~h wide, per token
for name, p_b, h, L in [("7B",7.6,3584,28),("70B",70,8192,80),("405B",405,16384,126)]:
    print(name, f"{wgb(p_b,2):.0f}GB", "minTP", min_tp(p_b,2), f"AR {ar_kb(h,L):.0f}KB/tok")
# 7B 14GB minTP 1 AR 392KB/tok | 70B 130GB minTP 8 AR 2560KB/tok | 405B 754GB minTP 64 AR 8064KB/tok
```

### Interviewer follow-ups

- *"Won't-fit vs too-slow — which family?"* → Won't fit = split (TP/PP/EP); too slow but fits = replicate (DP). Wrong family → OOM or wasted comm.
- *"Why not run TP across nodes?"* → Its all-reduce fires every layer, every token; on a slow link the GPUs starve on the wire. TP stays on NVLink within a node; PP crosses nodes.
- *"Does DP save memory?"* → No — full copy per replica. Throughput only, requires the model to fit.
- *"Why is TP degree a power of 2?"* → It splits attention heads; the degree must divide the head count evenly.
- *"How do you serve a 70B on 8×A100 across 2 nodes?"* → `--tensor-parallel-size 4 --pipeline-parallel-size 2` — TP within each node, PP across the two.
- *"When is EP relevant, and how big is it?"* → MoE only; experimental in 0.26.0; size auto = TP×DP, not set directly.
- *"A model fits but you still see TP>1 — why?"* → To cut latency (TTFT/TPOT); more GPUs crunch each token, at the cost of the all-reduce tax.

### Linked concepts

- Lesson: [Why Parallelize, and How: TP/PP/DP/EP](../part7/parallelism-strategies.md)
- Related: [VRAM budget & max concurrency](vram-capacity-planning.md) (the fit wall & KV budget), [Attention variants: MHA/MQA/GQA](attention-variants.md) (heads TP splits), [KV cache & throughput ceiling](kv-cache.md), [Quantization: why it speeds up inference](quantization-basics.md) (shrink before you split)
- Glossary: [Tensor/Pipeline/Data/Expert Parallelism, Collective communication, TP degree](../glossary.md)
