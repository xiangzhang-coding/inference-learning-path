# MoE inference: active vs total params & expert routing

!!! info "Baseline: **vLLM 0.26.0** · flags verified via Context7 (ADR-0004)"

**Module:** Part 7 · Multi-GPU & Distributed   ·   **Tests the lesson:** [Why Parallelize, and How: Tensor / Pipeline / Data / Expert Parallelism](../part7/parallelism-strategies.md)

---

## Q: For a Mixture-of-Experts model at serving time, explain active vs total parameters and how expert routing works — and what that means for memory, compute, and multi-GPU serving.

### Direct answer

An **MoE** replaces the dense FFN in each block with $N$ expert FFNs plus a small **router** (gating network). Per token, the router scores the experts and sends the token to its **top-$k$** (e.g. 2 of 8); only those experts run.

- **Total params** — *all* experts, all resident in VRAM. **Memory tracks total.**
- **Active params** — only the routed experts compute for a given token. **Compute (FLOPs) tracks active.**

So an MoE buys **dense-model quality at a fraction of the per-token compute**, but you still pay the **full memory** for every expert (all must be loaded). Example magnitudes (示例 / 量级参考): Mixtral 8×7B ≈ **47B total / ~13B active**; DeepSeek-V3 ≈ **671B total / 37B active**; Qwen3-MoE families sit on the same active≪total curve.

**At serving:** the router picks experts **per token, per layer**. Within a batch, different tokens route to different experts → the compute is an irregular, imbalanced gather. Across GPUs, **Expert Parallelism (EP)** (`--enable-expert-parallel`) shards *which experts live where*; an **all-to-all** ships each token to the GPU that owns its expert and ships the result back. EP exists because the thing that doesn't fit is the **experts' memory** — TP alone would shard every expert's matrices and pay an all-reduce per layer instead.

### Deep dive

- **The router** is a linear layer producing gate logits over $N$ experts; a top-$k$ softmax gives weights, and the token's output is the weighted sum of its $k$ experts' outputs.
- **Decode can be memory-bound worse than "active params" suggests.** At small batch, the weights actually *read* from HBM depend on how many **distinct** experts the batch touches — a scattered batch can pull far more than $k$ experts' worth of weights, so effective bandwidth ≠ active-param count.
- **Sizing VRAM uses total; estimating decode speed uses the weights read.** Two different columns — the classic MoE trap is quoting active params for a VRAM budget.
- **EP vs TP for the experts:** TP shards *each expert's* matmul (all-reduce per layer, NVLink-bound, within a node); EP shards *expert ownership* (all-to-all). Different collective, different scaling. vLLM sets **EP size = TP × DP** (see the [parallelism lesson](../part7/parallelism-strategies.md) §8).
- **Load balance matters at serving.** Routers can be lopsided; a hot expert becomes the bottleneck. Training adds an auxiliary balancing loss; serving relies on balanced routing / capacity so one GPU's experts don't stall the all-to-all.

### Code

A toy single-layer MoE forward — router top-$k$, gather, weighted sum — and an active-vs-total count (pure NumPy, no GPU):

```python
import numpy as np

def moe_layer(x, experts_W, router_W, k=2):
    # x: [d]; experts_W: [N, d, d]; router_W: [d, N]
    gate = x @ router_W                       # [N] expert logits
    topk = np.argsort(gate)[-k:]              # route to top-k experts
    w = np.exp(gate[topk] - gate[topk].max()); w /= w.sum()   # softmax weights
    y = sum(wi * (x @ experts_W[e]) for wi, e in zip(w, topk))
    return y, topk

N, d, k = 8, 16, 2
experts_W = np.random.randn(N, d, d) / d
total  = experts_W.size + d * N               # all experts resident (memory)
active = k * d * d + d * N                     # only k experts compute (FLOPs)
print(f"total={total}  active={active}  ratio={active/total:.2f}")  # 示例
```

### Interviewer follow-ups

- *"A 57B MoE — how much VRAM?"* → Track **total** (all experts resident): ~57B × bytes/param + KV/overhead. Never size on the active count.
- *"Why is MoE decode sometimes not as fast as its active size?"* → A scattered batch reads many distinct experts' weights; effective HBM traffic exceeds the active-param figure. It's still memory-bound.
- *"How does vLLM serve MoE across GPUs?"* → **EP** (`--enable-expert-parallel`) shards experts; an all-to-all routes tokens to the owning GPU. Combinable with TP/DP (EP size = TP × DP).
- *"How does the router choose experts?"* → Gate logits → top-$k$ → normalized weights → weighted sum of those experts' outputs, per token per layer.
- *"TP vs EP for the experts — what's the difference?"* → TP shards each expert's matmul (all-reduce, within a node); EP shards expert ownership (all-to-all). EP relieves the experts' memory footprint that TP would otherwise fight.

### Linked concepts

- Lesson: [Why Parallelize, and How: TP / PP / DP / EP](../part7/parallelism-strategies.md) — where EP and the fit-first decision tree live.
- Related: [Parallelism: TP/PP/DP/EP & when to use each](parallelism-strategies.md), [Quantizing & serving in practice](quantization-serving.md) (all experts resident → quantize to fit), [VRAM budget & max concurrency](vram-capacity-planning.md)
- Glossary: [MoE, Expert parallelism](../glossary.md)
