# Prefill vs decode

!!! info "Baseline: **vLLM 0.26.0** · verified via Context7 (ADR-0004)"

**Module:** Part 0 · Foundations   ·   **Tests the lesson:** [Inference Flow: Prefill & Decode](../part0/inference-flow.md)

---

## Q: Explain the prefill and decode phases of LLM inference. Which is compute-bound, which is memory-bound, and why? What does each imply for latency (TTFT/TPOT) and for batching?

### Direct answer

Inference has two phases because generation is **autoregressive**. **Prefill** is the single forward pass that ingests the whole prompt — all prompt tokens processed **in parallel**, producing the first output token and the prompt's KV cache. **Decode** is the loop that follows: one forward pass per output token, each **serial** (token *t+1* needs token *t*'s value) and each reusing the KV of everything before it.

Prefill is **compute-bound**: it does ~$2N$ FLOPs per token across many tokens while reading each weight from HBM only once, so arithmetic intensity is high and the GPU's math units saturate. Decode is **memory-bound**: each step re-reads *all* weights and the *entire* KV cache to emit a single token, so arithmetic intensity is ≈ 1 FLOP/byte and the GPU stalls on HBM bandwidth.

Consequences: **TTFT** (time to first token) is dominated by prefill; **TPOT** (time per output token) is dominated by decode. Batching helps decode a lot (running many sequences' steps together reuses weights across the batch, raising intensity) but does little for a single stream.

### Deep dive

- **Why the split is forced.** You can't compute token 51 before token 50 exists — decode is inherently serial. But the prompt is fully known up front, so prefill can be one wide parallel matmul. Same math, opposite parallelism profile.
- **Arithmetic intensity, precisely.** With $N$ params, $b$ bytes/weight, $\kappa$ KV bytes/token: prefill intensity $\approx \frac{2NS}{Nb + \kappa S} \approx \frac{2S}{b}$ while $\kappa S \ll Nb$ (so it grows ~linearly with prompt length $S$ across any realistic context); decode intensity $\approx \frac{2N}{Nb + \kappa S} \le \frac{2}{b} = 1$ for BF16. A GPU wants hundreds of FLOP/byte to stay busy — prefill clears it, decode never does.
- **Where the wall-clock goes.** A request's latency ≈ TTFT (one prefill over the prompt) + (#output tokens × TPOT). Long prompt → prefill dominates TTFT; long answer → decode dominates total time, roughly linearly.
- **What each phase's optimizations target.** Prefill: chunked prefill, PD disaggregation (keep TTFT bounded). Decode: continuous batching, PagedAttention, speculative decoding (fight the memory wall and the serial dependency).

### Code

The regimes fall out of a FLOP/byte estimate — no GPU needed:

```python
N = 7_615_000_000        # ~7.6B params (Qwen2.5-7B)
b = 2                    # BF16 bytes/weight
kappa = 57344            # KV bytes/token: 2*28*4*128*2

def prefill_I(S): return (2*N*S) / (N*b + kappa*S)   # grows with prompt length
def decode_I(S):  return (2*N)   / (N*b + kappa*S)   # pinned near 1

print(round(prefill_I(1024), 1), round(decode_I(1024), 2))   # ~1020.1  ~1.00
```

### Interviewer follow-ups

- *"Why can't you just parallelize decode?"* → serial data dependency: token *t+1* is conditioned on the sampled value of token *t*. You parallelize *across* requests (batching) or *guess* ahead (speculative decoding), never within one stream's own future.
- *"A user complains the first token is slow but streaming is fast. Diagnose."* → long prompt → prefill-heavy TTFT; the fast streaming is decode. Fixes: cap/trim prompt, prefix caching, chunked prefill.
- *"Does a bigger batch lower TPOT?"* → per-request TPOT may rise slightly, but *aggregate* token throughput rises a lot because decode's weight reads amortize across the batch — the point of continuous batching.
- *"Prefill or decode: which benefits more from higher HBM *bandwidth* vs more *FLOPs*?"* → decode wants bandwidth (it's memory-bound); prefill wants FLOPs (compute-bound).

### Linked concepts

- Lesson: [Inference Flow: Prefill & Decode](../part0/inference-flow.md)
- Related lesson: [Transformer, the Infra View](../part0/transformer-infra.md) (where the $2N$ FLOPs come from)
- Glossary: [Prefill, Decode, Memory-bound / Compute-bound, TTFT, TPOT, Roofline](../glossary.md)
