# Speculative decoding: guess-and-verify

!!! info "Baseline: **vLLM 0.26.0** · config verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [Speculative Decoding: Guess Many, Verify Once](../part5/speculative-decoding.md)

---

## Q: Explain speculative decoding. Why is it a "free lunch" only because decode is memory-bound, what sets the speedup, does it change outputs, and when does it backfire?

### Direct answer

A cheap **draft** proposes the next K tokens; the big **target** runs **one** forward pass over all K+1 positions and verifies them, accepting the longest correct prefix and emitting its own token at the first mismatch. So you get multiple tokens per expensive target weight-read — and the **output is bit-identical** to vanilla target decoding (verification makes it exact).

It's nearly free **only because [decode is memory-bound](../part0/inference-flow.md)**: a target forward over K+1 tokens reads the weights **once** (same as one token) and uses the GPU's otherwise-idle compute for the extra positions. Convert idle FLOPs → fewer HBM round-trips.

**Speedup = expected accepted run length** $\sum_{i=0}^{K}\alpha^i=(1-\alpha^{K+1})/(1-\alpha)$, where $\alpha$ is the per-token draft/target agreement. At $\alpha=0.7$, K=4 ≈ 2.77× fewer target passes.

**It backfires** as the batch grows compute-bound: the "idle" compute is now used serving the batch, so verifying extra tokens isn't free and the draft overhead can make you *slower*. It's a low-batch **latency** tool, not a throughput tool.

### Deep dive

- **The draft sources** (`method`): `ngram` (prompt-lookup, no draft model — great when output echoes input: summarization/RAG/editing), `eagle`/`eagle3` (tiny trained head, high acceptance, small VRAM — modern default), `draft_model` (small standalone model, more general but its own forwards cost more). `num_speculative_tokens` = K.
- **Acceptance rate is everything.** A cheap draft that rarely agrees (low $\alpha$) wins little; the design tension is a draft that's *both* cheap *and* agrees often (why EAGLE exists).
- **Diminishing K.** $\sum\alpha^i$ saturates; past a point extra draft tokens rarely survive but always cost draft compute. Tune K to $\alpha$.
- **VRAM.** `draft_model`/EAGLE checkpoints share GPU memory with the target, shrinking the [KV-cache budget](../part5/paged-attention.md); `ngram` is the zero-VRAM option.

### Code

The speedup as a deterministic function of acceptance rate:

```python
def tokens_per_target_pass(alpha, k):
    return sum(alpha**i for i in range(k+1))    # 1 + a + ... + a^k == (1 - a^(k+1))/(1 - a)
for a in (0.5, 0.7, 0.9):
    print(a, round(tokens_per_target_pass(a, 4), 2))   # 0.5→1.94  0.7→2.77  0.9→4.10
# vanilla = 1.00 token/pass, so the number IS the speedup in target forwards (before draft cost)
```

### Interviewer follow-ups

- *"Does it hurt quality?"* → No — verification makes output identical to the target alone. Pure latency.
- *"Why does it fade at large batch?"* → Decode turns compute-bound; the idle compute that made verification free is gone, so the draft overhead isn't repaid.
- *"Bigger K always better?"* → No — diminishing returns via $\sum\alpha^i$; large K on low $\alpha$ wastes draft compute. Match K to $\alpha$.
- *"Which method for summarization?"* → `ngram` — output echoes input, so prompt-lookup hits often at zero draft-model cost / no extra VRAM.
- *"What's the single biggest lever?"* → Acceptance rate $\alpha$ (draft/target alignment) — it dominates the whole $\sum\alpha^i$.

### Linked concepts

- Lesson: [Speculative Decoding](../part5/speculative-decoding.md)
- Related: [Prefill vs decode](prefill-vs-decode.md) (why decode is memory-bound — the premise), [Arithmetic intensity](arithmetic-intensity.md) (the compute ridge where it fades), [Static vs continuous batching](continuous-batching.md) (the throughput axis it does *not* address)
- Glossary: [Speculative decoding, Decode](../glossary.md)
