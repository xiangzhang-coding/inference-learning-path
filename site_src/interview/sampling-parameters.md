# Sampling parameters: temperature, top-p / top-k & throughput

!!! info "Baseline: **vLLM 0.26.0** · flags verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [The Tuning Knobs: Sweeping the Throughput/Latency Curve](../part5/tuning-knobs-sweep.md)

---

## Q: Explain temperature, top-p, and top-k; what is greedy decoding; and how does the choice of sampling parameters affect batching and throughput?

### Direct answer

Sampling parameters reshape the probability distribution the model draws from **each decode step** — they act on the logits *after* the forward pass, so they change *what* comes out, not *how fast* the matmuls run.

| Param | vLLM default | What it does |
|---|---|---|
| `temperature` $T$ | `1.0` | Scales logits $z_i \to z_i / T$ before softmax. $T<1$ sharpens (more deterministic), $T>1$ flattens (more random), $T=0$ → **greedy** (argmax). |
| `top_k` | `0` | Keep only the $k$ highest-probability tokens, renormalize, then sample. `0` (or `-1`) **disables** it — consider all tokens. |
| `top_p` | `1.0` | **Nucleus**: keep the smallest set of tokens whose cumulative probability $\ge p$, renormalize, sample. `1.0` disables it. |

**Greedy vs sampling:** greedy = `temperature=0` → pick the argmax token every step. It's deterministic and reproducible (best for eval / A-B). Sampling (`temperature>0`, optionally with `top_p`/`top_k`) trades reproducibility for diversity.

**Throughput impact — small, and not where people think.** The sampling step is an $O(\text{batch} \times \text{vocab})$ reduction over the logits; it's negligible next to the $O(\text{batch} \times \text{params})$ memory-bound forward pass. So raising `temperature` or turning on `top_p` **does not meaningfully cost throughput** — batch width and KV capacity set throughput, not the sampler. The one real coupling: vLLM applies logits processors at **batch granularity**, and argmax-invariant processors can be skipped only if *every* request in the batch is greedy — so a batch mixing greedy and sampled requests forgoes that small optimization.

### Deep dive

- **The per-step pipeline is batched:** forward → logits `[batch, vocab]` → penalties + temperature → `top_k`/`top_p` filter → softmax → sample. Every request in the running batch flows through it together.
- **`temperature=0` is a short-circuit**, not "divide by zero" — vLLM takes the argmax directly (no RNG), which is why greedy is deterministic given the same batch composition.
- **`top_k` and `top_p` compose** (both can be active); each just prunes the candidate set before the (re-normalized) softmax.
- **Defaults disable the truncators.** In vLLM 0.26.0 the OpenAI-compatible fallbacks are `temperature=1.0`, `top_p=1.0`, `top_k=0`, `min_p=0.0` — i.e. sample from the full softmax unless you narrow it.
- **For the sweep, pin sampling.** When you tune a serving knob you fix `temperature=0` + a `seed` so the **quality** axis is deterministic and the delta is attributable to the knob — the discipline the [tuning-knobs lesson](../part5/tuning-knobs-sweep.md) builds on.

### Code

The three transforms on one logits vector (pure NumPy, no GPU):

```python
import numpy as np

def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    if temperature == 0:                      # greedy: argmax, deterministic
        return int(np.argmax(logits))
    logits = logits / temperature             # temperature scaling
    if top_k > 0:                             # keep k highest logits
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p < 1.0:                           # nucleus: smallest set with cumprob >= p
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        keep = order[:np.searchsorted(cum, top_p) + 1]
        mask = np.zeros_like(probs); mask[keep] = 1
        probs = probs * mask; probs /= probs.sum()
    rng = rng or np.random.default_rng(0)
    return int(rng.choice(len(probs), p=probs))
```

### Interviewer follow-ups

- *"`temperature=0` vs `temperature=1`?"* → `0` = greedy/argmax, deterministic; `1` = sample from the raw softmax. Between them, lower `T` is steadier, higher `T` is more diverse.
- *"Does raising temperature hurt throughput?"* → Not meaningfully. Sampling is an $O(\text{batch}\times\text{vocab})$ reduction, tiny next to the forward pass. Throughput is set by batch width and KV, not the sampler.
- *"`top_k=0` in vLLM — does that keep zero tokens?"* → No — `0` (or `-1`) **disables** top-k and considers all tokens. Easy trap.
- *"How do you make an A/B eval reproducible?"* → `temperature=0` (greedy) + fixed `seed`, so only the config under test varies. Then measure the (quality, throughput, latency) triple.
- *"Any batching subtlety with sampling?"* → Logits processors run at batch level; argmax-invariant ones can be skipped only when the whole batch is greedy, so mixed greedy+sampled batches lose that small saving.

### Linked concepts

- Lesson: [The Tuning Knobs: Sweeping the Throughput/Latency Curve](../part5/tuning-knobs-sweep.md) — where fixed sampling makes the sweep's quality axis attributable.
- Related: [Latency vs throughput metrics](latency-throughput-metrics.md), [Static vs continuous batching](continuous-batching.md), [Guided / structured decoding](structured-decoding.md) (constrains the same logits), [Speculative decoding](speculative-decoding.md) (verification honors the target's sampling)
- Glossary: [Sampling parameters, Decode](../glossary.md)
