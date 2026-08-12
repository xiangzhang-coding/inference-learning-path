# Guided / structured decoding: masking tokens to a schema

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 6 · Advanced Inference Topics   ·   **Tests the lesson:** [Guided / Structured Decoding: Make Invalid Tokens Impossible](../part6/structured-decoding.md)

---

## Q: How does structured decoding force valid JSON/regex/grammar output, why is the guarantee hard rather than statistical, what does it cost, and what does it *not* fix?

### Direct answer

Structured decoding constrains generation to a **schema / regex / grammar / enum** by masking logits at **every decode step**. The schema is compiled once into a **finite-state machine**; at state $s$ it yields a binary mask $m^{(s)}\in\{0,1\}^{|V|}$ of allowed next tokens. vLLM adds $\log m^{(s)}$ to the logits before softmax, so a forbidden token's logit becomes $-\infty$ and its probability is **exactly 0** — uncsamplable at any temperature/top-p. After sampling, the FSM advances and the next step uses the new state's mask. The output is schema-valid **by construction**.

**Why hard, not statistical:** prompting only *lowers* the probability of bad tokens (so a few percent still slip through at scale); masking makes them *impossible* — there is no path to invalid output.

**Cost:** a one-time schema→automaton **compile** (amortized; can show as first-token latency for a brand-new schema) plus per-step masking, which the default **`xgrammar`** backend drives near-zero via precomputed per-state token sets. Alternative backend: **`guidance`**.

**What it does NOT fix:** it guarantees **shape, never truth** — a schema-valid answer can still be wrong.

### Deep dive

- **The masked softmax.** $z'_i = z_i + \log m^{(s)}_i$; since $e^{-\infty}=0$, forbidden tokens vanish from the distribution regardless of sampling settings.
- **Constraint types (`StructuredOutputsParams`):** `json` (JSON Schema / Pydantic), `regex` (backend uses Rust-style regex), `choice` (enum), `grammar` (EBNF CFG), `structural_tag`.
- **API rename to know:** `guided_json`/`guided_regex`/… were **deprecated and removed in vLLM 0.12.0**; on 0.26.0 use `structured_outputs` (online) / `StructuredOutputsParams` (offline). Seeing `guided_*` = pre-0.12.0 code.
- **Backend selection:** `--structured-outputs-config.backend` (default `auto` → `xgrammar`).

### Code

The verified 0.26.0 offline API for JSON + enum:

```python
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
class Review(BaseModel):
    sentiment: str; score: float
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
so = StructuredOutputsParams(json=Review.model_json_schema())    # or regex=/choice=/grammar=
out = llm.generate("Rate: 'vLLM is wonderful!'",
                   SamplingParams(temperature=0, max_tokens=64, structured_outputs=so))
print(out[0].outputs[0].text)    # {"sentiment": "positive", "score": 0.95}
```

Online: `response_format={"type":"json_schema", "json_schema":{"name":..., "schema":...}}` or `extra_body={"structured_outputs": {"regex": r"\d{3}-\d{3}-\d{4}"}}`.

### Interviewer follow-ups

- *"Prompting gets JSON 97% of the time — why bother?"* → Prompting biases; masking forbids. At scale the 3% is a stream of parse errors. Masking makes invalid output impossible, not just unlikely.
- *"Does it make answers correct?"* → No. It fixes **form, not content**. `{"capital": "Lyon"}` for France is schema-valid and wrong — validate values separately.
- *"Where's the latency?"* → One-time grammar **compile** (amortized across requests using that schema); per-step masking is near-free with xgrammar. Cold, giant, unique schemas spike first-token latency.
- *"`SamplingParams(guided_json=...)` errors on 0.26.0 — why?"* → Removed in 0.12.0. Use `structured_outputs=StructuredOutputsParams(json=...)`.
- *"A too-loose schema still gives garbage?"* → Yes — `{"answer": "string"}` barely constrains. Tighten with enums/regex/maxLength/required fields.

### Linked concepts

- Lesson: [Guided / Structured Decoding](../part6/structured-decoding.md)
- Related: [Static vs continuous batching](continuous-batching.md) (masking runs inside each sequence's decode step), [Number formats & precision](number-formats.md) (logits/softmax the mask acts on), [Tuning knobs](tuning-knobs.md) (backend choice as a serving knob)
- Glossary: [Guided / Structured decoding](../glossary.md)
