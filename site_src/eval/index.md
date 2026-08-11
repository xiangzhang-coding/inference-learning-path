# Eval Sets

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The harness code on these pages is verified against vLLM 0.26.0 via Context7 (ADR-0004). Everything here is a **learner-runnable resource** — following ADR-0004, the author does **not** execute it. Any accuracy or throughput number you produce is **yours**; this site prints none as fact.

---

## Why you need an eval set

Almost every optimization in this path — quantizing weights to INT4, dropping the KV cache to FP8, raising `gpu_memory_utilization`, turning on speculative decoding — buys you **throughput** by trading away something. Usually the thing traded is a little **output quality**. The whole game of inference infra is running that trade *on purpose*, with eyes open, instead of shipping a silent regression.

To do that you need a cheap, repeatable way to answer one question:

> *After this change, is the model still good enough — and how much faster did it get?*

That is a **quality-vs-throughput** measurement, and it requires a fixed set of inputs you can re-run before and after every change. This section gives you two of them:

| Set | Size | Purpose | Page |
|---|---|---|---|
| **Small** | ~20 items | 60-second smoke check — "did I obviously break it?" | [Small set (~20)](small.md) |
| **Large** | hundreds+ | Fuller signal from a standard public benchmark | [Large set](large.md) |

Both are consumed by the **[Quantization](../part4/index.md)** and **[Serving & Throughput](../part5/index.md)** parts, and by the **[Capstone](../capstone/index.md)** "before → after" report.

## The measurement loop

```text
        ┌─────────────────────────────────────────────┐
        │ 1. Baseline: run eval on the UNCHANGED model  │  → (quality_0, throughput_0)
        └─────────────────────────────────────────────┘
                              │
             apply ONE change (quantize / tune a knob)
                              ▼
        ┌─────────────────────────────────────────────┐
        │ 2. Re-run the SAME eval on the changed model  │  → (quality_1, throughput_1)
        └─────────────────────────────────────────────┘
                              │
                              ▼
        Δquality = quality_1 − quality_0     Δthroughput = throughput_1 − throughput_0
        Keep the change only if Δthroughput is worth Δquality.
```

Two rules make the loop trustworthy:

- **Change one thing at a time.** If you quantize *and* bump `gpu_memory_utilization` in the same step, you can't attribute the quality delta.
- **Fix the sampling.** Set `temperature=0` (greedy) and a `seed` so re-runs are comparable; a quality "regression" that's really just sampling noise will waste your time.

**Quality** here is accuracy — the fraction of items whose output passes that item's check:

$$
\text{acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\hat{y}_i \text{ passes check}_i\right]
$$

**Throughput** is output tokens per second, measured wall-clock over the whole set. Neither number is meaningful in isolation — you compare the *pair* before and after a change.

## Two ways to run the harness

Both eval sets reuse one of these two runner shapes. Pick by what you're testing.

### A) Offline `LLM.chat` — simplest

One process, no server, batched internally by vLLM. Best for the small set and quick experiments. Note the verified gotcha: `LLM.generate` does **not** apply the chat template — for an Instruct model you must use `LLM.chat`, which takes OpenAI-style `messages`.

```python title="harness_offline.py"
"""Offline eval harness skeleton (vLLM 0.26.0, verified via Context7)."""
import time
from vllm import LLM, SamplingParams

def run_offline(model: str, prompts: list[str], max_tokens: int = 256):
    llm = LLM(model=model)                      # add quantization=... in Part 4 experiments
    sp = SamplingParams(temperature=0.0,        # greedy -> deterministic, comparable runs
                        max_tokens=max_tokens,
                        seed=0)
    # LLM.chat applies the model's chat template; LLM.generate does NOT.
    convos = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.perf_counter()
    outputs = llm.chat(convos, sp)
    dt = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outputs]
    n_out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    return texts, {"gen_tokens_per_s": n_out_tok / dt, "wall_s": dt}
```

### B) OpenAI-compatible server + client — matches production

Start the server once, hit it with the official `openai` client. This is how you'll actually serve, so latency/throughput numbers here are closer to reality. Start it (in one terminal):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192
```

Then query it:

```python title="harness_server.py"
"""Server-mode eval harness skeleton (vLLM 0.26.0 OpenAI-compatible API)."""
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

def ask(model: str, prompt: str, max_tokens: int = 256) -> str:
    resp = client.chat.completions.create(
        model=model,                            # the name you served it under
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
```

For a true throughput number in server mode you'd fire requests **concurrently** (that's the point of continuous batching) and read vLLM's own logged metrics — you'll build exactly that load test in **[Part 5](../part5/index.md)**. For eval-quality checks, a simple loop is enough.

## Running on a single 4090 — and in 无卡 (no-GPU) mode

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB (AWQ/GPTQ 4-bit 7B leaves plenty of room; BF16 7B is ~15 GB of weights so it also fits with a short `max_model_len`)
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** small set ~2–5 min · large set (500 items) ~15–40 min · a few ¥ of GPU time (all **illustrative**)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** vLLM also has AMD ROCm and CPU builds; the harness logic is backend-agnostic, but startup time and `kv_cache_dtype` support differ — check your build's notes.

Follow the same budget discipline as the rest of this path (ADR-0001): **GPU billing only when the model actually runs.**

- **In 无卡 mode (¥0, no GPU):** download the datasets, write and syntax-check the harness, and validate the *scorer* against canned/mock outputs — everything except the forward pass. Loading the 7B model needs the GPU, so the real run happens with the card on.
- **With the card on:** run the baseline, then each changed variant. Keep the eval small enough that a full before/after cycle is minutes, not hours — that's why the small set exists.

## On the numbers

!!! warning "Every number you get is yours"
    This site states **no** accuracy or throughput figures as ground truth. Per ADR-0004 the author verifies APIs statically and does not execute. When you run these sets you'll get concrete numbers — treat them as measurements of *your* box, *your* model build, *your* prompts. In particular, compare **relative deltas** (baseline vs. changed) on one setup; do not compare your absolute accuracy against a public leaderboard, whose prompts, templates, and sampling differ from yours.
