# Large set

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Harness verified against vLLM 0.26.0 via Context7 (ADR-0004). This is a **learner-runnable resource**; the author does not execute it. Any number you get is **yours** — see the [overview](index.md#on-the-numbers).

---

## What this is

The [small set](small.md) tells you *"not obviously broken."* This set tells you *"how much did quality actually move,"* with enough items that the number is stable instead of noise. The honest way to get hundreds of well-vetted items is **not** to hand-write them — it's to pull from an established **public benchmark**. So the "data" here is a *recipe*: a small, seeded script that downloads a standard dataset and samples a fixed slice you re-use for every before/after run.

**Data source:** public datasets on the Hugging Face Hub, loaded with the `datasets` library. We use two classics that stress different axes:

| Dataset | Tests | Scoring | HF id |
|---|---|---|---|
| **GSM8K** | multi-step math reasoning | exact-match on the final number | `openai/gsm8k` (config `main`) |
| **MMLU** | broad factual knowledge | multiple-choice letter match | `cais/mmlu` (config `all`) |

GSM8K is the primary path below — its `#### <number>` gold format gives a clean, unambiguous exact-match metric, which is exactly what you want when comparing a quantized model against its baseline. MMLU is noted as an option for knowledge coverage.

!!! note "Confirm the dataset card"
    Dataset repo ids, config names, splits, and column names occasionally change on the Hub. The ids above are the canonical ones at time of writing; if a load call errors, open the dataset's card on huggingface.co and adjust `path` / `name` / `split` / field names. This is data plumbing, not a vLLM API, so it isn't pinned by our version baseline.

## Sample a fixed slice (无卡-friendly)

Downloading and sampling is pure network + CPU — do it in **无卡 mode for ¥0**, before you ever turn the card on. Seeding makes the slice identical across runs, so baseline and variant see the same questions.

```python title="sample_gsm8k.py"
"""Download GSM8K and freeze a seeded N-item slice to JSONL (CPU-only, offline after download)."""
import json
import random
from datasets import load_dataset

N = 500          # 200 is enough for a quick read; 500–1000 tightens the estimate
SEED = 0

ds = load_dataset("openai/gsm8k", "main", split="test")   # ~1.3k test items
idx = random.Random(SEED).sample(range(len(ds)), k=min(N, len(ds)))

with open("gsm8k_slice.jsonl", "w", encoding="utf-8") as f:
    for i in idx:
        row = ds[i]
        f.write(json.dumps({"id": f"gsm8k-{i}",
                            "prompt": row["question"],
                            "gold": row["answer"]}, ensure_ascii=False) + "\n")
print(f"wrote {len(idx)} items to gsm8k_slice.jsonl")
```

Each GSM8K `answer` ends with a line like `#### 42`; that trailing number is the gold label we score against.

## Exact-match scorer

Parse the gold number out of the `#### …` marker, and the model's answer out of the **last** number it emits (models put the final answer last after showing work). Pure CPU — testable in 无卡 mode against mock outputs.

```python title="score_gsm8k.py"
"""GSM8K exact-match scorer (pure CPU, offline)."""
import json
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gold_number(answer: str) -> str:
    # GSM8K gold format: reasoning ... "\n#### 42"
    after = answer.split("####")[-1]
    m = _NUM.search(after)
    return m.group().replace(",", "") if m else ""


def pred_number(output: str) -> str:
    # take the LAST number in the model's output (the final answer)
    nums = _NUM.findall(output)
    return nums[-1].replace(",", "") if nums else ""


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(items: list[dict], outputs: list[str]) -> dict:
    correct = sum(gold_number(it["gold"]) == pred_number(out)
                  for it, out in zip(items, outputs))
    return {"accuracy": correct / len(items), "correct": correct, "total": len(items)}
```

**Line-by-line:**

- `_NUM` — matches an optionally-signed integer or decimal, tolerating thousands separators (`1,024`), which we strip before comparing.
- `gold_number` — everything after the last `####` is the answer region; pull the number there.
- `pred_number` — the model reasons out loud, so the *last* number is its final answer. This is the standard GSM8K heuristic; it's not perfect, but it's the same rule applied to baseline and variant, so the **delta** stays fair.
- `score` — plain exact-match accuracy, the $\text{acc}$ formula from the [overview](index.md#the-measurement-loop).

## Run it end to end

Reuse the [offline runner](index.md#a-offline-llmchat-simplest). Give math a longer `max_tokens` so the model can show its work before the final number.

```python title="run_gsm8k.py"
import json
from vllm import LLM, SamplingParams
from score_gsm8k import load, score

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"   # swap to the variant under test

items = load("gsm8k_slice.jsonl")
llm = LLM(model=MODEL, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=512, seed=0)   # room to reason; greedy

convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
outputs = [o.outputs[0].text for o in llm.chat(convos, sp)]

print(json.dumps(score(items, outputs), indent=2))
```

vLLM batches the whole slice internally via continuous batching, so 500 items is one `llm.chat` call, not 500 round-trips.

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~15–40 min for 500 items (math answers are long) · a few ¥ of GPU time (**illustrative** — depends heavily on `max_tokens` and the model variant)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** the download and scorer run anywhere; only `LLM(...)` needs a supported vLLM backend.

## The industry-standard tool

Rolling your own harness (above) is the best way to *understand* what an eval does. In practice, teams reach for **[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)** (EleutherAI) — it ships hundreds of tasks with vetted prompts and metrics, and has a native vLLM backend so you can evaluate the exact serving engine you'll deploy. It's heavier to set up and its CLI flags evolve, so check its current docs rather than trusting a flag from memory (ADR-0004). Use it once you want *comparable* numbers across models; use the DIY path when you want a fast, transparent loop you fully control.

## Common pitfalls

- **Comparing your accuracy to a leaderboard.** Published GSM8K/MMLU scores use specific prompt templates, few-shot counts, and answer parsers. Yours will differ. Only the **relative delta** (your baseline vs. your quantized run, same harness) is meaningful.
- **`max_tokens` too small for math.** If the model gets truncated mid-reasoning, `pred_number` grabs a wrong intermediate number and accuracy craters — for the model, not for the quantization. Give reasoning tasks room (`512`+).
- **Re-sampling the slice.** If you don't freeze `gsm8k_slice.jsonl`, each run sees different questions and the delta is meaningless. Sample once, commit the file, reuse it.
- **Forgetting the small set.** Don't burn 30 GPU-minutes to discover a change was catastrophic. Run the [small set](small.md) first; escalate here only when it passes.

## Next

- Plug these deltas into **[Quantization](../part4/index.md)** and **[Serving & Throughput](../part5/index.md)**, and into the **[Capstone](../capstone/index.md)** before → after report.
- Need the fast gate instead? Back to the **[Small set (~20)](small.md)**.
