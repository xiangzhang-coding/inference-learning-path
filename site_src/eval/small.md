# Small set (~20)

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    Harness verified against vLLM 0.26.0 via Context7 (ADR-0004). This is a **learner-runnable resource**; the author does not execute it. Any number you get is **yours** — see the [overview](index.md#on-the-numbers).

---

## What this is

Twenty short items with a **programmatic check** each — no LLM judge, no network, fully deterministic. It is a **smoke test**, not a benchmark: it answers *"did I obviously break the model?"* in under a minute, so you can iterate on a quantization or tuning change without waiting on the [large set](large.md). It is deliberately coarse — a substring match on free-form output will occasionally pass a wrong answer or fail a right-but-differently-phrased one. That's an acceptable trade for speed; the large set is where you get a trustworthy signal.

The items span the failure modes that quantization and aggressive tuning tend to hit first: factual recall, multi-step arithmetic/reasoning, code, instruction-following / output format, and **bilingual** (Chinese) behavior — important because our baseline model is `Qwen2.5-7B-Instruct`.

**Data source:** hand-authored for this course, released under the site's **CC BY-SA 4.0**. It contains no proprietary or benchmark-derived data, so you can copy, edit, and extend it freely.

## The data

Copy this into `small_eval.jsonl` (one JSON object per line). Each item has an `id`, a `category`, the `prompt`, and a `check` describing how to score the output:

- `substr` — output contains `value` (case-insensitive)
- `all` — output contains **every** string in the `value` list (case-insensitive)
- `regex` — `value` matches the output (case-insensitive `re.search`)

```jsonl title="small_eval.jsonl"
{"id": "fact-01", "category": "factual", "prompt": "What is the capital of France? Answer with just the city name.", "check": {"type": "substr", "value": "Paris"}}
{"id": "fact-02", "category": "factual", "prompt": "What is the chemical symbol for gold?", "check": {"type": "substr", "value": "Au"}}
{"id": "fact-03", "category": "factual", "prompt": "In which year did the first humans land on the Moon? Four-digit year only.", "check": {"type": "substr", "value": "1969"}}
{"id": "math-01", "category": "math", "prompt": "Compute 17 * 23. Give only the number.", "check": {"type": "substr", "value": "391"}}
{"id": "math-02", "category": "math", "prompt": "What is the square root of 144?", "check": {"type": "substr", "value": "12"}}
{"id": "math-03", "category": "math", "prompt": "What is 15% of 200? Number only.", "check": {"type": "substr", "value": "30"}}
{"id": "reason-01", "category": "reasoning", "prompt": "Tom has 5 apples. He buys 2 boxes with 6 apples each. How many apples does he have now? Answer with the number.", "check": {"type": "substr", "value": "17"}}
{"id": "reason-02", "category": "reasoning", "prompt": "A train leaves at 9:00 and arrives at 11:30 the same morning. How many minutes is the trip? Number only.", "check": {"type": "substr", "value": "150"}}
{"id": "reason-03", "category": "reasoning", "prompt": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Answer yes or no.", "check": {"type": "substr", "value": "yes"}}
{"id": "reason-04", "category": "reasoning", "prompt": "What number comes next in the sequence 2, 4, 8, 16, ...? Number only.", "check": {"type": "substr", "value": "32"}}
{"id": "code-01", "category": "code", "prompt": "Write a Python expression that reverses the string in variable s.", "check": {"type": "substr", "value": "[::-1]"}}
{"id": "code-02", "category": "code", "prompt": "In Python, what does len([]) evaluate to?", "check": {"type": "substr", "value": "0"}}
{"id": "code-03", "category": "code", "prompt": "Which Python keyword is used to define a function?", "check": {"type": "substr", "value": "def"}}
{"id": "format-01", "category": "format", "prompt": "Reply with exactly one word: the color of a clear daytime sky.", "check": {"type": "substr", "value": "blue"}}
{"id": "format-02", "category": "format", "prompt": "Output only valid JSON with a single key \"ok\" whose value is the boolean true.", "check": {"type": "regex", "value": "\"ok\"\\s*:\\s*true"}}
{"id": "format-03", "category": "format", "prompt": "List the three additive primary colors of light, comma-separated, lowercase.", "check": {"type": "all", "value": ["red", "green", "blue"]}}
{"id": "inst-01", "category": "instruction", "prompt": "Translate to French: \"Good morning\". Give only the translation.", "check": {"type": "substr", "value": "Bonjour"}}
{"id": "zh-01", "category": "bilingual", "prompt": "用中文回答：中国的首都是哪座城市？只回答城市名。", "check": {"type": "substr", "value": "北京"}}
{"id": "zh-02", "category": "bilingual", "prompt": "用中文回答：一年有多少个月？只回答数字。", "check": {"type": "substr", "value": "12"}}
{"id": "zh-03", "category": "bilingual", "prompt": "Translate to English: 我喜欢机器学习。", "check": {"type": "all", "value": ["machine", "learning"]}}
```

## The scorer

Pure CPU, offline-runnable — you can test this half of the harness in 无卡 mode against mock outputs before spending any GPU time.

```python title="score.py"
"""Coarse programmatic scorer for the small eval set (pure CPU, offline)."""
import json
import re
from collections import defaultdict


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def passes(check: dict, output: str) -> bool:
    out = output.lower()
    kind, value = check["type"], check["value"]
    if kind == "substr":
        return value.lower() in out
    if kind == "all":
        return all(v.lower() in out for v in value)
    if kind == "regex":
        return re.search(value, output, re.IGNORECASE) is not None
    raise ValueError(f"unknown check type: {kind}")


def summarize(items: list[dict], outputs: list[str]) -> dict:
    per_cat = defaultdict(lambda: [0, 0])   # category -> [passed, total]
    passed = 0
    for item, out in zip(items, outputs):
        ok = passes(item["check"], out)
        passed += ok
        c = per_cat[item["category"]]
        c[0] += ok
        c[1] += 1
    return {
        "accuracy": passed / len(items),
        "passed": passed,
        "total": len(items),
        "by_category": {k: v[0] / v[1] for k, v in per_cat.items()},
    }
```

**Line-by-line:**

- `load_items` — reads the JSONL; UTF-8 matters because of the Chinese items.
- `passes` — the three check kinds from above. Matching is case-insensitive; `regex` keeps the raw output so patterns like `"ok"\s*:\s*true` see real casing/whitespace.
- `summarize` — overall accuracy (the $\text{acc}$ formula from the [overview](index.md#the-measurement-loop)) plus a per-category breakdown, which is what actually tells you *what* a change broke (e.g. "math held but `format` collapsed").

## Run it end to end

Combine the [offline runner](index.md#a-offline-llmchat-simplest) with the scorer:

```python title="run_small.py"
import json
from vllm import LLM, SamplingParams
from score import load_items, summarize

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"   # swap to the variant under test (Part 4)

items = load_items("small_eval.jsonl")
llm = LLM(model=MODEL, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # greedy, comparable

convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
outputs = [o.outputs[0].text for o in llm.chat(convos, sp)]     # chat() applies the template

report = summarize(items, outputs)
print(json.dumps(report, indent=2, ensure_ascii=False))
```

Run the baseline first, save the report, apply **one** change, run again, and diff the two reports. A drop concentrated in one category is your signal.

!!! gpu "GPU Lab"
    - **Min VRAM:** 24 GB
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** ~2–5 min including model load · well under ¥1 of GPU time (**illustrative**)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** the scorer is pure Python and runs anywhere; only the `LLM(...)` load needs a supported vLLM backend.

## Common pitfalls

- **Treating the score as a benchmark.** It's a smoke test. A pass means "not obviously broken," not "production-ready." Confirm real regressions on the [large set](large.md).
- **Non-greedy sampling.** With `temperature > 0` a re-run can differ by chance and masquerade as a regression. Keep `temperature=0.0` and a fixed `seed` for comparisons.
- **`generate` instead of `chat`.** `LLM.generate` skips the chat template, so an Instruct model gets a malformed prompt and scores look terrible for the wrong reason.
- **Substring false-positives.** `"12"` also matches `"120"`; `"Au"` matches `"August"`. Fine for a coarse gate — just don't over-read a single item. Widen or tighten `check` values as you learn your model's phrasing.

## Next

- Got a real regression signal? Move to the **[Large set](large.md)** for a trustworthy quality number.
- See where this plugs in: **[Quantization](../part4/index.md)** (quality vs. bits) and **[Serving & Throughput](../part5/index.md)** (quality vs. knobs).
