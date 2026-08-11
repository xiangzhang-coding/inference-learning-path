# 小评测集（~20）

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    Harness 已通过 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。这是**学习者可自行运行的资源**；作者不执行。你跑出的任何数字都是**你自己的**——见[总览](index.md)。

---

## 这是什么

二十条短题，每条配一个**程序化检查**——无 LLM 评委、无网络、完全确定性。它是一个**冒烟测试**，不是基准：它在一分钟内回答*「我是不是把模型明显搞坏了？」*，好让你在迭代一次量化或调参改动时，不必等 [大评测集](large.md)。它刻意做得很粗——对自由文本输出做子串匹配，偶尔会放过一个错答案，或误杀一个对但换了说法的答案。为换取速度，这个取舍可以接受；可信的信号交给大集。

题目覆盖量化与激进调参最先命中的几类失效：事实召回、多步算术/推理、代码、指令遵循 / 输出格式，以及**双语**（中文）行为——这很重要，因为我们的基线模型是 `Qwen2.5-7B-Instruct`。

**数据来源：** 为本课程手写，按本站 **CC BY-SA 4.0** 发布。其中不含任何专有或源自基准的数据，你可以自由复制、修改、扩充。

## 数据

把下面内容拷进 `small_eval.jsonl`（每行一个 JSON 对象）。每条含 `id`、`category`、`prompt`，以及描述如何打分的 `check`：

- `substr` —— 输出包含 `value`（不区分大小写）
- `all` —— 输出包含 `value` 列表里的**每一个**字符串（不区分大小写）
- `regex` —— `value` 能匹配输出（不区分大小写的 `re.search`）

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

## 打分器

纯 CPU、离线可跑——你可以在无卡模式下拿假造输出先把 harness 的这一半测通，再花 GPU 时间。

```python title="score.py"
"""小评测集的粗粒度程序化打分器（纯 CPU、离线）。"""
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
    per_cat = defaultdict(lambda: [0, 0])   # category -> [通过数, 总数]
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

**逐行讲解：**

- `load_items` —— 读 JSONL；因为有中文题，UTF-8 很关键。
- `passes` —— 上面那三种检查。匹配不区分大小写；`regex` 保留原始输出，让 `"ok"\s*:\s*true` 这类模式看到真实的大小写/空白。
- `summarize` —— 总体精度（[总览](index.md)里的 $\text{acc}$ 公式）加上按类别的拆解——后者才真正告诉你一次改动*具体*坏在哪（比如「math 保住了但 `format` 崩了」）。

## 端到端跑起来

把[离线 runner](index.md) 和打分器拼起来：

```python title="run_small.py"
import json
from vllm import LLM, SamplingParams
from score import load_items, summarize

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"   # 换成待测变体（Part 4）

items = load_items("small_eval.jsonl")
llm = LLM(model=MODEL, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # 贪心、可比

convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
outputs = [o.outputs[0].text for o in llm.chat(convos, sp)]     # chat() 套用模板

report = summarize(items, outputs)
print(json.dumps(report, indent=2, ensure_ascii=False))
```

先跑基线、存下报告，施加**一个**改动，再跑一次，然后 diff 两份报告。集中在某一类别上的下降，就是你的信号。

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 含模型加载 ~2–5 分钟 · 远不到 ¥1 的 GPU 时费（**示例/量级参考**）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** 打分器是纯 Python，哪里都能跑；只有 `LLM(...)` 加载需要受支持的 vLLM 后端。

## 常见坑

- **把分数当基准。** 它是冒烟测试。通过只意味着「没明显坏」，不代表「可上生产」。真正的回退要到[大集](large.md)上确认。
- **非贪心采样。** `temperature > 0` 时重跑可能偶然不同，伪装成回退。做对比时保持 `temperature=0.0` 加固定 `seed`。
- **用 `generate` 而非 `chat`。** `LLM.generate` 跳过 chat 模板，于是 Instruct 模型拿到畸形提示，分数难看却是错怪了它。
- **子串误报。** `"12"` 也会匹配 `"120"`；`"Au"` 会匹配 `"August"`。作为粗闸够用——只是别对单条题过度解读。随着你摸清模型的措辞，再放宽或收紧 `check` 的取值。

## 下一步

- 拿到了真实的回退信号？转向 **[大评测集](large.md)** 获得可信的质量数字。
- 看它接在哪里：**[量化](../part4/index.md)**（质量 vs 比特）与 **[服务化与吞吐](../part5/index.md)**（质量 vs 旋钮）。
