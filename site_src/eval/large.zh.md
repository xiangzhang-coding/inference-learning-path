# 大评测集

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    Harness 已通过 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。这是**学习者可自行运行的资源**；作者不执行。你跑出的任何数字都是**你自己的**——见[总览](index.md)。

---

## 这是什么

[小集](small.md)告诉你*「没明显坏」*。这套告诉你*「质量到底动了多少」*，且题量足够大，让数字稳定而非噪声。要拿到数百条经过良好审校的题目，诚实的做法**不是**手写——而是从成熟的**公开基准**里取。所以这里的「数据」是一份*配方*：一个带 seed 的小脚本，下载标准数据集并抽取一个固定切片，供你每次前后对比复用。

**数据来源：** Hugging Face Hub 上的公开数据集，用 `datasets` 库加载。我们用两个经典、各压不同的轴：

| 数据集 | 考察 | 打分 | HF id |
|---|---|---|---|
| **GSM8K** | 多步数学推理 | 对最终数字做精确匹配 | `openai/gsm8k`（config `main`） |
| **MMLU** | 广域事实知识 | 多选字母匹配 | `cais/mmlu`（config `all`） |

下面以 GSM8K 为主路径——它的 `#### <数字>` 标准答案格式给出干净、无歧义的精确匹配指标，这正是拿量化模型对比其基线时你想要的。MMLU 作为覆盖知识面的可选项列出。

!!! note "请核对数据集卡片"
    数据集仓库 id、config 名、split、列名在 Hub 上偶尔会变。上表是撰写时的规范 id；若某次加载报错，去 huggingface.co 打开该数据集的卡片，调整 `path` / `name` / `split` / 字段名。这属于数据管线、不是 vLLM API，因此不受我们的版本基线钉定。

## 抽取固定切片（对无卡友好）

下载与抽样是纯网络 + CPU——在**无卡模式下用 ¥0** 完成，先于你开卡。给定 seed 让切片跨运行一致，于是基线与变体看到同一批题。

```python title="sample_gsm8k.py"
"""下载 GSM8K 并把带 seed 的 N 条切片冻结到 JSONL（仅 CPU；下载后可离线）。"""
import json
import random
from datasets import load_dataset

N = 500          # 200 条足够快速一读；500–1000 条能收紧估计
SEED = 0

ds = load_dataset("openai/gsm8k", "main", split="test")   # test 约 1.3k 条
idx = random.Random(SEED).sample(range(len(ds)), k=min(N, len(ds)))

with open("gsm8k_slice.jsonl", "w", encoding="utf-8") as f:
    for i in idx:
        row = ds[i]
        f.write(json.dumps({"id": f"gsm8k-{i}",
                            "prompt": row["question"],
                            "gold": row["answer"]}, ensure_ascii=False) + "\n")
print(f"wrote {len(idx)} items to gsm8k_slice.jsonl")
```

每条 GSM8K 的 `answer` 以形如 `#### 42` 的一行结尾；那个末尾数字就是我们据以打分的标准答案。

## 精确匹配打分器

从 `#### …` 标记里解析出标准答案数字，从模型输出的**最后一个**数字里解析出预测（模型先展示推理、把最终答案放在最后）。纯 CPU——可在无卡模式下拿假造输出测通。

```python title="score_gsm8k.py"
"""GSM8K 精确匹配打分器（纯 CPU、离线）。"""
import json
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gold_number(answer: str) -> str:
    # GSM8K 标准答案格式：推理 ... "\n#### 42"
    after = answer.split("####")[-1]
    m = _NUM.search(after)
    return m.group().replace(",", "") if m else ""


def pred_number(output: str) -> str:
    # 取模型输出里的【最后一个】数字（最终答案）
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

**逐行讲解：**

- `_NUM` —— 匹配可带符号的整数或小数，容忍千分位（`1,024`），比较前把逗号去掉。
- `gold_number` —— 最后一个 `####` 之后就是答案区；取那里的数字。
- `pred_number` —— 模型会把推理写出来，所以*最后*一个数字是它的最终答案。这是 GSM8K 的标准启发式；它不完美，但对基线与变体施加的是同一规则，因此**差值**依然公平。
- `score` —— 朴素的精确匹配精度，即[总览](index.md)里的 $\text{acc}$ 公式。

## 端到端跑起来

复用[离线 runner](index.md)。给数学更长的 `max_tokens`，让模型先展示推理再给最终数字。

```python title="run_gsm8k.py"
import json
from vllm import LLM, SamplingParams
from score_gsm8k import load, score

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"   # 换成待测变体

items = load("gsm8k_slice.jsonl")
llm = LLM(model=MODEL, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=512, seed=0)   # 留出推理空间；贪心

convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
outputs = [o.outputs[0].text for o in llm.chat(convos, sp)]

print(json.dumps(score(items, outputs), indent=2))
```

vLLM 通过 continuous batching 在内部批处理整个切片，所以 500 条是一次 `llm.chat` 调用，而不是 500 次往返。

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 500 条 ~15–40 分钟（数学答案较长）· 几元 GPU 时费（**示例/量级参考**——与 `max_tokens` 及模型变体强相关）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** 下载与打分器哪里都能跑；只有 `LLM(...)` 需要受支持的 vLLM 后端。

## 业界标准工具

自己搭 harness（如上）是*理解*评测在做什么的最好方式。实践中，团队会用 **[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)**（EleutherAI）——它内置数百个任务，配有审校过的提示词与指标，并有原生的 vLLM 后端，让你评测的正是将要部署的那个服务引擎。它搭起来更重、CLI flag 也在演进，所以请查它的最新文档、而不要凭记忆用某个 flag（ADR-0004）。当你想要跨模型**可比**的数字时用它；当你想要一个快速、透明、完全受控的循环时用 DIY 路径。

## 常见坑

- **拿你的精度去比榜单。** 公开的 GSM8K/MMLU 分数用的是特定的提示模板、few-shot 数量与答案解析器。你的会不同。只有**相对差值**（你的基线 vs 你的量化运行、同一套 harness）才有意义。
- **数学的 `max_tokens` 太小。** 若模型推理中途被截断，`pred_number` 会抓到一个错误的中间数字，精度暴跌——怪的是量化，其实不是。给推理任务留足空间（`512`+）。
- **重新抽样切片。** 如果不冻结 `gsm8k_slice.jsonl`，每次运行看到不同题目，差值就没意义。抽一次、把文件存下、反复复用。
- **忘了小集。** 别烧 30 GPU-分钟才发现某个改动是灾难性的。先跑[小集](small.md)；它通过后再升级到这里。

## 下一步

- 把这些差值接入 **[量化](../part4/index.md)** 与 **[服务化与吞吐](../part5/index.md)**，以及 **[Capstone](../capstone/index.md)** 的优化前 → 后报告。
- 需要那个快速闸门？回到 **[小评测集（~20）](small.md)**。
