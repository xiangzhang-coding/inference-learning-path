# 评测集

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    本页的 harness 代码已通过 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。这里的一切都是**学习者可自行运行的资源**——遵循 ADR-0004，作者**不执行**它。你跑出来的任何精度或吞吐数字都是**你自己的**；本站不把任何数字当作事实给出。

---

## 为什么你需要一个评测集

这条学习路径里几乎每一个优化——把权重量化到 INT4、把 KV cache 降到 FP8、调高 `gpu_memory_utilization`、开启投机解码——都是拿**吞吐**去换某样东西。换掉的通常是一点点**输出质量**。推理 Infra 的全部功夫，就在于**有意识地、睁着眼**做这笔交易，而不是悄悄上线一个已经劣化的模型。

要做到这一点，你需要一个便宜、可重复的办法回答一个问题：

> *这次改动之后，模型是否还够好——又快了多少？*

这就是一次**质量 vs 吞吐**的测量，它需要一组固定输入，让你能在每次改动前后重跑。本节给你两套：

| 评测集 | 规模 | 用途 | 页面 |
|---|---|---|---|
| **小集** | ~20 条 | 60 秒冒烟核对——「我是不是把它明显搞坏了？」 | [小评测集（~20）](small.md) |
| **大集** | 数百条以上 | 用标准公开基准得到更充分的信号 | [大评测集](large.md) |

两套都会被 **[量化](../part4/index.md)** 与 **[服务化与吞吐](../part5/index.md)** 两部分、以及 **[Capstone](../capstone/index.md)** 的「优化前 → 后」报告使用。

## 测量闭环

```text
        ┌─────────────────────────────────────────────┐
        │ 1. 基线：在【未改动】的模型上跑评测           │  → (quality_0, throughput_0)
        └─────────────────────────────────────────────┘
                              │
              只施加【一个】改动（量化 / 调一个旋钮）
                              ▼
        ┌─────────────────────────────────────────────┐
        │ 2. 在改动后的模型上跑【同一套】评测           │  → (quality_1, throughput_1)
        └─────────────────────────────────────────────┘
                              │
                              ▼
        Δ质量 = quality_1 − quality_0     Δ吞吐 = throughput_1 − throughput_0
        只有当 Δ吞吐 值得 Δ质量 时，才保留这次改动。
```

有两条规则让闭环可信：

- **一次只改一样。** 如果你同一步里既量化又调高 `gpu_memory_utilization`，就无法把质量变化归因到哪一项。
- **固定采样。** 设 `temperature=0`（贪心）并给定 `seed`，让重跑可比；否则一个其实只是采样噪声的「质量回退」会白白浪费你的时间。

这里的**质量**就是精度——输出通过该条检查的比例：

$$
\text{acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\hat{y}_i \text{ 通过 check}_i\right]
$$

**吞吐**是每秒输出 token 数，按整套评测的墙钟时间测得。任何一个数字孤立看都没意义——你比较的是一次改动前后的**这一对**数。

## 两种运行 harness 的方式

两套评测集都复用下面两种 runner 之一。按你要测什么来选。

### A）离线 `LLM.chat`——最简单

单进程、无需起服务，由 vLLM 内部批处理。适合小集与快速实验。注意这条已核实的坑：`LLM.generate` **不会**套用 chat 模板——对 Instruct 模型你必须用 `LLM.chat`，它接收 OpenAI 风格的 `messages`。

```python title="harness_offline.py"
"""离线评测 harness 骨架（vLLM 0.26.0，经 Context7 核实）。"""
import time
from vllm import LLM, SamplingParams

def run_offline(model: str, prompts: list[str], max_tokens: int = 256):
    llm = LLM(model=model)                      # Part 4 实验里在这里加 quantization=...
    sp = SamplingParams(temperature=0.0,        # 贪心 -> 确定性、可比
                        max_tokens=max_tokens,
                        seed=0)
    # LLM.chat 会套用模型的 chat 模板；LLM.generate 不会。
    convos = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.perf_counter()
    outputs = llm.chat(convos, sp)
    dt = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outputs]
    n_out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    return texts, {"gen_tokens_per_s": n_out_tok / dt, "wall_s": dt}
```

### B）OpenAI 兼容服务 + 客户端——贴近生产

起一次服务，用官方 `openai` 客户端去打。这就是你实际的服务形态，所以这里的延迟/吞吐更接近真实。先起服务（在一个终端里）：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192
```

然后查询它：

```python title="harness_server.py"
"""服务模式评测 harness 骨架（vLLM 0.26.0 OpenAI 兼容 API）。"""
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

def ask(model: str, prompt: str, max_tokens: int = 256) -> str:
    resp = client.chat.completions.create(
        model=model,                            # 你起服务时用的模型名
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
```

服务模式下要拿到真实吞吐数字，你得**并发**发请求（这正是 continuous batching 的意义），并读 vLLM 自己记录的指标——你会在 **[Part 5](../part5/index.md)** 亲手搭这样一个压测。对评测「质量」而言，一个简单循环就够了。

## 在单张 4090 上跑——以及无卡模式

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB（AWQ/GPTQ 4-bit 的 7B 余量很足；BF16 7B 权重约 15 GB，配一个短 `max_model_len` 也放得下）
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 小集 ~2–5 分钟 · 大集（500 条）~15–40 分钟 · 几元 GPU 时费（均为**示例/量级参考**）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** vLLM 也有 AMD ROCm 与 CPU 版本；harness 逻辑与后端无关，但启动耗时与 `kv_cache_dtype` 支持有差异——请查你所用版本的说明。

遵循与本路径其余部分一致的预算纪律（ADR-0001）：**只在模型真正运行时才计 GPU 费。**

- **在无卡模式下（¥0，无 GPU）：** 下载数据集、写好并做语法检查、用假造/mock 输出验证**打分器**——除了前向推理之外的一切都能做。加载 7B 模型需要 GPU，所以真正的运行放到开卡时。
- **开卡之后：** 先跑基线，再跑每个改动后的变体。把评测规模控制得足够小，让一整轮前后对比是几分钟而非几小时——这正是小集存在的理由。

## 关于数字

!!! warning "你跑出的每个数字都是你自己的"
    本站**不**把任何精度或吞吐数字当作真值给出。遵循 ADR-0004，作者只做静态核实、不执行。当你运行这些评测集时会得到具体数字——把它们当作**你的**机器、**你的**模型构建、**你的**提示词上的测量。尤其是：只在同一套设置上比较**相对差值**（基线 vs 改动后）；不要拿你的绝对精度去比公开榜单——榜单的提示词、模板、采样都和你不同。
