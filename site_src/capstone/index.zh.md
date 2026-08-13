# Capstone：在单张 4090 上把 Qwen2.5-7B 吞吐拉满 —— 优化前 → 后报告

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    本 Capstone 拼装的每个 flag 都已用 Context7 对照 vLLM 0.26.0 核实（ADR-0004）：预构建 AWQ 检查点（`Qwen/Qwen2.5-7B-Instruct-AWQ`，量化**自动检测**、无需 `--quantization`）、`--kv-cache-dtype fp8`、`--gpu-memory-utilization`（默认 **0.92**）、`--max-num-seqs`（**128**）、`--max-num-batched-tokens`（**2048**，自动调）、`--enable-chunked-prefill`（**True**）、`--enable-prefix-caching`（由模型 config 解析）、`--max-model-len`、`--tensor-parallel-size`（单卡为 **1**），以及 **`vllm bench throughput`** 与 **`vllm bench serve`**。**作者不执行任何一步**（ADR-0004）：下文每个显存 / tokens-per-second / req/s / 准确率数字都是**示例 / 量级参考**。Capstone 的全部意义就在于：真正的优化前→后数字，是**你**在自己的 AutoDL 机器上测出来的那些。

---

## 1 · 直觉 & 为什么重要

Part 0–8 里的一切都是一个旋钮。Capstone 就是你在单张 4090 上**按顺序、一次一个**地拧它们，并产出那个真正能帮你拿到 offer 的产物：一份**优化前 → 后报告**，说清楚*你改了什么、移动了多少、以及你怎么知道它没把模型搞坏*。

新手掉进的坑是「我把它做快了」。这不是工程结论——是感觉。专业结论是一张**带基线的表**：FP16 做到 *X* output tokens/s、准确率 *Y*；上了 AWQ + FP8 KV + 容量调优后做到 *3.2X*、准确率 *Y − 0.02*，p99 TTFT 仍在 SLO 之下。每个数字都可归因到恰好一次改动。这张表就是面试官说「讲讲你做过的一次优化」时想听的东西，也正是本项目要搭出来的。

要内化的唯一纪律——就是 Part 5 里那个 [sweep 循环](../part5/tuning-knobs-sweep.md)，现在端到端地跑一遍：**先测基线，只改一个东西，测量（质量、吞吐、延迟）三元组，归因这个 delta，然后保留或回退。** 机制你早已掌握（量化、PagedAttention、continuous batching、prefix caching、各种旋钮）；Capstone 是把它们变成可辩护结果的那套*方法*。→ 度量词汇见[术语表](../glossary.md)。

## 2 · 心智模型

抓住两件事：一条**有序的优化阶梯**，和一根在每一级都要跑的**测量脊柱**。

```text
  优化阶梯（按此顺序往上爬——最大、最安全的杠杆先上）              测量脊柱（每一级都跑）
                                                                     ┌───────────────────────────┐
  第0级  基线              FP16 Qwen2.5-7B，默认配置                 │ 1. 质量（eval A/B，       │
             │             ── 在这里把一切都测一遍 ──                │    greedy + seed，分类）  │
             ▼                                                       │ 2. 吞吐（output tok/s，   │
  第1级  量化              AWQ INT4 权重（自动检测）                 │    vllm bench）           │
             │             省下 ~10 GB，加速带宽受限的 decode        │ 3. 延迟/拐点（p99 TTFT/   │
             ▼                                                       │    TPOT，bench serve 扫）  │
  第2级  FP8 KV 缓存       --kv-cache-dtype fp8 → KV 容量 ~2×         └───────────────────────────┘
             │                                                                    │
             ▼                                                          每一级：
  第3级  花掉省下的显存    --gpu-memory-utilization ↑, --max-num-seqs ↑          只改一个东西
             │             （省下的显存 → 更大的 continuous batch）              ▼
             ▼                                                          测量三元组
  第4级  前缀缓存          --enable-prefix-caching（若有共享前缀）              ▼
             │                                                          归因 delta
             ▼                                                                    ▼
  第5级  批形状调优        --max-num-batched-tokens（TTFT ↔ ITL）      保留吗？（质量闸门）
             │                                                                    ▼
             ▼                                                          不过闸就回退
  第6级  （多卡：TP）      单卡 4090 范围之外——只标注，本项目不做（Part 7 / A100）
             │
             ▼
        最终报告           基线行 → 每保留一级一行 → 汇总
```

要记住的三个形状：

- **顺序有讲究，且非任意。**[量化](../part4/quantization-lab.md)排第一，因为它是最大、最接近免费的杠杆——它同时抬高*两道闸*（省出显存给更多 [KV cache](../part2/kv-cache-math.md)，*同时*加速[带宽受限的 decode](../part2/roofline-analysis.md)）。你从「两头都帮、还便宜」往下爬到「必须调的真实 trade-off」，于是每一级都踩在下一级省出的资源上。把显存花在 batch 宽度上（第3级）只有在量化省出显存（第1–2级）*之后*才说得通。
- **脊柱在每一级都一样——这才让 delta 可归因。** 同一个[eval 集](../eval/small.md)、同一套固定采样（`temperature=0`、固定 `seed`）、同一个 benchmark 形状。级与级之间改了 eval 或形状，你的优化前→后表就成了拿苹果比橘子。
- **质量是闸门，不是填完就忘的一列。** 一级若抬了吞吐却过不了 eval A/B（或某一分类塌了），就**回退或退让**——把这个决定说出口，是整份报告里最资深的动作。「我试了 FP8 KV，数学准确率掉了 8 分，所以我保留 BF16 KV」比一个更大的吞吐数字是*更好*的答案。

## 3 · 原理 —— 方法

Capstone 不教新机制；它用一套严谨的方法把你已有的机制组合起来。五条规则。

### 3.1 先测基线 —— delta 需要一个「前」

没有你改进*之前*的那个数字，你就报不出改进。第0级是未优化模型在三个轴上的测量：FP16 `Qwen2.5-7B-Instruct`、默认配置，跑[eval 集](../eval/small.md)和 `vllm bench`。之后的一切都是相对这一行的 delta。跳过基线，是「3× 加速」变成不可证伪的最常见方式。

### 3.2 三个轴，每次都用同样的方法测

- **质量** —— [量化实操](../part4/quantization-lab.md)里的 A/B：用 `temperature=0.0` + 固定 `seed`、经 `LLM.chat` 跑[小 eval 集](../eval/small.md)，比较**分类**准确率。Greedy + seed 才让重跑可比；分类拆分才告诉你一级*把什么*搞坏了。
- **吞吐** —— `vllm bench throughput`（或内联计时）给出的 **output tokens/s**，在一个固定的 decode-heavy 形状上。output tok/s 是整套栈想移动的那个 decode 数字。
- **延迟 / 拐点** —— 你的 SLO 下的 **p99 TTFT 与 TPOT**，通过把 `vllm bench serve --request-rate` 往上扫得到（[压测那一课](../part8/load-testing-knee.md)）。拐点——仍满足 SLO 的最高供给负载——才是诚实的容量数字，而非裸吞吐。

### 3.3 一次只改一个，并归因

每一级只改一个旋钮。若你在同一步里既开 AWQ *又*抬 `gpu-memory-utilization`，吞吐翻倍你也学不到是哪个干的——而且如果质量掉了，你分不清该怪谁。这更慢，也是让报告有意义的唯一办法。

### 3.4 质量闸门

在每一级，eval A/B 都是一个**通过/不通过的闸门**，不是装饰。整体准确率小幅下降是预期内、也没问题；某个*分类塌陷*（如 `math` 1.0 → 0.3，或 `bilingual` 项掉链子——对一个中文能力基座模型很关键）是**回退**信号。一级过不了闸，你就退让：更多比特、保留 BF16 KV、换校准集——并且**把你这么做了写下来**。报告的可信度，既来自它的胜绩，也来自它的回退。

### 3.5 预算纪律 —— 这里大部分是免费的

¥500 预算（ADR-0001）很宽裕——*前提是*你别把 GPU 时间烧在不需要 GPU 的事上。所有下载和任何自量化都在 AutoDL **无卡模式**（无 GPU，近乎免费）做。只在测量时开 GPU——eval A/B 和 `vllm bench` 运行。一份完整的优化前→后报告是几十分钟 GPU 时长、个位数 ¥——预算是给你*迭代*用的，不是给单次运行用的。

## 4 · 完整可跑代码 + 逐行讲解

产物是一份报告，所以代码是一个**报告生成器**：在固定 eval 集上测一个配置的（质量、吞吐），然后驱动阶梯、产出一张 Markdown 表。它复用[小 eval 集](../eval/small.md)里的 `score.py`（`load_items`/`summarize`）——别重造 scorer。**你在 4090 上可跑；作者不执行**——注释里的数字是示例。

**第 1 步 —— 测一级**（单个配置的质量 A/B + decode 吞吐）：

```python title="capstone_stage.py"
"""在固定 eval 集上测一个配置：分类质量 + decode 吞吐。
复用小 eval 集 (#3) 的 load_items/summarize。API 已对照 vLLM 0.26.0 核实。
作者不执行（ADR-0004）；返回的数字是你的。"""
import time
from vllm import LLM, SamplingParams
from score import load_items, summarize          # 来自小 eval 集页面

ITEMS = load_items("small_eval.jsonl")            # 固定输入——级与级之间绝不改
CONVOS = [[{"role": "user", "content": it["prompt"]}] for it in ITEMS]
SP = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # greedy + seed => 各级可比

def measure(model: str, **engine_kwargs) -> dict:
    """用这些旋钮构建引擎、跑 eval、返回一行报告。"""
    llm = LLM(model=model, max_model_len=4096, **engine_kwargs)   # vLLM 从 config 自动检测 AWQ
    t0 = time.perf_counter()
    outs = llm.chat(CONVOS, SP)                   # chat() 套用 Instruct 模板
    dt = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outs]
    report = summarize(ITEMS, texts)              # 整体 + 分类准确率
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    row = {
        "accuracy": round(report["accuracy"], 3),
        "by_category": report["by_category"],
        "throughput_tok_s": round(out_tokens / dt, 1),   # 这个小集上的粗略 decode tok/s
    }
    del llm                                        # 下一级前释放显存（两个 7B 无法共存）
    return row
```

**逐行（第 1 步）：** `ITEMS`/`CONVOS`/`SP` 放在模块级，好让**每一级共享完全相同的输入与采样**——这是 delta 可归因的前提（§3.2）。`measure` 用该级的 `engine_kwargs` 构建引擎，跑 `LLM.chat`（套模板——`generate` 会喂给 Instruct 模型一个畸形 prompt，为错误的原因拉垮质量），返回一行：整体准确率、分类拆分（*闸门*，§3.4）、粗略 output tok/s。`del llm` 释放显存，好让下一级的 `LLM(...)` 不 OOM。这个小集吞吐是快速的相对信号；*权威的*吞吐/拐点数字来自 `vllm bench`（第 3 步）。

**第 2 步 —— 驱动阶梯、产出报告表：**

```python title="capstone_report.py"
"""跑优化阶梯，打印优化前->后的 Markdown 表。
每一级相对上一行只改一个东西（§3.3）。数字是示例。"""
from capstone_stage import measure

FP16 = "Qwen/Qwen2.5-7B-Instruct"
AWQ  = "Qwen/Qwen2.5-7B-Instruct-AWQ"     # 预构建 INT4；无卡模式下载

# 阶梯：(标签, 模型, engine_kwargs) —— 每行相对上一行只改恰好一个杠杆。
LADDER = [
    ("0 · 基线 (FP16)",          FP16, {}),
    ("1 · AWQ INT4 权重",        AWQ,  {}),
    ("2 · + FP8 KV 缓存",        AWQ,  {"kv_cache_dtype": "fp8"}),
    ("3 · + 花掉省下的显存",     AWQ,  {"kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.94,
                                        "max_num_seqs": 256}),
    ("4 · + 前缀缓存",           AWQ,  {"kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.94,
                                        "max_num_seqs": 256, "enable_prefix_caching": True}),
]

rows = []
baseline_tps = None
for label, model, kw in LADDER:
    r = measure(model, **kw)
    if baseline_tps is None:
        baseline_tps = r["throughput_tok_s"]
    speedup = r["throughput_tok_s"] / baseline_tps
    rows.append((label, r["accuracy"], r["throughput_tok_s"], speedup))

print("| 阶段 | eval 准确率 | output tok/s | 相对基线加速 |")
print("|---|---|---|---|")
for label, acc, tps, sp in rows:
    print(f"| {label} | {acc:.3f} | {tps:.0f} | {sp:.2f}× |")
# 示例输出（你的会不同）：
#   | 0 · 基线 (FP16)       | 0.95 |  620 | 1.00× |
#   | 1 · AWQ INT4 权重     | 0.90 | 1180 | 1.90× |   <- decode 提速 + 省出显存
#   | 2 · + FP8 KV 缓存     | 0.90 | 1210 | 1.95× |   <- 更多 KV 空间；检查分类质量！
#   | 3 · + 花掉省下的显存  | 0.90 | 1820 | 2.94× |   <- 更大 batch 填满省出的显存
#   | 4 · + 前缀缓存        | 0.90 | 2050 | 3.31× |   <- 仅当工作负载共享前缀时
```

**逐行（第 2 步）：** `LADDER` 就是把 §2 的阶梯写成数据——顺着 `engine_kwargs` 往下看，每行相对上一行只加**恰好一个** key（§3.3），于是该行的 delta 可归因到那个 key。循环测每一级、算相对基线行的加速、打印一张可直接粘进报告的 Markdown 表。**质量列是闸门**：横着看它——若某级准确率骤降或某分类塌陷（查 `r["by_category"]`），这级就回退（§3.4），不上线。前缀缓存（第4级）只在你的流量真的共享前缀（系统提示 / few-shot）时才帮忙；在唯一 prompt 上它是空操作——测量，别假设。注意这个离线阶梯止于第4级：第0–4级是能从这个离线批处理上读出的**容量/吞吐**杠杆，而**第5级（批形状调优，`max_num_batched_tokens`）是一个 TTFT↔ITL 的*服务*旋钮**——它的效果显现在 `vllm bench serve` 拐点扫描（第3步）里，而非离线 decode tok/s，所以你在那里对着 p99 调它，并用*那个*测量填报告的第5级行。

**第 3 步 —— 权威的吞吐与拐点**（shell；`pip install vllm[bench]`）：

```bash title="bench.sh"
# decode 吞吐，基线 vs 全调优。两次同形状 => 可比。(vllm bench throughput)
vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct       --num-prompts 200 --input-len 256 --output-len 256
vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct-AWQ   --num-prompts 200 --input-len 256 --output-len 256 \
    --kv-cache-dtype fp8
# 输出行："Throughput: X requests/s, Y total tokens/s, Z output tokens/s" —— 比较两次的 Z。

# 你 SLO 下的拐点：serve 调优后的配置，然后扫到达率（见压测那一课）。
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --max-num-seqs 256 --enable-prefix-caching &
#   然后：python sweep_knee.py   （来自压测课——步进 --request-rate，读 p99 TTFT/goodput）
```

**逐行（第 3 步）：** `vllm bench throughput` 是权威的 decode 数字（真实的批处理负载，而非那个小 eval 集）——在基线和调优配置上用**相同**的 `--input-len`/`--output-len`/`--num-prompts` 跑，比较 **output tokens/s**。用 decode-heavy 形状（长 `--output-len`）；prefill-heavy 形状会低估 AWQ 的收益，因为 weight-only 量化不削减 prefill FLOPs。至于*延迟*那一半，serve 调优配置并跑[拐点扫描](../part8/load-testing-knee.md)（`sweep_knee.py`）得到 p99 TTFT/TPOT 与 SLO 受限的 req/s——那才是你真正上报的容量数字。

## 5 · Lab —— 产出你的优化前 → 后报告

!!! gpu "Capstone Lab（单卡，预算有界）"
    - **最低显存：** 24 GB。FP16 `Qwen2.5-7B`（~15 GB 权重 + KV）是最紧的一级；每个优化后的级都装得下、还有富余。
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）—— ADR-0001 基线。**不用多卡**（第6级 / TP 出范围；为 [Part 7](../part7/nccl-and-launching-tp-pp.md) 标注）。
    - **预估耗时 / 花费：** 下载 + 任何自量化在**无卡模式**（近乎免费）；GPU 时长 ≈ **30–60 分钟**跑完整阶梯 A/B + 两次 bench + 一次短拐点扫描 · **≈ ¥3–8** GPU 时费（**示例**；远在 ¥500 内，而 ¥500 是给你*迭代*用的，不是单次跑）。
    - **平台：** NVIDIA CUDA（默认）。AWQ Marlin + FP8 tensor-core 路径需 Ampere+/Ada——4090（Ada）没问题。
    - **非 NVIDIA：** scorer 与报告生成器是纯 Python、哪都能跑；`LLM(...)` / `vllm bench` 步骤需要受支持的 vLLM 后端，FP8 KV 需要实现了它的后端。

**运行顺序：**（1）无卡模式下，下载 `Qwen/Qwen2.5-7B-Instruct` 与 `Qwen/Qwen2.5-7B-Instruct-AWQ`，并从[eval 集](../eval/small.md)拷来 `small_eval.jsonl` + `score.py`；（2）开 GPU，跑 `capstone_report.py`（阶梯 A/B），逐级读质量列；（3）跑 `bench.sh` 拿权威 decode 吞吐 + 一次短拐点扫描；（4）填下面的模板；（5）**关机。**

拷走它，填进**你的**实测数字——空格是你的，示例值仅供参考：

```markdown title="before_after_report.md"
# 优化前 → 后：单张 RTX 4090 上的 Qwen2.5-7B

SLO：p99 TTFT ≤ ____ ms，p99 TPOT ≤ ____ ms   ·   工作负载：____-in / ____-out，共享前缀？____
基线 vLLM 0.26.0 · greedy（temperature=0, seed=0）· eval = 小集（20 项）

| 阶段（每级只改一个）      | eval 准确率 | 分类回归        | output tok/s | 加速   | SLO 下拐点 req/s | 保留？ |
|---------------------------|-----------:|-----------------|-------------:|-------:|-----------------:|:-----:|
| 0 · 基线 (FP16)           |    ____    | —               |     ____     |  1.00× |       ____        |  n/a  |
| 1 · AWQ INT4 权重         |    ____    | ____            |     ____     |  ____  |       ____        | ____  |
| 2 · + FP8 KV 缓存         |    ____    | ____            |     ____     |  ____  |       ____        | ____  |
| 3 · + 花掉省下的显存      |    ____    | ____            |     ____     |  ____  |       ____        | ____  |
| 4 · + 前缀缓存            |    ____    | ____            |     ____     |  ____  |       ____        | ____  |
| 5 · 批形状调优            |    ____    | ____            |     ____     |  ____  |       ____        | ____  |

## 我保留了什么、为什么
- 第 __ 级：保留——<吞吐收益> 换 <质量代价>，为 <SLO> 值得。
- 第 __ 级：回退——<什么退化了>（如 FP8 KV 掉了数学 8 分）；退让到 <备选>。

## 一句话结论（可辩护）
从 FP16（___ tok/s，准确率 ___）起步，保留的栈达到 ___ tok/s（___× ）、准确率 ___，
p99 TTFT ___ ms / p99 TPOT ___ ms、___ req/s —— 在我的 4090 上测得，花费 ≈ ¥___。
```

那句结论就是面试答案：一个基线、一个倍数、一个准确率 delta、一个延迟预算、一个花费——每个数字都是你的，每一级都可归因。

## 6 · 常见坑 / 反直觉点

- **没有基线行。** 相对「无」的「3× 加速」不可证伪。第0级（FP16、默认）不容商量——它是每个 delta 的分母。
- **每级改好几个旋钮。** AWQ *和* `gpu-memory-utilization` 一起翻，你就无法归因结果——若质量掉了，也分不清怪哪个杠杆。永远一级一改（§3.3）。
- **跳过质量闸门。** 量化和 FP8 KV *悄悄地*退化——流利但更错。「看着还行」不是信号；greedy + seed 的 [eval A/B](../part4/quantization-lab.md) 才是。某分类塌陷（盯 `math`、`format` 和 `bilingual` 项）就回退。
- **叠容量旋钮引发的 OOM 连锁。** 第2–3级先省出、再*花掉*显存；把 `gpu-memory-utilization` 推到 1.0 或 `max-num-seqs` 太高，你会在启动时或突发下 OOM。小步抬、盯着「# GPU blocks」那行。
- **报吞吐却没 SLO。**「2050 tok/s」没有「p99 TTFT ≤ X」就没意义。诚实的容量数字是**拐点处的 goodput（有效吞吐）**（[压测](../part8/load-testing-knee.md)），不是 `--request-rate inf` 的饱和值。
- **测冷启动的 server / 跑太短。** 冷的 CUDA graphs 和热身瞬态不是稳态。先热身 server、用足够多的 prompt，再信一个数字。
- **追别人的魔法值。** 某博客的 `max-num-batched-tokens=16384` 是为*他们的*模型/GPU/流量调的。抄*方法*（阶梯 + 扫描），永远别抄量级——那是你机器的属性。
- **在单卡上试 TP。** `--tensor-parallel-size > 1` 需要 ≥2 GPU；单张 4090 上就是 1。多卡是 [Part 7](../part7/nccl-and-launching-tp-pp.md)（A100 地界，ADR-0001），明确在单卡 Capstone 范围之外——标为未来工作，别假装做了。
- **在有利形状上宣布胜利。** AWQ 的 decode 收益在长输出上显现；prefill-heavy 形状会藏住它。固定并上报 input/output 拆分，否则数字不可迁移。

## 7 · 面试连线

- [系统设计：给推理服务定容与设计](../interview/system-design.md) —— Capstone 端到端演练的那道长题：*给定模型、硬件、SLO 与峰值 QPS，做餐巾纸估算、设计整套服务、并为每个 trade-off 辩护*——含数个完整的做过一遍的设计。Capstone 是动手那一半，那页是白板那一半。
- [容量规划：从一张卡的吞吐到一个集群](../part8/capacity-planning.md) —— 把你实测的单实例拐点换算成 GPU 台数。
- [调参旋钮：哪个对哪个 SLO](../interview/tuning-knobs.md) —— 阶梯所应用的每个旋钮的方向/权衡。

## 8 · 小结 & 延伸阅读

**一句话：** Capstone 就是一次大型的**优化前 → 后 sweep**——爬一条有序阶梯（量化 → FP8 KV → 把省出的显存花在 batch 宽度 → 前缀缓存 → 批形状调优；多卡/TP 出单卡 4090 范围），每级只改**一个**杠杆，每级都跑同一根脊柱（在固定[eval 集](../eval/small.md)上用 greedy+seed 做质量 A/B、用 `vllm bench throughput` 测 output tok/s、用 `vllm bench serve` 扫得 p99/拐点），每级都以质量为闸（某分类塌陷就回退、并*说出来*），最后产出一份结论可辩护的报告——基线 → 倍数 → 准确率 delta → 延迟预算 → 花费——每个数字都在你自己的 4090 上、¥500 之内测得。

延伸阅读：

- [Part 4 · 把 Qwen2.5-7B 量化成 INT4](../part4/quantization-lab.md) —— 第1级，以及整根脊柱复用的质量-A/B 纪律；[FP8 KV 缓存](../part4/quantization-methods.md)是第2级。
- [Part 5 · 调参旋钮 sweep](../part5/tuning-knobs-sweep.md) —— 本项目端到端跑的那个 sweep，以及每一级所暴露的 [continuous batching](../part5/continuous-batching.md) / [PagedAttention](../part5/paged-attention.md) / [前缀缓存](../part5/prefix-caching.md)机制。
- [Part 8 · 压测拐点](../part8/load-testing-knee.md)与 [SLO 驱动调优](../part8/slo-driven-tuning.md) —— 脊柱的延迟那一半，以及「调优 vs 扩容」的决策。
- [Part 8 · 容量规划](../part8/capacity-planning.md) —— 从你实测的拐点到一个集群。
- [评测集](../eval/index.md) —— 每一级都倚赖的测量循环与工具（`score.py`、[小](../eval/small.md) / [大](../eval/large.md)集）。

## 9 · 自测小问

??? question "同事秀出一个 benchmark：「AWQ + FP8 KV + 大 batch 配置做到 2050 output tok/s」。在相信这是一次胜利之前，你会问哪三件事？"
    （1）**相对什么基线？** 2050 tok/s 没有它改进*之前*的 FP16 默认数字、以及两者*相同的 benchmark 形状*，就毫无意义——否则不可证伪（§3.1）。（2）**在什么质量下？** 量化和 FP8 KV 悄悄退化；要 greedy + seed 的 [eval A/B](../part4/quantization-lab.md) 和*分类*拆分——一个塌了 `math` 或 `bilingual` 项的 2050-tok/s 配置是回归，不是胜利（§3.4）。（3）**在什么 SLO 下？** 裸吞吐忽略延迟；要 p99 TTFT/TPOT 和[拐点](../part8/load-testing-knee.md)（SLO 下的 goodput / 有效吞吐），因为饱和的 server 吞吐很大、可每个用户都在等。还有第四件：**是一次改动还是好几个？** 若 AWQ、FP8 KV、batch 旋钮一起翻，这数字无法归因——你不知道哪个杠杆挣来的、也不知道回退哪个会崩。

??? question "为什么阶梯把量化放第一、把「花掉省出的显存」放后面，而不是反过来？"
    因为这些级有**依赖顺序**，不是随意挑。量化第一，因为它是最大、最接近免费的杠杆：它一次抬高*两道闸*——AWQ 省出 ~10 GB 权重显存（更多空间给 [KV cache](../part2/kv-cache-math.md)），*同时*缩小[带宽受限 decode](../part2/roofline-analysis.md) 的权重读取（更快出字）——只付一点点可测量的质量代价。「花掉省出的显存」（抬 `gpu-memory-utilization` / `max-num-seqs` 换更大的 [continuous batch](../part5/continuous-batching.md)）是量化*产出*的资源的*消费者*：先做它、在 FP16 上，要么 OOM、要么只装下少得多的序列。所以你从「两头都帮、便宜、还产出资源」往下爬到「花掉那资源的 trade-off 旋钮」。每一级踩在下一级上——这也是为什么每级之后都要测：第3级的收益之所以存在，正因为第1–2级为它腾出了空间。

??? question "面试里你展示这份报告，并提到你*回退*了 FP8 KV 缓存、因为它掉了数学准确率。承认一次被回退的优化是弱点吗？它证明了什么？"
    恰恰相反——这是报告里最资深的信号。它证明你（1）**测的是质量、不只是速度**——否则你根本抓不到一次悄悄的、分类特定的退化；（2）**把质量当闸门**，于是你优化出的是*可上线*的配置、而非一个更大但坏掉的数字；（3）**能把回归归因到一个杠杆**，而这只有在你一次只改一个东西时才可能。一个只报单调胜利的候选人，要么运气好、要么没看——真实的优化工作满是不划算的杠杆，知道该丢*哪个*、还有数字佐证，才是本事。结论「3.3×、准确率 −0.02，且因 8 分数学下降而否掉了 FP8 KV」远比一句光秃秃的「3.5×」有力。
