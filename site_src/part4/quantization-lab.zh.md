# 动手：把 Qwen2.5-7B 量化成 INT4、在 vLLM 里服务、对比质量与吞吐

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    工具——**llm-compressor**（`oneshot` + `GPTQModifier(scheme="W4A16")`）、vLLM 对压缩 checkpoint 的自动检测、`LLM.chat`、`kv_cache_dtype="fp8"`、以及 `vllm bench throughput`——经 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。**作者不运行其中任何一样**（ADR-0004）：下面每个 VRAM / tokens-per-second / 精度数字都是**示例 / 量级参考**。你在自己 4090 上得到的数字才是真的——本 Lab 的要点就是让*你*去测。

---

## 1 · 直觉 & 为什么重要

Part 4 的一切在这里汇合：拿 `Qwen2.5-7B-Instruct`、把它变 INT4、服务它，并**证明两件事**——它变小/变快了，*且*它没变笨。第二半是新手会跳过、面试官会追问的部分。量化**悄悄地**失败：模型仍吐出流畅文本，所以「看着没问题」毫无价值。专业做法是 A/B：在 FP16 基线与 INT4 模型上用完全相同的贪心设置跑一个固定评测，比较一个*数字*。若质量守住，你白赚了 ~4× 更少的权重内存与更快的 decode；若掉了，你退一步（更高比特、更好方法、per-group、不同校准）。

要内化的一个工作流：**量化一次（离线），服务多次，永远对着 FP16 做 A/B。** 量化是一次性离线成本；此后每个请求都复用那个 checkpoint。且因为 [decode 是 memory-bound](../part2/roofline-analysis.md)，INT4 权重读取正是加速来处——这也意味着赢面出现在 *decode 吞吐*，而非 prefill FLOPs。→ 术语见[术语表](../glossary.md)；本课把[方法](quantization-methods.md) 与[基础](quantization-basics.md) 课落到实操。

## 2 · 心智模型

端到端路径，以及每个 Part-4 想法插在哪：

```text
 [FP16 model]  ──quantize (OFFLINE, once)──►  [INT4 checkpoint]  ──serve──►  [measure]
  Qwen2.5-7B     llm-compressor:                compressed-tensors            A/B vs FP16:
  ~15 GB wts     GPTQModifier(W4A16),           ~4–5 GB wts                   • quality: small eval (#3)
                 calibration set                vLLM auto-detects it            greedy, seed, per-category
                 (or: grab a prebuilt            (no flag needed)              • speed: vllm bench throughput
                  Qwen2.5-7B-Instruct-AWQ)                                       output tokens/s
                                                                              • memory: freed VRAM → more KV
```

两个要抓住的形状：

- **两种拿到 INT4 checkpoint 的方式。** 最快最省：下载一个**预量化**社区 checkpoint（`Qwen/Qwen2.5-7B-Instruct-AWQ`）——零量化算力，在 AutoDL 无卡模式做。或在你需要特定 scheme/校准时**自己量化**（llm-compressor）。无论哪种，vLLM 从 checkpoint config 自动检测格式——不需要 `--quantization` flag。
- **量化结果是一张表，不是一种感觉。** 交付物是 before→after 对比：VRAM、decode 吞吐（output tokens/s）、每类别评测精度。那张表告诉你 INT4 是否值得——也正是 Capstone 要的「优化前后报告」。

## 3 · 原理 —— 四步

### 3.1 拿到 INT4 checkpoint

**路径 A（便宜，建议起步）：** 用预量化 checkpoint。`Qwen/Qwen2.5-7B-Instruct-AWQ` 是官方 AWQ INT4 构建；vLLM 直接加载。*下载*它不需要 GPU——在无卡模式做。

**路径 B（自量化）：** 用一个 `W4A16` 的 `GPTQModifier` 在小**校准**集上跑 llm-compressor 的 `oneshot`。这是[方法那课](quantization-methods.md) 的 GPTQ recipe 落地：权重量化成 4-bit per-group、逐层校正误差，并 `ignore=["lm_head"]`。用 `save_compressed=True` 保存，得到 vLLM 认识的 compressed-tensors checkpoint。

### 3.2 在 vLLM 里服务

把 vLLM 指向 checkpoint。它从 config 读量化方法、选对 INT4 kernel（Ampere+ 上的 Marlin）；你不传 flag。用 `LLM.chat(...)`（不是 `generate`）以套上 Instruct chat 模板——`generate` 会喂畸形 prompt、因无关原因拖垮质量。

### 3.3 测质量 —— 小评测集（#3）

量化的失败模式是悄悄掉质量，所以你**测**。复用[小评测集](../eval/small.md)：20 条带确定性程序化检查。在 FP16 与 INT4 上用 `temperature=0.0` + 固定 `seed` 跑（好让重跑不会偶然不同），diff 每类别精度。集中在某一类别的下降（如 `math` 或 `format`）就是你退一步的信号。

### 3.4 测吞吐与内存

用 `vllm bench throughput`（来自 `pip install vllm[bench]`）在各 checkpoint 上跑、比较 **output tokens/s**——量化要推动的 decode 吞吐数字。也留意释放的 VRAM：INT4 权重（~4–5 GB vs ~15 GB）为 [KV cache](../part0/kv-cache.md) 腾出多得多的空间，于是你能提并发——并可选叠 **FP8 KV cache**（`kv_cache_dtype="fp8"`）再省。

## 4 · 完整可跑代码 + 逐行讲解

三块：量化（或直接用预量化）、A/B 质量、benchmark 吞吐。**你在 4090 上可跑；作者不执行**——注释里的数字是示例。

**第 1 步 —— 用 llm-compressor 量化成 INT4**（路径 B；用预量化 AWQ 则跳过）：

```python title="quantize_int4.py"
"""用 llm-compressor 把 Qwen2.5-7B-Instruct 量化成 INT4（W4A16）。跑一次、离线。
API 形状经 vLLM 0.26.0 量化文档核实；作者不执行（ADR-0004）。"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SAVE_DIR = "Qwen2.5-7B-Instruct-W4A16-G128"
NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN = 512, 2048

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 校准集：几百条有代表性的 prompt，先套 chat 模板再 tokenize。
# 偏离分布的校准会选错范围——用真实文本。
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(lambda s: tokenizer(s["text"], max_length=MAX_SEQ_LEN, truncation=True,
                                add_special_tokens=False), remove_columns=ds.column_names)

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])   # 4-bit 权重、per-group
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=MAX_SEQ_LEN, num_calibration_samples=NUM_CALIBRATION_SAMPLES)

model.save_pretrained(SAVE_DIR, save_compressed=True)   # compressed-tensors checkpoint
tokenizer.save_pretrained(SAVE_DIR)
print(f"INT4 checkpoint saved to {SAVE_DIR}")           # 磁盘上 ~4–5 GB vs FP16 ~15 GB（示例）
```

**逐行讲解（第 1 步）：** 正常加载 FP16 模型 + tokenizer。**校准集**是来自 `ultrachat_200k` 的 512 条套过 chat 模板的样本（用像你部署流量的文本——其分布要紧，§6），经两次 `ds.map` tokenize。`GPTQModifier(scheme="W4A16", ignore=["lm_head"])` 是[方法那课](quantization-methods.md) 的 GPTQ 放到轴上：4-bit 权重、FP16 激活、per-group，且绝不量化输出投影。`oneshot(...)` 跑逐层量化 + 误差校正；`save_pretrained(save_compressed=True)` 写出 vLLM 自动检测的 checkpoint。要跳过这一切，在第 2 步把 `MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"`。

**第 2 步 —— 在小评测集上 A/B 质量**（复用[小评测集](../eval/small.md) 的 `score.py`）：

```python title="ab_quality.py"
"""在 FP16 vs INT4 上用完全相同的贪心设置跑小评测集；diff 精度。
复用小评测集（#3）的 load_items/summarize。数字是示例。"""
import json
from vllm import LLM, SamplingParams
from score import load_items, summarize          # 来自小评测集（#3）

CHECKPOINTS = {
    "fp16": "Qwen/Qwen2.5-7B-Instruct",
    "int4": "Qwen2.5-7B-Instruct-W4A16-G128",    # 或 "Qwen/Qwen2.5-7B-Instruct-AWQ"
}
items = load_items("small_eval.jsonl")
convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # 贪心 + seed => 可比

for tag, model in CHECKPOINTS.items():
    llm = LLM(model=model, max_model_len=4096)   # vLLM 从 config 自动检测 INT4
    outs = [o.outputs[0].text for o in llm.chat(convos, sp)]   # chat() 套模板
    report = summarize(items, outs)
    print(tag, json.dumps(report["by_category"], ensure_ascii=False), "acc=", round(report["accuracy"], 3))
    del llm                                       # 加载下一个模型前释放 VRAM
# 示例：
#   fp16 {...} acc= 0.95
#   int4 {...} acc= 0.90   <- 小而可容忍的下降；某类别大崩塌则退一步
```

**逐行讲解（第 2 步）：** 同样 20 条、同样**贪心**采样（`temperature=0.0`、固定 `seed`）对两个模型——唯一变量是权重。`LLM.chat` 套 Instruct 模板；`summarize` 给总体 + 每类别精度。加载一个模型、评测、`del` 释放 VRAM、再下一个（两个 7B 模型在 24 GB 上共存不下）。比较两份报告：小的总体下降是预期且没问题；某*类别*崩塌（比如 `format` 1.0 → 0.3）是提比特或换方法的信号。

**第 3 步 —— benchmark 吞吐与内存**（shell；`pip install vllm[bench]`）：

```bash title="bench.sh"
# 比较 decode 吞吐：output tokens/s，FP16 vs INT4。（已核实：`vllm bench throughput`）
vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct            --num-prompts 200 --input-len 256 --output-len 256
vllm bench throughput --model Qwen2.5-7B-Instruct-W4A16-G128      --num-prompts 200 --input-len 256 --output-len 256
# 输出行形如：
#   Throughput: 7.15 requests/s, 4656.00 total tokens/s, 1072.15 output tokens/s
# 比较两次运行的 "output tokens/s"（INT4 更高，示例 decode-heavy 形状 ~1.5–3x）。
#
# 可选：叠 FP8 KV cache 以装下更多并发序列（释放 KV 内存）：
#   vllm serve Qwen2.5-7B-Instruct-W4A16-G128 --kv-cache-dtype fp8
```

**逐行讲解（第 3 步）：** `vllm bench throughput` 驱动一批请求、报告 requests/s 与 tokens/s。用*相同*形状（`--input-len`/`--output-len`/`--num-prompts`）在 FP16 与 INT4 checkpoint 上跑、比较 **output tokens/s**——decode 数字。用 decode-heavy 形状（长 `--output-len`）来看权重读取赢面；prefill-heavy 形状显示更少，因为 INT4 weight-only 不砍 prefill FLOPs。可选的 `--kv-cache-dtype fp8` 是[方法那课](quantization-methods.md) 里正交的 KV 杠杆。

合起来，交付物是一张 before→after 表（示例——**你的会不同**）：

```text
                     FP16 baseline      INT4 (W4A16)      note
  weight memory      ~15 GB             ~4–5 GB           ~3–4x smaller
  output tokens/s    1.0x (ref)         ~1.5–3x           decode-heavy shape; memory-bound win
  small-eval acc     ~0.95              ~0.90             A/B with greedy+seed; watch per-category
  max concurrency    baseline           higher            freed VRAM → more KV cache (+ FP8 KV)
```

## 5 · Lab —— 自己跑出 before→after 报告

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB（FP16 `Qwen2.5-7B` 需 ~15 GB 权重 + KV/开销；INT4 模型轻松装下）。自量化（第 1 步）也在 24 GB 内。
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）—— ADR-0001 基线。
    - **预估耗时 / 花费：** 在**无卡模式**下载/量化（免费）；GPU 运行 ~15–25 分钟做完两次 A/B eval + 两次吞吐 bench · **≈ ¥2–4** GPU 时长（**示例**；取决于卡价/速度）。
    - **平台：** NVIDIA CUDA（默认）。INT4 Marlin kernel 需 Ampere+（4090 是 Ada——没问题）。
    - **非 NVIDIA：** 评分器是纯 Python、到处能跑；`LLM(...)`/`vllm bench` 步骤需支持的 vLLM 后端，FP8 tensor-core 路径需 Hopper/Ada。

**运行顺序：**（1）无卡模式下下载 FP16 模型，并下载 `Qwen/Qwen2.5-7B-Instruct-AWQ` 或跑 `quantize_int4.py`；（2）开 GPU，跑 `ab_quality.py`、读每类别 diff；（3）跑两条 `vllm bench throughput`；（4）填你自己的 before→after 表。**质量门：** 若总体精度守住且无类别崩塌，INT4 可留。若某类别崩了，那是提示去试 INT8/`W8A8`、别的方法或更好校准——你现在有[方法那课](quantization-methods.md) 的框架去选。

## 6 · 常见坑 / 反直觉点

- **完全不测质量。** 量化*悄悄*退化——流畅但更错的输出。「看着没问题」不是信号；[小评测](../eval/small.md) A/B 才是。这是最常见的错误。
- **非贪心 A/B。** `temperature > 0` 时，重跑会偶然不同，你没法把差异归因于量化。两个模型都用 `temperature=0.0` + 固定 `seed`。
- **用 `generate` 而非 `chat`。** 跳过 chat 模板会喂 Instruct 模型畸形 prompt；质量因与量化无关的原因看着糟。
- **在错的形状上测吞吐。** INT4 weight-only 加速 **decode**（memory-bound 权重读取）。benchmark 一个 decode-heavy 形状（长 `--output-len`）；prefill-heavy 形状低估赢面，因为 prefill 是 compute-bound、权重反正被反量化回 FP16。
- **量化 `lm_head`。** 对精度敏感且很小——总是 `ignore=["lm_head"]`。量化它是无谓的精度损失。
- **偏离分布的校准。** GPTQ/AWQ 从校准集选 scale；垃圾或跨域校准 → 错的范围 → 更差精度。用几百条有代表性、套 chat 模板的 prompt。
- **忘了花掉释放的 VRAM。** INT4 释放 ~10 GB；若你不提 `--max-num-seqs` / `--gpu-memory-utilization`（或加 FP8 KV），你就买了没用的内存——并发收益是可选项。
- **不需要时传 `--quantization`。** vLLM 从预量化/压缩 checkpoint 的 config 自动检测方法。显式 flag 是给在线量化的（如 `fp8`、`bitsandbytes`）。

## 7 · 面试连线

- [实操量化与服务：量化 → 服务 → 验证](../interview/quantization-serving.md) —— 这节课为你准备的高频题：*你怎么量化一个模型、服务它、并证明质量守住；你测什么、用什么设置？*

## 8 · 小结 & 延伸阅读

**一句话：** 把 `Qwen2.5-7B` 量化成 INT4 一次（llm-compressor `W4A16`，或拿一个预量化 AWQ checkpoint）、在 vLLM 里服务（自动检测、无 flag），然后**对着 FP16 做 A/B**——质量用小评测集（贪心 + seed、按类别）、速度用 `vllm bench throughput`（output tokens/s）——产出一张 before→after 表，证明这个小 ~4×、decode 更快的模型没掉质量。

延伸阅读：

- [方法那课](quantization-methods.md) —— GPTQ/AWQ/SmoothQuant/FP8/LLM.int8() 作为设计空间里的点，好让你在 INT4 质量下滑时知道*该伸手拿什么*。
- [小评测集](../eval/small.md) —— 本 Lab 复用的质量 harness；有信号后用[大评测集](../eval/large.md) 拿可信数字。
- llm-compressor 示例（W4A16、W8A8）与 vLLM 量化 + `vllm bench` 文档 —— 确切的 recipe 与 flag。
- Part 5（服务与吞吐）—— 你把释放的 VRAM 花在并发上的地方，以及 Capstone 的 before→after 报告。

## 9 · 自测小问

??? question "你把 Qwen2.5-7B 量化成 INT4，它仍产生流畅答案。为什么这还不够，你怎么做？"
    流畅不等于正确——量化**悄悄**退化，所以模型可以听着没问题却答错更多。你必须**对着 FP16 基线做 A/B**、用一个固定评测：在两个模型上用完全相同的**贪心**设置（`temperature=0.0`、固定 `seed`）跑[小评测集](../eval/small.md)，好让唯一变量是权重，并**按类别**比较精度。小的总体下降可接受；某类别崩塌（如 `math` 或 `format`）意味退一步——更多比特、别的方法、per-group 粒度或更好校准。交付物是一个数字，不是一种印象。

??? question "INT4 量化后，你的 benchmark 里 decode 吞吐几乎没提升。给两个可能的原因。"
    （1）**Prefill-heavy 的 benchmark 形状。** INT4 weight-only 加速的是 *memory-bound decode* 的权重读取；若你用短 `--output-len`（大多是 prefill）来测，赢面被压平，因为 prefill 是 compute-bound、INT4 权重反正为 matmul 反量化回 FP16。用 decode-heavy 形状（长输出）。（2）**你没用上释放的 VRAM / 该 batch 下你不 memory-bound。** 若并发很低，decode 可能没在打满 HBM 带宽，于是砍权重字节帮得少；提 `--max-num-seqs`（用上 INT4 释放的 ~10 GB）来推更多并发 decode，才是把带宽节省变成吞吐的办法。（也检查 INT4 Marlin kernel 真的启用了、且你测的是稳态而非模型加载。）

??? question "走一遍在 vLLM 里上线一个 INT4 Qwen2.5-7B 的完整工作流，说出工具与检查。"
    （1）**拿 checkpoint** —— 要么下载一个预量化的（`Qwen/Qwen2.5-7B-Instruct-AWQ`），要么用 **llm-compressor** `oneshot` + `GPTQModifier(scheme="W4A16", ignore=["lm_head"])` 在小的有代表性校准集上自量化、用 `save_compressed=True` 保存。（2）**服务** —— 在 vLLM 里指向 checkpoint，vLLM 从 config **自动检测**量化（无 `--quantization` flag），用 INT4 Marlin kernel。（3）**验证质量** —— 用 `LLM.chat`、贪心 + seed 在 FP16 vs INT4 上跑小评测集、diff 每类别精度。（4）**测速度/内存** —— 在两者上 `vllm bench throughput`、比较 output tokens/s（decode-heavy 形状），留意释放的 VRAM；可选加 `--kv-cache-dtype fp8` 并提并发。交付物：一张 before→after 表（VRAM、output tokens/s、精度）。
