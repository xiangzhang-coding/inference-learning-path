# 实操量化与服务：量化 → 服务 → 验证

!!! info "基线：**vLLM 0.26.0** · 工具/flag 经 Context7 核实（ADR-0004）"

**模块：** Part 4 · 量化   ·   **对应课程：** [动手：把 Qwen2.5-7B 量化成 INT4、在 vLLM 里服务、对比质量与吞吐](../part4/quantization-lab.md)

---

## Q：让你在单张 4090 上上线一个 INT4 版的 Qwen2.5-7B。走一遍量化它、服务它、并证明它没掉质量。你到底测什么、用什么设置？

### 直接答案

四步：

1. **拿到 INT4 checkpoint** —— 要么预量化的（`Qwen/Qwen2.5-7B-Instruct-AWQ`），要么用 **llm-compressor** 自量化：`oneshot` + `GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])` 在小的有代表性校准集上，`save_pretrained(save_compressed=True)`。（这是一次性离线成本，在无卡模式做。）
2. **服务** —— 在 vLLM 里指向 checkpoint，vLLM 从 config **自动检测**方法（无 `--quantization` flag），用 INT4 Marlin kernel。
3. **验证质量** —— 用 `LLM.chat`（套 chat 模板）、**完全相同的贪心设置**（`temperature=0.0`、固定 `seed`）在 FP16 vs INT4 上 A/B 跑[小评测集](../eval/small.md)、diff **每类别**精度。
4. **测速度/内存** —— 在两者上 `vllm bench throughput`、比较 **output tokens/s**（decode-heavy 形状）；留意释放的 VRAM（~15 GB → ~4–5 GB 权重），可选加 `--kv-cache-dtype fp8` + 提并发。

**交付物：** 一张 before→after 表（VRAM、output tokens/s、每类别精度）。

### 深入原理

- **量化悄悄失败。** 模型保持流畅却答错更多，所以「看着没问题」毫无价值——你需要固定评测和一个*数字*。贪心 + seed 让 FP16 与 INT4 直接可比（唯一变量是权重）。
- **量化一次，服务多次。** 量化一趟是一次性离线成本；checkpoint 被每个请求复用。绝不按请求量化。
- **赢面在 decode，不在 prefill。** INT4 weight-only 砍 memory-bound 权重读取 → decode 吞吐。测 decode-heavy 形状；prefill（compute-bound）几乎不动，因为权重为 matmul 反量化回 FP16。
- **花掉释放的 VRAM。** 释放 ~10 GB → 提 `--max-num-seqs` / `--gpu-memory-utilization`（更多 KV cache → 更多并发），或叠 FP8 KV cache。并发收益是可选项。

### 代码

用完全相同贪心设置 A/B 质量（复用评分器）：

```python
from vllm import LLM, SamplingParams
from score import load_items, summarize          # 来自小评测集

items = load_items("small_eval.jsonl")
convos = [[{"role": "user", "content": it["prompt"]}] for it in items]
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)   # 贪心 + seed => 可比

for tag, model in {"fp16": "Qwen/Qwen2.5-7B-Instruct",
                   "int4": "Qwen/Qwen2.5-7B-Instruct-AWQ"}.items():
    llm = LLM(model=model, max_model_len=4096)   # vLLM 自动检测 INT4
    outs = [o.outputs[0].text for o in llm.chat(convos, sp)]
    print(tag, "acc=", round(summarize(items, outs)["accuracy"], 3)); del llm
# 示例：  fp16 acc= 0.95   int4 acc= 0.90
```

### 面试官追问

- *「为什么贪心 + 固定 seed？」* → 好让重跑不会偶然不同；任何精度差都可归因于量化、而非采样噪声。这是唯一公平的 A/B。
- *「为什么 `chat` 不是 `generate`？」* → `chat` 套 Instruct chat 模板；`generate` 喂原始 prompt，于是 Instruct 模型因与量化无关的原因得分很糟。
- *「总体精度守住但 `format` 从 1.0 掉到 0.4——怎么办？」* → 某类别崩塌意味 INT4 伤了那个能力；退一步——INT8 / `W8A8`、别的方法、更细粒度或更好校准。按类别正是你不能只看总体数字的原因。
- *「vLLM 运行——传 `--quantization` 吗？」* → 预量化/压缩 checkpoint 不传；vLLM 从 config 自动检测。flag 是给在线量化的（`fp8`、`bitsandbytes`）。
- *「怎么让吞吐增益显现？」* → 测 decode-heavy 形状，并真的用上释放的 VRAM（更高并发 / `--max-num-seqs`），否则你买了没花的内存。

### 关联概念

- 课程：[动手：把 Qwen2.5-7B 量化成 INT4](../part4/quantization-lab.md)
- 相关：[量化方法（该伸手拿什么）](quantization-methods.md)、[数值格式与精度](number-formats.md)、[显存预算与最大并发](vram-capacity-planning.md)（花掉释放的内存）、[延迟与吞吐度量](latency-throughput-metrics.md)（测什么）
- 术语表：[Quantization、PTQ/QAT、KV-cache quantization](../glossary.md)
