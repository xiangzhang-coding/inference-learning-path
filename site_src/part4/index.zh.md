# Part 4 · 量化

> 量化为何能提吞吐、代价是多少精度、真实场景怎么选型。

## 本 Part 覆盖

- **[量化为何有助吞吐](quantization-basics.md)**，以及**精度权衡**——仿射映射与它的误差界
- **[Weight-only vs weight+activation](quantization-schemes.md)**、粒度（per-tensor/channel/group）、对称/非对称，以及 PTQ vs QAT
- 方法族：**GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** 与 **KV-cache 量化** *（下一张票）*
- 一段完整可跑路径：量化 `Qwen2.5-7B-Instruct` 并在 vLLM 里跑 INT4 *（下一张票）*

KV-cache 量化直接连回 **[KV 缓存](../part0/kv-cache.md)**。

## 课程

- **[量化为何能加速推理：仿射映射与精度权衡](quantization-basics.md)** —— 为什么更少权重比特意味更快的 memory-bound decode（带宽、非 FLOPs）、仿射量化映射（$\hat{x}=\text{scale}\cdot(q-z)$），以及被 outlier 抬高的误差界（$\le \text{scale}/2$）。
- **[量化的选择：粒度、对称性、量化什么，以及 PTQ vs QAT](quantization-schemes.md)** —— 在低比特下把误差压小的四个工程选择：per-tensor/channel/group、对称/非对称、weight-only（`W4A16`）vs weight+activation（`W8A8`），以及为什么推理用 PTQ。

!!! note "脚手架状态"
    Part 4 的原理那一半已就位（票 #10）：[量化基础](quantization-basics.md) 与 [四个方案选择](quantization-schemes.md)，各带一道双向链接的面试题。具体方法族（GPTQ/AWQ/SmoothQuant/FP8/LLM.int8()、KV-cache 量化）与动手把 `Qwen2.5-7B` → INT4 在 vLLM 里跑随后落地（#11）。量化术语见 **[术语表](../glossary.md)**，链接题集见 [面试题库](../interview/index.md)。
