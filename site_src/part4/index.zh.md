# Part 4 · 量化

> 量化为何能提吞吐、代价是多少精度、真实场景怎么选型。

## 本 Part 覆盖

- **[量化为何有助吞吐](quantization-basics.md)**，以及**精度权衡**——仿射映射与它的误差界
- **[Weight-only vs weight+activation](quantization-schemes.md)**、粒度（per-tensor/channel/group）、对称/非对称，以及 PTQ vs QAT
- **[方法族](quantization-methods.md)**：**GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** 与 **KV-cache 量化**
- **[一段完整可跑路径](quantization-lab.md)**：量化 `Qwen2.5-7B-Instruct` 并在 vLLM 里跑 INT4，对比质量与吞吐

KV-cache 量化直接连回 **[KV 缓存](../part0/kv-cache.md)**。

## 课程

- **[量化为何能加速推理：仿射映射与精度权衡](quantization-basics.md)** —— 为什么更少权重比特意味更快的 memory-bound decode（带宽、非 FLOPs）、仿射量化映射（$\hat{x}=\text{scale}\cdot(q-z)$），以及被 outlier 抬高的误差界（$\le \text{scale}/2$）。
- **[量化的选择：粒度、对称性、量化什么，以及 PTQ vs QAT](quantization-schemes.md)** —— 在低比特下把误差压小的四个工程选择：per-tensor/channel/group、对称/非对称、weight-only（`W4A16`）vs weight+activation（`W8A8`），以及为什么推理用 PTQ。
- **[量化方法族：GPTQ、AWQ、SmoothQuant、FP8、LLM.int8()、KV-cache](quantization-methods.md)** —— 每个方法都是设计空间里的一个点外加一个抗 outlier 巧招，以及如何为给定瓶颈选一个。
- **[动手：把 Qwen2.5-7B 量化成 INT4、在 vLLM 里服务、对比质量与吞吐](quantization-lab.md)** —— 完整可跑路径：用 llm-compressor（或预量化 AWQ checkpoint）量化、服务（自动检测）、在小评测集上 A/B 质量并测吞吐。

!!! note "Part 4 完成"
    四节课全部写就，各带一道双向链接的面试题——原理（[基础](quantization-basics.md)、[方案](quantization-schemes.md)）与应用那一半（[方法族](quantization-methods.md)、[动手 INT4 实操](quantization-lab.md)）。手写 CUDA kernel 仍不在范围（ADR-0002）；把释放的 VRAM 花在并发上是 **Part 5**。所有 vLLM flag/API 经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**。量化术语见 **[术语表](../glossary.md)**，链接题集见 [面试题库](../interview/index.md)。
