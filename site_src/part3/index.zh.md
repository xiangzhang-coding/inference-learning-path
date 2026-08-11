# Part 3 · GPU 编程（Triton）

> 只学到「能**推理** kernel 为何快、能**读** vLLM kernel」的程度——不是一门 CUDA C++ 课程（深度边界见 ADR-0002）。

## 本 Part 覆盖

- **CUDA 执行模型**与访存，用心智模型而非死记细节
- 动手写几个简单 **Triton** kernel——获得「能写一点」的底气
- 被带着导读 vLLM 的 **PagedAttention** kernel，建立读源码能力

GPU 编程术语见 **[术语表](../glossary.md)**。

!!! note "脚手架状态"
    本 Part 课程在后续票落地。手写 / 优化 CUDA C++ 明确不在范围（ADR-0002）。
