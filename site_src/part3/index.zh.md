# Part 3 · GPU 编程（Triton）

> 只学到「能**推理** kernel 为何快、能**读** vLLM kernel」的程度——不是一门 CUDA C++ 课程（深度边界见 ADR-0002）。

## 本 Part 覆盖

- **[CUDA 执行模型](cuda-execution-model.md)**——grid/block/thread、warp、SIMT divergence、occupancy——用心智模型而非死记细节
- **[访存](memory-access.md)**——coalescing、shared memory、bank conflict：一个 warp *应该*怎样触碰内存
- 动手写几个简单 **Triton** kernel——获得「能写一点」的底气 *（下一张票）*
- 被带着导读 vLLM 的 **PagedAttention** kernel，建立读源码能力 *（下一张票）*

GPU 编程术语见 **[术语表](../glossary.md)**。

## 课程

- **[CUDA 执行模型：线程、Warp 与 Occupancy](cuda-execution-model.md)** —— GPU 如何靠驻留许多 32 线程的 warp 来隐藏访存时延；为什么 warp 内部的 SIMT divergence 会串行化；以及为什么 occupancy 是*隐藏时延的余量*，不是速度旋钮。
- **[访存：Coalescing、Shared Memory 与 Bank Conflict](memory-access.md)** —— 为什么一个 warp 的 32 条 lane 应触碰连续地址（uncoalesced 访问搬多达 32× 的字节）、shared memory 靠复用换来什么，以及 bank 冲突这个坑与它一列 padding 的修法。

!!! note "脚手架状态"
    Part 3 的心智模型那一半已就位（票 #8）：[CUDA 执行模型](cuda-execution-model.md) 与 [访存](memory-access.md)，各带一道双向链接的面试题。动手 **Triton** 与 **PagedAttention** kernel 导读随后落地。手写 / 优化 CUDA C++ 明确不在范围（ADR-0002）。链接题集见 [面试题库](../interview/index.md)。
