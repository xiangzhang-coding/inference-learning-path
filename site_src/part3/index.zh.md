# Part 3 · GPU 编程（Triton）

> 只学到「能**推理** kernel 为何快、能**读** vLLM kernel」的程度——不是一门 CUDA C++ 课程（深度边界见 ADR-0002）。

## 本 Part 覆盖

- **[CUDA 执行模型](cuda-execution-model.md)**——grid/block/thread、warp、SIMT divergence、occupancy——用心智模型而非死记细节
- **[访存](memory-access.md)**——coalescing、shared memory、bank conflict：一个 warp *应该*怎样触碰内存
- 动手写几个简单 **[Triton](triton-basics.md)** kernel——获得「能写一点」的底气
- 被带着导读 vLLM 的 **[PagedAttention kernel](paged-attention-kernel.md)**，建立读源码能力

GPU 编程术语见 **[术语表](../glossary.md)**。

## 课程

- **[CUDA 执行模型：线程、Warp 与 Occupancy](cuda-execution-model.md)** —— GPU 如何靠驻留许多 32 线程的 warp 来隐藏访存时延；为什么 warp 内部的 SIMT divergence 会串行化；以及为什么 occupancy 是*隐藏时延的余量*，不是速度旋钮。
- **[访存：Coalescing、Shared Memory 与 Bank Conflict](memory-access.md)** —— 为什么一个 warp 的 32 条 lane 应触碰连续地址（uncoalesced 访问搬多达 32× 的字节）、shared memory 靠复用换来什么，以及 bank 冲突这个坑与它一列 padding 的修法。
- **[Triton：写你的第一个 GPU kernel](triton-basics.md)** —— 用「作用在块上的 Python」写 vector add → fused softmax → simple matmul：`program_id`、offset、mask、片上归约、`tl.dot`、autotuning，warp 与 coalescing 交给编译器。
- **[读 vLLM 的 PagedAttention kernel](paged-attention-kernel.md)** —— 虚拟内存思想（KV 块 + block table）、kernel 如何逐块 gather KV 并用 online softmax 折进来，以及一份 paged == dense attention 的纯 Python 证明。

!!! note "脚手架状态"
    Part 3 已完成：四节课（票 #8、#9）全部写就，各带一道双向链接的面试题——GPU 编程心智模型（执行模型、访存）与动手那一半（Triton kernel、读 PagedAttention kernel）。手写 / 优化 CUDA C++ 仍不在范围（ADR-0002）；PagedAttention 的*服务*深挖（continuous batching、block manager）在 **Part 5** 落地。所有 vLLM flag/API 经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**。链接题集见 [面试题库](../interview/index.md)。
