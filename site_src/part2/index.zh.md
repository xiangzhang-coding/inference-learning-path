# Part 2 · 单卡推理性能

> 先建立硬件心智模型，再学会在信任任何优化前先**量化**它。

## 本 Part 覆盖

- **GPU 硬件模型**：SM / warp、**HBM vs SRAM**、带宽 vs 算力
- **度量与测量**：TTFT、TPOT/ITL、throughput、goodput
- **Roofline 与算术强度**：attention / GEMM 受算力还是受带宽限制？
- **FlashAttention**：IO-aware 思想（tiling、online softmax）与它为何更快
- **Kernel fusion 与 CUDA graphs**：decode 阶段 launch overhead 为何致命

性能术语见 **[术语表](../glossary.md)**。

!!! note "脚手架状态"
    本 Part 课程在后续票落地，依赖 **[Part 0](../part0/index.md)**。
