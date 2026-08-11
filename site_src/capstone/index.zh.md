# Capstone 实战

> 收官之作：在**单张 RTX 4090**、**¥500 AutoDL 预算**内，把 `Qwen2.5-7B-Instruct` 的吞吐拉到极限——并产出**「优化前 → 优化 → 优化后」报告**。

## 目标

把 Part 0–8 的一切用起来，拧那些真正移动数字的旋钮：

- 量化（Part 4）以在 24 GB 里塞下更多 KV cache
- Continuous batching + PagedAttention 调优（Part 5）
- 工作负载有共享前缀时用 prefix caching
- 为你的 SLO 选对 `gpu-memory-utilization`、`max-model-len` 与批处理设置

你要产出一份在自己 AutoDL 环境上测得的、**优化前后**吞吐与延迟对比报告。

!!! gpu "Capstone Lab"
    - **最低显存：** 24 GB
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）
    - **预估耗时 / 花费：** 几小时 GPU 时长，远在 ¥500 内（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** 相关技术处标注兼容性差异

!!! warning "关于数字"
    这里所有目标与数字都是**示例 / 量级参考**（ADR-0004）。真正的「优化前 → 后」数字，是**你**测出来的那些。

!!! note "脚手架状态"
    完整 Capstone 说明在其依赖的 Part 落地后给出。它直接建立在 **[Part 5](../part5/index.md)** 与 **[Part 8](../part8/index.md)** 之上。
