# Part 7 · 多卡与分布式

> 当一张 4090 不够用：如何把模型切到多卡，并会读、会配。

## 本 Part 覆盖

- **Tensor / Pipeline / Data / Expert parallelism** 与 **NCCL** 集合通信
- vLLM 里如何开 **TP / PP**、如何选 **TP 度**
- **压测**并找到并发**拐点（knee）**——服务真实的吞吐天花板

!!! gpu "多卡说明"
    遵循 ADR-0001，主线在单张 RTX 4090 上跑。1–2 个真正需要多卡的专题（如 TP/PP 演示）用 A100「开机即关」，其余一律单卡。

!!! note "脚手架状态"
    本 Part 课程在后续票落地。见 **[术语表](../glossary.md)**。
