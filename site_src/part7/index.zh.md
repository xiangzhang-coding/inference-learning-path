# Part 7 · 多卡与分布式

> 当一张 4090 不够用：如何把模型切到多卡，并会读、会配。

## 本 Part 覆盖

- **Tensor / Pipeline / Data / Expert parallelism** 与 **NCCL** 集合通信
- vLLM 里如何开 **TP / PP**、如何选 **TP 度**
- **压测**并找到并发**拐点（knee）**——服务真实的吞吐天花板

!!! gpu "多卡说明"
    遵循 ADR-0001，主线在单张 RTX 4090 上跑。1–2 个真正需要多卡的专题（如 TP/PP 演示）用 A100「开机即关」，其余一律单卡。

## 课程

- **[为什么要并行，以及怎么并行：张量 / 流水线 / 数据 / 专家并行](parallelism-strategies.md)** —— 离开单卡的两个理由（**装不下** → 切模型；**太慢** → 复制它），以及四种刀法各切什么、通信代价多大：**TP** 切每层的矩阵、每层两次 all-reduce（吃带宽 → NVLink → *节点内*），**PP** 把层切成 stage、便宜的点对点交接（→ *跨节点*，代价是流水线气泡），**DP** 复制整个模型提吞吐（要求装得下），**EP** 切 MoE 专家、用 all-to-all——外加决策树（`tensor_parallel_size` / `pipeline_parallel_size` / `--data-parallel-size` / `--enable-expert-parallel`，已对照 vLLM 0.26.0 核实）。

!!! note "脚手架状态"
    **[并行策略](parallelism-strategies.md)** 课（票 #17）已落地，与其面试题（[并行策略：TP/PP/DP/EP](../interview/parallelism-strategies.md)）双向链接。仍待后续票落地：**NCCL 集合通信 + 真正在多卡上启动 TP/PP**（本概念课的上手篇），随后是压测与并发**拐点（knee）**。所有 vLLM flag/API 均按 ADR-0004 用 Context7 核实；基线为 **vLLM 0.26.0**，一切性能数字均为**示例 / 量级参考**。见 **[术语表](../glossary.md)** 与 [面试题库](../interview/index.md)。
