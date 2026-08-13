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
- **[NCCL 集合通信与在 vLLM 里启动 TP/PP](nccl-and-launching-tp-pp.md)** —— 上手篇：**all-reduce / all-gather / reduce-scatter** 各搬什么、为何 **ring all-reduce**（= reduce-scatter + all-gather）代价约为消息的 2 倍且*与卡数无关*、vLLM 如何用 **NCCL**（`PyNcclCommunicator`）+ **GLOO** 跑它们，以及你怎么真正启动 TP/PP——单节点（`mp`）vs 多节点（`ray`，`--nnodes` / `--node-rank` / `--master-addr` / `--headless`）——配 `torchrun` 的 NCCL 自检与调那些真正咬人的 init 卡死的 `NCCL_DEBUG` / `NCCL_SOCKET_IFNAME` 工具箱。

!!! note "脚手架状态"
    两节课已落地——**[并行策略](parallelism-strategies.md)**（票 #17，*为什么与选哪种*）与 **[NCCL + 启动 TP/PP](nccl-and-launching-tp-pp.md)**（票 #18，多卡上手）——各与其面试题（[并行策略：TP/PP/DP/EP](../interview/parallelism-strategies.md)、[NCCL 集合通信与启动 TP/PP](../interview/nccl-collective-communication.md)）双向链接。压测与并发**拐点（knee）**随生产服务的票落地。所有 vLLM flag/API 均按 ADR-0004 用 Context7 核实；基线为 **vLLM 0.26.0**，一切性能数字均为**示例 / 量级参考**。见 **[术语表](../glossary.md)** 与 [面试题库](../interview/index.md)。
