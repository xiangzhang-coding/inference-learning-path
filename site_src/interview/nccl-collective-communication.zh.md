# NCCL 集合通信、ring all-reduce 与启动 TP/PP

!!! info "基线：**vLLM 0.26.0** · API 已按 ADR-0004 用 Context7 核实"

**模块：** Part 7 · 多卡与分布式   ·   **对应课程：** [NCCL 集合通信与在 vLLM 里启动 TP/PP](../part7/nccl-and-launching-tp-pp.md)

---

## Q：all-reduce / all-gather / reduce-scatter 各做什么？为何 ring all-reduce 约为消息的 2 倍且与 N 无关？TP 用哪个集合通信、多频繁？vLLM 单机 vs 多机怎么启动 TP/PP（含调试卡死）？

### 直接回答

**集合通信**（一个*集合*=组内每个 rank 一起跑的一个操作，不是一串 send 的循环）：

- **all-reduce** —— 把每张 GPU 都持有的张量合并（求和）；**所有** GPU 都以结果收尾。**TP 的主力**——attention 后一次、FFN 后一次 = **每层两次、每个 token**。
- **all-gather** —— 各持一个分片 → 每张 GPU 得到完整拼接。
- **reduce-scatter** —— all-gather 的对偶：跨 GPU 求和，每张只保留和中**自己那一片**。
- **点对点** send/recv —— PP 的 stage 间激活交接。

**ring all-reduce = reduce-scatter + all-gather**，每阶段每卡搬消息的 $\frac{N-1}{N}$ → 总计约 $2\cdot\frac{N-1}{N}\cdot M \to 2M$、**与 N 无关**（带宽最优）。所以代价的驱动不是卡数——而是 TP 的 all-reduce **每层、每个 token** 都发，这就是它必须走 **NVLink**、TP 待在**节点内**的原因。

**NCCL** 在 NVIDIA GPU 上跑这些（vLLM 经 `PyNcclCommunicator`）；**GLOO** 做 CPU 侧协调。

**在 vLLM 里启动：**
- 单节点 → 后端 **`mp`**：`vllm serve M --tensor-parallel-size 4`。
- 多节点 → **`ray`**（默认）或 **`mp`**：`mp` 路径在**每个**节点启动、带 `--nnodes` / `--node-rank` / `--master-addr`（worker 加 `--headless`）；`ray` 路径先组集群（`ray start`）再在头节点**只启动一次**、带 `--distributed-executor-backend ray`。法则 **TP = 节点内 GPU 数，PP = 节点数**。

**调试卡死**（多卡以 init 卡死告终、而非崩溃）：用 `torchrun` 跑 `torch.distributed` NCCL `all_reduce` 自检；`NCCL_DEBUG=TRACE`；用 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` 钉住接口；多网卡网络用 `VLLM_HOST_IP`；`NCCL_P2P_DISABLE=1` 仅作临时绕过。

### 深入原理

- **TP 为何用 all-reduce。** Megatron 的按列-再按行切分让每张 GPU 拿到层输出的一个*部分和*；all-reduce 把部分和变成人人往下带的完整激活。每个 transformer block 两次。
- **带宽最优。** ring 把数据切成 $N$ 块流水，让每条链路都忙；每卡流量 → $2M$、与 $N$ 无关。朴素的「汇集到 rank 0 再广播」是 $O(N)$、会卡在单条链路。
- **mp vs ray。** `mp` 每张本地 GPU spawn 一个进程（单节点）。`ray` 跨节点协调进程。用 `--distributed-executor-backend` 选。
- **TP 度。** 整除（KV）头数的 2 的幂；≤ 节点内 GPU 数（NVLink）；能装下/达标的最小值——TP 越大、每 token all-reduce 越多、收益次线性。
- **卡死。** `init_process_group` 会阻塞到所有 rank 会合；错的网卡（InfiniBand 上常见）会静默卡住——所以有那些接口环境变量。

### 代码

最小 NCCL 自检——对全 1 的一次 all-reduce 必须求和为 `world_size`：

```python
import torch, torch.distributed as dist
dist.init_process_group(backend="nccl")             # 网卡错时卡在这里
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
data = torch.ones(128, device="cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)          # 跨 rank 求和；结果落在每张 GPU
assert data.mean().item() == dist.get_world_size()   # N 个 rank × 1.0 → N；否则硬件/驱动/网络故障
# 启动：torchrun --nproc-per-node=2 nccl_sanity.py   （裸 python → world_size=1 → 什么都没测）
```

### 面试官追问

- *「all-reduce 会随卡数 N 倍变慢吗？」* → 不会——ring all-reduce 每卡约 $2M$、与 N 无关。代价是每 token/每层的频率 × 链路带宽。
- *「TP 用哪个集合通信、多频繁？」* → all-reduce，每层两次（attention 后、FFN 后），每个 token 都发（含 decode）。
- *「all-reduce 由哪些更简单的操作搭成？」* → 先 reduce-scatter 再 all-gather。
- *「单机 vs 多机后端？」* → `mp`（单机）vs `ray`（多机）；`--distributed-executor-backend`。
- *「多机服务启动卡死——第一步？」* → 用 `torchrun` 跑自检；`NCCL_DEBUG=TRACE`；设 `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`；检查每个节点共享 `--master-addr`、worker 是 `--headless`。
- *「为何 TP 待在节点内？」* → 它每 token 的 all-reduce 要 NVLink 带宽；跨节点会爬——用 PP。
- *「裸 `python` 跑自检过了——能证明健康吗？」* → 不能；`world_size==1` 让 all-reduce 成空操作。必须用 `torchrun`。

### 关联概念

- 课程：[NCCL 集合通信与启动 TP/PP](../part7/nccl-and-launching-tp-pp.md)
- 相关：[并行策略：TP/PP/DP/EP 及各自适用场景](parallelism-strategies.md)（这些集合通信实现的策略）、[CUDA graphs 与 kernel fusion](cuda-graphs-fusion.md)（vLLM 的 NCCL + CUDA-graph 路径）、[显存预算与最大并发](vram-capacity-planning.md)（TP 为 KV 腾显存）
- 术语表：[集合通信、TP 度](../glossary.md)
