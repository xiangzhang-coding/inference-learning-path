# NCCL 集合通信与在 vLLM 里启动 TP/PP

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` / `Llama-3.1-70B-Instruct` · 多卡 Lab 用 A100"
    已按 ADR-0004 用 Context7 对照 vLLM 0.26.0 核实：vLLM 的 GPU 集合通信走 **NCCL**（经 `vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`）、CPU 集合通信走 **GLOO**；分布式执行后端为 **`mp`**（原生多进程，单节点默认）或 **`ray`**（多节点默认运行时），经 `--distributed-executor-backend` 选择；多节点也可走 `mp`，用 `--nnodes` / `--node-rank` / `--master-addr`（worker 加 `--headless`）；GPU 通信**自检**用 `torch.distributed.init_process_group(backend="nccl")` + `all_reduce`，以 `torchrun` 启动；用 **`NCCL_DEBUG=TRACE`** 调试，用 **`NCCL_SOCKET_IFNAME`** / **`GLOO_SOCKET_IFNAME`** / **`VLLM_HOST_IP`** / **`NCCL_P2P_DISABLE`** 修 init 卡死。本节是 [为什么要并行，以及怎么并行](parallelism-strategies.md) 的上手篇——打开 TP「每层一次 all-reduce」的黑盒，并真正启动 TP/PP。所有数字均为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

[上一节](parallelism-strategies.md) 反复说 TP「每层、每个 token 一次 all-reduce」、PP「一次点对点交接」。那些是**集合通信 (collective communication)** 操作，而在 NVIDIA GPU 上跑它们的库是 **NCCL**（NVIDIA Collective Communications Library，读作「nickel」）。本节打开这个黑盒，然后真正启动多卡推理。

面试官期望你*上过手*、而非只读过的两件事：

1. **集合通信原语。** 当四张卡各自算出一个部分结果、需要在所有卡上得到其和时，那是一次 **all-reduce**——不是一个 `for` 循环的 send。NCCL 把一小撮词汇——**all-reduce**、**all-gather**、**reduce-scatter**，外加点对点 send/recv——实现成**拓扑感知、带宽最优**的算法（ring all-reduce 是最著名的那个）。知道每个原语搬什么、以及 all-reduce = reduce-scatter + all-gather，就是「TP 慢是因为……网络？」和一个真答案之间的差别。
2. **怎么真正开起来。** TP/PP 不是永远好使的魔法 flag——它们启动一个**进程组 (process group)**，每个 rank 绑一张 GPU，NCCL 去发现互联。单节点上是 `--tensor-parallel-size N` 配 `mp` 后端；跨节点是 `ray` + 每个节点匹配的 `--master-addr`，而现实中头号故障是 **init 卡死**——因为 NCCL 挑错了网络接口。能跑自检、会读 `NCCL_DEBUG=TRACE` 的工程师，才是能把多卡落地的人。

所以：先是原语（网线上搬什么），再是启动（怎么跑起来、卡死时怎么调）。→ 术语见 [术语表](../glossary.md) 的 *集合通信、TP 度*。

## 2 · 心智模型

两层：**集合通信原语**（什么数据在动）与**启动拓扑**（有哪些进程、它们怎么找到彼此）。

```text
集合通信 —— 4 张 GPU，各自起始有数据；最终什么落在哪：

ALL-REDUCE  （TP 的主力：把部分和加起来，人人得到总和）
   入:  g0=[a]  g1=[b]  g2=[c]  g3=[d]
   出: 每张 GPU = [a+b+c+d]                        ← 这就是「每层一次 all-reduce」

ALL-GATHER  （各持一个分片 → 人人得到完整拼接）
   入:  g0=[a]  g1=[b]  g2=[c]  g3=[d]
   出: 每张 GPU = [a,b,c,d]

REDUCE-SCATTER  （跨 GPU 求和，但每张只保留自己那一片和）
   入:  g0=[a0,a1,a2,a3]  g1=[b0..]  g2=[c0..]  g3=[d0..]
   出: g0=[Σ_0]  g1=[Σ_1]  g2=[Σ_2]  g3=[Σ_3]      (Σ_k = a_k+b_k+c_k+d_k)

RING ALL-REDUCE = 先 REDUCE-SCATTER 再 ALL-GATHER
   每卡代价 ≈ 2·(N-1)/N · 消息   → N 增大时趋于 ~2·消息，与 N「无关」（带宽最优）
   这就是好的 all-reduce 不会随卡数增多而 N 倍变慢的原因

启动拓扑 —— 进程、rank，以及它们如何 rendezvous（会合）：

  单节点 (backend = mp):                多节点 (ray 默认，或 mp):
    每卡 1 进程，本地 spawn                node0 (rank 0..3) ── 网络 ── node1 (rank 4..7)
    NCCL 走 NVLink/PCIe                    --master-addr 让每个节点都指向 node0（mp 路径）
    vllm serve --tensor-parallel-size 4    每节点内 TP=4 · 跨 2 节点 PP=2
                                           NCCL 节点内走 NVLink，跨节点走 IB/以太网
```

三个要记住的形状：

- **一次集合通信是一个融合操作，不是一串 send 的循环。**「每张 GPU 都要那个和」是*一次* all-reduce，由 NCCL 按真实链路调度成 ring/tree——而不是 N² 次点对点。这就是它快、以及你永远不该手写它的原因。
- **all-reduce = reduce-scatter + all-gather**，其代价约为消息的 $2\times$、*与 N 无关*。更多 GPU 不会让实现良好的 all-reduce 成比例变慢——但消息仍然每层、每个 token 都发，所以它必须走快链路。
- **启动 = 一个进程组 + 一次会合。** 每个 rank 必须对 `--master-addr` 达成一致、并找到一个能用的网络接口，否则 NCCL 会在生成第一个 token 之前就 **init 卡死**。单节点 `mp`「开箱即用」；多节点 `ray` 才是网络咬人的地方。

## 3 · 原理

### 3.1 集合通信原语

一个**集合 (collective)** 是组内每个 rank 一起参与的操作。推理里要紧的四个：

- **all-reduce** —— 把每张 GPU 都持有的张量合并（通常求和），使**所有** GPU 都以规约结果收尾。这正是 TP 的按列-再按行切分所需：每张 GPU 算了层输出的一个部分和，all-reduce 把这些部分和变成每张 GPU 都带着往下走的完整激活。*attention 后一次，FFN 后一次——每层两次。*
- **all-gather** —— 每张 GPU 持有一个**分片**；之后每张 GPU 都持有**完整拼接**。用来重建被切散在多卡上的张量（如 sequence-parallel 激活、汇集各 GPU 的输出）。
- **reduce-scatter** —— all-gather 的对偶：每张 GPU 持有完整张量，之后每张只保留**逐元素和中属于自己的那一片**。
- **点对点** send/recv —— 一张 GPU 到一张 GPU。这是 PP 的 stage 间交接。

关键恒等式：**一次 ring all-reduce 就是一次 reduce-scatter 接一次 all-gather。** 每个阶段每卡搬运消息的 $\frac{N-1}{N}$，所以每卡总流量是

$$
\text{ring all-reduce 代价} \approx 2\cdot\frac{N-1}{N}\cdot M
$$

其中 $M$ 是消息字节数。N 增大时它趋于 $2M$——**与 GPU 数无关**。这就是要点：好的 all-reduce 是*带宽最优*的，所以加卡不会让每次 all-reduce 成比例变慢。TP 的痛不在于 all-reduce 随 $N$ 扩展得差；而在于 all-reduce **每层、每个 token** 都发，所以它必须坐在**高带宽链路**上。

### 3.2 NCCL —— 跑集合通信的东西

**NCCL** 是 NVIDIA 实现这些集合通信的库，其算法（ring、tree）针对真实硬件拓扑调优——节点内 GPU 间的 NVLink、退化时的 PCIe、跨节点的 InfiniBand/以太网。vLLM 通过一层薄封装 **`PyNcclCommunicator`**（`vllm.distributed.device_communicators.pynccl`）驱动它，并用 **GLOO** 做 CPU 侧协调（会合、元数据）。你很少直接调 NCCL——但你*确实*会调试它，因为当 GPU 之间说不上话时，NCCL 就是它冒出来的地方。

### 3.3 在 vLLM 里启动 TP/PP

vLLM 选一个**分布式执行后端**：**`mp`**（原生 Python 多进程）是**单节点**默认；**多节点**默认运行时是 **`ray`**，但 vLLM 也支持多节点走 `mp`。用 `--distributed-executor-backend` 选择。

**单节点** —— 每卡一进程，NCCL 走 NVLink/PCIe：

```bash
# 单节点 4 卡，张量并行
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4
```

**多节点走 `mp`** —— 在*每个*节点上启动，`--nnodes` / `--master-addr` 保持一致；每个节点带自己的 `--node-rank`，worker 加 `--headless`：

```bash
# 2 节点 × 4 卡 = 8 卡，节点内 TP=4，跨节点 PP=2
# 头节点（node 0）：
vllm serve /models/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --nnodes 2 --node-rank 0 --master-addr $HEAD_NODE_IP
# worker 节点（node 1）：
vllm serve /models/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --nnodes 2 --node-rank 1 --master-addr $HEAD_NODE_IP --headless
```

**Ray** 那条路子先组集群（每个节点 `ray start`），再在头节点上**只启动一次** vLLM、带 `--distributed-executor-backend ray` 与相同的 TP/PP 度（没有逐节点的 `--node-rank`）。无论哪种，这都是上一节 [决策树](parallelism-strategies.md) 的落地：**TP = 节点内 GPU 数，PP = 节点数。**

### 3.4 选 TP 度

三条约束，按顺序：

1. **必须整除（KV）头数** —— TP 切注意力头，所以度数（几乎总是）是整除头数的 **2 的幂**。Qwen2.5-7B 有 28 个注意力头 / 4 个 KV 头；TP ∈ {1, 2, 4} 能让 KV 头保持整数。
2. **保持 ≤ 节点内 GPU 数** —— TP 每层的 all-reduce（§3.1）必须走 **NVLink**，所以 TP 不应跨节点边界；跨节点交给 **PP**。
3. **够用就好** —— TP 越大，每 token 的 all-reduce 流量越多、收益递减；用能让模型装下（或达到延迟目标）的最小 TP，然后用 PP 做跨节点容量、用 DP 做吞吐。

### 3.5 调试启动（真正咬人的部分）

多卡很少以崩溃告终——它**在初始化时卡死**，因为 NCCL/GLOO 建不起进程组。已核实的工具箱：

- **`NCCL_DEBUG=TRACE`** —— 打开 NCCL 详细日志，看 init 卡在哪。
- **`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`** —— 钉住**网络接口**（如 `eth0`）。尤其在 InfiniBand 集群上，torch 的组发现可能挑错接口而卡死；把初始会合强制走以太网可修复。
- **`VLLM_HOST_IP`** —— 当网络复杂/多网卡时，覆盖 vLLM 探测到的 IP。
- **`NCCL_P2P_DISABLE=1`** —— 当点对点（NVLink/PCIe P2P）有硬件/驱动故障时的*临时*绕过；真正的修复是驱动/拓扑。
- **自检脚本（§4）** —— 在怪 vLLM *之前*先跑它。如果裸的 PyTorch NCCL all-reduce 卡死或崩溃，问题是硬件/驱动/网络，不是引擎。

## 4 · 完整可跑代码 + 逐行讲解

经典的 **GPU 通信自检**——vLLM 自己的排障脚本，裁到 PyTorch-NCCL 核心。它只做一次集合通信（对全 1 向量 all-reduce）并检查算术：跨 `world_size` 张 GPU 求和后，每个元素必须等于 `world_size`。这个过了，你的 NCCL/驱动/互联就是健康的，任何多卡 vLLM 故障就是配置、不是硬件。

```python title="nccl_sanity.py"
"""最小 NCCL all-reduce 自检（vLLM 排障脚本，裁剪版）。
在 N 张 GPU 上跑：  torchrun --nproc-per-node=2 nccl_sanity.py
这是正确性检查，不是 benchmark。"""
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")            # 用 NCCL 组成进程组
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)                  # 把「本 rank」绑到自己的 GPU

data = torch.FloatTensor([1.0] * 128).to("cuda")   # 每个 rank 起始是一条全 1 向量
dist.all_reduce(data, op=dist.ReduceOp.SUM)        # 那个集合通信：跨所有 rank 求和
torch.cuda.synchronize()                           # 等 GPU 算完

value = data.mean().item()                         # 现在每个元素应等于 world_size
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"   # 正确性闸门
print(f"[rank {dist.get_rank()}] PyTorch NCCL all-reduce OK (value={value}, world={world_size})")

dist.destroy_process_group()
```

**逐行讲解：**

- `init_process_group(backend="nccl")` —— `torchrun` 启动的每个进程都加入一个**进程组**；`nccl` 选 GPU 集合通信后端。这一步就是网络接口错时会**卡死**的地方（§3.5）——会合永不完成。
- `local_rank = get_rank() % device_count()` 再 `set_device(local_rank)` —— 每个 rank **绑一张不同的 GPU**。rank 0 → GPU 0，rank 1 → GPU 1……不做这步，两个 rank 会抢一张 GPU。
- `data = [1.0]*128 .to("cuda")` —— 每个 rank 把*相同*的全 1 向量放上 GPU，于是期望和就是 `world_size`。
- `all_reduce(data, op=SUM)` —— 被测的那一个集合通信：NCCL 跨所有 rank 逐元素求和 `data`（一次 ring reduce-scatter + all-gather，§3.1），并把结果留在**每张** GPU 上。
- `assert value == world_size` —— `world_size` 个 rank 各贡献 `1.0`，每个元素必须是 `world_size`。这里值不对或卡死是硬件/驱动/网络故障，*不是* vLLM 的 bug——这就是它全部的诊断价值。
- 用 `torchrun --nproc-per-node=2 nccl_sanity.py` 启动；**不用** `torchrun` 时 `world_size==1`、all-reduce 是空操作，你什么都没测到。

2 卡上的预期输出（示例）：

```text
[rank 0] PyTorch NCCL all-reduce OK (value=2.0, world=2)
[rank 1] PyTorch NCCL all-reduce OK (value=2.0, world=2)
```

两个 rank 各贡献 `1.0`，在两张 GPU 上都求和为 `2.0`——NCCL 与互联健康。（vLLM 完整脚本继续做一个 GLOO/CPU 测试和一个 `PyNcclCommunicator` + CUDA-graph 测试——同样的思路，锻炼 vLLM 自己的 NCCL 封装与 decode 用到的 CUDA-graph replay 路径。）

## 5 · Lab —— 先跑集合通信，再在 2 卡上启动 TP

!!! gpu "GPU Lab（多卡——租、跑、关机）"
    - **最低显存 / 卡数：** 需要 **2 张 GPU**（自检在单卡上是空操作）。一个 2× A100（40/80 GB）节点能从容跑完 NCCL 测试与一次真正的 `Qwen2.5-7B-Instruct` TP=2 服务。
    - **建议 AutoDL 卡型：** **2× A100**，**「开机即关」**（ADR-0001）——这是少数真正需要多卡的专题之一。*别*让它一直开着。4090 没有 NVLink 对端，做不了本 Lab。
    - **预估耗时 / 花费：** 上手约 30–40 分钟 · 短时 2×A100 会话 **约 ¥8–20**（示例；A100 租金远高于单 4090 默认——开起来、跑完、拆掉）。
    - **平台：** NVIDIA CUDA + **NCCL**（默认）。**非 NVIDIA：** AMD ROCm 用 **RCCL**（一个 NCCL API 兼容的重实现），集合通信相同；启动 flag 一致，环境变量是 ROCm 对应项。

从网线往引擎逐层做上去：

1. **跑集合通信。** 在 2 卡节点上 `torchrun --nproc-per-node=2 nccl_sanity.py`——两个 rank 都应 `value=2.0`。这在你碰 vLLM *之前*证明 NCCL + 互联。
2. **故意让它卡死。** 用一个假接口重跑：`NCCL_SOCKET_IFNAME=doesnotexist torchrun --nproc-per-node=2 nccl_sanity.py`。它会卡在 `init_process_group`——头号多机故障的特征。杀掉它、取消该变量、确认又能过。
3. **启动真正的 TP=2。** `vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2`。在启动日志里确认它 spawn 了 **2 个 worker**（`mp` 后端）、初始化 NCCL、并报出比 TP=1 更大的 KV-cache 块池（权重现在切在 2 卡上，为 KV 腾了显存）。发一个请求，确认输出与单卡跑一致——TP 改变*算在哪*，不改变*算什么*。
4. **读流量。** 用 `NCCL_DEBUG=TRACE vllm serve … --tensor-parallel-size 2` 重启，看 NCCL 宣告它的 ring/算法与 channel——§3.1 的 all-reduce，活的。然后**关掉实例。**

## 6 · 常见坑 / 反直觉点

- **把网络的锅甩给 vLLM。** 如果启动卡在 NCCL init，先跑 §4 自检。若*那个*卡死，是硬件/驱动/网络——把 `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` 设成正确接口（如 `eth0`），或在多网卡网络里设 `VLLM_HOST_IP`。vLLM 修不了一次坏的会合。
- **不用 `torchrun` 跑自检。** 裸 `python nccl_sanity.py` 的 `world_size==1`，all-reduce 是空操作、assert 平凡通过——你什么都没验证。必须用 `torchrun --nproc-per-node=N` 启动。
- **worker 节点忘了 `--headless`（或 `--master-addr` 不一致）。** 多机服务里每个节点都必须指向*同一个*头 IP；非头节点跑 `--headless`。不一致会让整个集群 init 卡死。
- **让 TP 跨节点边界。** TP 的 all-reduce 每层、每个 token 都发，必须走 NVLink；一个跨节点的 TP 组会在更慢的节点间链路上爬。保持 **TP ≤ 节点内 GPU 数**、跨节点用 **PP**。
- **指望 TP=N 给 N 倍吞吐。** TP 主要买**容量**（装下模型）和一些**延迟**；它每 token 加 all-reduce 流量，所以吞吐是次线性的。裸吞吐来自 **DP 副本**，不是更多 TP。
- **以为 all-reduce 会随卡数 N 倍变慢。** ring all-reduce 每卡约 $2M$、*与 N 无关*（§3.1）——带宽最优。代价的驱动是**每 token、每层的频率**与**链路带宽**，不是卡数。
- **把 `NCCL_P2P_DISABLE=1` 当「修复」长期开着。** 它是会拖垮性能的诊断绕过；真正的修复是驱动/拓扑。别带着它上线。
- **让 A100 实例一直开着。** 按 ADR-0001 本 Lab 是开机即关。多卡租金烧预算飞快——集合通信与 TP 服务确认后就拆掉。

## 7 · 面试连线

- [NCCL 集合通信、ring all-reduce 与启动 TP/PP](../interview/nccl-collective-communication.md) —— 本节课为你准备的高频题：*all-reduce / all-gather / reduce-scatter 各搬什么、为何 ring all-reduce 约为消息的 2 倍且与 N 无关、TP 用哪个集合通信及多频繁、以及 vLLM 单机 vs 多机如何启动 TP/PP（mp vs ray、那些 flag、以及调 init 卡死）。*

## 8 · 小结 & 延伸阅读

**一句话：** [上一节](parallelism-strategies.md) 的「每层一次 all-reduce」是一个由 **NCCL** 跑的**集合通信**原语——词汇是 **all-reduce**（求和、人人得到——TP 的主力，每层两次）、**all-gather**（分片 → 处处得到完整副本）、**reduce-scatter**（求和、各留自己那片）与点对点（PP 的交接）；一次 **ring all-reduce = reduce-scatter + all-gather**，每卡代价约 $2\cdot\frac{N-1}{N}\cdot M \to 2M$、**与 N 无关**（带宽最优，所以痛点是每 token 的频率、不是卡数）——你在 vLLM 里用 `--tensor-parallel-size` / `--pipeline-parallel-size` 启动它，走 **`mp`** 后端（单节点，或多节点经 `--nnodes` / `--node-rank` / `--master-addr` + `--headless`）或跨节点的 **`ray`** 集群，用 `torchrun` 下的 `torch.distributed` NCCL `all_reduce` 测试验网线，并用 `NCCL_DEBUG=TRACE` + `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`VLLM_HOST_IP` 调 init 卡死。

延伸阅读：

- 百度 *Bringing HPC Techniques to Deep Learning*（2017）—— ring all-reduce，及其代价与 GPU 数无关的原因。
- NVIDIA **NCCL** 文档 —— 集合通信算法（ring/tree），以及 §3.5/§6 引用的 `NCCL_*` 环境变量。
- vLLM `docs/usage/troubleshooting.md` —— §4 那段精确的 GPU/CPU 通信自检脚本与调试环境变量。
- vLLM `docs/serving/parallelism_scaling.md` 与 `docs/serving/expert_parallel_deployment.md` —— `mp`/`ray` 后端、多机 flag，以及此处引用的 `GLOO_SOCKET_IFNAME` 网络配置。
- [上一节](parallelism-strategies.md) —— 为什么并行（这里的集合通信是 TP/PP *怎么*为它付账）。
- [容量规划](../part8/capacity-planning.md) 课（Part 8）—— TP/PP 的选择如何喂给 VRAM 与集群规模估算。

## 9 · 自测小问

??? question "all-reduce 做什么？为何一个好的（ring）all-reduce 不会随加卡而成比例变慢？"
    **all-reduce** 取每张 GPU 都持有的张量，跨所有 rank 逐元素合并（通常求和），并把**规约结果留在每张 GPU 上**——这正是 TP 把每张 GPU 的部分层输出变成完整激活所需（attention 后一次、FFN 后一次）。一次 **ring all-reduce** 被实现为一次 **reduce-scatter**（每张 GPU 收尾时有一片和）接一次 **all-gather**（广播这些片），每阶段每卡搬消息的 $\frac{N-1}{N}$，总计约 $2\cdot\frac{N-1}{N}\cdot M \to 2M$（N 增大时）——**与卡数无关**。所以它*带宽最优*：更多 GPU 不会让每次 all-reduce 成比例变慢。伤 TP 的不是 $N$；是 all-reduce **每层、每个 token** 都发，所以它必须走高带宽链路（NVLink）——因此 TP 待在节点内。

??? question "你启动一个 2 节点 × 4 卡的 vLLM 服务，它在启动时卡死、没有报错。走一遍你怎么诊断。"
    启动时静默卡死几乎总是**进程组 / NCCL 会合**失败、不是模型问题。(1) 跨节点用 `torchrun` 跑 **自检**（§4）——若裸 PyTorch NCCL `all_reduce` 卡死，是硬件/驱动/**网络**、不是 vLLM。(2) 打开 **`NCCL_DEBUG=TRACE`** 看 init 卡在哪。(3) 常见元凶是错的**网络接口**：把 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 设成真实以太网接口（如 `eth0`）——在会挑错接口的 InfiniBand 集群上尤其关键；若探测到的 IP 不对就设 `VLLM_HOST_IP`。(4) 检查每个节点是否指向**同一个 `--master-addr`**、worker 是否跑 **`--headless`**（多机后端 `ray`）。只有在集合通信过了之后，才怀疑像 TP 度不整除头数这类配置。

??? question "你的模型在单张 A100 上装得下，但延迟太高，于是你用 `--tensor-parallel-size 4` 服务、指望约 4 倍吞吐。为何这个预期错了，你会先查什么？"
    TP 主要买**容量**（装下太大的模型）和一些**延迟**下降（更多卡一起算每个 token）——它**不**给线性吞吐，因为它每层、每个 token 加一次 **all-reduce**（§3.1），纯通信开销随 TP 度增长、收益递减。裸吞吐来自 **DP 副本**、不是更多 TP。先查：确认 **TP ≤ 节点内 GPU 数**，让 all-reduce 走 **NVLink**（跨 PCIe/节点会爬）；确认度数**整除头数**（2 的幂）；若目标是 QPS 而非单请求延迟，改用**数据并行副本**。用 `NCCL_DEBUG` 与一次延迟/吞吐 sweep 去测量，而不是假设。
