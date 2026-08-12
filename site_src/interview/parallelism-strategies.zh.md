# 并行策略：TP/PP/DP/EP 及各自适用场景

!!! info "基线：**vLLM 0.26.0** · API 已按 ADR-0004 用 Context7 核实"

**模块：** Part 7 · 多卡与分布式   ·   **对应课程：** [为什么要并行，以及怎么并行：张量 / 流水线 / 数据 / 专家并行](../part7/parallelism-strategies.md)

---

## Q：到底为什么要并行？TP/PP/DP/EP 各切什么、通信代价多大？为什么 TP 待在节点内而 PP 跨节点？怎么选策略？

### 直接回答

你并行只为**恰好两个理由**，而说清是哪一个很重要：

1. **容量——装不下。** 权重 + [KV 缓存](../part0/kv-cache.md) + 激活超过单卡（70B 模型 FP16 约 130 GB、INT4 约 33 GB——都超过 24 GB 卡）。你必须**切模型**。
2. **吞吐——太慢。** 模型装得下但单卡打不到 QPS/SLO。你**复制**它。

四种刀法，各自不同的轴和通信税：

- **张量并行 (TP)** —— 切每层的矩阵（Megatron 按列/按行切），**每层两次 all-reduce、每个 token**。吃带宽 → **NVLink → 单节点内**。`tensor_parallel_size`。度数是 2 的幂（切注意力头）。也压延迟。
- **流水线并行 (PP)** —— 把*层*切成 stage；每个边界只有一次**点对点激活交接** → **能跨节点**；代价是**流水线气泡** $\frac{p-1}{m+p-1}$，靠很多 microbatch 掩盖。`pipeline_parallel_size`。
- **数据并行 (DP)** —— *复制*整个模型、切请求。每卡零显存节省 → 纯**吞吐**杠杆，**要求模型装得下**。`--data-parallel-size`。
- **专家并行 (EP)** —— **仅 MoE**：把专家切到多卡、**all-to-all** token 路由。`--enable-expert-parallel`，0.26.0 中为实验特性，度数 = `tensor_parallel_size × data_parallel_size`。

**决策树：** 单卡装得下 → 单卡（+DP 提吞吐）；超单卡但装进节点 → **TP**（≤ 节点内 GPU 数）；超节点 → **节点内 TP + 跨节点 PP**（TP = 节点内 GPU 数，PP = 节点数）；MoE → 加 **EP**。

### 深入原理

- **TP 机制。** $Y=XW$ 把 $W=[W_1|\dots|W_N]$ 按列切（GPU $k$ 得 $Y_k=XW_k$）；下一个线性层按行切、吃进这些切片、产出部分和，由一次 **all-reduce** 补全——attention 后一次、FFN 后一次 = **每层 2 次**。每次 all-reduce 流量 $\approx \text{tokens}\times d_\text{hidden}\times b$，在*每个* token（含每个 decode 步）上发生 → 小、频、对延迟敏感 → 要 NVLink、限制在节点内。
- **为什么这样分拓扑。** TP 每 token 的 all-reduce 受带宽限制 → NVLink → 节点内。PP 只在 stage 边界点对点传激活 → 容忍慢/高延迟链路 → 跨节点。因此 **TP = 节点内 GPU 数，PP = 节点数**。
- **PP 气泡。** $p$ 个 stage、$m$ 个 microbatch → 空转比例 $\frac{p-1}{m+p-1}$；喂 $m \gg p$ 来摊薄。
- **DP ≠ 显存。** 每个副本都是完整副本；它永远帮不了「装不下」，只帮「太慢」。
- **EP 是自动定尺寸的。** 你设 TP 和 DP，EP 就是它俩的乘积。只对 MoE 有意义。
- **它们组合。** 真正的大规模服务 = TP（节点内）× PP（跨节点）× DP（副本）× EP（MoE），在正交的轴上。

### 代码

装得下还是要并行的决策，加上 TP 每 token 的 all-reduce 税（Qwen/Llama 形状；显存/通信模型）：

```python
GPU, KV = 24, 6; AVAIL = GPU - KV                        # 一张 24GB 卡；约 6GB 给 KV+激活（示例）
wgb  = lambda p_b, b: p_b*1e9*b/1024**3                  # 权重 GB：参数量(B) x 字节/参数
def min_tp(p_b, b):                                      # 让一个切片装得下的最小 2 的幂 TP
    need, tp = wgb(p_b, b), 1
    while need/tp > AVAIL: tp *= 2                        # 2 的幂 -> 头数整除均分
    return tp
ar_kb = lambda h, L, b=2: 2*L*h*b/1024                   # 每层 2 次 all-reduce，约 h 宽，每 token
for name, p_b, h, L in [("7B",7.6,3584,28),("70B",70,8192,80),("405B",405,16384,126)]:
    print(name, f"{wgb(p_b,2):.0f}GB", "minTP", min_tp(p_b,2), f"AR {ar_kb(h,L):.0f}KB/tok")
# 7B 14GB minTP 1 AR 392KB/tok | 70B 130GB minTP 8 AR 2560KB/tok | 405B 754GB minTP 64 AR 8064KB/tok
```

### 面试官追问

- *「装不下 vs 太慢——哪个工具族？」* → 装不下 = 切（TP/PP/EP）；太慢但装得下 = 复制（DP）。抓错族 → OOM 或白费通信。
- *「为什么不能跨节点跑 TP？」* → 它的 all-reduce 每层、每个 token 都发；在慢链路上 GPU 都在等网线。TP 待在节点内 NVLink；PP 跨节点。
- *「DP 省显存吗？」* → 不省——每副本一份完整副本。只提吞吐，且要求模型装得下。
- *「为什么 TP 度是 2 的幂？」* → 它切注意力头；度数必须整除头数。
- *「怎么在跨 2 节点的 8×A100 上服务 70B？」* → `--tensor-parallel-size 4 --pipeline-parallel-size 2`——每节点内 TP，跨两节点 PP。
- *「EP 何时相关、多大？」* → 仅 MoE；0.26.0 实验特性；度数自动 = TP×DP，不直接设。
- *「模型装得下却还用 TP>1——为什么？」* → 为压延迟（TTFT/TPOT）；更多卡一起算每个 token，代价是 all-reduce 税。

### 关联概念

- 课程：[为什么要并行，以及怎么并行：TP/PP/DP/EP](../part7/parallelism-strategies.md)
- 相关：[显存预算与最大并发](vram-capacity-planning.md)（容量墙与 KV 预算）、[注意力变体：MHA/MQA/GQA](attention-variants.md)（TP 切的头）、[KV 缓存与吞吐上限](kv-cache.md)、[量化：为何加速推理](quantization-basics.md)（切之前先缩）
- 术语表：[张量/流水线/数据/专家并行、集合通信、TP 度](../glossary.md)
