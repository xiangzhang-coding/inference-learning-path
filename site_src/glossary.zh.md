# 术语表

> 本页镜像自仓库的双语术语表（`CONTEXT.md`）。**只放术语定义**，不含实现细节、决策、配置或预算。交叉引用用 →。术语一律保留英文原名（即使在中文页也是），以对齐面试与源码阅读语境。

## 推理流程 Inference Flow

- **Prefill / 预填充** — 一次性处理输入 prompt 的全部 token、算出其 KV 并产出第一个输出 token 的阶段；通常 → Compute-bound。→ Decode、→ TTFT
- **Decode / 解码** — 自回归逐个生成后续 token 的阶段，每步只算一个新 token 并追加其 KV；通常 → Memory-bound。→ Prefill、→ TPOT
- **Autoregressive / 自回归** — 每个新 token 的生成都以此前所有 token 为条件。

## 显存与缓存 Memory & Cache

- **KV cache / KV 缓存** — 缓存已生成 token 的 Key/Value 张量，避免逐步重算历史注意力；是推理显存占用与吞吐上限的核心矛盾。→ PagedAttention、→ GQA
- **HBM / SRAM** — GPU 的高带宽显存 (HBM) 与片上高速缓存/寄存器 (SRAM)；二者带宽差一个数量级，是 IO-aware 优化的前提。
- **Memory-bound / Compute-bound（带宽受限 / 算力受限）** — 瓶颈在数据搬运还是在计算。→ Roofline

## 架构 Architecture

- **MHA / MQA / GQA** — Multi-Head / Multi-Query / Grouped-Query Attention；KV 头数依次减少，直接缩小 → KV cache、抬高吞吐上限。
- **FFN / MLP** — Transformer 中承载大部分 FLOPs 与权重显存的前馈层。
- **RoPE / 旋转位置编码** — 其外推特性是 → 长上下文推理的基础。
- **MoE / 专家混合** — 每 token 只激活部分专家的稀疏结构。→ Expert parallelism

## 性能度量 Metrics

- **TTFT** — Time To First Token，首 token 延迟；主要由 → Prefill 决定。
- **TPOT / ITL** — Time Per Output Token / Inter-Token Latency，出字速度；主要由 → Decode 决定。
- **Throughput / 吞吐** — 单位时间处理的 token 数或请求数。
- **Goodput / 有效吞吐** — 满足 → SLO 约束下的有效吞吐，而非裸吞吐。

## 单卡性能 Single-GPU Performance

- **Roofline / Arithmetic Intensity（算术强度）** — 以「计算量/访存量」判断算子受算力还是受带宽限制的模型。
- **FlashAttention** — IO-aware 的注意力算法，用 tiling + online softmax 减少 HBM 读写。
- **CUDA graphs** — 把一串 kernel 启动录制成图重放，摊薄 → Decode 阶段的启动开销。
- **Kernel fusion / 算子融合** — 合并多算子为一个 kernel，减少访存与启动开销。

## GPU 编程 GPU Programming

- **SM / Warp / Occupancy** — 流多处理器 / 32 线程的调度单位 / 占用率。
- **Coalescing / Shared memory / Bank conflict** — 访存合并 / 片上共享内存 / bank 冲突。
- **Triton** — 基于 Python 的 GPU kernel 语言。

## 量化 Quantization

- **PTQ / QAT** — 训练后量化 / 量化感知训练。
- **Weight-only vs Weight+Activation** — 只量化权重 vs 权重与激活都量化。
- **Per-tensor / per-channel / per-group** — 量化粒度。
- **GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** — 主流量化方法族。
- **KV-cache quantization / KV 量化** — 对 → KV cache 本身量化以省显存。

## 服务化与吞吐 Serving & Throughput

- **Static / Dynamic / Continuous batching** — 静态 / 动态 / 连续批处理；连续批处理（Orca 风格）随到随入、随完随出，是推理吞吐的关键。
- **PagedAttention** — 把 → KV cache 按 block 管理、像虚拟内存分页一样分配，消除碎片、提升利用率。
- **Chunked prefill / 分块预填充** — 把长 prefill 切块，与 → Decode 交织调度，平衡 TTFT 与吞吐。
- **PD disaggregation / PD 分离** — 把 → Prefill 与 → Decode 拆到不同资源上分别优化。
- **Prefix caching / 前缀缓存** — 复用相同前缀的 KV，省去重复 prefill。
- **Speculative decoding / 投机解码** — 用小 draft model 猜多个 token、大模型一次校验，加速 → Decode。

## 进阶专题 Advanced Topics

- **LoRA / Multi-LoRA serving** — 低秩适配器；一份基座 + 多 adapter 动态切换的服务形态。
- **Guided / Structured decoding / 约束解码** — 用 JSON / 正则 / 语法约束输出。
- **长上下文推理 / Long-context inference** — → RoPE 外推、attention sink、KV 压缩与长序列的显存及调度问题。

## 分布式 Distributed

- **Tensor / Pipeline / Data / Expert Parallelism** — 张量 / 流水线 / 数据 / 专家并行。→ MoE
- **Collective communication / 集合通信** — all-reduce / all-gather / reduce-scatter 等原语（NCCL）。
- **TP degree / TP 度** — 张量并行切分的 GPU 数。

## 生产 Production

- **SLO** — Service Level Objective，服务的延迟/可用性目标；驱动 → Goodput 与调优。
- **Knee / 并发拐点** — 吞吐-并发曲线开始劣化的点，压测要找的关键位置。
- **KV-cache aware routing / KV 感知路由** — 依据实例上已缓存前缀路由请求，提升前缀缓存命中。
