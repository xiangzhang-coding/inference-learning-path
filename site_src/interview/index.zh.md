# 面试题库

> 一个持续增长的**高频**面试题库，按模块（Part 0–8）归类。每题遵循同一 schema：**直接答案 → 深入原理 → 代码（若适用）→ 面试官追问 → 关联知识点**。

每题都反链回它考察的课程，每节课的「面试连线」也链到这里——学练闭环。

## 按模块

- **Part 0 · 基础**
    - [Prefill vs decode](prefill-vs-decode.md) — 哪个阶段 compute- vs memory-bound，为什么。
    - [注意力变体：MHA/MQA/GQA](attention-variants.md) — KV 头如何决定 KV 缓存与吞吐上限。
    - [KV 缓存与吞吐上限](kv-cache.md) — 为什么瓶颈通常是 KV cache 而非算力。
    - [GPU 内存层级与 roofline](gpu-memory-hierarchy.md) — 走一遍内存层级，用 roofline 解释为何 decode 是 memory-bound。
    - [延迟与吞吐度量](latency-throughput-metrics.md) — TTFT/TPOT/ITL/throughput/goodput，如何测，以及 batch size 的权衡。
    - [数值格式与精度](number-formats.md) — FP16/BF16/FP8/INT8/INT4，范围 vs 精度，以及低比特为何加速 decode。
- **Part 2 · 单卡推理性能**
    - [GEMM 与 attention 的算术强度](arithmetic-intensity.md) — 仅凭形状推导一个算子的强度、decode attention 为何与上下文无关、以及越过拐点的 batch。
    - [显存预算与最大并发](vram-capacity-planning.md) — 走一遍完整显存预算、估最大并发；达到并发目标的旋钮。
    - [FlashAttention 与 IO-aware attention](flash-attention.md) — 同 FLOPs 为何更快、online softmax、以及它在哪帮得上/帮不上。
    - [CUDA graphs 与 kernel fusion](cuda-graphs-fusion.md) — decode 启动开销、为何伤 decode 而非 prefill、`enforce_eager` 权衡什么。
- **Part 3 · GPU 编程（Triton）**
    - [CUDA 执行模型：warp、SIMT 与 occupancy](cuda-execution-model.md) — 什么是 warp、SIMT divergence 的代价，以及为何拉满 occupancy 未必更快。
    - [Memory coalescing、shared memory 与 bank conflict](memory-coalescing.md) — 什么让一次访问 coalesced、uncoalesced 的代价，以及 shared memory 与 bank conflict 是什么。
    - [Triton 编程模型](triton-programming.md) — 一个 Triton program 映射到什么、`program_id`/offset/mask、FP32 累加，以及何时选 Triton。
    - [PagedAttention kernel 与 block table](paged-attention-kernel.md) — 为何 KV 存成块、block table 做什么、kernel 如何 gather KV，以及它为何等于稠密 attention。
- **Part 4 · 量化**
    - [量化：为何加速推理](quantization-basics.md) — 量化为何提吞吐（内存、非计算）、仿射映射，以及误差被什么界住。
    - [量化方案：粒度、对称性、PTQ/QAT](quantization-schemes.md) — per-tensor/channel/group、对称 vs 非对称、W4A16 vs W8A8，以及为何推理用 PTQ。
    - [量化方法：GPTQ/AWQ/SmoothQuant/FP8](quantization-methods.md) — 把每个方法放到轴上、它的抗 outlier 巧招，以及为瓶颈选哪个。
    - [实操量化与服务](quantization-serving.md) — 量化 → 服务 → 验证：工具、设置，以及测什么。
- **Part 5 · 服务化与吞吐（vLLM 核心）**
    - [Static vs continuous batching](continuous-batching.md) — 为何 static batching 浪费 GPU、迭代级调度是什么意思，以及到底什么限制 batch 大小。
    - [PagedAttention：block manager 与碎片](kv-cache-block-manager.md) — 为何连续 KV 会碎片、block manager 做什么、`num_gpu_blocks` 怎么定，以及分页如何变成吞吐。
    - [Chunked prefill 与 PD 分离](chunked-prefill-pd.md) — 为何长 prefill 拖停 decode、chunked prefill 换什么、`max_num_batched_tokens` 旋钮，以及何时分离。
    - [Prefix caching](prefix-caching.md) — 块哈希如何让复用安全、为何只整块缓存、何时有用、以及为何输出不变。
    - [Speculative decoding](speculative-decoding.md) — 猜测-校验、为何只因 decode memory-bound 才免费、什么设定加速、以及何时反噬。
    - [追踪一个请求穿过 vLLM 架构](vllm-architecture.md) — V1 组件（API server / engine core / worker）、端到端追踪、以及哪个优化住在哪个盒子。
    - [调参旋钮：哪个对哪个 SLO](tuning-knobs.md) — 哪个旋钮移动吞吐/延迟曲线的哪一端、它的权衡、以及要跑的 sweep。
- **Part 6 · 进阶推理专题**
    - [Multi-LoRA serving：一基座，多 adapter](multi-lora-serving.md) — 为何 LoRA adapter 很小、vLLM 如何用 grouped GEMM batch 异构 adapter、以及决定同服多少个的旋钮（`max_lora_rank`、`max_loras`、动态加载）。
    - [Guided / structured decoding](structured-decoding.md) — schema 如何变成每步 logit 掩码、为何保证是硬的而非统计的、它的代价、以及为何它修形状却从不修真值。
    - [长上下文推理：位置、sink 与 KV 墙](long-context-inference.md) — 为何模型超出训练长度就崩、RoPE 缩放（PI/NTK/YaRN）如何修、attention sink 是什么、以及为何 KV 缓存——而非算力——是长上下文的上限。
- **Part 7 · 多卡与分布式**
    - [并行策略：TP/PP/DP/EP 及各自适用场景](parallelism-strategies.md) — 并行的两个理由、TP/PP/DP/EP 各切什么与通信代价、为何 TP 待在节点内而 PP 跨节点、以及如何从模型大小和拓扑选出策略。
    - [NCCL 集合通信与启动 TP/PP](nccl-collective-communication.md) — all-reduce / all-gather / reduce-scatter 各搬什么、为何 ring all-reduce 约为消息的 2 倍且与卡数无关、TP 用哪个集合通信及多频繁、以及 vLLM 单机 vs 多机（mp vs ray）如何启动 TP/PP——含调试 init 卡死。
- **Part 8 · 生产部署与系统设计**
    - [HTTP 服务化：OpenAI 兼容 server 及其 endpoints](openai-server-deployment.md) — `vllm serve` 暴露什么、`/v1/chat/completions` vs `/v1/completions`、`/health` 保证与不保证什么、鉴权怎么工作、以及接口 vs 容量 flag。
    - [压测与并发拐点（Little 定律）](load-testing-knee.md) — knee 是什么、曲线为何折弯、开环 vs 闭环负载、Little 定律怎么解释过 knee 后的失控、以及为何报 goodput（不是裸吞吐）。
    - [路由、自动扩缩与 KV 感知路由](routing-autoscaling.md) — 前缀感知路由为何胜过 round-robin（每实例缓存）、为何按 `num_requests_waiting` 而非 GPU 利用率扩缩、以及冷启动与排空怎么塑造安全策略。
    - [可观测性与 profiling：指标、trace 与 kernel 时间线](observability-profiling.md) — 三个缩放层级（指标 → trace → profile）、你对哪些 vLLM 指标告警、prefill/decode 分叉、以及怎么捕获 torch/Nsight profile 而不淹死在数据里。
    - [SLO 驱动调优：goodput、绑定约束与闭环](slo-driven-tuning.md) — 为何对着 SLO 优化 goodput、怎么从指标读绑定约束（队列/prefill/decode/KV）、哪个旋钮缓解哪个、以及一次一个旋钮的闭环。
    - [服务生态：选 vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](framework-comparison.md) — 共享基线 vs 分歧轴、一个可辩护的默认与例外、以及靠在你自己 workload 上、SLO 下 OpenAI 兼容地压测来决定。
    - [系统设计：给推理服务定容与设计](system-design.md) — **长题**演练：框架（厘清 → 草稿纸算术 → 架构 → 瓶颈 → 取舍）与多道完整带解设计（为 X QPS、Y 延迟的 chat API、多租户 LoRA 平台、长上下文 RAG）。
- **Part 1** — 各题随对应课程在后续票落地。

!!! note "脚手架状态"
    Part 0（票 #2、#4、#5）、Part 2（票 #6、#7）、Part 3（票 #8、#9）、Part 4（票 #10、#11）、Part 5（票 #12、#13、#14）、Part 6（票 #15、#16）、Part 7（票 #17、#18）与 Part 8 全部七题（票 #19、#20、#21——最后一张是一组系统设计长题）已入库，每题与它考察的课程双向链接。完整 ~100 道题库随各 Part 落地增长。难度档 / 频率标签 / 权重暂不在范围。
