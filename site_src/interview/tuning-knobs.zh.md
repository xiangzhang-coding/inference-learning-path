# 调参旋钮：哪个对哪个 SLO

!!! info "基线：**vLLM 0.26.0** · flag 经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [调参旋钮：扫过吞吐/延迟曲线](../part5/tuning-knobs-sweep.md)

---

## Q：给定 vLLM 上的 TTFT / 吞吐 / OOM 问题，说出旋钮、它在吞吐↔延迟曲线上的方向、它的权衡——并描述你会跑的 sweep。

### 直接答案

**没有普遍意义上快的配置**——只有为某个 SLO 调出的。每个旋钮移动吞吐↔延迟曲线的一端并换一样东西：

| 旋钮 | 方向 | 换什么 |
|---|---|---|
| `gpu_memory_utilization ↑` | 吞吐（更多 KV block → 更大批） | VRAM 余量 → OOM 风险 |
| `max_num_seqs ↑` | 吞吐（更宽批） | 饱和时每请求延迟 |
| `max_num_batched_tokens ↑` | TTFT + 吞吐 | 更差 ITL（prefill 干扰） |
| `quantization` INT4/AWQ | **两端**（腾 VRAM *且*加速 decode） | 一点输出质量 |
| `kv_cache_dtype=fp8` | 吞吐（~2× KV 容量） | 一点 KV 精度 |
| `enable_prefix_caching` | **两端**（共享前缀上） | ~无（V1 默认开） |
| `enforce_eager=True` | ↓ 吞吐/延迟（无 CUDA graphs） | 省 VRAM/启动 |
| `tensor_parallel_size ↑` | **两端**（余量 + 切分计算） | 多 GPU + 通信代价 |

**诊断 → 旋钮：** TTFT 高 → `max_num_batched_tokens ↑` 或 prefix caching；启动 OOM → `gpu_memory_utilization ↓` / 量化 / `max_model_len ↓`；吞吐低 → 容量旋钮（量化、FP8 KV、`gpu_memory_utilization ↑`）；单流 decode 慢 → 保留 CUDA graphs（别 `enforce_eager`）或 speculative decoding。

**sweep：** 固定[评测集](../eval/index.md)、一次改**一个**旋钮、在几个值上、固定采样（`temperature=0`、`seed`）、测（质量、吞吐、延迟）**三元组**、只在权衡划算时保留。

### 深入原理

- **容量旋钮是总闸。** 任何装更多 KV 的——`gpu_memory_utilization`、[量化](../part4/index.md)、[FP8 KV](../part4/quantization-methods.md)——都抬高并发天花板，既提吞吐*又*削排队延迟。
- **近乎免费的旋钮**（量化、prefix caching、TP）帮*两端*——先伸手拿它们。纯权衡旋钮（`max_num_seqs`、`max_num_batched_tokens`）在后。
- **方向迁移，量级不迁移。** 旋钮往哪推是机制的属性；推多远是你模型/GPU/流量的属性——所以你测量，从不复制别人的数字。

### 代码

旋钮→方向地图（纯 Python）：

```python
KNOBS = {  # 旋钮: (吞吐, 延迟)
    "gpu_memory_utilization↑": ("↑", "≈"), "max_num_batched_tokens↑": ("↑", "↑ITL"),
    "quantization": ("↑", "↓"), "kv_cache_dtype=fp8": ("↑", "≈"),
    "enable_prefix_caching": ("↑", "↓"), "enforce_eager": ("↓", "↑"),
}
both = [k for k,(t,l) in KNOBS.items() if t=="↑" and l=="↓"]  # ['quantization','enable_prefix_caching']
```

### 面试官追问

- *「TTFT 太高——第一步？」* → 若 prompt 共享前缀，`enable_prefix_caching`（免费）。否则 `max_num_batched_tokens ↑`（换 ITL）。若是排队，加容量。
- *「启动 OOM？」* → `gpu_memory_utilization ↓`、量化权重、或 `max_model_len ↓`——都缩小溢出的 KV 池定尺。
- *「哪些旋钮帮两端？」* → 量化、prefix caching、FP8 KV、TP——它们不坐在权衡上；换的是质量/硬件。
- *「为何不直接发布最优值？」* → 它们取决于模型/GPU/流量；只有方向迁移。复制 sweep 方法，不是数字。
- *「除速度外每次 sweep 必须测什么？」* → **质量**——加速却拉垮 accuracy 的配置是回归。测三元组。

### 关联概念

- 课程：[调参旋钮：扫过吞吐/延迟曲线](../part5/tuning-knobs-sweep.md)
- 相关：[Chunked prefill 与 PD](chunked-prefill-pd.md)（`max_num_batched_tokens`）、[PagedAttention：block manager](kv-cache-block-manager.md)（`gpu_memory_utilization`/`num_gpu_blocks`）、[Prefix caching](prefix-caching.md)、[Speculative decoding](speculative-decoding.md)、[vLLM 架构](vllm-architecture.md)（哪个旋钮拧哪个盒子）、[显存预算与最大并发](vram-capacity-planning.md)
- 术语：[SLO、Goodput、TTFT、TPOT/ITL](../glossary.md)
