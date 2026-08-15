# 采样参数：temperature、top-p / top-k 与吞吐

!!! info "基线：**vLLM 0.26.0** · flag 经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [调参旋钮：扫过吞吐/延迟曲线](../part5/tuning-knobs-sweep.md)

---

## Q：解释 temperature、top-p、top-k；什么是 greedy 解码；采样参数的选择又如何影响批处理与吞吐？

### 直接答案

采样参数重塑模型**每个 decode 步**采样的概率分布——它们作用在前向算完的 logits *之后*，改变的是*出什么字*，而不是*矩阵乘跑多快*。

| 参数 | vLLM 默认 | 作用 |
|---|---|---|
| `temperature` $T$ | `1.0` | softmax 前把 logits 缩放为 $z_i \to z_i / T$。$T<1$ 更尖锐（更确定），$T>1$ 更平（更随机），$T=0$ → **greedy**（argmax）。 |
| `top_k` | `0` | 只保留概率最高的 $k$ 个 token，重新归一化后采样。`0`（或 `-1`）**关闭**——考虑所有 token。 |
| `top_p` | `1.0` | **Nucleus（核采样）**：保留累积概率 $\ge p$ 的最小 token 集合，重新归一化后采样。`1.0` 关闭。 |

**Greedy vs 采样：** greedy = `temperature=0` → 每步取 argmax。它确定、可复现（最适合 eval / A-B）。采样（`temperature>0`，可叠加 `top_p`/`top_k`）用可复现性换多样性。

**对吞吐的影响——很小，而且不在你以为的地方。** 采样这一步是对 logits 做的 $O(\text{batch} \times \text{vocab})$ 规约，相比 $O(\text{batch} \times \text{params})$ 的带宽受限前向可忽略。所以调高 `temperature` 或开 `top_p` **并不会明显吃吞吐**——决定吞吐的是 batch 宽度与 KV 容量，不是采样器。唯一真实的耦合：vLLM 在 **batch 粒度**上施加 logits processor，只有当 batch 里*每个*请求都是 greedy 时才能跳过 argmax-invariant 的处理器——所以 greedy 与采样混在同一 batch 会放弃这点小优化。

### 深入原理

- **每步的流水线是成批的：** 前向 → logits `[batch, vocab]` → 惩罚项 + temperature → `top_k`/`top_p` 过滤 → softmax → 采样。运行中 batch 的每个请求一起流过。
- **`temperature=0` 是短路**，不是「除以零」——vLLM 直接取 argmax（无 RNG），所以在相同 batch 构成下 greedy 是确定的。
- **`top_k` 与 `top_p` 可叠加**（都能开）；各自只是在（重新归一化的）softmax 之前裁剪候选集。
- **默认值把截断器关掉。** vLLM 0.26.0 的 OpenAI 兼容回退是 `temperature=1.0`、`top_p=1.0`、`top_k=0`、`min_p=0.0`——即除非你收窄，否则从完整 softmax 采样。
- **做 sweep 时把采样钉死。** 调服务化旋钮时固定 `temperature=0` + `seed`，让**质量**这一轴确定、delta 可归因到旋钮——正是[调参旋钮那课](../part5/tuning-knobs-sweep.md)所依赖的纪律。

### 代码

对一条 logits 向量做三种变换（纯 NumPy，无 GPU）：

```python
import numpy as np

def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    if temperature == 0:                      # greedy：argmax，确定
        return int(np.argmax(logits))
    logits = logits / temperature             # temperature 缩放
    if top_k > 0:                             # 保留最高的 k 个 logit
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p < 1.0:                           # nucleus：累积概率 >= p 的最小集合
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        keep = order[:np.searchsorted(cum, top_p) + 1]
        mask = np.zeros_like(probs); mask[keep] = 1
        probs = probs * mask; probs /= probs.sum()
    rng = rng or np.random.default_rng(0)
    return int(rng.choice(len(probs), p=probs))
```

### 面试官追问

- *「`temperature=0` vs `temperature=1`？」* → `0` = greedy/argmax，确定；`1` = 从原始 softmax 采样。两者之间，越低越稳、越高越发散。
- *「调高 temperature 会伤吞吐吗？」* → 不明显。采样是 $O(\text{batch}\times\text{vocab})$ 规约，相比前向微不足道。吞吐由 batch 宽度与 KV 决定，不是采样器。
- *「vLLM 里 `top_k=0` 是保留零个 token 吗？」* → 不是——`0`（或 `-1`）**关闭** top-k，考虑所有 token。经典陷阱。
- *「怎么让 A/B eval 可复现？」* → `temperature=0`（greedy）+ 固定 `seed`，只让被测配置变化。再测（质量、吞吐、延迟）三元组。
- *「采样在批处理上有什么细节？」* → logits processor 在 batch 级运行；只有整批都是 greedy 时才能跳过 argmax-invariant 的那些，所以 greedy+采样混批会损失这点节省。

### 关联概念

- 课程：[调参旋钮：扫过吞吐/延迟曲线](../part5/tuning-knobs-sweep.md)——固定采样让 sweep 的质量轴可归因。
- 相关：[延迟 vs 吞吐度量](latency-throughput-metrics.md)、[静态 vs 连续批处理](continuous-batching.md)、[约束 / 结构化解码](structured-decoding.md)（约束的是同一批 logits）、[投机解码](speculative-decoding.md)（校验遵循目标模型的采样）
- 术语：[Sampling parameters、Decode](../glossary.md)
