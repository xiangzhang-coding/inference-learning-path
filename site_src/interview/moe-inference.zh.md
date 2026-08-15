# MoE 推理：激活参数 vs 总参数与专家路由

!!! info "基线：**vLLM 0.26.0** · flag 经 Context7 核实（ADR-0004）"

**模块：** Part 7 · 多卡与分布式   ·   **考察课程：** [为什么要并行，以及怎么并行：张量 / 流水线 / 数据 / 专家并行](../part7/parallelism-strategies.md)

---

## Q：对一个 Mixture-of-Experts 模型，在 serving 期解释激活参数 vs 总参数、专家路由如何工作——以及这对显存、算力、多卡服务意味着什么？

### 直接答案

**MoE** 把每个 block 里的 dense FFN 换成 $N$ 个 expert FFN 加一个小 **router（门控网络）**。每个 token 由 router 打分，送到它的 **top-$k$** 个专家（如 8 选 2）；只有这些专家运行。

- **总参数（total）**——*所有*专家，全部常驻 VRAM。**显存看总参数。**
- **激活参数（active）**——某个 token 只有被路由到的专家参与计算。**算力（FLOPs）看激活参数。**

于是 MoE 用**每 token 一小部分算力换来 dense 模型级的质量**，但每个专家的**显存照付**（全部得加载）。量级示例（示例 / 量级参考）：Mixtral 8×7B ≈ **47B 总 / ~13B 激活**；DeepSeek-V3 ≈ **671B 总 / 37B 激活**；Qwen3-MoE 系列同样落在 active≪total 这条曲线上。

**serving 期：** router **逐 token、逐层**挑专家。一个 batch 里不同 token 路由到不同专家 → 计算是一次不规则、不均衡的 gather。跨卡时，**Expert Parallelism（EP）**（`--enable-expert-parallel`）切分*哪些专家在哪张卡*；一次 **all-to-all** 把每个 token 送到拥有其专家的卡、再把结果送回。EP 存在的理由正是那个装不下的东西是**专家的显存**——只靠 TP 会去切每个专家的矩阵、每层付一次 all-reduce。

### 深入原理

- **router** 是一个线性层，产出对 $N$ 个专家的门控 logits；top-$k$ softmax 给出权重，token 输出 = 其 $k$ 个专家输出的加权和。
- **decode 可能比「激活参数」暗示的更受带宽限制。** 小 batch 时，真正从 HBM *读*的权重取决于该 batch 触及多少**不同**专家——分散的 batch 可能拉进远超 $k$ 个专家的权重，所以有效带宽 ≠ 激活参数量。
- **算 VRAM 用总参数；估 decode 速度用实际读的权重。** 两栏不同——MoE 经典陷阱就是拿激活参数去估 VRAM 预算。
- **EP vs TP 切专家：** TP 切*每个专家的*矩阵乘（每层 all-reduce，NVLink 受限，节点内）；EP 切*专家归属*（all-to-all）。不同集合通信、不同扩展性。vLLM 设 **EP size = TP × DP**（见[并行那课](../part7/parallelism-strategies.md) §8）。
- **serving 期负载均衡要紧。** router 可能偏斜；热门专家成瓶颈。训练加辅助均衡损失；serving 靠均衡路由 / capacity，别让某张卡的专家拖垮 all-to-all。

### 代码

一个玩具单层 MoE 前向——router top-$k$、gather、加权和——外加激活 vs 总参数计数（纯 NumPy，无 GPU）：

```python
import numpy as np

def moe_layer(x, experts_W, router_W, k=2):
    # x: [d]; experts_W: [N, d, d]; router_W: [d, N]
    gate = x @ router_W                       # [N] 专家 logits
    topk = np.argsort(gate)[-k:]              # 路由到 top-k 专家
    w = np.exp(gate[topk] - gate[topk].max()); w /= w.sum()   # softmax 权重
    y = sum(wi * (x @ experts_W[e]) for wi, e in zip(w, topk))
    return y, topk

N, d, k = 8, 16, 2
experts_W = np.random.randn(N, d, d) / d
total  = experts_W.size + d * N               # 所有专家常驻（显存）
active = k * d * d + d * N                     # 只有 k 个专家算（FLOPs）
print(f"total={total}  active={active}  ratio={active/total:.2f}")  # 示例
```

### 面试官追问

- *「一个 57B MoE 要多少 VRAM？」* → 看**总参数**（所有专家常驻）：~57B × 每参数字节 + KV/开销。绝不按激活数估。
- *「MoE 的 decode 为什么有时没它激活规模那么快？」* → 分散的 batch 会读很多不同专家的权重；有效 HBM 流量超过激活参数值。它仍是带宽受限。
- *「vLLM 怎么跨卡服务 MoE？」* → **EP**（`--enable-expert-parallel`）切专家；all-to-all 把 token 路由到归属卡。可与 TP/DP 组合（EP size = TP × DP）。
- *「router 怎么选专家？」* → 门控 logits → top-$k$ → 归一化权重 → 这些专家输出的加权和，逐 token 逐层。
- *「切专家用 TP 还是 EP，区别是什么？」* → TP 切每个专家的矩阵乘（all-reduce，节点内）；EP 切专家归属（all-to-all）。EP 缓解的是 TP 反而要硬扛的专家显存占用。

### 关联概念

- 课程：[为什么要并行，以及怎么并行：TP / PP / DP / EP](../part7/parallelism-strategies.md)——EP 与「先装下」的决策树在此。
- 相关：[并行：TP/PP/DP/EP 及各自何时用](parallelism-strategies.md)、[量化与上线实操](quantization-serving.md)（所有专家常驻 → 量化以装下）、[VRAM 预算与最大并发](vram-capacity-planning.md)
- 术语：[MoE、Expert parallelism](../glossary.md)
