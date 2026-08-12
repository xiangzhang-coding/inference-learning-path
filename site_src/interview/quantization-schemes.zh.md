# 量化方案：粒度、对称性、量化什么、PTQ vs QAT

!!! info "基线：**vLLM 0.26.0** · `WxAy` 命名与 `quantization=` flag 经 Context7 核实（ADR-0004）"

**模块：** Part 4 · 量化   ·   **对应课程：** [量化的选择：粒度、对称性、量化什么，以及 PTQ vs QAT](../part4/quantization-schemes.md)

---

## Q：走一遍量化的设计选择。per-tensor vs per-channel vs per-group；对称 vs 非对称；weight-only vs weight+activation（W4A16 vs W8A8）；以及为什么推理用 PTQ。

### 直接答案

四个选择，都在缩小每个 scale 必须覆盖的范围（误差 $\le \text{scale}/2$，$\text{scale}=\text{range}/(2^b-1)$）：

- **粒度** —— per-tensor（每矩阵一个 scale）→ per-channel（每行一个）→ per-group（每 ~128 一个）。更细把 outlier 隔离到更小区域，于是干净权重得到细步长；代价是更多存的 scale（更高有效比特）。per-group 是 INT4 权重的标准。
- **对称性** —— 对称（$z=0$，无偏移、matmul 更快）适配零中心**权重**；非对称（$z\ne0$）适配偏斜**激活**（如全正 post-ReLU），那里对称浪费半个格点。
- **量化什么**（`WxAy`）—— **`W4A16`** = 4-bit 权重、FP16 激活 → weight-only，**内存/decode** 赢面（反量化→FP16 matmul；AWQ/GPTQ）。**`W8A8`** = 两者都 8-bit → INT8 tensor core，**计算**赢面（帮 prefill/大 batch）但激活 outlier 使其难（SmoothQuant）。
- **怎么** —— **PTQ** 量化已训练模型（± 校准），便宜，推理默认；**QAT** 训练时模拟量化以求最佳低比特精度，但需训练流水线。推理聚焦 PTQ。

### 深入原理

- **为何更细粒度有效。** 误差 ∝ 一个 scale 覆盖的范围。per-tensor 让一个 outlier 通道粗化整个矩阵；per-channel/group 把它限住，给干净通道 ~10–30× 更细的步长、仅 ~0.1 额外比特/权重（per-group 128）。
- **权重易、激活难。** 权重静态、近对称 → weight-only 量化易且流行。激活动态、重尾 → `W8A8` 需要 outlier 迁移（SmoothQuant）或动态 per-tensor 缩放（vLLM 的 FP8 路径每次前向缩放激活，无需校准）。
- **KV-cache 量化是独立的轴** —— 量化存的 KV 以装下更多序列（有助[显存预算](vram-capacity-planning.md)）；与权重/激活量化正交。
- **校准。** PTQ 常跑一个小校准集来选范围（有 outlier 时百分位裁剪胜过朴素 min/max）——便宜、无梯度更新。

### 代码

粒度对步长的影响（纯 Python）：

```python
def step(xs, bits):                                  # 非对称步长 = range / (2^b-1)
    return (max(xs) - min(xs)) / ((1 << bits) - 1)

W = [[0.1, -0.2, 0.15, -0.05], [-0.3, 0.25, -0.1, 0.2], [8.0, -0.1, 0.05, 0.2]]
flat = [x for row in W for x in row]
print(f"per-tensor step  {step(flat, 4):.4f}")               # 0.5533（8.0 给全体定的）
print(f"per-channel row0 {step(W[0], 4):.4f}")               # 0.0233（干净行细 ~24×）
```

per-channel 靠不与 `8.0` outlier 共享 scale，给干净行 ~24× 更细的步长。

### 面试官追问

- *「AWQ / GPTQ / SmoothQuant / FP8 各落在这些轴的哪里？」* → AWQ/GPTQ：weight-only INT4（`W4A16`）、per-group、PTQ。SmoothQuant：`W8A8`（权重+激活 INT8），把激活 outlier 迁进权重使 per-tensor 可行、PTQ。FP8：常是 `W8A8` 式的 FP8 格式；vLLM 的动态 FP8 对权重按 per-tensor 量化、对激活每次前向 per-tensor 缩放。（细节在方法课。）
- *「为什么 per-group 而非 per-element？」* → per-element 每权重存一个 scale——没有压缩。per-group（~128）以 ~0.1 额外比特/权重捕获局部 outlier；是精度/大小甜点。
- *「权重用对称还是非对称？激活呢？」* → 权重用对称（大致零中心；matmul 更简单）。偏斜激活（如 post-ReLU）用非对称，免得浪费半个格点。
- *「为什么 `W8A8` 比 `W4A16` 更难，尽管它对权重『没那么激进』？」* → 因为它量化**激活**，而激活动态且多 outlier；朴素 INT8 激活掉精度、需要 outlier 处理。`W4A16` 让激活留在 FP16、绕开那个问题。
- *「何时选 QAT 而非 PTQ？」* → 只在 PTQ 在目标比特宽度下守不住质量时（如激进的 sub-4-bit）；QAT 要花一次训练。

### 关联概念

- 课程：[量化的选择：粒度、对称性、量化什么、PTQ vs QAT](../part4/quantization-schemes.md)
- 相关：[量化：为何加速推理](quantization-basics.md)（这些调的仿射映射/误差）、[显存预算与最大并发](vram-capacity-planning.md)（KV-cache 量化省回什么）、[数值格式与精度](number-formats.md)（FP16/INT8/INT4/FP8）
- 术语表：[per-tensor/channel/group、PTQ/QAT、weight-only vs weight+activation](../glossary.md)
