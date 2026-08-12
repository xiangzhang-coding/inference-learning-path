# 量化方法：GPTQ vs AWQ vs SmoothQuant vs FP8 vs LLM.int8()

!!! info "基线：**vLLM 0.26.0** · 工具/flag 经 Context7 核实（ADR-0004）"

**模块：** Part 4 · 量化   ·   **对应课程：** [量化方法族](../part4/quantization-methods.md)

---

## Q：比较主流量化方法——GPTQ、AWQ、SmoothQuant、FP8、LLM.int8()。把每个放到设计空间轴上、说出它的巧招、并为给定瓶颈选一个。

### 直接答案

每个方法是[设计空间](../part4/quantization-schemes.md)（比特、粒度、量化什么）里的一个点外加一个抗 outlier 巧招：

| 方法 | Scheme | 巧招 | 主要赢面 |
|---|---|---|---|
| **GPTQ** | W4A16, per-group | 逐层**误差校正**（Hessian） | 内存 / **decode** |
| **AWQ** | W4A16, per-group | 缩放**显著**权重通道（激活感知） | 内存 / **decode** |
| **SmoothQuant** | W8A8 | **迁移**激活 outlier → 权重 | 计算 / **prefill+batch** |
| **FP8 (E4M3)** | W8A8 | 浮点格式 = 更大**动态范围**（常无校准） | 计算+内存（**Hopper/Ada**） |
| **LLM.int8()** | W8A8 | **outlier 维保留 FP16**、其余 INT8 | 内存（**精度安全** INT8） |

**选择：** decode-bound 服务（常见情形）→ **AWQ 或 GPTQ** INT4（weight-only，大内存/decode 赢面，容易）。compute-bound prefill / 大 batch → **SmoothQuant 或 FP8**（INT8/FP8 tensor core）。Hopper/Ada 硬件 → **FP8**（最佳现代默认）。精度攸关的 INT8 → **LLM.int8()**。正交地，**FP8 KV cache** 用于长上下文 / 更多序列。

### 深入原理

- **Weight-only（W4A16）vs weight+activation（W8A8）是主要分野。** AWQ/GPTQ 让激活留 FP16 → 内存/decode 赢面，容易（权重静态、近对称）。SmoothQuant/FP8/LLM.int8() 量化激活 → 计算赢面，但必须驯服激活 **outlier**——这正是各自贡献的巧招。
- **GPTQ vs AWQ。** 都是 INT4 weight-only、per-group、带校准的 PTQ；GPTQ 逐层校正舍入误差，AWQ 保护与大激活挂钩的权重。实践中质量相当；两者都是 INT4 标准。
- **FP8 为何容忍 outlier。** 指数让 FP8 在 8 比特下比 INT8 有大得多的动态范围，于是不用逐通道迁移就能表示 outlier——故常无校准（动态 per-tensor 缩放）。需 FP8 tensor core（Hopper/Ada）。
- **工具。** llm-compressor 是当前路径（`GPTQModifier(scheme="W4A16"/"W8A8")`、`SmoothQuantModifier`）；**AutoAWQ 已废弃**并入它。vLLM 从 config **自动检测**预量化 checkpoint 的方法。

### 代码

方法作为设计空间表（纯 Python）：

```python
METHODS = {  # name: (W-bits, A-bits, primary_win)
    "GPTQ": (4, 16, "memory / decode"), "AWQ": (4, 16, "memory / decode"),
    "SmoothQuant": (8, 8, "compute / prefill+batch"),
    "FP8 (E4M3)": (8, 8, "compute + memory (Hopper/Ada)"),
    "LLM.int8()": (8, 8, "memory (accuracy-safe INT8)"),
}
decode = [n for n, (w, a, win) in METHODS.items() if "decode" in win]
print(decode)   # ['GPTQ', 'AWQ']  <- weight-only INT4，decode 打法
```

### 面试官追问

- *「4090 上 decode-bound 服务——用哪个方法？」* → INT4 weight-only，**AWQ 或 GPTQ**（`W4A16`）：激活留 FP16、权重字节降 ~4×、decode 加速；容易且支持好。
- *「为什么不能直接朴素 INT8 激活？」* → 激活有大的逐通道 outlier，单个 INT8 scale 存不下。SmoothQuant 把它们迁进权重；LLM.int8() 把 outlier 维留 FP16；FP8 用指数取范围。这就是 `W8A8` 方法存在的全部理由。
- *「何时 FP8 而非 INT4？」* → 在 Hopper/Ada 上，当你还想要*计算*加速（prefill/大 batch）且要近乎无损、无需校准时——FP8 是 FP8 tensor core 上的 `W8A8`。INT4 是任意 Ampere+ 上可移植的内存/decode 打法。
- *「KV-cache 量化加了什么？」* → 正交内存：FP8 KV（`kv_cache_dtype="fp8"`）为更长上下文 / 更多序列释放显存；叠在 INT4 权重上。
- *「用哪个工具，AutoAWQ 还是吗？」* → llm-compressor（recipe = scheme）。AutoAWQ 已废弃并入 llm-compressor；用 recipe 或预量化 checkpoint。

### 关联概念

- 课程：[量化方法族](../part4/quantization-methods.md)
- 相关：[量化方案（轴）](quantization-schemes.md)、[量化基础（outlier 问题）](quantization-basics.md)、[实操量化与服务](quantization-serving.md)、[显存预算](vram-capacity-planning.md)（KV-cache 量化省回什么）
- 术语表：[GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()](../glossary.md)
