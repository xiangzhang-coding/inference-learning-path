# 量化：为何加速推理与精度权衡

!!! info "基线：**vLLM 0.26.0** · 量化接口经 Context7 核实（ADR-0004）"

**模块：** Part 4 · 量化   ·   **对应课程：** [量化为何能加速推理：仿射映射与精度权衡](../part4/quantization-basics.md)

---

## Q：量化为何提升 LLM 推理吞吐？它加速计算还是内存？解释仿射映射与误差被什么界住。

### 直接答案

量化提吞吐是因为 **decode 是 memory-bound**——一步的时间由从 HBM 读字节主宰，主要是模型权重。把权重存成更少比特缩小那份流量：FP16（2 B/参数）→ INT4（0.5 B/参数）是 ~4× 更少权重字节，于是 memory-bound 的一步 decode 快到 ~4×。

**计算还是内存？** 对常见的 **weight-only** 路径（`W4A16`，AWQ/GPTQ），只是**内存**赢面：权重在片上反量化回 FP16、matmul 保持 FP16，故 FLOPs 不变。加速*计算*需要连**激活**也量化（INT8 tensor core 上的 `W8A8`）。

**仿射映射：** 以 $q_{\max}=2^b-1$ 跨范围 $[\ell,h]$，

$$\text{scale}=\frac{h-\ell}{q_{\max}},\quad z=\operatorname{round}(-\ell/\text{scale}),\quad \hat{x}=\text{scale}\cdot(q-z)$$

$z$（zero-point）是真实 $0$ 的整数（非对称）；对称去掉它。

**误差界：** 四舍五入到最近格点给出 $|x-\hat{x}|\le \text{scale}/2=(h-\ell)/(2(2^b-1))$。每多一比特减半；更宽的范围（outlier）为一切共享 scale 者抬高它。

### 深入原理

- **为何「memory-bound ⇒ 比特 ≈ 速度」。** 在 [roofline](arithmetic-intensity.md) 上，decode 远靠左（强度 ≈ 1）；时间 ∝ 搬的字节。权重字节 ∝ 比特/参数，故 decode 加速 ≈ 相对 FP16 的 `16/bits`（示例）。prefill（compute-bound）从 weight-only 量化受益较少。
- **有效比特 > 名义。** 你还每组存 `(scale, zero_point)`；每 128 一组加 ~0.125 比特/权重，所以「INT4」是 ~4.1 有效比特、文件比 `N/2` 字节略大。
- **Outlier 是敌人。** 误差 ∝ 范围。一个大权重拉伸范围、粗化该 scale 下所有值的步长——per-channel/group 粒度与 outlier-aware 方法的理由。
- **零精确量化（非对称）。** 因为 $z$ 映到真实 $0$，padding/ReLU/剪枝的零被无误差表示——非对称适配偏斜、多零数据的一个原因。

### 代码

仿射映射 + 误差界，纯 Python：

```python
def quantize_dequantize(xs, bits):
    qmax = (1 << bits) - 1
    lo, hi = min(xs), max(xs)
    scale = (hi - lo) / qmax
    zero = round(-lo / scale)
    out = [scale * (min(max(round(x / scale) + zero, 0), qmax) - zero) for x in xs]
    return out, scale

w = [-0.9, -0.2, 0.1, 0.5, 3.0]          # 3.0 这个 outlier 拉伸了范围
_, s4 = quantize_dequantize(w, 4)
print(f"INT4 step {s4:.2f}, error <= {s4/2:.2f}")   # step 0.26, error <= 0.13
```

`3.0` 给每个权重定了 0.26 的步长（0.13 误差界）——去掉它步长减半。

### 面试官追问

- *「INT4 让 GEMM 快 4× 吗？」* → 不——weight-only INT4 反量化回 FP16，故 GEMM 的 FLOPs 不变。这个 4× 是 HBM 流量，即 memory-bound decode 受限的东西。计算加速需要 INT8 激活 + tensor core。
- *「为什么量化帮 decode 多于 prefill？」* → decode 是 memory-bound（读权重+KV、计算很小）；prefill 是 compute-bound。weight-only 量化砍字节，故最帮 memory-bound 阶段。
- *「什么定量化误差？」* → `scale/2`，且 `scale = range/(2^b−1)`。所以误差由比特宽度（每比特减半）与范围（outlier 抬高）驱动。
- *「4-bit 模型正好是 FP16 的 1/4 大小吗？」* → 略大——你还存每组 scale/zero-point，故有效比特略高于 4。
- *「zero-point 有什么用？」* → 它让真实 0 映到精确整数、并适配偏斜（非对称）范围；对称量化把它设在格点中心、省掉偏移以简化 matmul。

### 关联概念

- 课程：[量化为何能加速推理](../part4/quantization-basics.md)
- 相关：[量化方案：粒度、对称性、PTQ vs QAT](quantization-schemes.md)（如何在低比特下把误差压小）、[数值格式与精度](number-formats.md)（FP16/INT8/INT4 是什么）、[GEMM 与 attention 的算术强度](arithmetic-intensity.md)（为何 decode memory-bound）
- 术语表：[Quantization、PTQ/QAT、per-tensor/channel/group](../glossary.md)
