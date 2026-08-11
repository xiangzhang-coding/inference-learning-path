# 数值格式与精度

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 0 · 基础   ·   **考察课程：** [数值格式：FP16 · BF16 · FP8 · INT8 · INT4](../part0/number-formats.md)

---

## Q：按位布局与范围-vs-精度权衡比较 FP16、BF16、FP8、INT8、INT4。为什么 BF16 在深度学习里胜过 FP16？E4M3 vs E5M2 是什么？为什么低比特格式尤其能加速 decode？

### 直接答案

每种格式都在一份比特预算里切分**范围**（浮点的指数位，或整数的共享 scale）与**精度**（尾数位，或整数宽度）：

- **FP16**（1+5+10）：精度细，但 max ≈ 65504 → 溢出成 `inf`。
- **BF16**（1+8+7）：FP32 的指数，于是 FP32 的范围（≈ $3.4\times10^{38}$），精度粗约 8 倍。
- **FP8 E4M3**（1+4+3，max ≈ 448）与 **E5M2**（1+5+2，max ≈ 57344）：需缩放因子的 8 位浮点。
- **INT8** / **INT4**：整数，按 $r \approx s(q-z)$ 重建；256 档 vs 16 档。

**BF16 胜过 FP16** 是因为模型张量跨越巨大动态范围——溢出到 `inf` 腐蚀一切，而多余的舍入噪声可以扛，所以保住 FP32 的范围（BF16）比 FP16 多出的尾数位更重要。**E4M3 vs E5M2**：E4M3 用范围换精度（权重/激活），E5M2 用精度换范围（动态范围要紧处）；二者都需 scale。**低比特加速 decode** 是因为 decode 是 [memory-bound](../part0/gpu-hardware.md)——时间由搬运字节（每个数 bits/8）决定，而非 FLOPs——所以 BF16→INT4 是每步字节的 ~4× 削减，几乎直接流入 ~4× 的 decode 吞吐。

### 深入原理

- **浮点公式。** $x = (-1)^s(1+m/2^M)\,2^{e-\text{bias}}$：指数宽度定范围（$\sim2^{2^{E-1}}$），尾数宽度定相对精度（$\approx 2^{-M}$）。FP16 与 BF16 都是 16 位，但重分配为 5/10 vs 8/7——差别全在这。
- **整数量化误差有界。** 对称 scale $s=\max|w|/(2^{b-1}-1)$ 下，误差 $\le s/2$。更*小*的 scale（更紧的范围，或更细的[粒度](../glossary.md)——per-tensor → per-channel → per-group）意味着更小误差——这就是为什么 per-group INT4 能与 per-tensor INT8 掰手腕。
- **为何 INT4 只量化权重。** 16 档适合权重（平滑、有界），却在激活上崩掉——激活的逐 token *离群值*需要 16 档在不产生巨大误差下覆盖不了的范围——于是只量化权重的 INT4 或离群值感知方法（[Part 4](../part4/index.md)）。
- **显存链条。** 字节/参数 = bits/8；$N$ 参数的权重与 [KV 缓存](../part0/kv-cache.md) 都按同样比例缩小。在带宽受限的阶段，更少字节 ≈ 成比例更多吞吐——这就是量化成为*那个* decode 杠杆的原因。

### 代码

真实值上的范围-vs-精度跷跷板（需 `numpy`）：

```python
import numpy as np
# 1/3 测精度；1e5 测范围
print(float(np.float16(1/3)), float(np.float16(1e5)))   # 0.33325...  inf   <- 精确，但溢出
# 截断式 BF16（fp32 的高 16 位）：FP32 的范围，粗尾数
u = (np.float32(1e5).view(np.uint32) >> 16) << 16
print(float(u.view(np.float32)))                        # 99840.0            <- 有范围
```

### 面试官追问

- *「模型跑 BF16 但你设了 `--kv-cache-dtype fp8`。合法吗？为何这么做？」* → 合法——计算 dtype（`--dtype`）与 KV 缓存存储 dtype（`--kv-cache-dtype`）是独立旋钮。FP8 KV 缓存把每 token KV 占用 ~减半，在同样 VRAM 里塞下更多并发序列。注意：vLLM 警告它「没有合适的缩放因子可能导致精度下降」。
- *「为什么不能所有东西都用 INT4？」* → 激活有离群值会炸穿 16 档；朴素低比特激活跌下精度悬崖。需要只量化权重的 INT4、per-group scale、或离群值处理（SmoothQuant）——[Part 4](../part4/index.md)。
- *「激活幅度大的模型选 FP16 还是 BF16？」* → BF16——FP16 会在超过 65504 时溢出。幅度大时范围安全胜过精度。
- *「减半比特总能减半延迟吗？」* → 只对 memory-bound 部分，且要减去开销（反量化、非张量算子）。decode 受益最大（带宽受限）；compute-bound 的 prefill 受益不那么直接。

### 关联知识点

- 课程：[数值格式：FP16 · BF16 · FP8 · INT8 · INT4](../part0/number-formats.md)
- 相关课程：[GPU 硬件心智模型](../part0/gpu-hardware.md)（为何更少字节 → 更多 decode 吞吐）
- 术语：[FP8 / INT8 / INT4、Per-tensor / per-channel / per-group、KV-cache quantization](../glossary.md)
