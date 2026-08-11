# 数值格式：FP16 · BF16 · FP8 · INT8 · INT4

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    本页的 `--dtype` 与 `--kv-cache-dtype` flag 均针对 vLLM 0.26.0 经 Context7 核实（ADR-0004）。位布局事实与范围/精度算术是*精确*的；任何加速数字都是**示例 / 量级参考**。本课讲*格式*；量化*方法*（GPTQ/AWQ/SmoothQuant）是 [Part 4](../part4/index.md)。

---

## 1 · 直觉 & 为什么重要

LLM 里的每个权重、激活、KV 缓存项，归根结底都是一小撮**比特**。用*多少*比特、以及如何*切分*，一次决定三件事：模型吃多少 VRAM、每步有多少字节跨过 HBM 那条线、以及你注入多少数值误差。因为推理是 [memory-bound（带宽受限）](gpu-hardware.md)，中间那件事是决定性的——**把每个数的比特数减半，大致就把搬运的字节减半，从而大致让 decode 吞吐翻倍。** 数值格式*就是*那个吞吐杠杆；量化只是在不弄坏模型的前提下把它拉下来的艺术。

所以在碰任何一种量化方法之前，你需要对格式本身有清晰的图景：老牌的 **FP32/FP16/BF16**、激进的 **FP8**（两种口味）、以及权重量化所栖身的整数 **INT8/INT4**。整场游戏是**范围**（能表示多大/多小的数）与**精度**（能多细地区分相邻的数）之间的权衡——而浮点与整数以根本不同的方式做这个权衡。把本课学对，[Part 4](../part4/index.md) 就变成「哪种方法保住精度」，而非「等等，e4m3 是啥？」。→ 见[术语表](../glossary.md)的 *FP8 / INT8 / INT4*、*Per-tensor / per-channel / per-group*、*KV-cache quantization*。

## 2 · 心智模型

一个浮点数是 **符号 · 尾数 · 2^指数**——指数买*范围*，尾数买*精度*。一个整数格式是 **一个共享 scale · 一个小整数**——所有范围都住在 scale 里，所有值共享它。

```text
浮点（每个值自带指数：范围与精度都烘进每个数里）
  位布局                S = 符号,  E = 指数（范围）,  M = 尾数（精度）
  FP32   S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM   1+8+23   范围 ~1e38   参考基准
  FP16   S EEEEE MMMMMMMMMM                   1+5+10   范围 ~6e4    精确，范围小 -> 溢出风险
  BF16   S EEEEEEEE MMMMMMM                   1+8+7    范围 ~1e38   FP32 的范围，精度粗
  FP8e4m3  S EEEE MMM                         1+4+3    范围 ~448    FP8 用于权重/激活
  FP8e5m2  S EEEEE MM                         1+5+2    范围 ~6e4    FP8 范围更大，精度更低

整数（一个共享 scale s 覆盖整个 tensor/channel/group）
  INT8    [ -128 .. 127 ]   真实值  r ≈ s·(q - z)      8 比特/数
  INT4    [   -8 ..   7 ]   真实值  r ≈ s·(q - z)      4 比特/数  <- 权重量化
                             q = 存储的整数, s = scale, z = zero-point
```

要握住两个形状：

- **固定位宽下，范围与精度是一架跷跷板。** FP16 与 BF16 *都*是 16 位，但 FP16 把 5 位给指数 / 10 位给尾数（精确，但超过 ~65504 就溢出），而 BF16 给 8/7——*和 FP32 一样的指数*，于是它在 FP32 不溢出处永不溢出，代价是更粗的尾数。正是这一次重分配让 BF16 在深度学习里胜出：模型张量跨越巨大的动态范围，溢出到 `inf` 是致命的，而一点舍入噪声是能扛的。
- **浮点每个值自带范围；整数共享一个 scale。** 一个 INT8 张量是 `−s·128` 到 `s·127` 之间 256 个等距档位。这极其高效（8 位、无指数）——*前提是*值被良好缩放、且不被少数离群值主宰——而这恰是 [Part 4](../part4/index.md) 量化方法要管理的张力（per-channel/per-group scale、离群值处理）。

## 3 · 原理与数学

一个浮点值，符号 $s\in\{0,1\}$、$E$ 个指数位（bias $2^{E-1}-1$）、$M$ 个尾数位，对规格化数而言：

$$
x = (-1)^{s}\,\Bigl(1 + \tfrac{m}{2^{M}}\Bigr)\, 2^{\,e - \text{bias}}, \qquad 0 \le m < 2^{M}
$$

**指数宽度定范围**——最大幅度大约 $\sim 2^{2^{E-1}}$——而**尾数宽度定精度**——相邻可表示数之间的相对间隙 $\approx 2^{-M}$。比较 16 位格式：

- **FP16**（$E=5, M=10$）：max ≈ $65504$，相对步长 $\approx 2^{-10}\approx 0.001$。精确，但超过 65504 就成 `inf`。
- **BF16**（$E=8, M=7$）：max ≈ $3.39\times10^{38}$（FP32 的范围），相对步长 $\approx 2^{-7}\approx 0.008$。比 FP16 粗八倍，但基本不可溢出——当张量跨越很多数量级时的正确权衡。

**FP8** 把这推到 8 位，用两种标准化切分：**E4M3**（max ≈ $448$，精度更高）用于权重/激活，**E5M2**（max ≈ $57344$，范围更大）用于梯度/需要范围之处。二者都需要一个**缩放因子**把真实张量映进它们微小的范围——这正是为什么 vLLM 的 FP8 KV 缓存警告它「没有合适的缩放因子可能导致精度下降」。

**整数**格式彻底丢掉逐值指数。真实值 $r$ 由存储整数 $q$、共享 **scale** $s$、**zero-point** $z$ 重建：

$$
r \approx s\,(q - z), \qquad s = \frac{\max|w|}{2^{\,b-1}-1}\ \text{（对称，} b\text{ 位）}
$$

量化误差被半个步长界住，$|r - \hat r| \le s/2$，所以更*小*的 scale（更紧的值域，或更细的粒度——[per-tensor vs per-channel vs per-group](../glossary.md)）意味着更小的误差。INT8 给 $2^8=256$ 档；INT4 只 $2^4=16$ 档，这就是为什么 4-bit 几乎总是**只量化权重**（权重能扛；带离群值的激活通常不能）。

最后是显存的故事，也正是全部要点：**每个数的字节数 $= \text{bits}/8$**。$N$ 参数的权重耗 $N\cdot\text{bits}/8$ 字节；[KV 缓存](kv-cache.md) 在 FP8 存储时按同样比例缩小。BF16 → INT4 是 VRAM 占用与每步搬运字节的 **4× 削减**——而既然 decode 是带宽受限的，这个比例几乎直接流入吞吐。

## 4 · 完整可跑代码 + 逐行讲解

这段**可离线运行**——纯 CPU，只需 `numpy`。它打印格式参考表、在真实值上演示 BF16-vs-FP16 的范围/精度跷跷板、再做一次 INT8 量化往返，让你*看见*误差界 $s/2$。

```python title="number_formats.py"
"""数值格式探索器：布局表、浮点范围/精度、INT8 往返（CPU）。"""
import numpy as np

# --- Part 1：参考表（位宽 + 标准最大值都是精确事实）---
FORMATS = [
    # name,      bits, sign, exp, mant, ~max            note
    ("FP32",     32,   1,    8,   23,   "3.40e+38",     "reference"),
    ("FP16",     16,   1,    5,   10,   "6.55e+04",     "precise, small range"),
    ("BF16",     16,   1,    8,    7,   "3.39e+38",     "FP32 range, coarse"),
    ("FP8 E4M3",  8,   1,    4,    3,   "4.48e+02",     "FP8 weights/act"),
    ("FP8 E5M2",  8,   1,    5,    2,   "5.73e+04",     "FP8 more range"),
    ("INT8",      8,   0,    0,    0,   "s * 127",      "integer + scale"),
    ("INT4",      4,   0,    0,    0,   "s * 7",        "integer + scale"),
]
print(f"{'format':9} {'bits':>4} {'S':>2} {'E':>2} {'M':>3} {'~max':>10}   note")
for name, bits, s, e, m, mx, note in FORMATS:
    e_s = str(e) if e else "-"
    m_s = str(m) if m else "-"
    print(f"{name:9} {bits:>4} {s:>2} {e_s:>2} {m_s:>3} {mx:>10}   {note}")

# --- Part 2：BF16 vs FP16 —— 范围与精度的跷跷板 ---
def to_bf16(x):
    """截断式 BF16（向零舍入）。真实硬件是就近舍入；
    这里的要点是 *字段宽度*，不是最后一位。"""
    u = np.float32(x).view(np.uint32)
    u = (u >> 16) << 16                       # 丢掉低 16 个尾数位
    return u.view(np.float32)

print("\nvalue 1/3 (precision test):")
print(f"  fp32={float(np.float32(1/3)):.6g}  "
      f"fp16={float(np.float16(1/3)):.6g}  bf16={float(to_bf16(1/3)):.6g}")
print("value 1e5 (range test):")
print(f"  fp32={float(np.float32(1e5)):.6g}  "
      f"fp16={float(np.float16(1e5)):.6g}  bf16={float(to_bf16(1e5)):.6g}")

# --- Part 3：INT8 对称量化往返 ---
w = np.array([-2.5, -0.3, 0.0, 0.8, 3.1, 12.0], dtype=np.float32)
scale = np.abs(w).max() / 127                 # per-tensor 对称 scale s
q = np.round(w / scale).astype(np.int8)       # 存储整数，落在 [-128, 127]
deq = q.astype(np.float32) * scale            # 重建 r ≈ s*q（z = 0）
print(f"\nINT8 scale s = {scale:.5f}   (error bound s/2 = {scale/2:.5f})")
print(f"  q   = {q.tolist()}")
print(f"  max abs error = {np.abs(w - deq).max():.5f}")
```

**逐行讲解：**

- **Part 1** — 参考表。位宽与标准最大值（FP16 65504、FP8-E4M3 448、FP8-E5M2 57344）是固定事实，作为数据打印，没有需要算错的算术。把 `S/E/M` 列读成「这 16（或 8）位去了哪」。
- `to_bf16` — 靠*截断* FP32 的低 16 尾数位来模拟 BF16（BF16 字面上就是「少 16 个尾数位的 FP32」）。真实硬件就近舍入；截断已足够展示宽度效应。
- **Part 2** — 两个探针。`1/3` 测**精度**：FP16 的 10 位尾数落得比 BF16 的 7 位更近。`1e5` 测**范围**：它在 FP32 与 BF16 里没问题，却**在 FP16 里溢出成 `inf`**（超过 65504）。每个测试各有一个赢家——这就是跷跷板。
- **Part 3** — 对称 INT8：`scale = max|w| / 127` 把最大幅度映到 ±127；`round(w/scale)` 存整数；`q*scale` 重建。报告的最大误差保持在 `s/2` 之下，正是 §3 的界。

预期输出（精确算术，非跑分）：

```text
format    bits  S  E   M       ~max   note
FP32        32  1  8  23   3.40e+38   reference
FP16        16  1  5  10   6.55e+04   precise, small range
BF16        16  1  8   7   3.39e+38   FP32 range, coarse
FP8 E4M3     8  1  4   3   4.48e+02   FP8 weights/act
FP8 E5M2     8  1  5   2   5.73e+04   FP8 more range
INT8         8  0  -   -     s * 127   integer + scale
INT4         4  0  -   -       s * 7   integer + scale

value 1/3 (precision test):
  fp32=0.333333  fp16=0.333252  bf16=0.332031
value 1e5 (range test):
  fp32=100000  fp16=inf  bf16=99840

INT8 scale s = 0.09449   (error bound s/2 = 0.04724)
  q   = [-26, -3, 0, 8, 33, 127]
  max abs error = 0.04409
```

范围测试里的 `fp16=inf` 正是训练与推理默认选 BF16 的全部理由，而 `max abs error = 0.04409 < s/2` 把量化误差界具体化了。

## 5 · Lab —— 在 vLLM 里翻转格式、看 VRAM 挪动

!!! gpu "GPU Lab"
    - **最低显存：** 24 GB（加载 `Qwen2.5-7B-Instruct`）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）
    - **预估耗时 / 花费：** ~15 分钟 · ~¥1 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** FP8 KV 缓存支持受硬件门控——据 vLLM，CUDA 11.8+ 支持 `fp8`/`fp8_e5m2`，ROCm 支持 `fp8`（= `fp8_e4m3`）；其他后端不同。请查你平台的 vLLM 构建。

两个已核实的 `vllm serve` flag 让你在真实权重与真实 KV 缓存上感受范围/精度权衡。用 `--dtype` 设**计算 dtype**（权重/激活）：

```bash
# BF16（本模型默认）：FP32 的范围，FP32 一半的字节
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 8192
```

独立地，用 `--kv-cache-dtype` 把 **KV 缓存**存成 FP8（已核实字面量：`auto`、`fp8` = `fp8_e4m3`、`fp8_e5m2`）：

```bash
# KV 缓存用 FP8 e4m3：KV 缓存 ~2x 更小 -> 24 GB 里能塞更多并发序列
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8_e4m3 --max-model-len 8192
```

**观察什么：** 用 FP8 KV 缓存，每 token 的 KV 占用（[KV 缓存](kv-cache.md) 的 $\kappa$）大致减半，于是 vLLM 能在同样 24 GB 里放下更多序列的 KV——在 Part 0A 指出的那个瓶颈上直接赢下吞吐。看启动日志报告的 KV 缓存 block 数增长，并留意 vLLM 自己的警告：FP8 KV 缓存「没有合适的缩放因子可能导致精度下降」。把**权重**量化到 INT4/INT8（AWQ/GPTQ）是一种*方法*，不只是翻个 dtype——那是 [Part 4](../part4/index.md)；这里你已感受了 dtype 杠杆，下一步学会安全地拉动它。

## 6 · 常见坑 / 反直觉点

- **「比特越少总是越快、且基本免费。」** 更快，常常是；免费，不是。低于 ~8 位、尤其对*激活*（有离群值），朴素量化会跌下精度悬崖——这正是 [Part 4](../part4/index.md) 方法（per-group scale、SmoothQuant、离群值处理）存在的原因。
- **把 BF16 与 FP16 混为一谈。** 同样大小，相反权衡：BF16 = FP32 的范围、粗精度；FP16 = 细精度、小范围（超过 65504 溢出）。深度学习选 BF16 是因为溢出致命而舍入噪声不致命。
- **无 scale 的 FP8 是一把走火枪。** E4M3 顶到 ~448；喂未缩放的张量，一切饱和。FP8 永远带一个缩放因子上路——就是 vLLM 警告的那个「合适的缩放因子」。
- **INT4 用于激活。** INT4 只量化*权重*常见且能用；INT4 *激活*几乎从不幸存，因为离群值把 16 档预算炸穿。搞清你在量化哪个张量。
- **`--dtype`（计算）≠ `--kv-cache-dtype`（存储）。** 它们是分开的旋钮：你可以跑 BF16 权重配 FP8 KV 缓存。`--dtype auto` 只是选模型声明的 dtype。
- **「位宽决定精度。」** 粒度同样要紧：per-group INT4 在某些张量上能胜过 per-tensor INT8，因为更紧的 scale 意味着更小的误差界 $s/2$。

## 7 · 面试连线

- [数值格式与精度](../interview/number-formats.md) —— 本课为你准备的高频题：*按位布局与范围-vs-精度权衡比较 FP16/BF16/FP8/INT8/INT4，说为什么 BF16 在 DL 里胜过 FP16，并解释为什么低比特格式能加速 memory-bound 的 decode。*

## 8 · 小结 & 延伸阅读

**一句话：** 一个数值格式在固定比特预算下，用**范围**（指数位，或 scale）换**精度**（尾数位，或整数宽度）——BF16 为了安全保住 FP32 的范围，FP8/INT8/INT4 缩小字节来买 memory-bound decode 的吞吐，而量化（[Part 4](../part4/index.md)）随后努力保住精度。

延伸阅读：

- Kalamkar 等 —— *A Study of BFLOAT16 for Deep Learning Training* —— 8 位指数为何胜出。
- OCP / NVIDIA-Arm-Intel —— *FP8 Formats for Deep Learning* —— E4M3 / E5M2 规格及其最大值。
- vLLM 文档 —— *Engine arguments*（`--dtype`）与 KV 缓存 dtype 配置，基线 v0.26.0。
- [GPU 硬件](gpu-hardware.md) 那节课 —— 为什么更少的字节/数几乎直接映射到 decode 吞吐。

## 9 · 自测小问

??? question "FP16 与 BF16 都是 16 位。它们之间到底权衡了什么，深度学习为何偏爱 BF16？"
    它们重分配同样的 16 位：FP16 = 5 指数 / 10 尾数（精度细，但 max ≈ 65504 → 之外溢出成 `inf`）；BF16 = 8 指数 / 7 尾数（和 FP32 一样的*范围*，≈ $3.4\times10^{38}$，但精度粗约 8 倍）。DL 偏爱 BF16 是因为模型张量跨越巨大动态范围——溢出到 `inf` 会腐蚀整个计算，而多一点舍入噪声可以容忍。这里范围安全胜过精度。

??? question "为什么量化到更低比特格式尤其能加速 LLM *decode*，BF16 → INT4 大约快多少？"
    Decode 是 [memory-bound](gpu-hardware.md)：每步时间由跨 HBM 搬运的字节（权重 + KV 缓存）决定，而非 FLOPs。字节/数 $= \text{bits}/8$，所以 BF16（16 位）→ INT4（4 位）是每步搬运字节的 **4× 削减**，对带宽受限的工作负载而言，这几乎直接流入 ~4× 的 decode 吞吐（示例；真实增益因开销与反量化而更低）。FLOP 数几乎无关紧要，因为 GPU 本来就在等内存空转。

??? question "E4M3 与 E5M2 有何区别，为什么 INT4 通常只用于权重、而非激活？"
    E4M3（4 指数 / 3 尾数，max ≈ 448）精度更高；E5M2（5 指数 / 2 尾数，max ≈ 57344）范围更大——动态范围要紧时选 E5M2，精度要紧时选 E4M3，二者都需缩放因子。INT4 只给 16 档，所以对**权重**（平滑、有界的分布）没问题，但通常在**激活**上崩掉——激活的逐 token *离群值*需要 16 档在不产生巨大误差下覆盖不了的范围——由只量化权重的 INT4 或 [Part 4](../part4/index.md) 的离群值感知方法处理。
