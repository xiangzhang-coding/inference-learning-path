# 量化为何能加速推理：仿射映射与精度权衡

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    这里引用的 vLLM 量化接口（`LLM(..., quantization="fp8")`、`vllm serve --quantization fp8`）经 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。§3–§4 的仿射量化映射是**数学，不是库调用**——可跑代码是纯 Python、离线的。内存/加速数字是**示例 / 量级参考**；量化误差界（`≤ step/2`）是模型上的精确算术。具体方法族（GPTQ/AWQ/…）与动手把 `Qwen2.5-7B` 跑成 INT4 是**下一课**的内容。

---

## 1 · 直觉 & 为什么重要

你已从 [数值格式](../part0/number-formats.md) 那课知道 FP16/INT8/INT4 *是什么*，从 [算子 roofline](../part2/roofline-analysis.md) 知道 **decode 是 memory-bound**：每步的时间由从 HBM 读字节主宰，而非做运算。把两者合起来，量化的回报立刻显现。decode 一步读的字节主要是**模型权重**——一个 7B 模型 FP16 下每步要经 HBM 流过 ~14 GB。把权重改存 INT4（0.5 字节/参数），你读 ~3.5 GB：**流量少 ~4×，于是 memory-bound 的一步 decode 快到 ~4×**（示例）。这就是全部的吞吐杠杆。

要锚定一切的一个想法：**量化拿精度换带宽。** 你用更少的比特表示每个权重，缩小 HBM 流量（在 memory-bound 负载上即速度），代价是表示得不那么精确（精度）。因为 decode 缺带宽，省下的字节几乎直接变成速度——这就是为什么低比特权重近乎免费的性能，也是为什么每个正经部署都量化。于是整个游戏是：**在精度损失开始伤害质量之前，你能用多少比特？**——这正是本 Part 要回答的。→ 术语 *Quantization、PTQ/QAT、per-tensor/channel/group* 见[术语表](../glossary.md)。

!!! note "Weight-only 量化加速的是*内存*，不是*计算*"
    一个关键细微处：流行的 INT4 weight-only 路径（AWQ/GPTQ，vLLM 的 `W4A16`）把权重存成 4 比特，但在片上**把它反量化回 FP16** 再做 matmul——运算仍是 FP16。所以收益是**更少的 HBM 字节**（这正是 memory-bound decode 所需），*而非*更少的 FLOPs。加速计算需要连*激活*也量化（INT8 tensor core 上的 `W8A8`）——[下一课](quantization-schemes.md) 里更难的那条路。

## 2 · 心智模型

量化把一段连续范围映到一个小整数格点上，再映回来：

```text
FP16 权重（连续）              INT4 格点（16 级）              反量化（回到 FP16）
  -0.9 ─────────── 3.0          0  1  2 ... 15                 x̂ = scale·(q − zero)
   实值落在一条线上       ──►   ▏──▏──▏── ... ──▏     ──►      落到最近的格点
                               └ step = scale ┘                 误差 ≤ scale/2

存储：整数 q（4 比特）  +  每组一个 (scale, zero_point)      ← 唯一保留的浮点
读回：x̂ = scale·(q − zero_point)      ← 「反量化」，在 matmul 前于片上完成
```

三个要抓住的形状：

- **一个量化值是指向等间距格点的整数索引。** `scale` 是间距（一步的真实大小）；`zero_point` 是映到真实 0 的那个整数。存这些小整数加少数几个 `(scale, zero_point)` 浮点——这就是压缩。
- **步长由范围与比特数决定：** $\text{scale} = \text{range}/(2^b-1)$。比特越多 → 格点越细 → 误差越小。范围越宽（一个大 outlier）→ 格点越粗 → 人人受累。这一条关系驱动整个 Part。
- **反量化廉价、在使用时发生。** Weight-only 量化让*存储*形态小；它在 matmul 前才重建 FP16。省的是穿过 HBM 线的东西，而那恰是 decode 的瓶颈。

## 3 · 原理与数学

### 3.1 仿射（均匀）量化映射

选 $b$ 比特的目标，给出整数级 $[0, q_{\max}]$，$q_{\max}=2^b-1$。对跨 $[\ell, h]$ 的实值，**非对称**（仿射）量化是：

$$
\text{scale} = \frac{h-\ell}{q_{\max}}, \qquad z = \operatorname{round}\!\left(\frac{-\ell}{\text{scale}}\right)
$$

$$
q = \operatorname{clamp}\!\big(\operatorname{round}(x/\text{scale}) + z,\ 0,\ q_{\max}\big), \qquad \hat{x} = \text{scale}\cdot(q - z)
$$

$z$（zero-point）是表示真实 $0$ 的整数，于是 $0$ 被精确量化——这很重要，因为零无处不在（padding、ReLU 输出、被剪枝的权重）。**对称**量化去掉偏移（$z$ 固定在格点中心 / 对有符号格点为 $0$），并设 $\text{scale}=\max|x|/q_{\max}^{\text{signed}}$：更简单、matmul 里无偏移项，代价是若数据不以 $0$ 为中心就浪费格点（[下一课](quantization-schemes.md)）。

### 3.2 精度代价：误差被半个步长界住

四舍五入到最近格点意味着每个值的重建误差至多半步：

$$
|x - \hat{x}| \le \frac{\text{scale}}{2} = \frac{h-\ell}{2\,(2^b-1)}
$$

两个你会一直用到的推论：

- **每多一比特，误差减半。** $b \to b+1$ 大致翻倍 $q_{\max}$，把 `scale` 与误差界减半。INT8 vs INT4 是 ~16× 更细的步长。
- **Outlier 是毒药。** 误差随*范围* $h-\ell$ 变化。单个大幅值权重拉伸范围，抬高*每个*共享该 scale 的值的 `scale`。这就是为什么有粒度（per-channel/group）与 outlier-aware 方法——正是下一课与 #11 方法族的全部主题。

### 3.3 为什么更少比特 → 更高吞吐（以及有效比特数）

为权重搬的内存随每权重比特数变化，而 decode 是 memory-bound，于是吞吐大致与比特宽度成反比：

$$
\text{weight bytes} = N_{\text{params}}\times \frac{\text{bits}}{8}, \qquad
\text{decode speedup} \approx \frac{\text{FP16 bytes}}{\text{quantized bytes}} = \frac{16}{\text{bits}}\ \text{（示例，memory-bound）}
$$

于是 INT4 ≈ 比 FP16 少 4× 的权重流量。一个诚实核对：存的 `(scale, zero_point)` 浮点带来开销，所以*有效*每权重比特略高于名义值——例如 INT4 每 128 个权重一个 per-group scale，增加 ~$16/128=0.125$ 比特，得 ~4.1 有效比特而非 4。很小，但这就是「4-bit」模型比 $N/2$ 字节略大的原因。

## 4 · 完整可跑代码 + 逐行讲解

仿射量化/反量化，纯 Python、离线。它展示 INT8 与 INT4 的步长与误差界、outlier 如何同时抬高两者，以及一次具体的往返。

```python title="affine_quantization.py"
"""仿射量化：拿比特换带宽，代价是有界的精度损失。
纯 Python、离线——仿射映射是通用数学，不是库调用。"""

def quantize_dequantize(xs, bits):
    """非对称仿射 quant -> dequant。返回 (重建值, scale)。"""
    qmax = (1 << bits) - 1                       # 2^b - 1 级：255（INT8）、15（INT4）
    lo, hi = min(xs), max(xs)
    scale = (hi - lo) / qmax                     # 一个格点步长的真实大小
    zero = round(-lo / scale)                    # 映到真实 0 的整数
    out = []
    for x in xs:
        q = round(x / scale) + zero              # 到最近的格点索引……
        q = min(max(q, 0), qmax)                 # ……夹到 [0, qmax]
        out.append(scale * (q - zero))           # 反量化：回到一个实值
    return out, scale

def report(label, xs):
    lo, hi = min(xs), max(xs)
    print(f"range [{lo:.2f}, {hi:.2f}] (width {hi - lo:.2f}) — {label}")
    for bits in (8, 4):
        _, scale = quantize_dequantize(xs, bits)
        print(f"  INT{bits}: scale {scale:.4f}, max error <= {scale / 2:.4f}, "
              f"{16 / bits:.1f}x smaller than FP16")

if __name__ == "__main__":
    w_outlier = [-0.9, -0.2, 0.1, 0.5, 3.0]      # 一个大权重（3.0）拉伸了范围
    w_clean   = [-0.9, -0.2, 0.1, 0.5, 0.9]      # 同上，但没有 outlier
    report("one outlier at 3.0 stretches the range", w_outlier)
    report("no outlier", w_clean)

    recon, scale = quantize_dequantize(w_outlier, bits=4)
    err = max(abs(a - b) for a, b in zip(w_outlier, recon))
    print(f"\nINT4 round-trip of {w_outlier}:")
    print(f"  reconstructed: {[round(v, 2) for v in recon]}")
    print(f"  max abs error: {err:.2f}   (<= step/2 = {scale / 2:.2f}; the outlier forced a coarse step on all)")
```

**逐行讲解：**

- `quantize_dequantize` —— §3.1 映射的字面版。`qmax` 是级数；`scale` 把范围分到各级；`zero` 是真实 $0$ 的整数。循环把每个值四舍五入到最近格点索引、夹进范围、重建 $\hat{x}=\text{scale}\cdot(q-z)$。它返回反量化后的值，好让你直接看到精度损失。
- `report` —— 按比特宽度打印步长 `scale`、精确误差界 `scale/2`、以及相对 FP16 的压缩（`16/bits`）。在有/无 outlier 的权重向量上跑。
- `__main__` —— 两个向量只在是否含 `3.0` 上不同；看它如何改变 INT4 步长。末尾的往返展示实际重建值与界内的实测最大误差。

预期输出（精确算术，不是 benchmark）：

```text
range [-0.90, 3.00] (width 3.90) — one outlier at 3.0 stretches the range
  INT8: scale 0.0153, max error <= 0.0076, 2.0x smaller than FP16
  INT4: scale 0.2600, max error <= 0.1300, 4.0x smaller than FP16
range [-0.90, 0.90] (width 1.80) — no outlier
  INT8: scale 0.0071, max error <= 0.0035, 2.0x smaller than FP16
  INT4: scale 0.1200, max error <= 0.0600, 4.0x smaller than FP16

INT4 round-trip of [-0.9, -0.2, 0.1, 0.5, 3.0]:
  reconstructed: [-0.78, -0.26, 0.0, 0.52, 3.12]
  max abs error: 0.12   (<= step/2 = 0.13; the outlier forced a coarse step on all)
```

读出来：INT4 相对 INT8 是更粗的步长（更少比特），但相对 FP16 小 4× vs 2×——带宽赢面。而单个 `3.0` 让 INT4 误差为*每个*权重多翻一倍（界 0.13 vs 0.06），因为它们都共享一个由范围决定的 scale。这就是被量化出来的 outlier 问题——也是下一课转向更细粒度的理由。

## 5 · Lab —— 误差随比特宽度，与 outlier 税

这个 Lab 不需要 GPU——就是上面的纯 Python 模型，扫过比特宽度。（GPU 出现在[下一课](quantization-schemes.md)的动手量化与 #11。）

```python title="quant_error_sweep.py"
from affine_quantization import quantize_dequantize   # 来自 §4

weights = [-0.9, -0.2, 0.1, 0.5, 0.9]                  # 一个「干净」通道
print("bits  step(scale)  error_bound  compression")
for bits in (8, 6, 4, 3, 2):
    _, scale = quantize_dequantize(weights, bits)
    print(f"  {bits}     {scale:7.4f}     {scale/2:7.4f}      {16/bits:.1f}x")
```

**要观察什么：** 比特减半大致让步长与误差界翻倍，同时压缩以 `16/bits` 上升。这个扫描把权衡变具体——INT8 在这里近乎无损，INT4 通常是 LLM 权重的甜点（大内存赢面、可容忍的误差），INT2/INT3 开始咬人。给 `weights` 追加一个 `3.0` 重跑，每一行的误差大致翻倍：又是 outlier 税。这十行就是整个量化设计空间——本 Part 其余部分讲*如何*在低比特下把误差压小（粒度、outlier 处理、更好的方法）。

## 6 · 常见坑 / 反直觉点

- **「INT4 权重让运算快 4×」。** 不——weight-only INT4 在 matmul 前反量化回 FP16，所以 FLOPs 不变。这个 4× 是**内存流量**，即加速 *memory-bound decode* 的东西。计算加速需要量化激活（INT8 tensor core），是另一条更难的路。
- **忽略 outlier。** 误差随*范围*变化；单个大幅值抬高一切共享它的 scale。对带 outlier 的权重/激活做朴素 per-tensor 量化是头号精度杀手——也是 per-channel/group 与 outlier-aware 方法存在的理由。
- **忘了 scale/zero-point 开销。** 「4-bit」不精确等于 0.5 字节/参数——每组存的 `(scale, zero_point)` 会加一点。粒度越细 ⇒ scale 越多 ⇒ 有效比特越高。在精度之前就已有粒度-vs-大小的权衡。
- **把量化与低精度训练混了。** 这是对已训练模型做*事后*压缩以供推理（PTQ），不是用低精度训练。目标不同、失败模式也不同（下一课）。
- **以为更多比特总更安全/必要。** 在 memory-bound decode 上，多余比特是纯代价（更多流量、更慢）换你未必需要的精度。工程问题是保住质量的*最少*比特，而非最多。
- **处处对称。** 对称量化在偏斜数据（如全正激活）上浪费半个格点。何时用哪个是下一课。

## 7 · 面试连线

- [量化：为何加速推理与精度权衡](../interview/quantization-basics.md) —— 这节课为你准备的高频题：*量化为何提吞吐、它加速计算还是内存、仿射映射是什么、误差被什么界住？*

## 8 · 小结 & 延伸阅读

**一句话：** 量化经仿射映射把权重映到小整数格点（$\hat{x}=\text{scale}\cdot(q-z)$，误差 $\le \text{scale}/2$），拿精度换少得多的 HBM 字节——因为 decode 是 memory-bound，这几乎直接变成吞吐（INT4 ≈ 比 FP16 少 4× 权重流量）；设计挑战是在低比特下把精度损失压到最小。

延伸阅读：

- [数值格式](../part0/number-formats.md) 课 —— FP16/INT8/INT4 是什么及其范围-vs-分辨率权衡，本 Part 的输入。
- [算子 Roofline](../part2/roofline-analysis.md) 课 —— 为什么 decode 是 memory-bound，因而更少的权重字节 ≈ 更多速度。
- 下一节：[量化的选择](quantization-schemes.md) —— 粒度、对称性、量化什么、PTQ vs QAT；然后是 #11 的方法族（GPTQ/AWQ/…）与动手把 `Qwen2.5-7B` 跑成 INT4。
- *LLM.int8()*（Dettmers 等）—— 激活里的 outlier 问题，讲得很具体。

## 9 · 自测小问

??? question "Weight-only INT4 量化给出很大的 decode 加速。运算变快了吗？解释实际加速的是什么。"
    运算**没**变快——weight-only INT4 把权重存成 4 比特，但在 matmul 前于片上反量化回 FP16，所以 FLOPs 与 FP16 相同。变快的是 **HBM 流量**：每步 decode 少读 ~4× 权重字节。因为 decode 是 *memory-bound*（时间由从 HBM 读权重与 KV 主宰，而非计算），把字节砍 ~4× 让一步快到 ~4×。要加速*计算*，你还得量化激活并用 INT8 tensor core（W8A8），那更难，因为激活有 outlier。

??? question "写出仿射反量化公式，并说明量化误差被什么界住。为什么 outlier 让量化更糟？"
    反量化是 $\hat{x} = \text{scale}\cdot(q - z)$，其中 $q$ 是存的整数，`scale` 是步长，$z$（zero-point）是映到真实 $0$ 的整数。四舍五入到最近格点把误差界在半步：$|x-\hat{x}| \le \text{scale}/2 = (h-\ell)/(2(2^b-1))$。outlier 有害是因为误差随**范围** $h-\ell$ 变化：单个大幅值拉伸范围，从而抬高*每个*共享该 scale 的值的误差——连小而规矩的值也遭殃。

??? question "你把一个 7B 模型的权重从 FP16 量化到 INT4。估算权重内存的削减，并说出一个真实大小比朴素估计略大的原因。"
    FP16 是 2 字节/参数 → 7B 权重 ~14 GB；INT4 是 0.5 字节/参数 → ~3.5 GB，即 **~4× 削减**（$16/4$）。在 memory-bound decode 上这大致对应权重流量约 4× 的削减（示例）。真实量化模型比 3.5 GB 略大，因为你还得存 `(scale, zero_point)`——每个量化组一套——所以*有效*每权重比特略高于 4（如每 128 一组的 scale 增加 ~0.125 比特/权重）。
