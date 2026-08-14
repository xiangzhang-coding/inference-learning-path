# 量化的选择：粒度、对称性、量化什么，以及 PTQ vs QAT

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    这里用到的 vLLM 量化命名——`WxAy` 方案（如 `W4A16`、`W8A8`）、`LLM(..., quantization="fp8")` / `vllm serve --quantization fp8`、以及动态 FP8 对 Linear 权重按 per-tensor 量化、对激活按 per-tensor 动态缩放——经 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。§4 的粒度/对称性演示是**纯 Python 数学**、离线；误差界（`≤ step/2`）是精确的。任何精度/大小数字均为**示例 / 量级参考**。具体方法族（GPTQ/AWQ/SmoothQuant/…）与动手把 `Qwen2.5-7B` 跑成 INT4 是 **#11**。

---

## 1 · 直觉 & 为什么重要

[上一课](quantization-basics.md) 给了你唯一的映射——$\hat{x}=\text{scale}\cdot(q-z)$，误差 $\le \text{scale}/2$——与唯一的敌人：宽范围（outlier）抬高一切共享 scale 者的步长。量化里所有真实的东西，都是那套让步长在低比特下保持小的*工程选择*。共四个，面试官会期望你把任何方法（AWQ、GPTQ、FP8、SmoothQuant）放到它们上：

1. **粒度** —— 多少个值共享一个 `(scale, zero_point)`？per-tensor（全体一个）、per-channel（每行一个）、或 per-group（每 ~128 一个）。
2. **对称性** —— 对称（无 zero-point）还是非对称（有一个）？
3. **量化什么** —— 只权重，还是权重*与*激活？（vLLM 写作 `W4A16` vs `W8A8`。）
4. **怎么得到** —— PTQ（量化已训练模型）还是 QAT（训练时模拟量化）？

每个都是同一权衡上的滑块——**精度 ↔ 大小/速度 ↔ 复杂度**。它们存在的理由就是 outlier 问题：更细的粒度与对的对称性缩小每个 scale 必须覆盖的有效范围，于是你能降到 INT4 而不毁质量。→ 术语 *per-tensor/channel/group、PTQ/QAT、weight-only vs weight+activation* 见[术语表](../glossary.md)。

## 2 · 心智模型

四个选择，以及常见方法落在哪（一段语言中立的结构化对比，故用 ASCII，遵循 ADR-0005）：

```text
粒度  (粗 ─────────────────► 细；  更细 = 更小的 range/scale，更多 scale 存储)
   per-tensor            per-channel（每行）             per-group（如每 128）
   1 scale / 矩阵        1 scale / 输出通道             1 scale / 128 权重
   outlier 毁全体        隔离 outlier 通道              隔离 outlier 区域   ← INT4 甜点

对称性       对称 (z = 0，无偏移，matmul 快)   |  非对称 (z ≠ 0，适配偏斜数据)
             适合零中心的权重                  |  适合 post-ReLU / 偏斜激活

量化什么     W4A16 / W8A16 (weight-only)   |   W8A8 (weight + activation)
             反量化→FP16 matmul；内存赢面    |   INT8 tensor core；计算赢面，激活 outlier 难

怎么         PTQ (训练后，± 校准数据)          |   QAT (训练时模拟量化)
             便宜、不重训 —— 推理默认          |   精度最好、昂贵 —— 训练侧
```

放大看**粒度**——最高杠杆的那个选择——作为一个权重矩阵的空间切分（越细 = 每个 scale 覆盖的范围越小；图内标签保持英文）：

```text
  per-tensor              per-channel (per row)         per-group (block of ~128)
  ┌───────────┐           ┌───────────┐                 ┌─────┬─────┐
  │ s s s s s │           │ a a a a a │  scale a         │ p p │ q q │  scales p,q on row 0
  │ s s s s s │           │ b b b b b │  scale b         │ r r │ t t │  scales r,t on row 1
  │ s s s s s │           │ c c c c c │  scale c         │ u u │ v v │  scales u,v on row 2
  └───────────┘           └───────────┘                 └─────┴─────┘
  1 scale / matrix        1 scale / output row           1 scale / block   ← INT4 sweet spot
  outlier ruins all       isolates outlier rows          isolates outlier regions
```

两个要抓住的形状：

- **更细粒度用一点存储换精度。** 把一个 scale 拆成许多，缩小每个必须覆盖的范围，于是通道内或组内的 outlier 不再粗化整个张量。代价是更多存的 scale（更高有效比特）与略复杂的 kernel——真实但通常便宜的权衡。
- **「量化什么」决定你赢内存还是赢计算。** Weight-only（`W4A16`）砍 HBM 流量 → memory-bound *decode* 更快，且因为权重静态又规矩而容易。Weight+activation（`W8A8`）经 INT8 tensor core 也砍*计算*，但激活是动态且多 outlier 的，需要额外技巧（#11 的方法族）。

## 3 · 原理与四个选择

### 3.1 粒度：per-tensor → per-channel → per-group

回忆误差 $\le (h-\ell)/(2(2^b-1))$：它由*每个 scale 覆盖的范围*驱动。让一个 scale 跨整个权重矩阵（**per-tensor**）意味着单个 outlier 通道给所有通道定了粗步长。**per-channel**（每输出行一个 scale）把那个 outlier 隔离到它自己的行，于是规矩的通道得到细步长。**per-group**（每 ~128 个连续权重一个 scale）更细——INT4 LLM 权重的标准。代价是存储：per-group $g=128$ 增加 ~$16/128\approx0.125$ 有效比特/权重——为它买到的精度而言很便宜。§4 展示一个干净通道在 per-channel 下比 per-tensor 得到 ~24× 更细的步长。

### 3.2 对称 vs 非对称

**对称**把 $z$ 固定在格点中心（无 zero-point 偏移），于是反量化 matmul 无交叉项——更简单、硬件更快。它适配以 $0$ 为中心的数据，如多数**权重**。**非对称**保留 zero-point，花比特适配偏斜范围——是单侧**激活**（如 post-ReLU，全 $\ge 0$）的正确选择，那里对称会把一半的级浪费在从不出现的负范围上。§4 展示在全正数据上对称比非对称粗 ~2×。经验法则：**权重用对称，分布偏斜处用非对称。**

### 3.3 量化什么：weight-only vs weight+activation

vLLM 把方案命名为 `WxAy` = $x$ 比特权重、$y$ 比特激活：

- **Weight-only**（`W4A16`、`W8A16`）：量化权重、激活保持 FP16。权重在片上**反量化回 FP16**、matmul 跑 FP16——所以赢面是 **HBM 带宽**（更少权重字节 → memory-bound *decode* 更快），而非 FLOPs。容易且流行，因为权重静态、近对称；这是 AWQ/GPTQ 的 INT4 区间（#11）。
- **Weight+activation**（`W8A8`）：两者都量化，于是 matmul 跑在 **INT8 tensor core** 上——在内存赢面之上再加真正的**计算**加速，对 compute-bound *prefill* 与大 batch 有价值。难点：激活在运行时计算、有大 outlier，所以朴素 `W8A8` 掉精度——故有 SmoothQuant（把 outlier 迁到权重侧）等（#11）。vLLM 的动态 **FP8** 路径是中间地带：它对 Linear 权重按 per-tensor 量化、对激活*每次前向按 per-tensor 动态缩放*（无需校准），拿一些延迟收益换精度。
- **KV-cache 量化**是另一个独立的轴——量化存的 [KV cache](../part0/kv-cache.md) 以装下更多序列（有助[显存预算](../interview/vram-capacity-planning.md)）；它与权重/激活量化正交。

### 3.4 PTQ vs QAT

**PTQ（训练后量化）** 拿一个已训练模型来量化——可选用一个小**校准**集来选好范围/scale（百分位裁剪、outlier 处理）。不重训、几分钟到几小时，且它就是推理 infra 用的（GPTQ/AWQ/FP8 都是 PTQ）。**QAT（量化感知训练）** 在训练*期间*模拟量化，让模型学出对它稳健的权重——在极低比特下精度最好，但需要训练流水线、数据与算力。对服务，issue 的指引成立：**推理聚焦 PTQ**；只在 PTQ 在你需要的比特宽度下守不住精度时才动 QAT。

## 4 · 完整可跑代码 + 逐行讲解

两个演示，纯 Python、离线：在带 outlier 通道的矩阵上做 per-tensor vs per-channel，以及在偏斜数据上做对称 vs 非对称。

```python title="granularity_and_symmetry.py"
"""粒度与对称性：两个缩小每个 scale 覆盖范围的旋钮。
纯 Python、离线。每个值的误差界是 step/2（见上一课）。"""

def affine_step(xs, bits):
    """非对称步长 = range / (2^b - 1)。"""
    return (max(xs) - min(xs)) / ((1 << bits) - 1)

def symmetric_step(xs, bits):
    """对称步长把 [-amax, amax] 映到每侧 2^(b-1)-1 级的有符号格点。"""
    amax = max(abs(v) for v in xs)
    return amax / ((1 << (bits - 1)) - 1)

if __name__ == "__main__":
    W = [[0.1, -0.2, 0.15, -0.05],      # 干净通道
         [-0.3, 0.25, -0.1, 0.2],       # 干净通道
         [8.0, -0.1, 0.05, 0.2]]        # 一个带大 outlier（8.0）的通道

    # --- 粒度：全体一个 scale vs 每通道（行）一个 ---
    flat = [x for row in W for x in row]
    pt = affine_step(flat, bits=4)                       # per-tensor：全局范围定步长
    pcs = [affine_step(row, bits=4) for row in W]        # per-channel：每行自己的步长
    print(f"per-tensor  INT4 step: {pt:.4f}  (one scale for the whole matrix; the 8.0 outlier sets it)")
    print("per-channel INT4 step: " + "  ".join(f"row{i} {s:.4f}" for i, s in enumerate(pcs)))
    print(f"  clean row0 error bound: per-tensor <= {pt/2:.4f}  vs  per-channel <= {pcs[0]/2:.4f}  "
          f"(~{round((pt/2)/(pcs[0]/2))}x better)")

    # --- 对称性：偏斜、全正的数据（像激活后的值）---
    act = [0.0, 0.1, 0.4, 0.8, 2.0]
    asym, sym = affine_step(act, bits=4), symmetric_step(act, bits=4)
    print(f"\nskewed data {act}:")
    print(f"  asymmetric INT4 step {asym:.4f} (bound {asym/2:.4f})  vs  symmetric {sym:.4f} (bound {sym/2:.4f})")
    print(f"  -> symmetric wastes ~half the grid on unused negatives (~{round(sym/asym)}x coarser)")
```

**逐行讲解：**

- `affine_step` / `symmetric_step` —— 上一课的步长：非对称跨完整 `[min, max]`；对称跨 `[-amax, amax]` 于有符号格点，所以若数据单侧就浪费 $0$ 以下的级。
- **粒度块** —— `pt` 是整个矩阵一个步长（`8.0` outlier 定的）；`pcs` 给每行自己的。干净的 `row0` 的误差界从 per-tensor → per-channel 崩降 ~24×，因为它的 scale 不再要跨 outlier 的范围。
- **对称性块** —— 在全正的 `act` 上，对称的步长是非对称的 ~2×，因为它半个格点覆盖从不出现的负数。非对称把那些级花在数据真正所在处。

预期输出（精确算术，不是 benchmark）：

```text
per-tensor  INT4 step: 0.5533  (one scale for the whole matrix; the 8.0 outlier sets it)
per-channel INT4 step: row0 0.0233  row1 0.0367  row2 0.5400
  clean row0 error bound: per-tensor <= 0.2767  vs  per-channel <= 0.0117  (~24x better)

skewed data [0.0, 0.1, 0.4, 0.8, 2.0]:
  asymmetric INT4 step 0.1333 (bound 0.0667)  vs  symmetric 0.2857 (bound 0.1429)
  -> symmetric wastes ~half the grid on unused negatives (~2x coarser)
```

读出来：per-channel 给干净行 ~24× 更细的步长，同时把 `8.0` outlier 限在它*自己*行的 scale 里——这就是 per-channel/group 成为权重标准的全部理由。而非对称在偏斜数据上细 ~2×——这就是激活常想要 zero-point 的理由。两者是同一原理：**缩小每个 scale 必须覆盖的范围。**

## 5 · Lab —— 选一个方案，并把它连到真实 flag

数值 Lab 就是上面的演示（把每行切成 2 个一块、各自一个步长来试 per-group——误差再降）。*应用*那一半是识别这些选择在 vLLM 命名里的样子，读它不需要 GPU：

```python title="scheme_names.py"
# vLLM 用 WxAy 表达「量化什么」，并用 `quantization=` 选方法。
# （名称/flag 经 vLLM 0.26.0 核实；运行它们是 #11 的动手课。）
schemes = {
    "W4A16": "4-bit weights, 16-bit activations  -> weight-only; memory/decode win (AWQ/GPTQ INT4)",
    "W8A8":  "8-bit weights, 8-bit activations   -> INT8 tensor cores; compute win, needs outlier handling",
    "fp8":   "dynamic FP8: Linear weights per-tensor, activations scaled per-tensor per-forward (no calibration)",
}
for name, note in schemes.items():
    print(f"{name:6} {note}")
# 在 vLLM 中：  LLM(model, quantization="fp8")   或   vllm serve <model> --quantization fp8
```

**要观察什么：** `WxAy` 名字立刻告诉你一个方法是内存打法（`W4A16`，激活不动）还是计算打法（`W8A8`，两者都量化）。把你在 #11 遇到的任何方法映到这四个选择上——它的比特宽度（量化什么）、粒度、对称性、以及它是 PTQ——你不用死记就能推理它的精度/速度画像。真正在 `Qwen2.5-7B` 上*运行* `--quantization` 是 [#11 动手课](index.md)。

## 6 · 常见坑 / 反直觉点

- **对带 outlier 通道的权重用 per-tensor。** 一个 outlier 行给整个矩阵定了粗步长。用 per-channel 或 per-group——INT4 权重的标准——来隔离 outlier。这是低比特下最大的单个精度杠杆。
- **对偏斜激活用对称。** 对称把半个格点浪费在数据从不到访的范围上（如 post-ReLU 的负数）。分布单侧处用非对称；对称是给零中心权重的。
- **把 `W8A8` 当成「比 `W4A16` 量化得更多」。** 它们优化不同的东西：`W4A16` 是*内存/decode* 赢面（激活保持 FP16）；`W8A8` 是*计算*赢面（INT8 tensor core），但必须驯服激活 outlier。量化得更多 ≠ 严格更好——是不同的权衡。
- **为部署上 QAT。** QAT 需要训练流水线、对服务很少值得；PTQ（± 校准）是推理默认。只在 PTQ 在你目标比特下守不住质量时才用 QAT。
- **以为更细粒度免费。** 更多 scale = 更高有效比特、有时更慢的 kernel。per-group ~128 是常见甜点，而非 per-element。
- **忘了激活才是难点。** 权重静态、近对称（易）；激活动态、重尾（难）。这个不对称就是为什么 weight-only 量化这么流行、为什么激活量化需要 SmoothQuant 式技巧（#11）。
- **静态 vs 动态激活 scale —— 一条隐藏的轴。** 除对称性外，一个激活 scale 可以*静态*计算（从校准算一次）或*动态*计算（每次前向重算）。因为激活随输入剧烈摆动，动态缩放通常更能守住精度——vLLM 的 FP8 路径对激活*每次前向按 per-tensor 动态缩放*（无需校准），某些 `W8A8` 流水线更细到*per-token* scale。静态更简单/更快，但假设分布稳定，于是 off-distribution 流量会咬人。并非普适——在看重 kernel 简洁或固定 scale 处静态更优。

## 7 · 面试连线

- [量化方案：粒度、对称性、量化什么、PTQ vs QAT](../interview/quantization-schemes.md) —— 这节课为你准备的高频题：*per-tensor vs per-channel vs per-group、对称 vs 非对称、weight-only vs weight+activation（W4A16 vs W8A8），以及为什么推理用 PTQ。*

## 8 · 小结 & 延伸阅读

**一句话：** 四个选择驯服 outlier 问题、并安置每一个量化方法——粒度（per-tensor→channel→group 缩小每个 scale 覆盖的范围）、对称性（零中心权重用对称、偏斜激活用非对称）、量化什么（weight-only `W4A16` = 内存/decode 赢面，`W8A8` = 计算赢面但激活 outlier 难）、以及怎么得到（推理用 PTQ，只在 PTQ 守不住时才 QAT）。

延伸阅读：

- [量化基础](quantization-basics.md) 课 —— 这些选择调的那个仿射映射与误差界。
- 下一节（#11）：方法族——**GPTQ、AWQ、SmoothQuant、FP8、LLM.int8()**——作为这个设计空间里的具体点，加上动手把 `Qwen2.5-7B` → INT4 在 vLLM 里跑。
- *SmoothQuant*（Xiao 等）—— 把激活 outlier 迁进权重让 `W8A8` 可行；「激活才是难点」这个坑最清楚的动机。

## 9 · 自测小问

??? question "一个矩阵里某个通道的幅值远大于其余。为什么 per-tensor INT4 量化会伤到其他通道，per-channel 又如何修好？"
    量化误差被半步界住，而步长是 `range / (2^b − 1)`——由**scale 必须覆盖的范围**决定。用 per-tensor 时，一个 scale 跨*整个*矩阵，于是 outlier 通道的大幅值拉伸范围、给*每个*通道（包括小而规矩的）定了粗步长——它们的值被舍掉。per-channel 给每个输出行自己的 scale，于是干净行的步长只由*它自己*的（小）范围决定——常 ~10–30× 更细——而 outlier 被限在它自己行的粗 scale 里。per-group（每 ~128 权重一个 scale）进一步细化，是 INT4 权重的标准。

??? question "`W4A16` 与 `W8A8` 有何区别，各自加速 decode 还是 prefill？"
    `W4A16` = 4-bit 权重、16-bit（FP16）激活——**weight-only**：权重在片上反量化回 FP16、matmul 保持 FP16，所以赢面是 **HBM 带宽**（更少权重字节）。那加速 **memory-bound decode**。`W8A8` = 权重*与*激活都 8-bit——matmul 跑在 **INT8 tensor core** 上，是真正的**计算**加速、也有助 **compute-bound prefill** 与大 batch。权衡：`W8A8` 必须处理激活 outlier（动态、重尾），故需要 SmoothQuant 之类方法；而 `W4A16` 让激活留在 FP16、绕开了那个问题。

??? question "为什么推理侧量化压倒性地用 PTQ 而非 QAT？什么时候你仍会动 QAT？"
    PTQ（训练后量化）拿一个已训练模型来量化，可选用小校准集选好范围——不重训、便宜（几分钟到几小时），且能塞进服务流水线。这匹配推理 infra 的约束，主流方法（GPTQ、AWQ、FP8）都是 PTQ。QAT（量化感知训练）在训练时模拟量化，让模型学出量化稳健的权重——在极低比特下精度最好，但需要完整训练流水线、数据与算力。你只在 PTQ 在你需要的比特宽度下守不住可接受质量时（如激进的 sub-4-bit）才动 QAT，而这对服务通常瞄准的 INT4/INT8 区间并不常见。
