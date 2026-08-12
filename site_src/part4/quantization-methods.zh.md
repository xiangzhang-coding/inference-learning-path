# 量化方法族：GPTQ、AWQ、SmoothQuant、FP8、LLM.int8()、KV-cache

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    这里点名的工具——**llm-compressor**（`oneshot` + `GPTQModifier(scheme="W4A16"/"W8A8")`、`SmoothQuantModifier`）、vLLM 对预量化 checkpoint 的自动检测、以及 `kv_cache_dtype="fp8"`——经 Context7 对照 vLLM 0.26.0 核实（ADR-0004）。**AutoAWQ 已废弃**，其功能并入 llm-compressor（已核实）。§4 的方法→设计空间表是**分类，不是计算**（纯 Python、离线）。精度/速度数字均为**示例 / 量级参考**；动手实操在[下一课](quantization-lab.md)。

---

## 1 · 直觉 & 为什么重要

[方案那课](quantization-schemes.md) 给了你四个轴——粒度、对称性、量化什么（`W4A16` vs `W8A8`）、PTQ vs QAT。一个有名字的方法（GPTQ、AWQ、SmoothQuant、FP8、LLM.int8()）不过是**那个空间里的一个具体点，外加一个在低比特下保精度的巧招**。这样看，你不用背六个方法——把每个放到你已知的轴上，就能推理它的精度/速度画像。这正是面试技能：「讲讲 AWQ vs GPTQ vs SmoothQuant」是一道定位题，不是知识点默写题。

要带走的一个框架：**每个方法都在打[基础那课](quantization-basics.md) 里同一个敌人——抬高步长的 outlier。** 它们只在*怎么打*上不同。GPTQ 逐层校正舍入误差；AWQ 保护对输出最要紧的权重；SmoothQuant 把激活 outlier 迁进更好量化的权重里；FP8 用动态范围更大的格式；LLM.int8() 把少数 outlier 维保留在 FP16。同一个问题、五个巧招——外加 **KV-cache 量化**，它把整套思路用到另一个张量（[KV cache](../part0/kv-cache.md)）而非权重上。→ 方法名见[术语表](../glossary.md)。

## 2 · 心智模型

六个族，放在你已知的轴上：

```text
                 W-bits  A-bits  granularity        the trick                       primary win
  GPTQ            4 (8)   16      per-group          layer-wise error correction     memory / decode
  AWQ             4       16      per-group          scale by activation salience    memory / decode
  SmoothQuant     8       8       per-tensor/chan    migrate act. outliers → weights compute / prefill+batch
  FP8 (E4M3)      8       8       per-tensor         float format = more range       compute+memory (Hopper/Ada)
  LLM.int8()      8       8       per-chan+FP16 outl keep outlier dims in FP16        memory (accuracy-safe INT8)
  KV-cache FP8    —       —       (the KV tensor)    quantize K/V, not weights        memory (longer ctx / more seqs)

  weight-only (W4A16): AWQ, GPTQ  ── decode/内存打法，激活留 FP16，容易
  weight+activation (W8A8): SmoothQuant, FP8, LLM.int8() ── 计算打法，必须驯服激活 outlier
```

两个要抓住的形状：

- **Weight-only vs weight+activation 把这块领域一分为二。** AWQ/GPTQ 是 `W4A16`——内存/decode 赢面，容易，因为权重静态、近对称。SmoothQuant/FP8/LLM.int8() 是 `W8A8`——计算赢面（INT8/FP8 tensor core），但必须处理*激活* outlier，那正是各自解法不同的难点。
- **KV-cache 量化是独立的杠杆。** 它与权重/激活量化正交——你能在一个 INT4 权重模型上叠 FP8 KV cache。它买的是*更长上下文 / 更多并发序列*的内存（[显存预算](../interview/vram-capacity-planning.md)），而非权重读取速度。

## 3 · 原理 —— 六个族

### 3.1 GPTQ —— 逐层误差校正（weight-only）

GPTQ 一次量化一层权重，且在对每一列舍入后**更新剩余列以补偿**引入的误差（用一个小校准集做的二阶 / Hessian 校正）。结果是 INT4 权重（`W4A16`、per-group），比朴素舍入远更贴合 FP16 输出。代价：校准 + 误差校正一趟（几分钟）。是内存/decode 打法——激活留 FP16。

### 3.2 AWQ —— 激活感知的权重保护（weight-only）

AWQ 的洞见：不是所有权重同等重要——乘以大幅值激活通道的那些主宰输出。它在校准集上测激活幅值，并在量化前**放大那些显著权重通道**，让它们以更小误差挺过 INT4（把逆缩放折进下一个算子）。同样 `W4A16`、per-group、PTQ。实践中 AWQ 与 GPTQ 是两大主流 INT4 weight-only 方法；两者对 decode-bound 服务都很出色。

### 3.3 SmoothQuant —— 迁移 outlier 让 W8A8 可行

`W8A8` 的拦路虎是**激活有大的逐通道 outlier**（权重没有）。SmoothQuant 施加逐通道缩放，把「难度」**从激活转移进权重**——把激活除以因子 $s$、把对应权重乘 $s$（数学上恒等）——于是两者都变得好量化。现在 INT8 激活可行、matmul 跑 INT8 tensor core，你为 prefill/大 batch 拿到*计算*加速。PTQ、per-tensor/channel。

### 3.4 FP8 —— 动态范围更大的浮点格式

FP8（通常 **E4M3**：4 指数、3 尾数位）是浮点 8-bit 格式的 `W8A8`。因为它有指数，同样比特数下比 INT8 覆盖宽得多的动态范围——于是更能容忍 outlier，且常**无需校准**（激活每次前向按 per-tensor 动态缩放，如 vLLM 的 `--quantization fp8`）。它跑 FP8 tensor core（Hopper/Ada），拿到计算*与*内存双赢。硬件支持时的现代默认。

### 3.5 LLM.int8() —— 混合精度 outlier 分解

LLM.int8()（bitsandbytes 的方法）靠**拆分 matmul** 让 INT8 精度安全：把少数带 outlier 的激活维保留在 **FP16** 单独算，其余绝大多数跑 INT8；两部分结果相加。所以它是 `W8A8` 式的，但给会毁掉量化的那 ~0.1% 维留了 FP16 逃生口。重精度胜过纯速度。

### 3.6 KV-cache 量化 —— 另一个张量

上面全都量化*权重*（有时含激活）。**KV-cache 量化**转而压缩存的 [KV cache](../part0/kv-cache.md)——如经 vLLM 的 `kv_cache_dtype="fp8"` 做 FP8 K/V（无校准；scale 默认 1.0）。它不加速权重读取；它**释放显存**，从而买到更长上下文或更多并发序列。正交——叠在权重量化之上。

## 4 · 完整可跑代码 + 逐行讲解

一段把每个方法放到[方案那课](quantization-schemes.md) 轴上的纯 Python 定位，外加一个选择器——离线、无 GPU。它把「六个要背的方法」变成「一张你能重新生成的表」。

```python title="method_families.py"
"""把每个量化方法放到设计空间轴上（来自方案那课）。
纯 Python、离线——是分类，不是计算。"""

# name: (weight_bits, act_bits, granularity, calibration, primary_win)
METHODS = {
    "GPTQ":        (4, 16, "per-group",                 "yes (Hessian-based)",    "memory / decode"),
    "AWQ":         (4, 16, "per-group",                 "yes (activation-aware)", "memory / decode"),
    "SmoothQuant": (8,  8, "per-tensor/channel",        "yes (migrate outliers)", "compute / prefill+batch"),
    "FP8 (E4M3)":  (8,  8, "per-tensor",                "no (dynamic act.)",      "compute + memory (Hopper/Ada)"),
    "LLM.int8()":  (8,  8, "per-channel + FP16 outliers","no (runtime)",          "memory (accuracy-safe INT8)"),
}

def recommend(goal):
    """哪些方法瞄准某个目标（primary win 的子串）？"""
    return [name for name, (wb, ab, *_rest, win) in METHODS.items() if goal in win]

if __name__ == "__main__":
    for name, (wb, ab, gran, calib, win) in METHODS.items():
        print(f"{name}: W{wb}A{ab}, {gran}, calibration={calib}, win={win}")
    print()
    print("for 'decode':", recommend("decode"))     # weight-only INT4 methods
    print("for 'compute':", recommend("compute"))   # weight+activation methods
```

**逐行讲解：**

- `METHODS` —— 六个族，写成 `(weight_bits, act_bits, granularity, calibration, primary_win)`。读一行*就是*给方法定位：`W4A16`（AWQ/GPTQ）= weight-only/decode；`W8A8`（SmoothQuant/FP8/LLM.int8()）= weight+activation/compute。
- `recommend(goal)` —— 按 `primary_win` 字段过滤；`wb, ab, *_rest, win` 解包元组、忽略中间。这就是你如何从框架（而非记忆）回答「加速 decode 该用什么？」。
- `__main__` —— 打印表，再打印两个经典选择（decode → weight-only；compute → weight+activation）。

预期输出（分类表，不是 benchmark）：

```text
GPTQ: W4A16, per-group, calibration=yes (Hessian-based), win=memory / decode
AWQ: W4A16, per-group, calibration=yes (activation-aware), win=memory / decode
SmoothQuant: W8A8, per-tensor/channel, calibration=yes (migrate outliers), win=compute / prefill+batch
FP8 (E4M3): W8A8, per-tensor, calibration=no (dynamic act.), win=compute + memory (Hopper/Ada)
LLM.int8(): W8A8, per-channel + FP16 outliers, calibration=no (runtime), win=memory (accuracy-safe INT8)

for 'decode': ['GPTQ', 'AWQ']
for 'compute': ['SmoothQuant', 'FP8 (E4M3)']
```

选择器的答案直接从轴上掉出来：**decode/内存 → `W4A16` weight-only 方法**（AWQ、GPTQ）；**compute → `W8A8` 方法**（SmoothQuant、FP8）。你从不需要背哪个是哪个——设计空间告诉了你。

## 5 · Lab —— 一个方法就是一个 recipe

你不手写这些——你把一个 **recipe** 交给工具。现代的、vLLM 认可的工具是 **llm-compressor**（AutoAWQ 已废弃并入它）。读这些核实过的 recipe 能看到一个方法名如何变成配置；*运行*它们是[下一课](quantization-lab.md)，所以这是一个无 GPU 的阅读 Lab：

```python title="recipes.py"
# llm-compressor recipe —— 每个都是放到轴上的方法，写成代码。
# （API 经 vLLM 0.26.0 量化文档核实；在下一课运行它。）
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# INT4 weight-only（AWQ/GPTQ 地盘）：W4A16、per-group、绝不碰 lm_head。
w4a16 = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# INT8 weight+activation：SmoothQuant 驯服激活 outlier，再由 GPTQ 做 W8A8。
w8a8 = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
# FP8（无校准）更简单——vLLM 能在线做：quantization="fp8"。
print("W4A16 scheme:", w4a16.scheme)         # -> W4A16   (weight-only，decode 打法)
print("W8A8 stages :", [type(m).__name__ for m in w8a8])  # SmoothQuant then GPTQ
```

**要观察什么：** `scheme="W4A16"` / `"W8A8"` 字符串字面就是上一课的轴——工具说同一种语言。`ignore=["lm_head"]` 是通用的「别量化输出投影」规则（它很小且对精度敏感）。`W8A8` recipe 是*两*个阶段——先 SmoothQuant（迁移 outlier）再 GPTQ（舍入）——因为激活量化需要 outlier 修法；`W4A16` recipe 是一个阶段，因为 weight-only 不需要。那个结构差异*就是* §3.3。在[下一课](quantization-lab.md) 你会把像 `w4a16` 的 recipe 交给 `oneshot(...)`、存 checkpoint、在 vLLM 里服务、并测量它的代价。

## 6 · 常见坑 / 反直觉点

- **把这些方法当成一个排名。** 没有「最好」——AWQ/GPTQ 赢在 decode-bound 的 INT4 服务；SmoothQuant/FP8 赢在你需要计算时（prefill/大 batch）；LLM.int8() 赢在 INT8 精度至上时。把方法匹配到瓶颈。
- **新项目还用 AutoAWQ。** 它已废弃；功能现在在 llm-compressor 里。用 llm-compressor recipe（或预量化 checkpoint），别用老 AutoAWQ 路径。
- **忘了 FP8 需要硬件。** FP8 tensor-core 加速是 Hopper/Ada（compute capability ≥ 8.9）；老 GPU 上 FP8 可能是模拟或不支持。INT4 weight-only（AWQ/GPTQ）是可移植的内存打法。
- **把 KV-cache 量化与权重量化混了。** 它们针对不同张量、不同赢面：权重量化加速权重读取（decode）；KV-cache 量化释放显存（更长上下文 / 更多序列）。叠着用。
- **忽略校准分布。** GPTQ/AWQ/SmoothQuant 在小数据集上校准——若它与你的流量严重偏离分布，选出的 scale 就适配了错误的范围。用有代表性的 prompt。
- **量化 `lm_head`。** 输出投影小且对精度敏感；这里每个 recipe 都 `ignore` 它。量化它是经典的精度自摆乌龙。

## 7 · 面试连线

- [量化方法：GPTQ vs AWQ vs SmoothQuant vs FP8 vs LLM.int8()](../interview/quantization-methods.md) —— 这节课为你准备的高频题：*把每个方法放到轴上、说出它用的巧招、并为给定瓶颈选一个。*

## 8 · 小结 & 延伸阅读

**一句话：** 每个量化方法都是设计空间里的一个点外加一个抗 outlier 巧招——GPTQ（逐层误差校正）、AWQ（保护显著权重）是 `W4A16` 的 decode/内存打法；SmoothQuant（迁移 outlier）、FP8（浮点范围）、LLM.int8()（FP16 outlier 维）是 `W8A8` 的计算打法；而 KV-cache FP8 是正交的内存杠杆——所以你给方法*定位*而非背它，并把它的 recipe 交给 llm-compressor。

延伸阅读：

- [量化基础](quantization-basics.md) 与 [方案](quantization-schemes.md) 课 —— 每个方法所建于的 outlier 问题与四个轴。
- *GPTQ*、*AWQ*、*SmoothQuant*、*LLM.int8()* —— 原始论文；每个都是 §4 一行的深入。
- llm-compressor 文档（推荐工具）与 vLLM 量化指南 —— recipe 与支持的格式。
- 下一节：[动手 —— 把 Qwen2.5-7B 量化成 INT4](quantization-lab.md) —— 把 recipe 变成一个服务中的模型，测量质量 vs 吞吐。

## 9 · 自测小问

??? question "把 AWQ 与 SmoothQuant 放到设计空间轴上。哪个加速 decode、哪个加速 prefill，为什么？"
    **AWQ** 是 `W4A16`——weight-only INT4、per-group、PTQ；它的巧招是缩放*显著*权重通道（乘以大激活的那些）好让它们挺过 INT4。因为激活留 FP16、权重只是在 FP16 matmul 前反量化，它的赢面是 **HBM 带宽** → 它加速 **memory-bound decode**。**SmoothQuant** 是 `W8A8`——把激活 outlier 迁进权重，让 INT8 *激活*可量化，从而 matmul 跑 **INT8 tensor core**。那是**计算**赢面 → 它帮 **compute-bound prefill 与大 batch**。不同的轴（量化什么）、不同的瓶颈。

??? question "为什么 `W8A8` 需要 SmoothQuant（或 LLM.int8() 的 FP16 拆分）这种巧招，而 `W4A16` 不需要？"
    因为 `W8A8` 量化**激活**，而激活有大的逐通道 **outlier**，单个 INT8 scale 无法在不产生巨大误差下表示它（outlier 抬高了整通道的步长）。SmoothQuant 靠把激活缩小、权重放大（恒等变换）让两者都好量化；LLM.int8() 则把 ~0.1% 的 outlier 维保留在 FP16、只量化其余。`W4A16` 让激活完全留在 FP16，所以从不面对激活 outlier 问题——只有权重（静态、近对称）被量化，per-group 粒度就能应付它们更温和的 outlier。

??? question "什么时候你会加 FP8 KV-cache 量化，它买到权重 INT4 量化买不到的什么？"
    当你在 **KV cache 上 memory-bound** 时加 FP8 KV 量化（`kv_cache_dtype="fp8"`）——长上下文或多并发序列——因为它把 KV 字节相对 FP16 减半、释放显存给更多序列 / 更长上下文（抬高[并发上限](../interview/vram-capacity-planning.md)）。它与权重量化*正交*：INT4 权重砍**权重**读取（加速 decode 的权重流量），而 FP8 KV 砍 **KV** 占用（容量，非权重速度）。你叠着用——INT4 权重 + FP8 KV——一次赢下权重带宽与 KV 容量。
