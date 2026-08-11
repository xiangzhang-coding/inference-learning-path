# GPU 硬件心智模型

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    本页的硬件数字（峰值 FLOP/s、HBM 带宽、SM 数量）是消费级 RTX 4090 的**示例 / 量级参考**——精确值请查你自己卡的规格书或 `nvidia-smi`。roofline 的算术（一个 `min` 加一个除法）是*精确*的，不是跑分。

---

## 1 · 直觉 & 为什么重要

数据中心 GPU 用一个巨大的数字给自己打广告：**TFLOPS**，即峰值算术速率。这个数字对 LLM decode 几乎无关紧要。[推理流程](inference-flow.md) 那节课已表明 decode 跑在 ≈ 1 FLOP/字节 的算术强度上——即它每从内存拖出一个字节，只做大约一次乘加。而一张每字节带宽能做*几百* FLOPs 的 GPU，按定义就几乎在空转：它瞬间算完，然后每一步剩下的时间都在**等内存**。

所以你能带进后续每一章的最有用的东西，就是一个**围绕带宽、而非 FLOPs 建立的硬件心智模型**。一旦你能想象字节住在哪里（寄存器 → SRAM → HBM）、每一层搬得多快、以及 roofline 如何把「强度」变成「可达吞吐」，本路径里的每个优化就不再是要背的招式，而变成*你能画出来的图上的一步棋*：FlashAttention 把字节留在 SRAM；量化缩小字节；continuous batching 抬高强度让 FLOPs 终于派上用场。→ 见[术语表](../glossary.md)的 *SM / Warp / Occupancy*、*HBM / SRAM*、*Roofline*。

## 2 · 心智模型

脑中同时握住两张图：**内存金字塔**（字节住哪、搬多快）与**执行结构**（谁来算）。

```text
                    内存金字塔（以 RTX 4090 为例，示例数据）
                    ┌───────────────────────────────┐
   最快、最小        │  寄存器          ~KB/SM    ~数十 TB/s │  片上
        ▲           ├───────────────────────────────┤
        │           │  SRAM: L1 / 共享内存  ~100 KB/SM ~TB/s │  片上  <- FlashAttention 住这
        │           ├───────────────────────────────┤
        │           │  L2 缓存         ~72 MB    ~数 TB/s     │  片上
        │           ├───────────────────────────────┤
   最慢、最大        │  HBM/GDDR6X      24 GB     ~1 TB/s      │  片外  <- 权重 + KV 缓存住这
                    └───────────────────────────────┘
                     每个 decode 步都要把权重 + KV 拖过这条又慢又长的最底线。

                    执行结构
   GPU = 128 个 SM（流多处理器），每个 SM 跑很多 WARP（32 线程，锁步执行）。
   SM 靠 SWAP warp 隐藏内存延迟：warp A 等 HBM 时，warp B 在算。
   OCCUPANCY（占用率）= 有多少 warp 常驻可供切换。高占用率隐藏延迟；
   但它 *不* 抬高带宽天花板。
```

要刻进脑子的两个形状：

- **带宽悬崖。** 层与层之间的移动不是缓坡——HBM 大约比片上 SRAM 慢*一个数量级*。一个本可把数据留在 SRAM 却从 HBM 重读的算法，每次访问都要付这道悬崖。这一个事实就是 IO-aware kernel 的全部动机。
- **隐藏延迟 ≠ 带宽。** SM 极擅长靠玩 warp 来隐藏*延迟*（等第一个字节的时间），却造不出*带宽*（每秒字节数）。当 decode 被带宽饿死时，加 warp/占用率毫无用处——管子早已被「等待」这个错误的东西填满。

## 3 · 原理与数学 —— roofline

[roofline 模型](../glossary.md) 把两个硬件数字——峰值算力 $P$（FLOP/s）与内存带宽 $B$（字节/s）——加上某工作负载的[算术强度](inference-flow.md) $I$（FLOP/字节），变成**可达吞吐**：

$$
\text{attainable}(I) \;=\; \min\bigl(\,P,\; I \cdot B\,\bigr)
$$

把它读成在一个拐角处相接的两个状态。$I$ 小时，$I\cdot B < P$，你被钉在**带宽屋顶** $I\cdot B$ 上——*memory-bound*。$I$ 大时，$P$ 项胜出，你落在平的**算力屋顶**上——*compute-bound*。交点就是**拐点（ridge point）**：

$$
I^{*} \;=\; \frac{P}{B}
$$

任何 $I < I^{*}$ 的工作负载在这台机器上都 memory-bound；任何 $I > I^{*}$ 的都 compute-bound。代入 RTX 4090 的示例数字——$P \approx 165\ \text{TFLOP/s}$（BF16，稠密）与 $B \approx 1\ \text{TB/s}$（GDDR6X）：

$$
I^{*} \;\approx\; \frac{165 \times 10^{12}}{1 \times 10^{12}} \;\approx\; 165\ \text{FLOP/字节}
$$

现在叠上工作负载。Decode 的强度 ≈ 1 FLOP/字节——**比拐点低两个数量级**——所以 decode 只达到 ≈ $1 \cdot B$ = ~1 TFLOP/s，用掉这张卡 165 TFLOP/s 里的 **< 1%**。Prefill 的强度攀到几千，越过拐点，落在平的算力屋顶上。*同一张 GPU、同样的权重、相反的状态*——正是[推理流程](inference-flow.md) 预言的那个不对称，如今从硬件自己的 roofline 上读了出来。

这就是为什么广告上的 TFLOPS 对 decode 说了谎：你只有在 $I \ge I^{*}$ 时才够得着它，而 decode 离那儿远得很。真正能帮 decode 的杠杆，全都**把你往右推、或抬高带宽屋顶**：量化（更少字节 → 更高 $I$ *且* 更少要搬的字节）、批处理（跨批复用权重 → 更高 $I$）、IO-aware kernel（避开 HBM 往返 → 等效更多可用 $B$）。

## 4 · 完整可跑代码 + 逐行讲解

这个 roofline 计算器**可离线运行**——纯 CPU、无 GPU、无网络。它把 $\min(P, I\cdot B)$ 变成你能动手拨弄的数字，让「decode 浪费了 GPU」成为可核对的算术。

```python title="roofline.py"
"""Roofline 计算器：可达吞吐 vs 算术强度（纯 CPU）。"""
from dataclasses import dataclass


@dataclass
class GPU:
    name: str
    peak_flops: float   # P，FLOP/s（BF16 稠密；示例）
    bandwidth: float    # B，字节/s（HBM；示例）

    @property
    def ridge_point(self) -> float:
        return self.peak_flops / self.bandwidth        # I* = P / B（FLOP/字节）


def attainable(gpu: GPU, intensity: float) -> float:
    return min(gpu.peak_flops, intensity * gpu.bandwidth)   # roofline：min(P, I*B)


if __name__ == "__main__":
    # 示例 RTX 4090：~165 TFLOP/s BF16 稠密，~1 TB/s GDDR6X。
    gpu = GPU("RTX 4090", peak_flops=165e12, bandwidth=1.0e12)
    print(f"{gpu.name}: 拐点 I* = {gpu.ridge_point:.0f} FLOP/字节\n")

    # 扫一组把真实 LLM 阶段夹在中间的强度：
    #   decode ~1，某些融合算子 ~10，拐点本身，prefill ~1000+。
    for I in (1, 10, gpu.ridge_point, 1000):
        got = attainable(gpu, I)
        regime = "memory-bound" if I < gpu.ridge_point else "compute-bound"
        util = got / gpu.peak_flops
        print(f"I={I:7.0f} FLOP/字节 -> {got/1e12:6.1f} TFLOP/s "
              f"（{util:6.1%} of peak，{regime}）")
```

**逐行讲解：**

- `GPU` — 就 roofline 而言，一台机器就是*两个*数字：峰值算力 `peak_flops`（$P$）与内存 `bandwidth`（$B$）。其余一切（SM 数、频率）都汇入 $P$。
- `ridge_point` — $I^{*} = P/B$，斜的带宽屋顶与平的算力屋顶相接处的强度。它左边 = memory-bound；右边 = compute-bound。
- `attainable` — roofline 本身：`min(P, I*B)`。拐点以下 `I*B` 项胜出；以上 `P` 封顶。
- `__main__` — 扫四个把真实阶段夹在中间的强度。看 decode（$I=1$）只达到峰值的一个舍入误差，而 $I=1000$（类 prefill）把它打满。

预期输出（精确算术，非跑分）：

```text
RTX 4090: 拐点 I* = 165 FLOP/字节

I=      1 FLOP/字节 ->    1.0 TFLOP/s （  0.6% of peak，memory-bound）
I=     10 FLOP/字节 ->   10.0 TFLOP/s （  6.1% of peak，memory-bound）
I=    165 FLOP/字节 ->  165.0 TFLOP/s （100.0% of peak，compute-bound）
I=   1000 FLOP/字节 ->  165.0 TFLOP/s （100.0% of peak，compute-bound）
```

$I=1$ 的 decode 用掉这张卡的 **0.6%**。这 0.6% 不是你代码的 bug——它是 roofline 在告诉你：极限来自工作负载，而非 GPU。Part 4–7 里的一切，都是把这个比例往上抬的斗争。

## 5 · Lab —— 读出你卡的真实 roofline 输入

!!! gpu "GPU Lab"
    - **最低显存：** 任意 CUDA GPU（本 lab 只*读取*设备属性；不加载任何模型）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）——对齐基线
    - **预估耗时 / 花费：** ~5 分钟 · ~¥0.5 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** `torch.cuda` 属性是 NVIDIA/ROCm 专有的；AMD ROCm 上同样的 PyTorch 调用能跑，但字段名/单位可能不同，CPU/TPU/Neuron 后端有各自的工具——请查你所在平台的文档。

别信 PPT 上的 TFLOPS。读出你卡真正上报的两个 roofline 输入，再喂给上面的 `roofline.py`：

```python title="device_roofline.py"
import torch

props = torch.cuda.get_device_properties(0)
print("name:            ", props.name)
print("SM 数量:         ", props.multi_processor_count)   # 可供跨其切换 warp 的 SM 数
print("显存总量 (GB):   ", round(props.total_memory / 1024**3, 1))

# 峰值 FLOP/s 与 HBM 带宽 torch 并不暴露；从厂商规格书取值
# （示例 4090：~165 TFLOP/s BF16 稠密，~1 TB/s），把二者
# 都代入 roofline.py 得到 *你自己的* 拐点。
```

**观察什么：** `multi_processor_count` 是执行结构的宽度（4090 上是 128）；`total_memory` 是你的 KV 缓存 + 权重预算（[KV 缓存](kv-cache.md) 的上限）。PyTorch 故意*不*把峰值 FLOP/s 或带宽交给你——那是厂商的数字，而本课的全部要点恰是：主宰 decode 的是*带宽*那个数，不是 FLOP/s 那个数。进阶练习：给一个大 `torch.mm`（compute-bound）与一个大 `x.copy_()`（bandwidth-bound）计时，确认 copy 离峰值 FLOP/s 差得远——这就是野外的 roofline。

## 6 · 常见坑 / 反直觉点

- **「TFLOPS 越高推理越快。」** 只在拐点以上成立。Decode 住在 $I \approx 1 \ll I^{*}$，所以一张 FLOPS 翻倍但带宽*不变*的卡，decode 速度基本一样。为 decode 买带宽，别买 FLOPs。
- **盒子上的峰值 TFLOPS 常假设了稀疏或 FP8。** 4090 那个「≈330 TFLOP/s」头条数字是 2:4 稀疏值；稠密 BF16 大约是它的一半。引用峰值前永远先问「稠密还是稀疏？什么 dtype？」。
- **占用率不是吞吐。** 把常驻 warp 拉满隐藏的是*延迟*；它无法超过*带宽*屋顶。100% 占用率的 memory-bound kernel 依然 bandwidth-bound。
- **可达带宽 < 峰值带宽。** 真实 kernel 大概只摸到规格书 HBM 数字的 70–85%。roofline 的 $B$ 是天花板，不是承诺。
- **HBM vs SRAM 是全部胜负手。** 「GPU 慢」几乎总意味着「我在重读 HBM」。FlashAttention 之所以出名，恰恰因为它把注意力工作集留在 SRAM，而非在 HBM 里往返。

## 7 · 面试连线

- [GPU 内存层级与 roofline](../interview/gpu-memory-hierarchy.md) —— 本课为你准备的高频题：*走一遍内存层级与 SM/warp 模型，再用 roofline 及其拐点解释为什么 LLM decode 是 memory-bound。*

## 8 · 小结 & 延伸阅读

**一句话：** GPU 是一台披着算力机规格书的带宽机——roofline $\min(P, I\cdot B)$ 及其拐点 $I^{*}=P/B$ 告诉你 LLM decode（$I\approx1$）被钉在带宽屋顶上，于是每个 decode 优化都是抬高强度或绕开 HBM 的一步棋。

延伸阅读：

- Williams, Waterman, Patterson —— *Roofline: An Insightful Visual Performance Model*（该模型的源头）。
- Dao 等 —— *FlashAttention* —— 经典的「留在 SRAM」IO-aware kernel。
- 你卡的厂商白皮书 —— 取*稠密*峰值 FLOP/s 与 HBM 带宽喂给 roofline。
- [推理流程](inference-flow.md) 那节课 —— decode 的 $I\approx1$ 强度从哪来。

## 9 · 自测小问

??? question "写出 roofline 方程及其拐点，并用一句话说为什么 decode 在 4090 上 memory-bound。"
    可达吞吐 $= \min(P,\ I\cdot B)$，其中 $P$ = 峰值 FLOP/s，$B$ = 带宽，$I$ = 算术强度；拐点为 $I^{*}=P/B$（4090 在 ~165 TFLOP/s 与 ~1 TB/s 下 ≈165 FLOP/字节）。Decode 的 $I\approx1$ 比 $I^{*}$ 低约两个数量级，所以它坐在斜的带宽屋顶上、~1 TFLOP/s（< 1% 峰值）——memory-bound。

??? question "厂商把一张 GPU 的 TFLOPS 翻倍但 HBM 带宽不变。单条流 decode 会更快吗？prefill 呢？"
    单条流 decode：基本**不会**——它在拐点以下、被钉在 $I\cdot B$，而 $B$ 没变。Prefill：**会**——它 compute-bound（在拐点以上），于是骑上更高的 $P$ 屋顶。这就是为什么「decode 要带宽、prefill 要 FLOPs」，也是两个阶段受益于不同硬件的原因。

??? question "为什么把一个 memory-bound kernel 的占用率从 50% 提到 100% 常常并不提速？"
    占用率主宰*延迟隐藏*——一个 SM 在等内存时能在多少 warp 间切换。一旦 warp 多到足以让内存管子饱和，再多 warp 也加不出带宽；kernel 早已受限于字节/秒（$B$），而非空闲的 SM。你需要的是更高的 $B$（更好硬件）、更少的字节（量化）、或更高的 $I$（批处理、融合）——而非更多占用率。
