# Kernel Fusion 与 CUDA Graphs：干掉 decode 的 launch overhead

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    vLLM flag（`enforce_eager`、`compilation_config` / `cudagraph_mode`）与 PyTorch `torch.cuda.graph` API 经 Context7 对照 vLLM 0.26.0 / PyTorch 核实（ADR-0004）。kernel 数量、launch 开销与吞吐数字是**示例 / 量级参考**——自己测。§4 的 launch-overhead 算术是*精确*的（它是模型自身的定义）。

---

## 1 · 直觉 & 为什么重要

你已在 Part 2 证明 decode 是 [memory-bound](roofline-analysis.md) 的：每步搬很多字节、算很少。decode 上还有*第二重*、更隐蔽的税，与带宽无关——**kernel launch overhead（启动开销）**。一个 decode 步不是跑一个 kernel，而是跑*几百*个——RMSNorm、QKV projection、attention 算子、output projection、gate/up projection、SiLU、down projection、残差加法，每层两遍，跨 ~28 层。每一个都是独立的 GPU kernel，而每次启动都要花 **CPU** 几微秒去派发。

因为 decode kernel 很小（batch 1、一个 token），GPU 几乎瞬间算完，然后**空等 CPU 启动下一个**。几微秒 × 几百个 kernel 加起来，就是每步一笔固定税，可占到 15–30%——而且你越优化计算它*相对*越糟（把权重量化，GPU 工作缩小、launch 开销纹丝不动）。两种技术干掉它：**kernel fusion（算子融合）**把多个算子并成一个 kernel（更少启动、更少 HBM 往返），**CUDA graphs** 把整串 kernel 序列录制一次、用*单*次启动重放。这是 Part 2 最后一个单卡杠杆，也是 vLLM 默认捕获 CUDA graph 的原因。→ 见[术语表](../glossary.md)的 *CUDA graphs*、*Kernel fusion*。

## 2 · 心智模型

问题是一场 CPU↔GPU 的乒乓；两种解各自改变谁来启动：

```text
EAGER —— CPU 逐个启动 kernel；GPU 在小 kernel 之间空转
  CPU: [启动 k1]   [启动 k2]   [启动 k3]  ...  （每次提交 ~5 µs）
  GPU:    [k1]▪▪空▪▪  [k2]▪▪空▪▪  [k3]▪▪空▪▪       <- GPU 在等 CPU
          └ 小小的 memory-bound kernel；下一次启动还没到就算完了

KERNEL FUSION —— 合并算子，变成更少、更大的 kernel
  CPU: [启动 fused]           GPU: [ norm+proj+bias+act 融合 ]   <- 更少启动、
                                                                    更少 HBM 往返

CUDA GRAPH —— 把整串序列录制一次，用一次启动重放
  捕获（一次）:  把 k1..kN 录进一张图
  每一步:        CPU: [g.replay()]   GPU: [k1][k2][k3]...[kN]   <- 背靠背，
                                                                   无逐 kernel 间隙
```

要握住的两个形状：

- **启动是 CPU 的活儿；GPU 会因它挨饿。** 当一个 kernel 的 GPU 时间小于 CPU 的启动+派发延迟，GPU 算完就等。decode 的 kernel 恰恰这么小，所以这一步在 memory-bound 之上又变得 *launch-bound*。prefill kernel 很大（多 token），它们的启动开销只是舍入误差——这笔税是 **decode 专属**的。
- **fusion 缩数量；graphs 摊启动。** fusion 攻 $N$（以及融合算子间的中间 HBM 流量）；CUDA graphs 攻每次启动成本，用一次 CPU 提交重放 $N$ 个 kernel。它们可叠加：vLLM 既融合能融的，又把结果裹进一张图。

## 3 · 原理与数学

### 3.1 launch-overhead 模型

把一个 decode 步建模为 $N$ 个 kernel，每次启动 CPU 开销 $\tau$，加上真实 GPU 计算时间 $T_{\text{compute}}$（对 memory-bound 的 decode 本质就是 [字节 ÷ 带宽](roofline-analysis.md)）。**eager** 模式下小 kernel 藏不住启动，于是它们累加：

$$
T_{\text{eager}} \approx T_{\text{compute}} + N\,\tau
$$

一张 **CUDA graph** 用一次 CPU 提交重放全部 $N$ 个 kernel，于是 $N\tau$ 项塌缩：

$$
T_{\text{graph}} \approx T_{\text{compute}} + \tau, \qquad
\text{speedup} = \frac{T_{\text{compute}} + N\tau}{T_{\text{compute}} + \tau}
$$

speedup 在 $N\tau$ 与 $T_{\text{compute}}$ 相当时最大——即计算*小*时。这就是回到 roofline 的反直觉对接：你越缩小 GPU 工作（量化权重、小 batch），启动税的相对占比*越大*，CUDA graph 买到的越多。一个 7B 约 $430$ 个 kernel、$\tau\approx5\,\mu s$，得 $N\tau\approx2.15$ ms/步——对 prefill 微不足道，对 decode 却是决定性的。

### 3.2 为什么 fusion 帮两次忙

一个融合 kernel 一次做两件事。它**砍启动数**（$N$ 降），且——对逐元素/归约链——**避开 HBM 往返**：不再是 kernel A 把输出写 HBM、kernel B 读回，融合 kernel 把中间量留在寄存器/SRAM。这是启动收益之外的带宽收益，也是为什么融合大 GEMM *周围*的众多小算子（norm、bias、activation、residual）很重要——尽管 GEMM 本身已是一个 kernel。

### 3.3 CUDA graph 需要什么——以及为何有 `enforce_eager`

一张 CUDA graph 录制的是**在特定内存地址上运行的特定 kernel**。重放复用那些捕获的指针，于是：（1）形状必须**静态**——vLLM 为每个 batch-size 桶捕获一张图、并把运行 batch *补齐*到最近的已捕获尺寸；（2）每步输入必须拷进同一组静态 buffer；（3）捕获前必须**预热**（让分配/autotune 稳定）。动态控制流或新形状回退到 eager。捕获图还要花**显存**（捕获的 buffer），这就是 vLLM 暴露 `enforce_eager=True` 来禁用它的原因——拿 decode 吞吐换 VRAM 与灵活性。（同一份 VRAM 你本可花在 [KV cache](kv-cache-math.md) 上。）

## 4 · 完整可跑代码 + 逐行讲解

这段把 launch-overhead 模型变成数字——**纯 CPU、可离线运行**，无 GPU。它展示*为什么*量化模型从 CUDA graph 获益更多。

```python title="launch_overhead.py"
"""decode launch-overhead 模型：eager vs CUDA-graph 步时（纯 CPU，离线）。"""
from dataclasses import dataclass


@dataclass
class DecodeStep:
    weight_gib: float                 # 每 decode 步拉取的字节（权重主导）
    n_layers: int = 28                # Qwen2.5-7B
    kernels_per_layer: int = 15       # norm、projection、attn、act、adds（示例）
    launch_us: float = 5.0            # CPU 侧每 kernel 启动开销（示例）
    bandwidth_bps: float = 1.0e12     # ~1 TB/s HBM（示例 4090）

    @property
    def n_kernels(self) -> int:
        return self.n_layers * self.kernels_per_layer + 10        # + embed / final / lm_head

    @property
    def compute_ms(self) -> float:
        return self.weight_gib * 1024**3 / self.bandwidth_bps * 1e3   # 字节 / 带宽

    @property
    def launch_ms(self) -> float:
        return self.n_kernels * self.launch_us / 1e3              # N * tau

    def eager_ms(self) -> float:
        return self.compute_ms + self.launch_ms                   # T_compute + N*tau

    def graph_ms(self) -> float:
        return self.compute_ms + self.launch_us / 1e3             # T_compute + 一次提交


if __name__ == "__main__":
    for label, w in (("AWQ  权重 (~5.5 GiB)", 5.5), ("BF16 权重 (~14.2 GiB)", 14.2)):
        s = DecodeStep(weight_gib=w)
        e, g = s.eager_ms(), s.graph_ms()
        print(f"{label}: {s.n_kernels} 个 kernel | 计算 {s.compute_ms:5.1f} ms | "
              f"启动 {s.launch_ms:4.2f} ms")
        print(f"   eager {e:5.1f} ms -> {1000/e:5.1f} tok/s | "
              f"graph {g:5.1f} ms -> {1000/g:5.1f} tok/s | 加速比 {e/g:.2f}x")
```

**逐行讲解：**

- `n_kernels` —— 一个 decode 步的 kernel 数：每层 ~15 × 28 层，加 embedding/final-norm/lm_head。几百次启动，每个都很小。
- `compute_ms` —— memory-bound 的 decode 计算：拉取字节（权重主导）÷ 带宽。这是 Part 2 前面的 roofline 结论——decode 受带宽限，故权重字节定下限。
- `launch_ms` —— $N\tau$，eager 模式每步都付的固定 CPU 侧税。
- `eager_ms` vs `graph_ms` —— §3.1 的模型：eager 付全部 $N$ 次启动；图只付一次。差额是图收回的纯开销。
- `__main__` —— 对 AWQ vs BF16 权重各跑一遍，让你看到量化模型*更大*的相对收益。

预期输出（精确算术，非跑分）：

```text
AWQ  权重 (~5.5 GiB): 430 个 kernel | 计算   5.9 ms | 启动 2.15 ms
   eager   8.1 ms -> 124.1 tok/s | graph   5.9 ms -> 169.2 tok/s | 加速比 1.36x
BF16 权重 (~14.2 GiB): 430 个 kernel | 计算  15.2 ms | 启动 2.15 ms
   eager  17.4 ms ->  57.5 tok/s | graph  15.3 ms ->  65.6 tok/s | 加速比 1.14x
```

两者同样的 2.15 ms 启动税——但它在 AWQ 模型 5.9 ms 小计算上是 **36%**，在 BF16 模型 15.2 ms 上是 **14%**。把权重量化以求更快，CUDA graph 反而*更*重要，不是更不重要：你缩小了计算，固定启动开销就显得更大。这就是 vLLM 默认捕获图、且它在你会在 4090 上跑的量化、小 batch decode 上最见效的原因。

## 5 · Lab —— 量出启动税，再在 vLLM 里开关它

!!! gpu "GPU Lab"
    - **最低显存：** Part A 8 GB（小张量）；Part B 24 GB（加载模型）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）
    - **预估耗时 / 花费：** ~10 分钟 · ~¥1 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** `torch.cuda.graph` 仅 CUDA；ROCm 有 HIP-graph 对应物，vLLM 的 `enforce_eager` 开关在任何后端都可用，但 CUDA-graph 捕获路径是 NVIDIA/ROCm 专有的。

**Part A —— 原生 PyTorch 里的启动税。** 在循环里启动很多小 kernel，再把它们作为一张捕获的图重放：

```python title="cuda_graph_demo.py"
import torch

assert torch.cuda.is_available()
x = torch.zeros(1024, device="cuda")

def many_tiny_ops(x):                 # 代表一个 decode 步的几百个小 kernel
    for _ in range(200):
        x = x + 1.0                    # 每个都是独立、微小、launch-bound 的 kernel
    return x

def timed(fn, iters=100):
    for _ in range(10): fn()           # 预热
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); [fn() for _ in range(iters)]; e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms/iter

# Eager：每次迭代 200 次启动
eager = timed(lambda: many_tiny_ops(x.clone()))

# CUDA graph：把 200 个算子捕获一次，用单次提交重放
static = x.clone()
g = torch.cuda.CUDAGraph()
many_tiny_ops(static)                  # 捕获前预热确切的工作负载
with torch.cuda.graph(g):
    static_out = many_tiny_ops(static)
graph = timed(g.replay)

print(f"eager: {eager*1e3:6.1f} µs/iter   graph: {graph*1e3:6.1f} µs/iter "
      f"-> {eager/graph:.1f}x fewer launch stalls")
```

**Part B —— 同一杠杆，在 vLLM 端到端。** 开图（默认）vs 关图，比 decode 吞吐：

```bash
# CUDA graph 开（默认）—— vLLM 启动时按 batch 桶捕获图
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192

# CUDA graph 关 —— eager 模式（省 VRAM，每步付启动税）
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192 --enforce-eager
```

**观察什么：** Part A 的图重放每迭代快很多倍——那道差就是 §4 模型预测的启动开销。Part B 里 `--enforce-eager` 应*降低* decode 吞吐（tok/s）而*释放*一些 VRAM（无捕获图）——正是 §3.3 的权衡。比开关更细的控制，vLLM 暴露 `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'`（0.26.0 已核实）。（数字为示例；在你的机器上测。）

## 6 · 常见坑 / 反直觉点

- **指望 CUDA graph 帮 prefill。** 帮不上多少——prefill kernel 大且 compute-bound，启动开销可忽略（$N\tau \ll T_{\text{compute}}$）。这是 **decode** 优化，原因与 decode 是 memory-bound 阶段相同：每 kernel 的 GPU 工作太小。
- **量化 ⇒ 图更不重要？反了。** 缩小计算（AWQ、小 batch）让固定启动税成为一步中*更大*的占比——图帮*更多*。§4 显示 36% vs 14%。
- **忘了静态形状约束。** 图捕获固定形状与内存地址；vLLM 把 batch 补齐到捕获的桶。新形状或数据相关控制流回退到 eager——若吞吐没改善，检查是否真的捕获了。
- **捕获前跳过预热。** 捕获冷工作负载会录进 allocator/autotune 产物，可能崩溃或误捕。总要先把确切工作负载跑几遍（demo 里就是）。
- **`enforce_eager` 是免费显存，不是免费速度。** 它省下捕获图会用的 VRAM（可花在 [KV cache](kv-cache-math.md) 上）但每 decode 步付启动税。这是权衡——VRAM 紧张或调试时用它，别默认用。
- **fusion ≠ graphs。** fusion 减少 kernel *数量*（与中间 HBM 流量）；图减少剩下 kernel 的*启动成本*。`torch.compile` 自动做 fusion；vLLM 两者都做。混淆二者会导致「我融合了，为什么启动没降到一次？」——fusion 降 $N$，不把启动塌缩成单次提交。

## 7 · 面试连线

- [CUDA graphs 与 kernel fusion](../interview/cuda-graphs-fusion.md) —— 本课为你准备的高频题：*为什么 decode launch-bound 而 prefill 不是；CUDA graph 去掉什么、又需要什么；以及为什么量化模型获益更多？*

## 8 · 小结 & 延伸阅读

**一句话：** decode 每个 token 发几百个小 kernel，于是 CPU 启动开销（$N\tau$）成为每步固定税、且随你缩小计算而*相对*增大——kernel fusion 砍 kernel 数（与中间 HBM 流量），CUDA graph 用一次提交重放整串序列，这就是为什么二者都是 decode 阶段、对量化友好的吞吐收益。

延伸阅读：

- PyTorch 文档 —— *Accelerating PyTorch with CUDA Graphs* 与 `torch.cuda.graph` / `make_graphed_callables` API 说明。
- vLLM 文档 —— *CUDA graphs* 设计文档与 `compilation_config` / `cudagraph_mode`（基线 v0.26.0）；`enforce_eager` 见 *Conserving Memory*。
- [算子 Roofline](roofline-analysis.md) 那节课 —— 为什么 decode 每 kernel 的 GPU 工作一开始就那么小（因而 launch-bound）。
- [FlashAttention](flash-attention.md) 那节课 —— Part 2 另一个 kernel 级收益，融合 attention 算子本身。

## 9 · 自测小问

??? question "为什么 kernel 启动开销伤 decode 却几乎不碰 prefill？"
    启动开销是每 kernel 固定的 CPU 侧成本（$\tau$）。它的影响取决于与每 kernel GPU 时间的对比。**decode** kernel 很小（batch 1、一个 token、memory-bound），GPU 在 CPU 启动下一个前就算完——这一步变 launch-bound，$N\tau$（几百 kernel × µs）是一步的实打实占比。**prefill** kernel 一次处理多 token（大、compute-bound），故 $N\tau \ll T_{\text{compute}}$，启动藏在真实工作的阴影里。

??? question "CUDA graph 去掉什么，作为交换又需要什么？"
    它去掉**每 kernel 的 CPU 启动/派发开销**：一张捕获的图用单次提交重放全部 $N$ 个 kernel，把 $N\tau$ 塌缩到 $\approx\tau$。作为交换它需要**静态形状与固定内存地址**（故 vLLM 按 batch-size 桶各捕一张图并补齐）、每步输入拷进同一组静态 buffer、以及捕获前的**预热**。它还为捕获的 buffer 花 VRAM——这就是 `enforce_eager=True` 存在来禁用它的原因。

??? question "你把模型权重量化，decode 计算从每步 15 ms 降到 6 ms。CUDA graph 现在更重要还是更不重要？"
    **更重要。** 启动税 $N\tau$（约 ~2 ms）是固定的，于是它从 15 ms 步的 ~14% 变成 6 ms 步的 ~36%——你缩小计算时相对开销反而涨了。优化 GPU 工作让固定启动开销更突出，所以 CUDA graph（与 fusion）在量化、小 batch 的 decode 上带来更大的相对提速——正是你在单张 4090 上跑的场景。
