# Triton：写你的第一个 GPU kernel

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）· Triton（随近期 PyTorch 一起分发）"
    这里用到的全部 Triton API——`@triton.jit`、`tl.program_id`、`tl.arange`、带 `mask`/`other` 的 `tl.load`/`tl.store`、`tl.constexpr`、`triton.cdiv`、grid-lambda 启动、`tl.max`/`tl.sum`/`tl.exp`、`tl.dot`、`tl.zeros`、以及 `@triton.autotune` / `triton.Config`——均经 Context7 对照官方 Triton 教程核实（ADR-0004）。任何吞吐数字均为**示例 / 量级参考**；正确性通过把每个 kernel 的输出与 PyTorch 参考对比（`torch.allclose`）来验证，精确到浮点误差。

---

## 1 · 直觉 & 为什么重要

[执行模型](cuda-execution-model.md) 与 [访存](memory-access.md) 两节课给了你 kernel 为何快的*道理*——warp、occupancy、coalescing。Triton 是你终于*动手写*一个的地方。要点（按 ADR-0002）不是要成为 kernel 作者；而是写过几个之后带来的底气：你会把 vLLM 的 Triton kernel 当作自己也写得出来的代码来读，而非天书。

让 Triton 成为「调 PyTorch 算子」与「手写 CUDA C++」之间正确一档的，是这一点：**你写的是对整块数据做操作的 Python，编译器替你处理块内的线程编排、访存合并与 shared memory 中转。** 在 CUDA C++ 里你从*单个线程*的视角写代码，手工编排 32-lane 的 warp、shared-memory tile、无 bank 冲突的布局。在 Triton 里你从*单个 program*（一块工作）的视角、用类 NumPy 的数组操作作用在指针上；Triton 把它降级为一个高效的 SIMT kernel。你保留前两课的心智模型——它们告诉你*为什么*你的 Triton kernel 快或慢——但从逐线程的琐事里解放出来。这个取舍就是为什么 FlashAttention、许多 vLLM kernel、以及大多数现代融合算子实际上是这么写的。→ 术语 *Triton、SM / Warp / Occupancy、Coalescing* 见[术语表](../glossary.md)。

## 2 · 心智模型

Triton 的单位是 **program**（你 kernel 的一个实例，像一个 CUDA *block*），由 `tl.program_id` 标识。你启动一个 **grid** 的 program；每个处理输出的一个 tile。

```text
CUDA C++  : 你写 ONE THREAD 的视角          Triton   : 你写 ONE PROGRAM 的视角（一个 block）
  idx = blockIdx.x*blockDim.x + threadIdx.x     pid    = tl.program_id(0)
  if (idx < n) out[idx] = x[idx] + y[idx];      offs   = pid*BLOCK + tl.arange(0, BLOCK)   # 一个 VECTOR
  // 你手工管理 32-lane warp、shared mem、       mask   = offs < n
  // coalescing、bank 冲突                       x = tl.load(x_ptr+offs, mask=mask)         # 整块
                                                 tl.store(out_ptr+offs, x+y, mask=mask)
                                                 # 编译器选 warp、合并访存、中转 SRAM

一个 GRID 的 program 铺在数据上：
   data:  [ block 0 ][ block 1 ][ block 2 ] ... [ block G-1 ]
   pid:       0          1          2      ...      G-1        G = triton.cdiv(n, BLOCK)
```

三个要抓住的形状：

- **你按块（向量）思考，而非标量。** `tl.arange(0, BLOCK)` 造出一个偏移向量；`tl.load`/`tl.store` 一次搬一整块。编译器把这块映到 warp 上并合并访问——那些你在 CUDA 里要手工雕的 coalescing，Triton 从连续的偏移模式里自动做到。
- **`mask` 是你处理不齐边界的方式。** 当数据大小不是 `BLOCK` 的整数倍时，最后一个 program 的块会越过末尾；`mask = offs < n` 在 load 与 store 时守住越界的 lane（`other=` 为被屏蔽的 load 提供填充值）。
- **`tl.constexpr` 值是编译期的。** `BLOCK_SIZE: tl.constexpr` 被烘进编译出的 kernel（好让它给数组定形、展开循环）；改它就重编译。这正是 `@triton.autotune` 用来搜索 block 大小的旋钮。

## 3 · 原理与数学

### 3.1 SPMD 启动

Triton 是 **SPMD**（Single Program, Multiple Data）：同一个 kernel 体跨一个 grid 的 program 运行，每个由 `tl.program_id(axis)` 标识。对一个 `n` 元素、block 大小 `B` 的一维问题，你启动 `G = ⌈n/B⌉` 个 program（`triton.cdiv(n, B)`）；program `pid` 拥有元素 `[pid·B, (pid+1)·B)`。没有显式的线程循环——向量宽度 `B` *就是*一个 program 表达的并行度，编译器把它拆到 warp 上。

### 3.2 指针、偏移与 mask

一个 `torch.Tensor` 参数以**指向其首元素的指针**到达。你用 `base_ptr + tl.arange(0, B)`（二维再加行/列 stride）算出一个*向量*的地址，然后 `tl.load(ptrs, mask, other)` 取它们。mask $\text{offs} < n$ 是边界守卫；在被屏蔽的 lane 上，load 返回 `other`（如求和用 `0`、求最大用 `-inf`），store 被抑制。正确的 masking 就是让一个 kernel 无需单独的余数路径就能处理任意大小。

### 3.3 归约与 online-softmax 回声

**fused softmax** kernel 把一整行 load 进一个块，再沿它归约：`tl.max(row, axis=0)` 求稳定性用的最大值、`tl.exp`、`tl.sum(..., axis=0)` 求归一化因子——整行的 softmax 在片上一个 kernel 内算完，于是 $O(\text{rows}\times\text{cols})$ 的中间量绝不往返 HBM。那种「load 一次、片上归约、write 一次」正是 [FlashAttention](../part2/flash-attention.md) 用 online softmax 推广的同一个 IO-aware 动作。（Triton 的 `tl.exp` 是快但近似的，像 CUDA 的 `__expf`——对 softmax 没问题，值得知道。）

### 3.4 `tl.dot` 与 matmul 累加器

一个 tiled **matmul** program 拥有一个 `BLOCK_M × BLOCK_N` 的输出 tile。它以 `BLOCK_K` 为步走 K 维，load 一个 $M\times K$ 与一个 $K\times N$ 的子 tile、用 `tl.dot(a, b)` 相乘（它瞄准 [tensor core](../part0/gpu-hardware.md)），累加进一个 `float32` 寄存器 tile——`acc += tl.dot(a, b)`——最后转型并 store 一次。即使输入是 FP16 也在 `float32` 里累加，是标准的精度守卫。`@triton.autotune` 随后搜索 `BLOCK_M/N/K`、`num_warps`、`num_stages`，为每个问题形状找最快的配置——那些你本来要手调的东西。

## 4 · 完整可跑代码 + 逐行讲解

issue 点名的三 kernel 递进：**vector add → fused softmax → simple matmul**，每个都与 PyTorch 对比校验。

!!! gpu "GPU Lab —— Triton 需要 GPU"
    - **最低显存：** 2 GB（小测试张量；不加载模型）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）——对齐基线
    - **预估耗时 / 花费：** ~5 分钟（首次运行含 JIT 编译）· ~¥0.5 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）。**Triton 不在 CPU 上跑**——它需要一块 GPU 来编译与启动。
    - **非 NVIDIA：** Triton 有 AMD **ROCm/HIP** 后端（`triton.runtime.driver.active.get_current_target().backend == "hip"`）；同样的 kernel 能编译，但 autotune 的配置与 `num_warps`（AMD wavefront = 64）不同。无 TPU/Neuron 后端。

**Kernel 1 —— vector add**（「hello world」：grid、offset、mask、load/store）：

```python title="triton_vector_add.py"
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                     # 我是哪个 block？（一维 grid）
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 这个 block 的索引 VECTOR
    mask = offsets < n_elements                     # 守住不齐的最后一个 block
    x = tl.load(x_ptr + offsets, mask=mask)         # 合并 load 一整块
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)   # 带 mask 写回这块

def add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)   # G = ceil(n / BLOCK)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)            # 启动 grid
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(98_432, device="cuda"); y = torch.rand(98_432, device="cuda")  # 不是 1024 的整数倍
    out = add(x, y)
    print("max abs err vs torch:", (out - (x + y)).abs().max().item())            # ~0.0
```

**逐行讲解（kernel 1）：** `tl.program_id(0)` 是这个 program 在一维 grid 里的索引。`offsets` 是一个*向量*——`pid*BLOCK_SIZE` 移到这块的切片，`tl.arange(0, BLOCK_SIZE)` 铺满它。`mask = offsets < n_elements` 至关重要，因为 `98_432` 不是 `1024` 的整数倍，最后一个 program 的块会越过数组末尾；mask 在 load 与 store 时都抑制那些 lane。`tl.load`/`tl.store` 搬整块——编译器合并连续地址并分派 warp。启动器的 `grid` lambda 从可调的 `BLOCK_SIZE` 算出 program 数；索引 `add_kernel[grid](...)` 启动它。预期：`max abs err ~ 0.0`。

**Kernel 2 —— fused softmax**（一个 program 处理一行；片上归约）：

```python title="triton_softmax.py"
import torch, triton, triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                          # 一个 program 处理一行
    cols = tl.arange(0, BLOCK_SIZE)                 # BLOCK_SIZE >= n_cols（2 的幂）
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))  # 用 -inf 填充
    x = x - tl.max(x, axis=0)                       # 减去行最大值以求稳定
    num = tl.exp(x)                                 # 快但近似的 exp（像 __expf）
    y = num / tl.sum(num, axis=0)                   # 用行和归一化
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    block = 1
    while block < n_cols:                           # 大于等于 n_cols 的下一个 2 的幂
        block *= 2
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block)
    return out

if __name__ == "__main__":
    x = torch.randn(1823, 781, device="cuda")
    err = (softmax(x) - torch.softmax(x, axis=1)).abs().max().item()
    print("max abs err vs torch.softmax:", err)     # ~1e-6（快 exp 近似）
```

**逐行讲解（kernel 2）：** grid 是 `(n_rows,)`——一个 program 一行。`BLOCK_SIZE` 是大于等于 `n_cols` 的下一个 2 的幂，于是一整行落进一个块；`mask` + `other=-inf` 把尾部填掉，好让 max/sum 忽略它。`tl.max(x, axis=0)` 与 `tl.sum(..., axis=0)` 是对这块的**片上归约**——行从 HBM load 一次、在 SRAM/寄存器里 softmax、write 一次。这就是融合收益：没有单独的 max/exp/sum kernel，中间量不往返 HBM。

**Kernel 3 —— simple tiled matmul**（二维 grid、`tl.dot`、`float32` 累加器、autotune）：

```python title="triton_matmul.py"
import torch, triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],                            # 问题形状变了就重新 tune
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)                        # 二维 grid：这个 program 拥有 C 的 tile [pid_m, pid_n]
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)                         # 在 fp32 里累加
    for k in range(0, K, BLOCK_K):                                              # 走 K 维
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)                                                     # tensor-core matmul
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)                            # 转型 + write 一次

def matmul(a, b):
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c

if __name__ == "__main__":
    a = torch.randn(512, 768, device="cuda", dtype=torch.float16)
    b = torch.randn(768, 256, device="cuda", dtype=torch.float16)
    err = (matmul(a, b).float() - (a.float() @ b.float())).abs().max().item()
    print("max abs err vs torch matmul:", err)      # 小的 fp16 舍入，非算法误差
```

**逐行讲解（kernel 3）：** 一个**二维 grid**——`pid_m, pid_n` 命名这个 program 计算的输出 tile。`offs_m[:, None]` / `offs_k[None, :]` 把一维偏移向量广播成二维的指针块（stride 处理任意布局）。K-loop load 一个 `BLOCK_M×BLOCK_K` 与一个 `BLOCK_K×BLOCK_N` 子 tile、`tl.dot` 进一个 **`float32` 累加器**（即便输入是 FP16 也做精度守卫），每步把指针推进 `BLOCK_K`。循环后转成 FP16、在 M/N 边界带 mask store 一次。`@triton.autotune` 编译列出的每个 `triton.Config`，为给定的 `(M, N, K)` 挑最快的——那个你本来要手做的 block 大小 / `num_warps` / `num_stages` 搜索。

## 5 · Lab —— 验证 + 窥一眼 tuning

在 GPU 上跑上面三个 `__main__` 块：每个都打印出与其 PyTorch 参考近乎为零的误差——kernel 正确的证明。再观察 Triton 的编译与缓存行为以及 autotuner：

```python title="triton_lab_notes.py"
# 1) 每个 kernel 的第一次启动很慢（JIT 编译）；之后的启动命中缓存。
#    与 torch 比较时，计时第二次调用，而非第一次。
# 2) matmul autotuner 在新 (M,N,K) 键的第一次调用上 benchmark 每个 Config，
#    然后缓存赢家。改形状 -> 重新 tune。
# 3) 查看你在为哪个环境编译：
import triton
tgt = triton.runtime.driver.active.get_current_target()
print("backend:", tgt.backend)                      # NVIDIA 上是 'cuda'，AMD ROCm 上是 'hip'
```

**要观察什么：** 每个 kernel 的第一次调用付一次性 JIT 编译；与 PyTorch 公平比较时计时*第二次*。对 matmul，autotuner 在每个新 `(M, N, K)` 上跑一次配置扫描并缓存赢家——所以在固定形状上的 benchmark 循环从第一次迭代之后就用上了 tune 好的 kernel。`backend` 打印确认你在 CUDA 还是 ROCm/HIP（kernel 可移植，tune 出的配置不可移植）。这些 kernel 在裸 matmul 上都赢不了 cuBLAS/PyTorch——那不是重点；重点是你现在能*读懂并修改*这种形状的 kernel 了。

## 6 · 常见坑 / 反直觉点

- **忘了 `mask`。** 若数据大小不是 `BLOCK_SIZE` 的整数倍，没 mask 的 kernel 会越界读/写——垃圾数据或崩溃。mask（`offs < n`）不是可选的；`other=` 提供中性填充（求和 `0`、求最大 `-inf`）。
- **把 program 当线程。** `tl.program_id` 是*块*索引（像 CUDA `blockIdx`），不是线程。你在 Triton 里从不索引单个线程——你对整块向量操作、让编译器摆放 warp。
- **计时第一次启动。** 第一次调用会 JIT 编译；比稳态慢得多。总是先预热再计时——和 [CUDA graphs](../part2/kernel-fusion-cuda-graphs.md) 那课的 benchmark 提醒一样。
- **用输入 dtype 累加。** `tl.dot` 累加进 FP16 会很快丢精度；在 `float32` 里累加（`tl.zeros(..., dtype=tl.float32)`）、最后再转型。这呼应了 matmul 硬件为何在 FP32 里累加。
- **指望赢过 cuBLAS。** 教学 matmul 跑不过厂商库；Triton 的价值在于融合库不提供的自定义算子（如 attention），而非重造 GEMM。
- **`BLOCK_SIZE` 必须是 `constexpr`，且（softmax 那个）是大于等于行宽的 2 的幂。** 它是编译期形状；非 `constexpr` 的 block 大小编译不过，太小的 softmax block 会悄悄丢列。
- **Triton 需要 GPU。** 没有生产级 CPU 后端——没有 CUDA/ROCm 的机器连编译都做不到。在任意机器上读，在 GPU 上跑 Lab。

## 7 · 面试连线

- [Triton 编程模型](../interview/triton-programming.md) —— 这节课为你准备的高频题：*一个 Triton program 映射到什么、`program_id`/offset/mask 怎么工作、matmul 为何在 FP32 里累加，以及何时选 Triton 而非 PyTorch 或 CUDA C++？*

## 8 · 小结 & 延伸阅读

**一句话：** Triton 让你把 GPU kernel 写成对整块数据操作的 Python——你选 grid 与 block、算向量偏移、用 mask 守边界、在片上归约/`tl.dot`，而编译器处理 warp、coalescing 与 SRAM 中转；它是那个高产的中间一档（按 ADR-0002），让 vLLM 的 kernel 变得可读、可轻改。

延伸阅读：

- **官方 Triton 教程**——*Vector Addition*、*Fused Softmax*、*Matrix Multiplication*（这里的三个 kernel，更深入）与 *Fused Attention*（一个 Triton 版 FlashAttention）。
- [FlashAttention](../part2/flash-attention.md) 课——kernel-2 的归约所呼应的 online-softmax 思想。
- [CUDA 执行模型](cuda-execution-model.md) 与 [访存](memory-access.md) 课——block 大小、`num_warps`、合并 load 背后的*道理*。
- 下一节：[读 vLLM 的 PagedAttention kernel](paged-attention-kernel.md)——把这份读源码能力用到一个真实的生产 kernel 上。

## 9 · 自测小问

??? question "在 vector-add kernel 里，`tl.program_id(0)` 返回什么，为什么 `mask = offsets < n_elements` 是必要的？"
    `tl.program_id(0)` 返回这个 program 实例在一维启动 grid 里的索引——即*块*索引（类比 CUDA 的 `blockIdx.x`），不是线程索引。每个 program 拥有切片 `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)`。mask 是必要的，因为总大小通常不是 `BLOCK_SIZE` 的整数倍，最后一个 program 的偏移块会越过数组末尾；`mask = offsets < n_elements` 在 `tl.load` 与 `tl.store` 时都禁用越界 lane，无需单独的余数 kernel 就防止了非法访存。

??? question "fused softmax load 一行、做 `tl.max`/`tl.exp`/`tl.sum`、再 store——比调三个 PyTorch 算子的收益在哪？"
    融合。行从 HBM **load 一次**进一个块，减最大值、exp、求和归一化全在**片上**（SRAM/寄存器）发生，结果 **write 一次**。三个独立的 PyTorch 算子会各自把中间数组往返 HBM（写 exp 输出、读回来求和……）——对一个 memory-bound 的逐元素/归约模式，那些额外的 HBM 过程会主宰。把整行的 softmax 留在片上，正是 FlashAttention 用 online softmax 放大的同一个 IO-aware 原理。

??? question "为什么 matmul kernel 即便输入与输出都是 FP16，也在 `float32` 张量里累加？"
    为了保精度。matmul 把 `K` 个乘积相加；在 FP16 里累加会让舍入误差沿 K-loop 累积，可能明显破坏结果。在 `float32` 里累加（`tl.zeros(..., dtype=tl.float32)`）让运行和保持准确，最后 store 前才转成 FP16 一次。这呼应了 tensor-core 硬件本身也是低精度相乘、FP32 累加。
