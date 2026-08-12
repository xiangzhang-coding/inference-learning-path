# Triton 编程模型

!!! info "基线：**vLLM 0.26.0** · Triton API 经 Context7 核实（ADR-0004）"

**模块：** Part 3 · GPU 编程（Triton）   ·   **对应课程：** [Triton：写你的第一个 GPU kernel](../part3/triton-basics.md)

---

## Q：讲讲 Triton 的编程模型。一个 Triton「program」映射到什么、`program_id` / offset / mask 怎么工作、matmul 为何在 FP32 里累加，以及何时选 Triton 而非 PyTorch 或 CUDA C++？

### 直接答案

Triton 是一门嵌在 Python 里的 GPU kernel 语言 + 编译器。你从**一个 program**——一块工作，类比 CUDA *block*——的视角写，用类 NumPy 的操作作用在指针上；编译器处理块内线程、warp 分派、访存合并与 shared-memory 中转。

- **`tl.program_id(axis)`** 返回这个 program 在启动 grid 里的索引。你启动 `triton.cdiv(n, BLOCK)` 个 program；program `pid` 拥有切片 `[pid·BLOCK, (pid+1)·BLOCK)`。
- **offset** 是*向量*：`offs = pid*BLOCK + tl.arange(0, BLOCK)`。`tl.load(ptr + offs)` / `tl.store` 一次搬一整块；编译器合并连续地址。
- **mask** 守不齐边界：`mask = offs < n` 在 load/store 时禁用越界 lane，`other=` 提供填充（求和 `0`、求最大 `-inf`）——一个 kernel 处理任意大小，无余数路径。
- **FP32 累加**：matmul 把 `K` 个乘积相加；在 FP16 里累加会让舍入误差沿 K-loop 累积。用 `tl.zeros(..., dtype=tl.float32)`、最后转型一次——和 tensor core 低精度相乘、FP32 累加同理。

**何时用 Triton：** **融合库不提供的自定义算子**（attention 变体、fused norm+激活、量化 matmul）——你想要比组合 PyTorch 算子（每步之间往返 HBM）更快，但又不想付手写 CUDA C++ 的代价。不用于重造 GEMM（cuBLAS 更强），也不用于 PyTorch 已很好融合的算子。

### 深入原理

- **`tl.constexpr` 是编译期的。** `BLOCK_SIZE: tl.constexpr` 被烘进编译出的 kernel，好让它给数组定形、展开循环；改它就重编译。这是 `@triton.autotune` 按问题形状搜索的旋钮（`BLOCK_*`、`num_warps`、`num_stages`）。
- **你保留 CUDA 心智模型，丢掉琐事。** occupancy、coalescing、bank 冲突仍解释一个 Triton kernel *为何*快——但你表达一个块向量、让编译器摆 warp，而非写逐线程代码。
- **`tl.dot` 瞄准 tensor core。** tiled matmul load `BLOCK_M×BLOCK_K` 与 `BLOCK_K×BLOCK_N` 子 tile、`tl.dot` 进累加器——片上复用（shared memory）由编译器管理。
- **融合 = 更少 HBM 往返。** fused-softmax kernel load 一行一次、片上做 max/exp/sum、write 一次——对比三个 PyTorch 算子各自把中间量往返 HBM。和 FlashAttention 放大的同一个收益。
- **`tl.exp` 是快近似**（像 CUDA `__expf`）——对 softmax 没问题，若你需要 bit 级精确要知道。

### 代码

vector-add kernel —— 整个模型八行：

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # grid 里的 block 索引
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 一个索引 VECTOR
    mask = offs < n                              # 守住不齐的最后一个 block
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# 启动：grid = ceil(n / BLOCK) 个 program
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

### 面试官追问

- *「一个 Triton program 对应 CUDA 里的什么？」* → 一个 block（CTA），不是线程。`tl.program_id` ≈ `blockIdx`；你对块的整块数据向量操作、从不写逐线程代码。
- *「为什么需要 `mask`？」* → 数据大小通常不是 `BLOCK` 的整数倍，最后一个 program 的偏移向量会越过数组；mask 在 load/store 时禁用那些 lane，无需单独的余数 kernel 就避免越界。
- *「你的 Triton matmul 比 `torch.matmul` 慢——是 bug 吗？」* → 通常不是——cuBLAS 极度 tune 过。Triton 的回报在融合库不提供的算子，而非赢 GEMM。也确认你计时的是*第二次*调用（第一次 JIT 编译）且 autotuner 已为该形状跑过。
- *「Triton 怎么选 block 大小 / warp？」* → 它不自动选——你列 `triton.Config`，`@triton.autotune` 按 `(M,N,K)` 键 benchmark 它们、缓存赢家。`constexpr` 参数使那个搜索成为可能。
- *「Triton 能在 CPU 上跑吗？」* → 没有生产级 CPU 后端——它需要 GPU（CUDA，或 AMD ROCm/HIP）来编译与启动。

### 关联概念

- 课程：[Triton：写你的第一个 GPU kernel](../part3/triton-basics.md)
- 相关：[CUDA 执行模型：warp、SIMT 与 occupancy](cuda-execution-model.md)（编译器替你摆的东西）、[Memory coalescing、shared memory 与 bank conflict](memory-coalescing.md)（为何块 load 快）、[PagedAttention kernel](paged-attention-kernel.md)（一个可读的真实 kernel）、[FlashAttention 与 IO-aware attention](flash-attention.md)（大尺度的融合）
- 术语表：[Triton、SM / Warp / Occupancy](../glossary.md)
