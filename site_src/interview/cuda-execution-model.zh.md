# CUDA 执行模型：warp、SIMT 与 occupancy

!!! info "基线：**vLLM 0.26.0** · `torch.cuda` 设备查询 API 经 Context7 核实（ADR-0004）"

**模块：** Part 3 · GPU 编程（Triton）   ·   **对应课程：** [CUDA 执行模型：线程、Warp 与 Occupancy](../part3/cuda-execution-model.md)

---

## Q：带我走一遍 GPU 到底怎么跑一个 kernel。什么是 warp，SIMT divergence 花什么代价，SM 怎么隐藏访存时延，以及拉满 occupancy 是否总让 kernel 更快？

### 直接答案

一个 kernel 启动一个 **grid 的 block**；每个 block 被指派给**一个 SM**，并被切成**每 32 线程一个 warp**。warp 是真正的执行单位：scheduler **每周期为全部 32 条 lane 发射一条指令**——这就是 SIMT（Single Instruction, Multiple Threads）。

- **Divergence**：若一个 warp 的 32 条 lane 走了数据相关分支的*不同*侧，warp 会**串行两条路径**（跑每一侧时屏蔽掉其他 lane）——大致是 $T_{if}+T_{else}$ 而非一侧。代价按 warp 算：若 32 条 lane 全体一致，就没有惩罚。不同 warp 之间 divergence 免费。
- **时延隐藏**：一次 HBM load 要几百周期。SM 驻留许多 warp，某个 stall 时零代价切到就绪 warp（所有驻留 warp 的寄存器都活着）。时延被*隐藏*，从不被移除。
- **Occupancy** = 驻留 warp / 每 SM 最大 warp（compute capability 8.9 上是 48）。它是使时延隐藏成为可能的*余量*——受最先耗尽的每-SM 资源约束：寄存器/线程、shared-mem/block，或 block 数上限。

拉满 occupancy 总有帮助吗？**否。** 你要*足够*的 occupancy 去隐藏时延，之后再多毫无收益——而硬凑（如砍寄存器）可能引发溢出反而有害。一个 kernel 也可以在 100% occupancy 下仍 memory-bound。

### 深入原理

- **为什么到处是 32。** 32 线程的 warp 是硬件常量（NVIDIA 上）。启动一个 40 线程的 block，你仍占用两个 warp——第二个里 24 条 lane 闲坐。block 大小取 32 的倍数以免浪费 lane。
- **occupancy 是资源受限的上限。** 在每 SM 65,536 寄存器堆（cc 8.9）下，一个用 64 寄存器/线程的 kernel 上限是 $65536/64 = 1024$ 线程 = 32 warp → ≤67% occupancy。shared memory 对共驻 block 做同样的事。CUDA occupancy 计算器不过是解那个约束的绑定项。
- **「足够」取决于负载。** memory-bound kernel 需要更多驻留 warp（更多 stall 要隐藏）；compute-bound kernel 在更低 occupancy 就饱和。这就是为什么盲目最大化 occupancy 是错的目标——[roofline](../part2/roofline-analysis.md) 告诉你你在哪个状态。
- **连到 LLM 推理。** batch 1 的 decode 启动的是填不满 SM 的小 kernel——warp 太少隐藏不了时延，于是 GPU 闲着。那份利用不足正是 [continuous batching](../part5/index.md)（把序列打包 → 一次胖启动）与 [CUDA graphs](cuda-graphs-fusion.md)（干掉每次启动的开销）的全部动机。

### 代码

SIMT divergence 规则的纯 CPU 模型——代价按 warp 算，且只有被劈开的 warp 付双倍：

```python
WARP = 32
def branch_bodies(conditions):                      # conditions[i]：lane i 是否走 'if'？
    bodies = 0
    for s in range(0, len(conditions), WARP):
        warp = conditions[s:s + WARP]
        bodies += 1 if (all(warp) or not any(warp)) else 2   # 一致：一侧；divergent：两侧
    return bodies

n = 32 * 8                                           # 8 个 warp
interleaved = [i % 2 == 0 for i in range(n)]         # 每个 warp 都被劈开 -> 16 (2.0x)
aligned     = [(i // WARP) % 2 == 0 for i in range(n)]  # 每个 warp 一致 -> 8 (1.0x)
print(branch_bodies(interleaved), branch_bodies(aligned))   # 16 8
```

同样的 50/50 工作量；interleaved 布局纯粹因为分支劈开了每个 warp 而花 2×。

### 面试官追问

- *「为什么启动配置必须是 32 的倍数？」* → 硬件不管怎样都把 block 切成 32 线程的 warp；非倍数会在最后那个填不满的 warp 里浪费 lane（它们被屏蔽但仍占一个 warp 槽）。
- *「你会怎么修一个因 divergence 而慢的 kernel？」* → 让 warp 内部一致——把数据排序/分桶，使一个 warp 里的 lane 走同一分支，或重构使分支对齐到 32 线程边界。这个税来自一个 warp *内*的 lane 分歧。
- *「occupancy 从 50% 涨到 100% 但运行时间没变——为什么？」* → 它本已 memory-bandwidth-bound（或 50% 已足够隐藏时延）。occupancy 是隐藏 stall 的余量，不是吞吐倍增器；查 roofline / achieved 带宽。
- *「什么限制 occupancy？」* → 绑定的那项每-SM 资源：寄存器/线程、shared-memory/block、线程/block，或驻留 block 上限——最先耗尽的那个。
- *「`__syncthreads()` 是全局 barrier 吗？」* → 否——它只同步*一个 block* 内的线程。没有便宜的 kernel 内跨 grid barrier；全局同步意味着一次新的 kernel 启动。

### 关联概念

- 课程：[CUDA 执行模型：线程、Warp 与 Occupancy](../part3/cuda-execution-model.md)
- 相关：[Memory coalescing、shared memory 与 bank conflict](memory-coalescing.md)（这些 warp 应怎样触碰内存）、[GPU 内存层级与 roofline](gpu-memory-hierarchy.md)（SM/warp/HBM 层级）、[CUDA graphs 与 kernel fusion](cuda-graphs-fusion.md)（为什么微小的 decode 启动填不满 GPU）
- 术语表：[SM / Warp / Occupancy](../glossary.md)
