# Memory coalescing、shared memory 与 bank conflict

!!! info "基线：**vLLM 0.26.0** · `torch.cuda` 计时 API + 张量连续性语义经 Context7 核实（ADR-0004）"

**模块：** Part 3 · GPU 编程（Triton）   ·   **对应课程：** [访存：Coalescing、Shared Memory 与 Bank Conflict](../part3/memory-access.md)

---

## Q：对一个 memory-bound kernel，一个 warp 怎样触碰内存比你跑多少 warp 更要紧。解释 memory coalescing、uncoalesced 访问的代价、shared memory 与 bank conflict 是什么——并连到 FlashAttention 为什么快。

### 直接答案

内存以固定块搬运（**32 字节 sector**，128 字节 line）。一个 warp 的 load 由它 32 个 lane 地址触及的不同事务数来服务：

- **Coalesced**：lane $k$ 读字 $k$（连续）→ 32×4 B = 128 B 落在一条 line → **1 次事务，~100% 有用**。
- **Uncoalesced**：大步长把 lane 散进各自的 sector → 多达 **32 次事务**，每次大半浪费 → 效率低到 ~1/8–1/32，有效带宽按同因子下降。同一条指令，多达 32× 的 HBM 流量。常见成因：把一个 row-major 数组**沿列**读。

**Shared memory** 是每个 SM 上一小块快速、*由程序管理*的 SRAM 便签。它的活儿是**复用**：从 HBM 装入一个 tile 一次，再廉价地读许多次——把 HBM 字节按 ~复用倍数削减、抬高[算术强度](arithmetic-intensity.md)。

**Bank conflict** 是 shared-memory 的坑：shared memory 有 **32 个 bank**（字 $w$ → bank $w \bmod 32$）。若 32 条 lane 命中 32 个不同 bank 则无冲突；若 $n$ 条 lane 命中*同一* bank 里*不同*的字，那些会串行（$n$-way 冲突）。

**FlashAttention** 快正因如此：它把 Q/K/V tile 进 shared memory / 寄存器，把 $S\times S$ scores 留在片上，于是搬 $O(S)$ 的 HBM 字节而非 $O(S^2)$——一个纯粹的复用/coalescing 胜利，FLOPs 相同。

### 深入原理

- **Coalescing 是按 warp、按指令的**——不是随时间的缓存效应。它关乎*这个* warp 的 32 个同时地址是否落在少数事务里。修法：用 `threadIdx.x` 索引**变化最快**的轴，使相邻 lane 命中相邻地址。
- **列访问是经典 bug。** row-major $(r,c)$ 在偏移 $r\cdot W+c$：走列（固定 $c$、变 $r$）把 lane 隔开整整一行——多达 32× 流量。走行是连续的。
- **shared memory 不免费。** 只有当复用摊薄那次一次性 HBM 装入*且*无冲突时才划算。一次性中转拷贝纯是开销。
- **padding 技巧。** 一个 32 宽 shared tile 的列访问把所有 lane 送进 bank 0（32-way 冲突）。把 tile 声明为 `[N][33]` 让列走相隔 33 的地址；$33 \bmod 32 = 1$ → 32 个不同 bank → 无冲突，代价一列。
- **broadcast ≠ 冲突。** 全部 32 条 lane 读*同一个*字免费（硬件广播）。冲突是 lane 命中*同一* bank 里*不同*的字。

### 代码

两种隐患作为纯寻址算术（无 GPU）：每 warp 搬的 sector 数，与每个 shared-memory bank 的最差 lane 数。

```python
WARP, SECTOR_B, DTYPE_B, BANKS = 32, 32, 4, 32
def sectors(stride):                                 # 一个 warp 触及的不同 32 B sector
    return len({((k * stride) * DTYPE_B) // SECTOR_B for k in range(WARP)})
def max_per_bank(stride):                            # 最差 bank 的 lane 数（1 = 无冲突）
    c = {}
    for k in range(WARP):
        b = (k * stride) % BANKS; c[b] = c.get(b, 0) + 1
    return max(c.values())

for s in (1, 8, 32):                                 # coalescing：4、32、32 sector
    print(f"stride {s:>2}: {sectors(s):>2} sectors, {max_per_bank(s):>2}-way bank")
# stride  1:  4 sectors,  1-way bank   (理想)
# stride  8: 32 sectors,  8-way ... -> uncoalesced + 有冲突
# stride 32: 32 sectors, 32-way bank  (最坏情形)
```

### 面试官追问

- *「你怎么让一次访问 coalesced？」* → 确保相邻线程（lane）读相邻地址——用 `threadIdx.x` 索引变化最快（最内、单位步长）的维。对 row-major 矩阵，走行别走列。
- *「什么时候 shared memory *不*值得？」* → 数据只用一次（没有复用去摊薄中转装入），或访存模式重度 bank 冲突（串行化的读抹掉了收益）。
- *「一个 shared tile 上的 32-way bank 冲突——一句话修法？」* → 把内维 padding 一列（`[N][33]`）打破 2 的幂周期性，使一列命中 32 个不同 bank。
- *「coalescing 对 LLM decode 要紧吗？」* → 要——decode 是 memory-bound（每步重读 KV cache），所以有效带宽 = coalescing 质量直接封顶吞吐。这就是为什么 KV-cache 布局与 attention kernel 死磕连续、coalesced 的读。
- *「你把一个转置张量喂进自定义 kernel 结果慢了——为什么？」* → 转置视图是非连续的；kernel 现在步长地（沿列）读、uncoalesced。先调 `.contiguous()`（若复用值回拷贝），或把 kernel 写成走连续轴。

### 关联概念

- 课程：[访存：Coalescing、Shared Memory 与 Bank Conflict](../part3/memory-access.md)
- 相关：[CUDA 执行模型：warp、SIMT 与 occupancy](cuda-execution-model.md)（做访问的那些 warp）、[FlashAttention 与 IO-aware attention](flash-attention.md)（shared-memory tiling 实战）、[GPU 内存层级与 roofline](gpu-memory-hierarchy.md)（HBM/SRAM 层级）、[GEMM 与 attention 的算术强度](arithmetic-intensity.md)（复用抬高强度）
- 术语表：[Coalescing / Shared memory / Bank conflict](../glossary.md)
