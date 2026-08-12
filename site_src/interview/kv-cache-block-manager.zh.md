# PagedAttention：KV cache 即虚拟内存（block manager & 碎片）

!!! info "基线：**vLLM 0.26.0** · V1 block-manager 内部经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [PagedAttention：像虚拟内存一样管理 KV Cache](../part5/paged-attention.md)

---

## Q：为什么连续 KV cache 会碎片？PagedAttention 的 block manager 怎么应对？池大小（`num_gpu_blocks`）怎么定？分页如何变成吞吐？

### 直接答案

朴素引擎把每个序列的 [KV cache](../part0/kv-cache.md) 存成一块按**最大**长度开好的**连续**区域——它必须这样，因为 kernel 把它当一段读、且生成长度事先未知。所以一个可能到 512 token 的请求就算只吐 40 个也预留满 512：**内部碎片**。已完序列留下的可变大小洞再添**外部碎片**。

PagedAttention 应用**虚拟内存**：把 KV cache 切成固定大小 **block**（页，16 token），放进共享**池子**，给每序列一张 **block table**（逻辑块 → 任意位置的物理块）。**block manager** 随序列增长一次分一块（浪费 ≤ 一个半满块）、结束时把块还回空闲表、并让有共同前缀的序列指向**同一**物理块。

**这如何变成吞吐：** 消除碎片意味着同样 VRAM 装下多得多的序列 → [continuous batch](../part5/continuous-batching.md) 更大 → memory-bound 的 decode 摊薄更好。分页不加速计算（[kernel](../part3/paged-attention-kernel.md) 算出相同 attention）；它抬高**容量**，而容量就是并发天花板。

### 深入原理

- **`num_gpu_blocks` 启动时 profiling。** vLLM 取 `gpu_memory_utilization × VRAM`（默认 **0.92**）作预算，减去非 KV 内存（权重 + 峰值激活 + CUDA-graph），余下除以每块字节数：$\texttt{num\_gpu\_blocks}=\lfloor(\texttt{util}\cdot\text{VRAM}-\text{weights}-\text{act}-\text{cudagraph})/\text{每块字节}\rfloor$。这就是量化（权重更小）与 FP8 KV cache（每块字节更小）提并发的原因。
- **V1 结构。** `BlockPool` 持有 `KVCacheBlock`；`free_block_queue`（双向链表，驱逐顺序）给 O(1) 分配（弹头部）/ 释放（推尾部）；`cached_block_hash_to_block` 映射支撑 prefix caching；每块有 `ref_cnt`。按请求的 `SingleTypeKVCacheManager`（`req_to_blocks`）从共享池取。
- **生命周期。** 准入 → 为 prompt 弹空闲块；decode → 最后一块满时再弹一个；结束 → 把块推回（`ref_cnt` 减一，到 0 释放）；有压力 → 抢占（驱逐/重算），相当于 OS 页换出。
- **共享 → prefix caching。** 块由其 token 哈希（+ 父哈希）为键。相同前缀 → 相同哈希 → 第二个请求的 table 指向已存在的块（`touch()` 加 `ref_cnt`）；只有**整块**缓存，KV 逐字节相同故**输出不变**；分叉触发 copy-on-write。（调优是 Part 5 下一话题。）
- **这是 kernel 那课的另一半。** [Part 3](../part3/paged-attention-kernel.md) 是 kernel 如何 *gather* 散落块；这里是 manager 如何 *分配* 它们。

### 代码

碎片论证写成算术（纯 Python）——同一池子、两种策略：

```python
import math
BLOCK, POOL, MAX_LEN = 16, 128, 512
lens = [40,128,300,64,210,96,180,48,150,80,60,420,33,256,90,110]

reserve = math.ceil(MAX_LEN/BLOCK)                       # 连续：每序列恒 32 块
contig  = POOL // reserve                                 # -> 装下 4 个序列
used = paged = 0
for L in lens:                                            # 分页：每个 ceil(actual/BLOCK)
    need = math.ceil(L/BLOCK)
    if used + need <= POOL: used += need; paged += 1
print(contig, "vs", paged)   # 4 vs 13 —— 同样 VRAM 3 倍多并发，计算不变
```

### 面试官追问

- *「OS 类比具体是什么？」* → 虚拟内存 / 按需分页：block table 是页表（逻辑→物理），池子是物理 RAM，append 时分配是按需分页，抢占是页换出，前缀共享是两张页表映射同一物理页。
- *「KV block vs 线程 block？」* → 无关。KV block = cache 池的 16-token 页；线程 block = CUDA 调度单位。同词。
- *「分页改变输出或加速 kernel 吗？」* → 都不（输出与 dense 相同，[Part 3](../part3/paged-attention-kernel.md) 已证；也不加速计算）。收益是容量 → 更大批 → 吞吐。
- *「KV-bound 时怎么抬高并发天花板？」* → 增大 `num_gpu_blocks`：调高 `gpu_memory_utilization`（谨慎）、量化权重（腾预算）、量化 KV 到 FP8（每块字节减半）。三者都对应公式里的项。
- *「为什么 16-token 块——不是 1 或 256？」* → 权衡：块大 = 分配更粗（半满块浪费更多）但 block-table 条目更少、记账更便宜；块小 = 浪费少、开销大。16 是调好的默认。

### 关联概念

- 课程：[PagedAttention：像虚拟内存一样管理 KV Cache](../part5/paged-attention.md)
- 相关：[Static vs continuous batching](continuous-batching.md)（这份容量喂养的批）、[PagedAttention kernel & block table](paged-attention-kernel.md)（gather 侧）、[KV 缓存与吞吐上限](kv-cache.md)、[显存预算与最大并发](vram-capacity-planning.md)
- 术语：[PagedAttention、KV cache、Block table](../glossary.md)
