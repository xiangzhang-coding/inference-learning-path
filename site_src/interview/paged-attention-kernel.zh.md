# PagedAttention kernel：paged KV cache 与 block table

!!! info "基线：**vLLM 0.26.0** · kernel 布局与 `paged_attn.py` 经 Context7 核实（ADR-0004）"

**模块：** Part 3 · GPU 编程（Triton）   ·   **对应课程：** [读 vLLM 的 PagedAttention kernel](../part3/paged-attention-kernel.md)

---

## Q：讲讲 PagedAttention。为何把 KV cache 存成块、block table 做什么、attention 时 kernel 如何 gather KV，以及结果和稠密 attention 有区别吗？

### 直接答案

PagedAttention 把**虚拟内存**用到 [KV cache](../part0/kv-cache.md) 上。朴素引擎为每条序列预留一块按最大长度定尺的**连续** KV 区域——于是多数一直空着（内部碎片），限制并发。PagedAttention 改为：

- **把 KV 存成固定大小的块**（页，每块 ~16 token）在一个共享池里。序列一次长一块（近乎零浪费）、块住在任意处、相同的块能跨序列**共享**（prefix caching）。
- **保存一张按序列的 block table**，映射逻辑块索引 → 物理块号——即 KV 的页表。`slot_mapping` 是写侧等价物（token → 扁平物理槽）。
- **kernel 做 gather**：它走序列的 block table，load 每个物理块的 K/V、算那块的 $QK^\top$ scores、折进一个运行的 **online-softmax** 累加器（FlashAttention 技巧，按块）。

结果和稠密 attention 有区别吗？**没有**——到机器精度一致。paging 改变 KV *住在哪*、kernel *怎么够到它*，从不改变 attention *算什么*。收益是内存效率（无碎片、块共享），而非不同的输出。

### 深入原理

- **块为何干掉碎片。** 连续预留每条序列浪费 `max_len − actual_len`；块分配每条序列最多浪费一个部分块。省回的 VRAM 变成更多 KV cache → 更高并发（连到 [VRAM 预算](vram-capacity-planning.md)）。
- **K-cache 的多出一维。** vLLM 把 K cache 布成 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，其中 `x = 16 // element_size`（FP16 下 8）。尾部的 `x` 把 `head_size` 打包成 16 字节对齐的块，好让 kernel 的 load [合并](memory-coalescing.md)；V cache（`[num_blocks, num_kv_heads, head_size, block_size]`）沿另一个轴读、不需要这种打包。
- **v1 vs v2（分区）。** 一个 `PARTITION_SIZE` 模板参数把长 KV 序列切成并行计算、事后合并的分区——即输出里的 `max_num_partitions` 维。短上下文走不分区路径；长的走分区路径以抬高 occupancy（和 FlashDecoding 同思路）。
- **写与读的对称。** `write_to_paged_cache` → `ops.reshape_and_cache(..., slot_mapping, ...)` 把新 token 散射到物理槽；attention kernel 经 block table 把它们 gather 回来。一层间接，两个方向。

### 代码

paged attention 作为纯 Python 模型——block-table 走 + online-softmax 折，证明与稠密相等：

```python
import math
BLOCK = 4                                          # 每块 KV token 数（vLLM：16）

def paged_attention(q, block_table, k_pool, v_pool, seq_len):
    d = len(q); m, l, acc = -math.inf, 0.0, [0.0]*d; pos = 0
    for phys in block_table:                        # 逻辑 -> 物理块 id
        for t in range(BLOCK):
            if pos >= seq_len: break                # 最后一块可能是部分
            k, v = k_pool[phys][t], v_pool[phys][t]
            s = sum(qi*ki for qi, ki in zip(q, k)) / math.sqrt(d)
            m_new = max(m, s); corr = math.exp(m - m_new) if m != -math.inf else 0.0
            p = math.exp(s - m_new); l = l*corr + p
            acc = [acc[j]*corr + p*v[j] for j in range(d)]; m = m_new; pos += 1
    return [a/l for a in acc]                        # == 稠密 attention，到机器精度
```

`block_table = [3, 1]`（物理块乱序）仍精确复现稠密 attention——物理放置任意，表还原逻辑顺序。

### 面试官追问

- *「和操作系统的类比是什么？」* → 虚拟内存 / 分页：block table 是把逻辑（虚拟）KV 块映到共享池物理块的页表；增长 = 分配一页，共享 = 两张页表指向同一物理页。
- *「prefix caching 怎么从这里掉出来？」* → 若两条序列共享 prompt 前缀，它们的 block table 对那段前缀指向**同一批物理块**——只算、只存一次。块是共享单位。
- *「paging 相对连续 KV 的代价是什么？」* → kernel 必须逐块 **gather** KV（经 block table 间接）而非一次连续读，加上 block-table 记账。那是消除碎片、使能共享的价钱——而 gather 正是自定义 kernel 优化的东西。
- *「K cache 为何是那个怪形状 `head_size/x … x`？」* → 为了让 load 合并：`x` 把元素打包成 16 字节对齐块，好让每个线程读一个对齐向量。是内存布局优化，非语义。
- *「PagedAttention 改变 attention 输出吗？」* → 不——到机器精度与稠密一致。它纯是内存管理方案；kernel 仍用 online softmax 算标准 attention。

### 关联概念

- 课程：[读 vLLM 的 PagedAttention kernel](../part3/paged-attention-kernel.md)
- 相关：[KV 缓存与吞吐上限](kv-cache.md)（被分页的东西）、[显存预算与最大并发](vram-capacity-planning.md)（paging 省回的碎片）、[FlashAttention 与 IO-aware attention](flash-attention.md)（online-softmax 折）、[Memory coalescing、shared memory 与 bank conflict](memory-coalescing.md)（K-cache 的 `x` 打包为何存在）
- 术语表：[PagedAttention、KV cache、Block table](../glossary.md)
