# 读 vLLM 的 PagedAttention kernel

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB）"
    kernel 签名、paged KV-cache 布局（`k_cache`、`v_cache`）、`PagedAttention.split_kv_cache`、以及 `write_to_paged_cache` → `ops.reshape_and_cache(..., slot_mapping, ...)` 均引自 vLLM 源码及其 `docs/design/paged_attention.md`，经 Context7 核实（ADR-0004）。`block_tables` 参数与 `v1`/`v2` op 名在**概念层面**描述——确切的当前参数列表请读设计文档，它跨版本会变。本课建立*读源码*能力（ADR-0002）；服务侧的深挖（continuous batching、block manager、为何 paging 提吞吐）在 **Part 5**。

---

## 1 · 直觉 & 为什么重要

你已经写过 kernel（[Triton 基础](triton-basics.md)）；现在来读一个真正重要的。PagedAttention 是 vLLM 高吞吐背后的机制，「你能读 PagedAttention kernel 并讲清它做什么吗」是一道真实的资深 infra 面试探针。按 ADR-0002，目标正是这个：**读懂 + 理解生产 kernel**，而非自己写一个。

一个核心想法。朴素引擎把每条序列的 [KV cache](../part0/kv-cache.md) 存成一块按*最大*可能长度定尺的**连续**区域——于是一个可能到 4k token 的请求预留了 4k 份 KV，多数一直空着。这种内部碎片正是限制 VRAM 里能装多少序列的东西。PagedAttention 借来操作系统的**虚拟内存**技巧：把 KV cache 切成固定大小的**块**（页）、把它们存在共享池的**任意处**、并保存一张按序列的**块表**（block table），映射逻辑块索引 → 物理块号。序列一次长一块（几乎零浪费），块甚至能跨序列**共享**（prefix caching 的基础）。代价：attention 再也不能读一段连续的 KV——kernel 必须逐块**gather** KV。那次 gather 就是 PagedAttention kernel 存在的目的——高效地做它。→ 术语 *PagedAttention、KV cache、Block table* 见[术语表](../glossary.md)。

## 2 · 心智模型

虚拟内存类比，以及 kernel 的循环：

```text
LOGICAL 视图（attention 想要的）           PHYSICAL 视图（vLLM 如何存）
  seq A: [tok0 tok1 tok2 tok3 tok4 tok5]     KV 块池（固定大小、可共享）：
             │ block table 映射            ┌──────┬──────┬──────┬──────┬──────┐
             ▼  逻辑→物理                  │ blk0 │ blk1 │ blk2 │ blk3 │ blk4 │ ...
  A.block_table = [3, 1]  ───────────────► │ (B)  │ tok4 │ (C)  │ tok0 │      │
                                           │      │ tok5 │      │ tok1 │      │
      逻辑 blk 0 ─► 物理 blk 3             │      │      │      │ tok2 │      │
      逻辑 blk 1 ─► 物理 blk 1             │      │      │      │ tok3 │      │
                                           └──────┴──────┴──────┴──────┴──────┘
  （物理顺序是任意的；block table 还原逻辑顺序）

KERNEL（一个 query，它的 KV 散在各块）：
  for logical_blk in seq.block_table:        # 走这条序列的块
      phys = block_table[logical_blk]         # 逻辑 -> 物理
      K_blk, V_blk = k_cache[phys], v_cache[phys]
      s = Q · K_blkᵀ                          # 这一块 token 的 scores
      用 ONLINE SOFTMAX 更新运行 (m, l, acc)  # 和 FlashAttention 同一个技巧
  out = acc / l
```

三个要抓住的形状：

- **block table 是 KV 的页表。** 物理放置是任意的；block table 是把散落的块还原成逻辑序列的那层间接。增长即追加一块；共享即让两张表指向同一物理块。
- **kernel 先 gather，再做普通 attention。** 一旦一块的 K/V 被 load，数学就是你已知的 $QK^\top$ → softmax → $\cdot V$——用 **online softmax** 折进来，于是 kernel 绝不物化整行 score（[FlashAttention](../part2/flash-attention.md) 的思想，按块施用）。
- **这里的「block」是 KV 页，不是线程块。** vLLM 默认 KV 块大小是 16 个 token。别把它和 [执行模型](cuda-execution-model.md) 那课的 CUDA/Triton 线程块混了。

## 3 · 原理与读源码

要打开的文件：设计文档 `docs/design/paged_attention.md`（叙事）、`csrc/attention/` 里的 CUDA kernel、以及 Python 封装 `vllm/.../attention/ops/paged_attn.py`。这是地图。

### 3.1 paged KV-cache 布局

vLLM 把 K、V cache 存成块池。核实过的 kernel 签名给出形状：

```cpp
// k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
// v_cache: [num_blocks, num_kv_heads, head_size,   block_size]
//   num_blocks  = 物理池大小              block_size = 每块的 token 数（如 16）
```

`num_blocks`（池）是首维——块是分配单位。奇怪的部分是 K-cache 里的 **`x`**：`split_kv_cache` 算 `x = 16 // element_size`（FP16 下 `x = 8`），把 key cache 视作 `[num_blocks, num_kv_heads, head_size // x, block_size, x]`。这把 `head_size` 切成 `x` 个连续元素一组，好让每个线程读一个 **16 字节对齐**的块——正是上一课的 [coalescing](memory-access.md) 优化，烘进了布局里。V cache 不需要（它沿另一个轴读），故形状更简单。读源码提示：当一个 cache 形状看着怪时，尾部那个打包维几乎总是为了让 load 合并。

### 3.2 写进 cache

新 KV 由 `PagedAttention.write_to_paged_cache` 写入，它调 `ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping.flatten(), kv_cache_dtype, k_scale, v_scale)`。关键参数是 **`slot_mapping`**：对每个 token，给出它扁平的物理槽位（`物理块 × block_size + 块内偏移`）。所以写路径经 `slot_mapping` 把 token 散射到物理槽；读路径（kernel）经 block table 把它们 gather 回来。同一层间接，两个方向。

### 3.3 kernel 的结构

输入（核实过）：query `q [num_seqs, num_heads, head_size]`、两个 cache、以及输出 `out [num_seqs, num_heads, max_num_partitions, head_size]`。核心是对序列各块的循环，块内是一次累加——设计文档自己的伪代码：

```cpp
float accs[NUM_ROWS_PER_THREAD];
for ... {                 // 遍历这条序列的块
    logits_vec = ...      //   这一块的 scores（Q·Kᵀ，再 softmax 权重）
    for ... {             //   遍历行
        v_vec = ...
        accs[i] += dot(logits_vec, v_vec);   // 对 V 的加权和 —— online-softmax 累加
    }
}
```

那正是 FlashAttention 的 online-softmax 循环，限制到外层每步一个物理块。两个值得注意的模板参数：`BLOCK_SIZE`（每块 KV token 数）与 `PARTITION_SIZE`。当 `PARTITION_SIZE > 0` 时，kernel 把一条长 KV 序列切成并行计算、事后合并的分区——那就是 `out` 里的 `max_num_partitions` 维，以及两个 kernel 变体的区别（口语上「v1」不分区、用于短上下文；「v2」分区、用于长上下文——这个切分像 FlashDecoding 一样抬高 occupancy）。

### 3.4 读它时该追什么

跟着一个 query 走：(1) 找它的 `block_table` 行 → 物理块列表；(2) 对每个，在 `num_blocks` 槽上索引 `k_cache`/`v_cache`；(3) 盯住运行最大值 / 和 / 累加器（online softmax）；(4) 看最后的除法与写 `out`。若你能把这四步讲出来，你就能读这个 kernel——其余是 CUDA 线程分派的细节，而布局（§3.1）存在的目的正是让它合并。

## 4 · 完整可跑代码 + 逐行讲解

这是一个纯 Python 的 **paged attention**，镜像 kernel 的形状：KV 住在非连续的物理块里、一张 block table 还原逻辑顺序、attention 逐块 gather 并用 online softmax。把它与稠密（连续）attention 对比，证明 **paging 是存储方案，不是数学改变**——纯 CPU、可离线运行。

```python title="paged_attention_ref.py"
"""Paged attention == dense attention，但 KV 住在非连续的块里。
纯 CPU、离线——镜像 vLLM kernel 的块循环形状，而非其线程编排。"""
import math

BLOCK = 4                                             # 每块 KV token 数（vLLM 默认 16）

def paged_attention(q, block_table, k_pool, v_pool, seq_len):
    """经 block table 走序列的块；用 online softmax 把每块折进来。"""
    d = len(q)
    m, l, acc = -math.inf, 0.0, [0.0] * d             # 运行最大值、归一化因子、输出
    pos = 0
    for phys in block_table:                          # 逻辑块顺序 -> 物理块 id
        k_blk, v_blk = k_pool[phys], v_pool[phys]     # 一个物理块 = BLOCK 个 token 槽
        for t in range(BLOCK):
            if pos >= seq_len:                        # 最后一块可能只填了一部分
                break
            k, v = k_blk[t], v_blk[t]
            s = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)
            m_new = max(m, s)
            corr = math.exp(m - m_new) if m != -math.inf else 0.0
            p = math.exp(s - m_new)
            l = l * corr + p                          # online-softmax 更新（重标定 + 加）
            acc = [acc[j] * corr + p * v[j] for j in range(d)]
            m = m_new
            pos += 1
    return [a / l for a in acc]

def dense_attention(q, K, V):                         # 参考：一段连续的 KV
    d = len(q)
    s = [sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d) for k in K]
    m = max(s); e = [math.exp(x - m) for x in s]; Z = sum(e)
    return [sum(e[i] / Z * V[i][j] for i in range(len(V))) for j in range(d)]

if __name__ == "__main__":
    d, seq_len = 4, 6
    q = [0.5, -0.3, 0.8, 0.1]
    K = [[0.2, 0.1, -0.4, 0.6], [0.9, -0.2, 0.3, 0.0], [-0.5, 0.4, 0.7, -0.1],
         [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, -0.8, 0.2], [-0.3, 0.5, 0.2, 0.9]]
    V = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0],
         [0, 0, 0, 1.0], [.5, .5, .5, .5], [1.0, 1, 1, 1]]

    # 把 6 个 token 散到物理块（BLOCK=4）里、顺序 NON-contiguous，像真实分配器：
    # ceil(6/4)=2 个逻辑块，放在物理 id 3 与 1。block_table 还原顺序。
    block_table = [3, 1]
    k_pool, v_pool = {}, {}
    for i in range(seq_len):
        phys, slot = block_table[i // BLOCK], i % BLOCK
        k_pool.setdefault(phys, [[0.0] * d for _ in range(BLOCK)])
        v_pool.setdefault(phys, [[0.0] * d for _ in range(BLOCK)])
        k_pool[phys][slot], v_pool[phys][slot] = K[i], V[i]

    paged = paged_attention(q, block_table, k_pool, v_pool, seq_len)
    dense = dense_attention(q, K, V)
    diff = max(abs(a - b) for a, b in zip(paged, dense))
    print("paged :", [round(x, 6) for x in paged])
    print("dense :", [round(x, 6) for x in dense])
    print(f"max abs diff = {diff:.2e}   (paged == dense; blocks are just storage)")
```

**逐行讲解：**

- `paged_attention` —— kernel 形状的 Python 版。外层循环走 `block_table`（逻辑→物理）；每个 `phys` 索引块池，正如在 `num_blocks` 槽上索引 `k_cache`/`v_cache`。内层循环用 **online-softmax** 更新把每个 token 折进来（`corr` 在最大值移动时重标定运行归一化因子与累加器）——即 §3.3 的 `accs[i] += dot(...)` 循环。
- `pos >= seq_len` 守卫处理**部分填充的最后一块**——一个真实细节（6 token 的序列用 2 块各 4，第二块半空），也是为什么 kernel 需要真实序列长度、而非只有块数。
- 散射循环把序列的两个逻辑块放在**物理 id 3 与 1**——故意乱序，来说明物理放置是任意的；`block_table = [3, 1]` 是唯一还原逻辑顺序的东西。vLLM 里的 `slot_mapping` 是这个放置的扁平版。
- `dense_attention` —— 对一段连续 KV 的同一 attention，即真值。

预期输出（精确算术，不是 benchmark）：

```text
paged : [0.363083, 0.449897, 0.392487, 0.386499]
dense : [0.363083, 0.449897, 0.392487, 0.386499]
max abs diff = 1.11e-16   (paged == dense; blocks are just storage)
```

差是机器 epsilon。paging 改变 KV *住在哪*、以及 kernel *怎么够到它*——从不改变 attention *算什么*。这就是 PagedAttention 的全部许可证：近乎零的内存浪费与块共享，代价是一次 kernel 吸收掉的 gather。

## 5 · Lab —— 读真实的 kernel

!!! gpu "GPU Lab（可选验证）"
    - **最低显存：** *读*不需要；若想实测观察块分配，用 `Qwen2.5-7B-Instruct`（INT4/AWQ）跑 vLLM 约需 16 GB
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）
    - **预估耗时 / 花费：** 阅读 ~30 分钟（无卡模式免费）· 可选运行 ~10 分钟 · ~¥1（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** paged-KV *设计*与后端无关；CUDA kernel 是 NVIDIA 专有，ROCm 有自己的移植，FlashInfer/FlashAttention 后端以不同方式实现同一个 paged-KV 契约。

核心 Lab 是**读**，完全可在 AutoDL 无卡模式（免费）做：

```text
阅读清单 —— 跟着一个 query 走源码：
1. 打开 docs/design/paged_attention.md —— 读 "Inputs" 与布局章节。
2. 在 vllm/.../attention/ops/paged_attn.py 里找 split_kv_cache：
     - 确认 x = 16 // element_size，与 [num_blocks, num_kv_heads, head_size//x, block_size, x] 视图。
3. 在 csrc/attention/ 里找块循环：
     - 定位 block_table 把逻辑 -> 物理块索引的地方，
     - 定位 online-softmax 的运行最大值 / 和 / 累加器，
     - 定位 PARTITION_SIZE 分支（v1 vs v2 切分）与 max_num_partitions 输出维。
4. 把每处映到 §4 paged_attention() 的一行：block-table 走、逐块 gather、online-softmax 折。
```

可选 GPU 验证：用 `Qwen2.5-7B-Instruct` 和一个小 `--max-model-len` 启动 vLLM，发几个请求，在日志/指标里看 KV 块用量——你会看到块随序列增长按需分配，而非一次性预留。（完整服务图景是 Part 5 的课；这里只是确认块池的行为像 §4 的 `k_pool`。）

## 6 · 常见坑 / 反直觉点

- **把 KV「block」和线程「block」混了。** vLLM 的 KV 块是 cache 池里 ~16 个 token 槽的一页；CUDA/Triton 线程块是调度单位。同一个词，无关的概念。
- **以为 paging 改变 attention 结果。** 它不会——§4 证明 paged == dense 到机器精度。paging 是内存管理方案；kernel gather 散落的 KV 但算出完全相同的 attention。
- **没搞懂 K cache 为何多一个 `x` 维。** 它不是随意的——`x` 把 `head_size` 打包成 16 字节对齐的块，好让 kernel 的 load 合并。V cache 沿另一个轴读，不需要它。
- **以为 block table 是连续或有序的。** 物理块被分配到任何有空的地方；block table（和 `slot_mapping`）是唯一把它们系到逻辑顺序的东西。那份自由正是要点——它干掉碎片、使能共享。
- **读 kernel 时指望一次连续的 KV 读。** gather（块循环）是与稠密 KV attention kernel 的定义性区别；若你在找一次跨整条序列的 strided load，就会错过它的结构。
- **在这里过度声称服务故事。** paging *如何*提吞吐（continuous batching、block manager、prefix caching）是系统话题——本课只讲读使它成为可能的那个 kernel。

## 7 · 面试连线

- [PagedAttention kernel：paged KV cache 与 block table](../interview/paged-attention-kernel.md) —— 这节课为你准备的高频题：*为何把 KV 存成块、block table 做什么、kernel 如何 gather KV，以及它为何与稠密 attention 数学等价。*

## 8 · 小结 & 延伸阅读

**一句话：** PagedAttention 把 KV cache 存成共享池里的固定大小块、配一张按序列的 block table（逻辑→物理），于是序列以近乎零浪费增长、还能共享块；kernel 经那张表逐块 gather KV、用 online softmax 折进来——算出与连续-KV kernel 完全相同的 attention，这也是为什么读它主要是跟着那层间接走。

延伸阅读：

- Kwon 等 —— *Efficient Memory Management for LLM Serving with PagedAttention*（vLLM 论文）—— 虚拟内存框架与碎片数字。
- vLLM `docs/design/paged_attention.md` —— 本课映射的 kernel 走读；开着 §3 一起读。
- [FlashAttention](../part2/flash-attention.md) 课 —— 块循环复用的 online-softmax 累加。
- Part 5（服务）—— continuous batching、block manager、prefix caching 把这个 kernel 变成吞吐的地方。

## 9 · 自测小问

??? question "为什么 vLLM 把 KV cache 存成固定大小的块，而不是每条序列一段连续区域？"
    每条序列的连续 cache 必须按*最大*可能长度定尺，于是对更短或仍在增长的序列多数一直空着——内部碎片，限制 VRAM 里能装多少序列。固定大小的块（页）让序列一次长一块、近乎零浪费，让分配器把块放在共享池的任意处，还让不同序列**共享**相同的块（如共同的 prompt 前缀）。代价是 attention 再不能读一段连续 KV——kernel 必须经 block table 逐块 gather。

??? question "block table 是什么，attention 时 kernel 怎么用它？"
    block table 是一张按序列的列表，把每个**逻辑** KV 块索引映到共享 cache 池里的一个**物理**块号——即 KV cache 的页表。attention kernel 按逻辑顺序走序列的 block table，对每个条目在那个物理块的槽上索引 `k_cache`/`v_cache` 来 load 它的 K/V、算这块的 scores、折进一个运行的 online-softmax 累加器。写路径用其扁平等价物（`slot_mapping`）把新 token 散射到物理槽。

??? question "PagedAttention 是 attention 的近似吗？用 kernel 实际做的事来论证。"
    不是——它算出与稠密、连续-KV kernel 完全相同的 attention（如 §4 参考所示，到机器精度一致）。paging 唯一改变的是 KV *存在哪*（散落的固定大小块）与 kernel *怎么够到它*（经 block table gather，而非一次连续读）。内部它仍做 $QK^\top$ → softmax → $\cdot V$，用 online softmax 累加。paging 是内存管理方案、不是数学改变——这也是为什么收益纯在内存效率（更少碎片、块共享），而非不同的结果。
