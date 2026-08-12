# PagedAttention：像虚拟内存一样管理 KV Cache

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    本课是 PagedAttention 的 **serving / 内存管理**视角——block manager、分配、以及*为何分页提吞吐*。**读 kernel** 视角（gather 循环、cache 布局、online softmax）在 [Part 3 那一课](../part3/paged-attention-kernel.md)；我们交叉链接而非重复。这里点到的 V1 内部——`BlockPool`（用按驱逐顺序排列的 `free_block_queue` 做 allocate/free/cache）、带 `ref_cnt` 的 `KVCacheBlock`、`SingleTypeKVCacheManager`（`req_to_blocks`），以及 `num_gpu_blocks` 由 `gpu_memory_utilization`（默认 **0.92**）经显存 profiling 推导——均针对 vLLM 0.26.0 经 Context7 核实（ADR-0004）。§4 的分配器是**容量模型，不是 benchmark**（纯 Python、离线）。序列数是精确算术；任何吞吐数字为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

[continuous-batching 那一课](continuous-batching.md) 留了个悬念：批只能长到 **[KV cache](../part0/kv-cache.md) 空间**用尽为止，而那个容量——不是算力——才是并发的天花板。本课就讲*如何抬高那道天花板*。PagedAttention 是让 vLLM 吞吐成为可能的那一个改动，也是你操作系统用来跑下超过物理 RAM 的程序数量的同一招：**虚拟内存**。

它解决的问题是这样。朴素引擎把每个序列的 KV cache 存成一块**连续**区域，按*最大*可能长度开好——因为一旦 attention 开始读它就搬不动了，且它事先不知道序列会长到多长。所以一个*可能*到 512 token 的请求会一上来就预留 512 token 的 KV，哪怕它实际只吐 40 个。这块留了却空着的空间就是**内部碎片（internal fragmentation）**，它很残酷：在 24 GB 的 4090 上，按最大长度预留意味着你只能装*寥寥几个*序列，哪怕它们真实的 KV 能装下几十个。更糟的是，释放的不同大小区域留下的洞分配器没法复用——**外部碎片（external fragmentation）**。

PagedAttention 借用了 OS 的修法。把 KV cache 切成固定大小的 **block**（页，每页 16 token），放进一个共享**池子**，给每个序列一张 **block table**，把它的逻辑块 → 池子里任意位置的物理块。序列现在**随长随分，一次一块**（浪费永远 ≤ 一个半满块），块从共享空闲表取、用完还回去（无外部碎片），而且——因为块是共享的单位——两个有共同前缀的序列可以指向*同一*物理块。消除碎片，就能装下多得多的序列；序列多了，[continuous batch](continuous-batching.md) 就更大；批更大，就把 memory-bound 的 decode 摊薄得更好。**这条链就是为什么分页等于吞吐。** → 术语见 [Glossary](../glossary.md) 的 *PagedAttention、KV cache、Block table*。

## 2 · 心智模型

把 KV cache 当物理内存，block table 当页表：

```text
CONTIGUOUS 预留（朴素引擎）                          PAGED 分配（vLLM）
  seq A ┃■■■□□□□□□□□□□□□□┃  按 MAX_LEN 预留            共享 block 池（16 tok/块）：
  seq B ┃■■■■■■□□□□□□□□□□┃  （□ = 预留但空着）           ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐
  seq C ┃■□□□□□□□□□□□□□□□┃                               │0 │1 │2 │3 │4 │5 │6 │7 │… │
        └── 只装得下 3 个；大部分 VRAM 是 □ 浪费 ──┘      └──┴──┴──┴──┴──┴──┴──┴──┴──┘
                                                          A.table=[3,0]  B.table=[5,1,6]  C.table=[2]
  区域无法缩小或移动，也无法拆开借给别的序列。            每个序列只占它填满的块；
                                                          新 token 可能再抓一个空闲块；
                                                          结束时，块还回池子。
                                                          → 同样 VRAM 装下多得多的序列

BLOCK MANAGER 生命周期（每迭代，由调度器驱动）：
  admit(seq)   : 从池子取空闲块给 prompt              （分配）
  append(tok)  : 若最后一块满了，再取 1 个空闲块       （随长随分）
  finish(seq)  : 把该序列的块推回空闲池                （释放 → 下一步就被复用）
  prefix hit   : 把序列的 table 指向一个已存在的块，ref_cnt++（共享，不拷贝）
```

三个要记的形状：

- **block table 是 KV 的页表。** 物理位置任意；table 是恢复逻辑顺序的间接层。增长 = 抓一个空闲块；结束 = 还回块；共享 = 两张 table 引用同一物理块。
- **浪费从 `max_len − actual_len` 降到 `< 一个块`。** 连续预留浪费你*可能*用但没用的一切；分页最多浪费最后一块没填满的尾巴。收回来的那些 VRAM *就是*多出来的并发。
- **block manager 是让 continuous batching 落地的那一块。** 上一课的「准入一个等待请求」字面意思就是「block manager 能不能发出足够的空闲块？」用完即还把块归还池子，让下次准入成功。分页与 continuous batching 是同一机制的两半。

## 3 · 原理——block manager

### 3.1 一共有多少块——`num_gpu_blocks`

池子不是无限的；它的大小在启动时算好。vLLM 跑一趟**显存 profiling**：取 `gpu_memory_utilization × 总VRAM`（默认 **0.92**）作预算，减去*非 KV* 部分所需（模型权重 + 峰值激活 + CUDA-graph 内存），剩下的成为 KV 池。除以每块字节数，就得到 **`num_gpu_blocks`**——分配器管理的固定页数。即：

$$
\texttt{num\_gpu\_blocks} \;=\; \left\lfloor \frac{\,\texttt{gpu\_mem\_util}\cdot \text{VRAM} \;-\; (\text{weights} + \text{activations} + \text{cudagraph})\,}{\text{每块字节数}} \right\rfloor
$$

这就是为什么 [量化](../part4/index.md) *间接*提吞吐：缩小权重就把更多预算留给 KV 池 → 更多块 → 更多序列。同理 [FP8 KV cache](../part4/quantization-methods.md) 也帮忙——它把每块字节数减半，于是同一池子装下两倍 token。

### 3.2 池子、块、与空闲表

在 vLLM 的 V1 引擎里，池子是一个 **`BlockPool`**，持有 `num_gpu_blocks` 个 **`KVCacheBlock`** 对象。两个结构做记账：

- 一个 **`free_block_queue`**（双向链表空闲表）**按驱逐顺序**保存可用块。分配从头部弹出；释放推回尾部。两向都 O(1)。
- 一个 **`cached_block_hash_to_block`** 映射支持 prefix caching：用块内容的哈希找到一个已算好的块（§3.4）。

每个 `KVCacheBlock` 带一个 **`ref_cnt`**（引用计数）。一个按请求的 manager（**`SingleTypeKVCacheManager`**，每种 attention 类型一个，持 `req_to_blocks`）从共享池取块。整个设计是教科书式分配器：一个空闲表、引用计数对象、两端 allocate/free。

### 3.3 生命周期，绑定调度器

每个调度器迭代（[batching 课](continuous-batching.md) 的 admit→step→evict 循环）驱动 manager：

- **准入 / prefill：** 弹出足够空闲块装下新请求的 prompt；记进请求的 block table。
- **decode 步（增长）：** 每个新 token 填当前最后一块；满了就再弹**一个**空闲块。这就是「一次一块」增长——近乎零浪费。
- **结束（释放）：** 请求命中 EOS 或 `max_tokens` → 把它所有块推回 `free_block_queue`（`ref_cnt` 减一；计数到 0 时块才真正释放）。这些块*下一步*就能给下次准入用。
- **抢占（有压力时）：** 若池子耗尽而高优请求需要空间，vLLM 可以驱逐一个运行序列的块（之后重算或换回）——相当于 OS 把页换出。

### 3.4 块共享 → prefix caching（一个推论，稍后调）

因为一个块由其 **token 的哈希**（加上父块的哈希，所以位置也算进去）标识，两个以*相同*前缀开头的请求会产生*相同*的块哈希——于是第二个请求的 block table 可以指向那些**已算好的物理块**而不必重算。命中时，manager 调 `touch()` 把块的 `ref_cnt` 加一（它可能正躺在空闲队列里当驱逐候选）。只有**整块**可缓存，且 KV 逐字节相同，所以**输出不变**。当两个共享者之后分叉时，一次 **copy-on-write** 拆开共享块。整个这个特性——自动 prefix caching——是分页*使能的*；如何配置与利用它（`enable_prefix_caching`、命中率、KV 感知路由）是 [Part 5 下一个话题](prefix-caching.md)。这里只记住形状：**块是共享的单位，而共享就是免费复用。**

## 4 · 完整可跑代码 + 逐行讲解

一个纯 Python 容量模型：在同一个固定 KV 池上按两种策略填充——连续预留 vs 分页分配——数装得下多少序列。这是把碎片论证写成算术，不用 GPU。

```python title="paged_allocator.py"
"""PagedAttention 作为内存管理器：连续预留 vs 分页分配。
纯 Python、离线——数一个固定 KV 池装得下多少序列。不是 benchmark。"""
import math

BLOCK = 16                 # 每个 KV block 的 token 数（vLLM 的页粒度）
POOL_BLOCKS = 128          # 物理 KV 池：128 块 x 16 = 2048 token-slot
MAX_LEN = 512              # *连续*引擎必须为每个序列预留的长度

# 16 个请求，真实各异的实际长度（prompt + output），都 <= MAX_LEN：
ACTUAL_LENS = [40, 128, 300, 64, 210, 96, 180, 48, 150, 80, 60, 420, 33, 256, 90, 110]

def contiguous_admit(lens, pool, block, max_len):
    """为每个序列一上来就预留 max_len 份块（朴素引擎）。"""
    reserve = math.ceil(max_len / block)                 # 最坏情况块数，所有序列相同
    used = admitted = 0
    for L in lens:
        if used + reserve <= pool:
            used += reserve; admitted += 1
        else:
            break                                        # 池子耗尽——请求只能等
    return admitted, reserve, used

def paged_admit(lens, pool, block):
    """只分配 ceil(actual_len / block) 个块——随长随分（PagedAttention）。"""
    used = admitted = 0
    for L in lens:
        need = math.ceil(L / block)                      # 这个序列实际需要的块数
        if used + need <= pool:
            used += need; admitted += 1
        else:
            break
    return admitted, used

if __name__ == "__main__":
    ca, reserve, cused = contiguous_admit(ACTUAL_LENS, POOL_BLOCKS, BLOCK, MAX_LEN)
    pa, pused = paged_admit(ACTUAL_LENS, POOL_BLOCKS, BLOCK)
    really_used = sum(math.ceil(L / BLOCK) for L in ACTUAL_LENS[:ca])
    print(f"KV pool: {POOL_BLOCKS} blocks x {BLOCK} tok = {POOL_BLOCKS*BLOCK} token-slots")
    print(f"contiguous: reserve max_len={MAX_LEN} -> {reserve} blocks/seq -> admits {ca} seqs")
    print(f"            ({cused}/{POOL_BLOCKS} blocks reserved, only {really_used} actually used)")
    print(f"paged     : allocate actual length      -> admits {pa} seqs ({pused}/{POOL_BLOCKS} blocks)")
```

**逐行讲解：**

- `BLOCK`、`POOL_BLOCKS`、`MAX_LEN`——一个 128 块的池（§3.1 的 `num_gpu_blocks`）和一个连续引擎必须为之规划的每序列 `max_len`。`ACTUAL_LENS` 是序列*真正*用的——总远低于 `max_len`，正常情形。
- `contiguous_admit`——每个序列不论真实长度都预留 `ceil(max_len/block)` = 32 块。它一直准入到池子装不下再一个 32 块预留为止。这就是内部碎片：为最坏情况预留。
- `paged_admit`——每个序列只取 `ceil(actual_len/block)` 块——它真正需要的。同一池子、同样请求，但没有预留浪费。它准入多得多。
- `really_used` 那行量化浪费：连续策略*预留*了多少块 vs 它准入的序列会*实际*填多少块。

预期输出（精确算术，不是 benchmark）：

```text
KV pool: 128 blocks x 16 tok = 2048 token-slots
contiguous: reserve max_len=512 -> 32 blocks/seq -> admits 4 seqs
            (128/128 blocks reserved, only 34 actually used)
paged     : allocate actual length      -> admits 13 seqs (118/128 blocks)
```

同样 VRAM、同样请求：连续预留装下 **4** 个序列（128 个预留块只装了 34 块的真实 KV——约 73% 浪费）；分页分配装下 **13** 个——3 倍多的并发，纯粹靠不预留没人用的空间。把这个更大的运行集喂给 [continuous batching](continuous-batching.md)，吞吐随之而来。这就是 PagedAttention 全部的 serving 理由。

## 5 · Lab——看 block 池呼吸

!!! gpu "GPU Lab（可选验证）"
    - **最低显存：** 读代码不需要；跑 `Qwen2.5-7B-Instruct`（INT4/AWQ）并观察 KV-block 用量需 ~16 GB
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 读 ~20 分钟（免费，无卡模式）· 可选运行 ~10 分钟 · ~¥1（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** paged-KV *设计*与后端无关（AMD ROCm、TPU、CPU 版都对 KV cache 分页）；只有 gather 块的 attention kernel 不同（见 [Part 3](../part3/paged-attention-kernel.md)）。

读是免费的（无卡模式）；可选运行实时展示池子的大小与用量。

```python title="observe_blocks.py"
# API 针对 vLLM 0.26.0 核实（LLM、gpu_memory_utilization）。在有 GPU 的 AutoDL 上跑。
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="awq",              # 权重更小 -> 更多预算变成 KV blocks
    gpu_memory_utilization=0.90,     # KV 池预算旋钮（默认 0.92）；调低 = 更少块
    # block_size 默认 16 token/块——§2 的页粒度
)
# 启动时 vLLM 会打一行 "GPU KV cache size: N tokens" / "# GPU blocks: M"。
# 那个 M 就是 §3.1 的 num_gpu_blocks——free_block_queue 的大小。
print(llm.generate(["Explain PagedAttention in one sentence."])[0].outputs[0].text[:80])
```

**观察什么：**

1. **池子大小 vs 预算。** 启动时 vLLM 打印 GPU KV 块数。把 `gpu_memory_utilization` 从 0.90 调到 0.80 再跑——块数下降（预算少 → 块少 → 并发天花板低）。（谨慎地）调高它，块数增长。这是 §3.1 的可视化。
2. **量化 → 更多块。** 对比 AWQ（INT4）模型的块数与 FP16 运行（如果装得下）：INT4 权重腾出预算，所以块更多。这是 [量化 → 并发](../part4/index.md) 的具体链条。
3. **按需分配。** 用 `vllm serve … --quantization awq` 起服务，发一个长请求和一个短请求，看 metrics：块随序列增长而发出、随结束而归还——从不为 `max_model_len` 一上来就预留。（*读*这些散落块的 gather 是 [Part 3 kernel](../part3/paged-attention-kernel.md)。）

## 6 · 常见坑 / 反直觉点

- **和 kernel 那课混淆。** 两个不同层：[Part 3](../part3/paged-attention-kernel.md) 是 *kernel 怎么读*散落的 KV（gather + online softmax）；本课是 *manager 怎么分配*块（空闲表、引用计数、admit/free）。面试两个都会问——搞清你被问的是哪个。
- **KV「block」vs 线程「block」。** vLLM 的 KV block 是 cache 池里的 16-token 页；CUDA/Triton 线程块（[执行模型课](../part3/cuda-execution-model.md)）是调度单位。同词无关。
- **以为分页加速了计算。** 不——attention 结果与连续 KV 相同（[Part 3 §4](../part3/paged-attention-kernel.md) 已证）。赢在*容量*：浪费少 → 序列多 → 批更大 → 吞吐更高。分页是内存管理器，不是更快的 kernel。
- **假设 `block_size` 越大越好。** 更大的块意味着更粗的分配（最后一块半满时浪费更多）但 block-table 条目更少、记账更便宜；更小的块浪费少但开销大。16 是 vLLM 的默认平衡——别不测就照搬改动。
- **把 `gpu_memory_utilization` 设成 1.0。** 它给激活尖峰或 CUDA 分配器碎片不留余地，招 OOM。默认 0.92 有其道理；小步往上推并盯着 OOM。
- **指望 prefix caching 改变输出。** 它为共享前缀复用逐字节相同的 KV——结果不变。若开缓存后看到不同输出，那是 bug 不是特性。（细节在[下一课](prefix-caching.md)。）
- **把碎片当舍入误差。** 在真实长度分布上，连续预留浪费*大部分* VRAM（§4：73%）。这不是小低效——是 4 个与 13 个并发序列之别。

## 7 · 面试连线

- [KV cache 即虚拟内存：block manager & 碎片](../interview/kv-cache-block-manager.md)——本课为你准备的高频题：*为什么连续 KV 会碎片、block manager 做什么、`num_gpu_blocks` 怎么定、分页如何变成吞吐。*
- 相关，kernel 侧：[PagedAttention kernel & block table](../interview/paged-attention-kernel.md)——gather、cache 布局、为何等于 dense attention。

## 8 · 小结 & 延伸阅读

**一句话：** 朴素引擎把每个序列的 KV cache 按最大长度连续预留，把大部分 VRAM 浪费在内部碎片上、压低并发；PagedAttention 像虚拟内存一样管理 KV cache——共享池里的固定大小块（`num_gpu_blocks` 由 `gpu_memory_utilization` profiling 得出）、每序列一张 block table、随长随分一次一块、用完即还、前缀块共享——于是浪费降到每序列不足一块，装下多得多的序列，continuous batch 更大，而*那里*正是 vLLM 吞吐的来源。

延伸阅读：

- Kwon 等 —— *Efficient Memory Management for LLM Serving with PagedAttention*（SOSP '23，vLLM 论文）—— 虚拟内存框架与碎片测量。
- [continuous-batching 课](continuous-batching.md) —— 这份容量喂养的那个批；分页与 batching 是同一机制的两半。
- [Part 3：读 vLLM 的 PagedAttention Kernel](../part3/paged-attention-kernel.md) —— 另一半：kernel 如何 gather 这些散落块并算出相同的 attention。
- vLLM `docs/design/prefix_caching.md` —— 本课预告的哈希式块共享；如何调优是 Part 5 下一话题。

## 9 · 自测小问

??? question "为什么朴素（连续 KV）引擎必须按最大长度预留？这会造成哪两种碎片？"
    因为 KV 区域必须**连续**（attention kernel 把它当一段读），且引擎**事先无法知道最终长度**（生成是自回归的），一旦开始读也搬不动——所以只能按最坏情况、即最大长度、一上来就预留。这造成**内部碎片**：序列实际长度与最大长度之间那段预留却空着的空间（常常是区域的大部分）。而当不同大小的序列结束、留下可变大小的洞，就造成**外部碎片**：空间存在但没有一整块足够大的连续区给下次预留。PagedAttention 两者都消除——固定大小块让内部浪费 ≤ 一个半满块，且任何空闲块都能满足任何需求（无外部碎片）。

??? question "带 block manager 走一遍：准入一个请求、生成 token、结束。块从哪来、到哪去？"
    **准入/prefill：** manager 从共享池的 `free_block_queue` 弹出足够空闲块装下 prompt，记进请求的 block table。**decode：** 每个新 token 填当前最后一块；满了 manager 再弹**一个**空闲块（随长随分——浪费保持在一块以内）。**结束（EOS 或 max_tokens）：** manager 把请求所有块还回空闲队列（每块 `ref_cnt` 减一；计数到 0 才释放），使它们在下个迭代立刻可供下次准入。若涉及共享前缀块，释放只是减一个引用——块为另一共享者继续存活。这种用完即还正是让 [continuous batching](continuous-batching.md) 一有空间就能准入等待请求的原因。

??? question "你的吞吐被 KV 容量卡住。说出三个增加 `num_gpu_blocks`（或其可装 token 数）的杠杆，并各自对应到公式。"
    由 `num_gpu_blocks = ⌊(gpu_mem_util·VRAM − weights − activations − cudagraph) / 每块字节数⌋`：（1）**调高 `gpu_memory_utilization`**（如 0.90 → 0.94）——直接增大分子预算（谨慎，防 OOM）。（2）**量化权重**（INT4/AWQ）——缩小 `weights` 项，把更多预算留给 KV blocks——[量化→并发](../part4/index.md) 链条。（3）**量化 KV cache**（FP8，`kv_cache_dtype="fp8"`）——把 `每块字节数`减半，于是同一池子装下约 2 倍 token。（附加：减小 `max_model_len` 或用 GQA 缩小每 token KV，更小的 `enforce_eager`/CUDA-graph 占用释放 `cudagraph` 项。）它们买的都是同一样东西：运行集里更多序列，因而更高吞吐。
