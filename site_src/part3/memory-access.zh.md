# 访存：Coalescing、Shared Memory 与 Bank Conflict

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090（24 GB，Ada Lovelace，compute capability 8.9）"
    `torch.cuda` 计时 API（`Event`、`elapsed_time`、`synchronize`）与 `Tensor.contiguous` / `.t()` 语义经 Context7 对照 PyTorch 核实（ADR-0004）。**32 字节 sector / 128 字节 line** 的事务粒度与 **32 个 shared-memory bank（4 字节步长）** 是 CUDA *架构文档给定*的常量。所有带宽数字均为**示例 / 量级参考**——§4 里的 sector/bank 计数是模型上的精确算术（一个寻址模型，不是 benchmark）。

---

## 1 · 直觉 & 为什么重要

[执行模型](cuda-execution-model.md) 那节课说，GPU 靠让许多 warp 忙碌来隐藏访存时延。但对一个 **memory-bound** 的 kernel——而 LLM decode *就是* memory-bound（[roofline](../part2/roofline-analysis.md)：强度 ≈ 1）——最大的杠杆不是你跑多少 warp，而是**每个 warp 怎样触碰内存**。带宽只有在一个 warp 的 32 条 lane 恰好索要硬件能一把取回的字节时才是「峰值」；索要得不好，就为同样有用的数据搬多达 32× 的字节。

三个杠杆，一个主题——*别搬你不需要的字节*：

- **Coalescing（访存合并）**——当一个 warp 的 32 条 lane 读 32 个*连续*地址时，硬件把它们折成一次宽事务。把这些地址打散，它就发出许多事务、每次大半浪费。同一条指令，多达 32× 的 HBM 流量。
- **Shared memory（共享内存）**——每个 SM 上一小块快速、*由程序管理*的 [SRAM](../part0/gpu-hardware.md) 便签。从 HBM 装入一个 tile 一次，再从 shared memory 复用多次：同样 FLOPs 下更少的 HBM 字节 → 更高的[算术强度](../interview/arithmetic-intensity.md)。这正是 [FlashAttention](../part2/flash-attention.md) 用来把 $S\times S$ scores 留在片上的杠杆。
- **Bank conflict（bank 冲突）**——shared memory 只有在 32 条 lane 命中 32 个不同 bank 时才快。当几条 lane 撞在同一个 bank 上时，那些访问会串行化——这是 uncoalesced HBM 在 shared-memory 上的对应物。

能说出「这个 kernel 慢是因为访存 uncoalesced / 有 bank 冲突」，是读懂 vLLM 与 Triton 搬数据代码的核心。→ 术语 *Coalescing / Shared memory / Bank conflict* 见[术语表](../glossary.md)。

## 2 · 心智模型

一个 *warp* 的访存指令在硬件里变成什么：

```text
COALESCED  —— warp 的 32 条 lane 读 32 个连续 float（128 B，对齐）
  lane:  0  1  2  3 ... 31
  addr: [────────────────── 一条 128 字节 line ──────────────────]   ⇒ 1 次事务，100% 有用

UNCOALESCED —— lane 以大步长读（例如相隔一整行）
  lane 0 ─► [line A]······  lane 1 ─► [line B]······  lane 2 ─► [line C]······
            (32 B 里只用 4 B) ...                                  ⇒ 多达 32 次事务，
                                                                     只 ~1/8–1/32 有用

SHARED MEMORY —— 32 个 bank，4 字节字交错：bank = (word_index) mod 32
  conflict-free : lane k -> bank k                      ⇒ 1 个 shared-mem 周期
  2-way conflict: lane k -> bank (2k mod 32)            ⇒ 每 bank 2 条 lane -> 串行 x2
  broadcast     : 所有 lane -> 同一个字                  ⇒ 免费（硬件广播）
```

两个要抓住的形状：

- **访存模式是*warp* 的、按指令的属性。** coalescing 不关乎随时间的缓存；它关乎*这个* warp 的 32 个同时地址是否落在少数事务里。修法几乎总是「让相邻 lane 触碰相邻地址」——即用 `threadIdx.x` 索引*变化最快*的那一维。
- **shared memory 是拿 HBM 流量换复用。** 它不会让单次装入比一次 coalesced 的 HBM load 更快；它在一个 tile *被复用*到足以摊薄其一次性装入时才划算——这正是为什么 tiling + shared memory 是标准的 GEMM/attention 模式。

## 3 · 原理与数学

### 3.1 Coalescing —— 每 warp 的事务数

内存系统以固定块搬数据：**32 字节 sector**，聚成 128 字节 line。一个 warp 的 load 由它 32 个 lane 地址触及的不同 sector 数量来服务。定义**效率**为有用字节比上搬运字节：

$$
\text{efficiency} = \frac{\text{lane 实际用到的字节}}{\text{硬件搬运的事务里的字节}}
$$

- **连续、对齐**（lane $k$ 读字 $k$）：32 × 4 B = 128 B 落在一条 line → **搬 1 条 line，效率 ≈ 100%**。
- **步长 $s$ 字**：相邻 lane 相隔 $4s$ 字节。步长一大，每条 lane 就落进*自己*的 32 字节 sector，于是 warp 触及多达 32 个 sector = 搬 1024 B 却只用 128 B → **效率 ≈ 1/8**（而每个 32 B sector 里只用 4 B → 低到一条 line 的 **1/32**）。有效带宽按同样因子下降。

现实里最主要的成因：把一个 row-major 矩阵**沿列**读。row-major 意味着元素 $(r,c)$ 在偏移 $r\cdot W + c$；若相邻 lane 取相邻*行*（固定列），它们的地址相隔 $W$ 字——整整一行的步长 → 灾难性 uncoalesced。沿*一行*走（相邻列）就是连续的。同样的数据，转置的访存模式，多达 32× 的流量。

### 3.2 Shared memory —— 复用抬高强度

考虑一个被复用 $R$ 次的 tile。没有 shared memory 时每次使用都是一次 HBM 读：$R$ 次 HBM 触碰。有 shared memory 时：**一次** HBM 读进便签，然后 $R$ 次廉价的 shared-memory 读。HBM 字节降 ~$R$×，于是在 [roofline](../part2/roofline-analysis.md) 上算术强度升 ~$R$×——你靠缩小字节分母、而非改变 FLOPs 向算力屋顶攀升。这正是 [FlashAttention](../part2/flash-attention.md) 做的动作（Q/K/V tile 装入一次、跨 block 复用），也是 tiled GEMM 胜过朴素 GEMM 的原因。

### 3.3 Bank conflict —— shared-memory 的坑

shared memory 被切成 **32 个 bank**；相继的 4 字节字映到相继的 bank，于是字 $w$ 住在 bank $w \bmod 32$。一个 warp 的 shared-memory 访问在其 32 条 lane 命中 32 个*不同* bank 时无冲突（或全都读*同一个*字——硬件免费**广播**它）。当 $n$ 条 lane 瞄准*同一* bank 里*不同*的字时就冲突：那 $n$ 次访问会**串行化**，一个 $n$-way 冲突花 ~$n$×。

经典触发是 2 的幂步长。以步长 2 访问一个 shared 数组（lane $k$ → 字 $2k$）把 lane 映到 bank $2k \bmod 32$，于是 lane $k$ 与 $k+16$ 相撞 → 2-way 冲突。步长 32（一个 32 宽 shared tile 的一列）把*所有* lane 送进 bank 0 → 32-way 冲突，完全串行。教科书修法是**padding**：把 tile 声明宽一列（宽 33 而非 32），于是一「列」走相隔 $33$ 的地址——$33 \bmod 32 = 1$，命中全部 32 个 bank，无冲突。一列浪费的 shared memory 换回一次 32× 的串行化。

## 4 · 完整可跑代码 + 逐行讲解

这段无 GPU 地建模两种隐患：（1）数一个 warp 的步长访问触及多少个 32 字节 **sector**（coalescing tax），以及（2）数落在某个 shared-memory **bank** 上的最多 lane 数（冲突度）。纯 CPU、可离线运行、确定性。

```python title="coalescing_and_banks.py"
"""建模 SIMT 为之收费的两种访存隐患。纯 CPU、离线——这是硬件做的寻址算术，
不是计时 benchmark。"""

WARP, SECTOR_B, DTYPE_B, BANKS = 32, 32, 4, 32

def sectors_touched(stride_words, base_word=0):
    """一个 warp，lane k 读字 (base + k*stride)。数搬运的不同 32 字节 sector。
    连续（步长 1）-> 少数 sector；大步长 -> 多达 32。"""
    sectors = {((base_word + k * stride_words) * DTYPE_B) // SECTOR_B for k in range(WARP)}
    return len(sectors)

def max_lanes_per_bank(stride_words):
    """一个 warp，lane k 访问 shared 字 k*stride。返回最差 bank 的 lane 数
    （1 = 无冲突；n = n-way 冲突，串行 x n）。"""
    counts = {}
    for k in range(WARP):
        bank = (k * stride_words) % BANKS
        counts[bank] = counts.get(bank, 0) + 1
    return max(counts.values())

if __name__ == "__main__":
    print("HBM coalescing — 32-byte sectors moved per warp (ideal = 4 for 128 B):")
    for s in (1, 2, 8, 32):
        sec = sectors_touched(s)
        print(f"  stride {s:>2} words: {sec:>2} sectors  ({4/sec*100:5.1f}% efficiency)")

    print("\nShared-memory bank conflicts — worst-case lanes on one bank (ideal = 1):")
    for s in (1, 2, 32, 33):
        n = max_lanes_per_bank(s)
        print(f"  stride {s:>2} words: {n:>2}-way  ({'conflict-free' if n == 1 else f'serialized x{n}'})")
```

**逐行讲解：**

- `sectors_touched` —— 把每条 lane 的字索引化成字节地址，向下整除 32 字节的 `SECTOR_B` 得到其 sector id，再用 Python `set` 数*不同*的 sector。连续访问（步长 1）把 32 个字塞进 4 个 sector（128 B）；大步长把它们散进多达 32 个 sector——量化出的 coalescing tax。
- `max_lanes_per_bank` —— 用 `word % 32` 把每条 lane 的字映到 bank，返回最拥挤的 bank。`1` 表示每条 lane 命中不同 bank（无冲突）；`n` 表示 $n$ 条 lane 在一个 bank 上串行。
- `__main__` 的扫描既展示两种隐患*也*展示修法：步长 32 是 32-way bank 冲突，但步长 **33**（padding 技巧）又回到无冲突。

预期输出（寻址算术，不是 benchmark）：

```text
HBM coalescing — 32-byte sectors moved per warp (ideal = 4 for 128 B):
  stride  1 words:  4 sectors  (100.0% efficiency)
  stride  2 words:  8 sectors  ( 50.0% efficiency)
  stride  8 words: 32 sectors  ( 12.5% efficiency)
  stride 32 words: 32 sectors  ( 12.5% efficiency)

Shared-memory bank conflicts — worst-case lanes on one bank (ideal = 1):
  stride  1 words:  1-way  (conflict-free)
  stride  2 words:  2-way  (serialized x2)
  stride 32 words: 32-way  (serialized x32)
  stride 33 words:  1-way  (conflict-free)
```

两种效应都是纯寻址：*值*没变，只是 32 条 lane 索要哪些字节变了。步长 1 → 步长 32 的塌缩（100% → 12.5% HBM 效率；1-way → 32-way bank 串行）与步长 33 的恢复，就是 kernel 死磕布局的全部理由。

## 5 · Lab —— 在真实 HBM 上对比 coalesced 与 strided 带宽

!!! gpu "GPU Lab"
    - **最低显存：** 4 GB（分配几百 MB 的 float32 缓冲；不加载模型）
    - **建议 AutoDL 卡型：** RTX 4090（24 GB）——对齐基线
    - **预估耗时 / 花费：** ~3 分钟 · ~¥0.3 GPU 时长（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** coalescing *原理*在 ROCm/其他平台成立，但事务粒度与 shared-bank 数不同；有效带宽比值会变。

你无法从 PyTorch 手工摆放一个 warp，但你能给一个 kernel 喂同一份数据的**连续**视图与**步长（转置）**视图，看着有效带宽塌缩——这是 uncoalesced 访问在张量层面的影子。用已核实的 CUDA event 计时。

```python title="coalesced_vs_strided_bandwidth.py"
import torch
assert torch.cuda.is_available()

def bandwidth_gbps(read_tensor):
    """把 read_tensor 拷进一个新的连续输出；报告有效 GB/s。
    一个非连续（转置）源会强制步长、uncoalesced 的读。"""
    nbytes = read_tensor.numel() * read_tensor.element_size()
    out = torch.empty(read_tensor.shape, device="cuda", dtype=read_tensor.dtype)
    for _ in range(3):                                  # 预热
        out.copy_(read_tensor)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        out.copy_(read_tensor)                          # 读 + 写 -> 搬 2 * nbytes
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 50
    return 2 * nbytes / (ms / 1e3) / 1e9

n = 8192
x = torch.randn(n, n, device="cuda", dtype=torch.float32)   # 256 MiB，row-major
contiguous = x                                              # 沿行走 -> coalesced
strided    = x.t()                                          # 转置 VIEW -> 沿列读

print(f"contiguous copy : {bandwidth_gbps(contiguous):6.0f} GB/s  (coalesced reads)")
print(f"transposed copy : {bandwidth_gbps(strided):6.0f} GB/s  (strided, uncoalesced reads)")
print("(same 256 MiB of data; only the access pattern differs)")
```

**要观察什么：** 两次拷贝搬的都是同样 256 MiB，但转置源*沿列*读一个 row-major 数组——每个 warp 的 lane 相隔整整一行，于是读是 uncoalesced 的，有效带宽掉到连续情形的一小部分（常是数倍；确切比值随卡与规模而变，示例）。这就是为什么 PyTorch 有 `.contiguous()`，以及为什么把一个转置视图直接喂进自定义/Triton kernel 会悄悄拖垮它。这个原理放大后，正是 tiled + shared-memory kernel 存在要修的东西。

## 6 · 常见坑 / 反直觉点

- **遍历错了轴。** 头号真实 coalescing bug：索引方式让相邻线程跨过 row-major 张量的*慢*（行步长）维。让 `threadIdx.x` 索引**变化最快**的轴（相邻列），使相邻 lane 命中相邻地址。
- **「shared memory 总是更快」。** 只有当装入的 tile *被复用*到足以摊薄其 HBM 装入——且无 bank 冲突——才划算。一次性的经 shared memory 中转拷贝纯是开销。
- **2 的幂 shared 步长。** 一个 32 宽 shared tile 的列访问是 32-way bank 冲突。`[N][33]` padding 技巧（多声明一列）打破周期性——便宜，且在 transpose/GEMM kernel 里是标准做法。
- **未对齐即使步长 1 也破坏 coalescing。** 若基地址没对齐到事务大小，一个连续的 warp 也会跨进多一个 sector/line。对齐要紧，不只是连续。
- **broadcast 不是冲突。** 全部 32 条 lane 读*同一个* shared 字是免费的（硬件广播）。冲突是 lane 命中*同一* bank 里*不同*的字——别把两者混了。
- **coalescing ≠ L2 缓存。** coalescing 关乎*一个 warp* 每条指令的 32 个地址折成少数事务。一个步长 kernel 随时间也许仍有 L2 命中，但它已付了每-warp 的事务税——不同机制，不同修法。
- **`.contiguous()` 会拷贝。** 调它为下游 kernel 修好访存模式，但要花一整遍扫过数据；别盲目乱撒——先弄清复用是否值回这次拷贝。

## 7 · 面试连线

- [Memory coalescing、shared memory 与 bank conflict](../interview/memory-coalescing.md) —— 这节课为你准备的高频题：*什么让一次访问 coalesced，uncoalesced 的代价是什么，以及 shared memory 与 bank conflict 是做什么用的？*

## 8 · 小结 & 延伸阅读

**一句话：** 一个 warp 的 32 条 lane 应触碰 32 个连续地址，好让硬件把它们 coalesce 成一次宽事务（散乱访问搬多达 32× 的字节）；shared memory 是由程序管理的 SRAM 便签，拿一次 HBM 装入换许多次廉价复用——只有当它的 32 条 lane 命中 32 个不同 bank 时才快（否则串行）。

延伸阅读：

- *CUDA C++ Best Practices Guide* —— "Coalesced Access to Global Memory" 与 "Shared Memory"（bank 模型、padding），第一手。
- [FlashAttention](../part2/flash-attention.md) 课 —— shared-memory tiling 的实战（把 score tile 留在片上、复用、绝不触 HBM）。
- [GPU 硬件心智模型](../part0/gpu-hardware.md) 课 —— 这一切所依托的 HBM/SRAM 层级与带宽。
- [算子 Roofline](../part2/roofline-analysis.md) 课 —— 为什么抬高*有效*带宽与复用能推动一个 memory-bound kernel。

## 9 · 自测小问

??? question "为什么把一个 row-major 矩阵*沿列*读会摧毁内存带宽，而*沿行*读不会？"
    在 row-major 布局里，元素 $(r,c)$ 位于偏移 $r\cdot W + c$。沿行读意味着相邻线程（lane）读相邻 `c` → 相邻地址 → warp 的 32 次访问 **coalesce** 成 ~一次 128 字节事务（≈100% 有效）。沿列读意味着相邻 lane 在固定 `c` 上读相邻 `r` → 地址相隔整整一行（$W$ 字）→ 每条 lane 落进自己的 32 字节 sector，于是 warp 发出多达 32 次事务、每次只用一小部分。同样的数据，多达 32× 的 HBM 流量，有效带宽随之塌缩。

??? question "shared memory 是做什么用的，什么时候用它*不*值得？"
    shared memory 是每个 SM 上一小块快速、由程序管理的 SRAM 便签。它的用途是**复用**：从 HBM 装入一个 tile 一次，再从 shared memory 廉价地读许多次，把 HBM 字节按 ~复用倍数削减、抬高算术强度（FlashAttention / tiled-GEMM 模式）。当数据只用一次时*不*值得（中转装入纯是开销，没有复用可摊薄），或当访存模式造成大量 bank 冲突、把 shared 读串行化时也不值得。

??? question "你把一个 32×32 的 tile 存进 shared memory，每个 warp 读它的一*列*，结果很慢。发生了什么，一句话修法是什么？"
    读一个 32 宽 shared tile 的一列意味着 32 条 lane 访问相隔 32 的字；因为 bank = 字 mod 32，**全部 32 条 lane 映到同一个 bank** → 一个 32-way bank 冲突，把访问串行化 ~32×。修法是 **padding**：把 tile 声明为 `[32][33]`（多一列）。现在一「列」走相隔 33 的地址，且 $33 \bmod 32 = 1$，于是 32 条 lane 命中 32 个不同 bank——无冲突——代价是一列没用上的 shared memory。
