# 从 Static 到 Continuous Batching：吞吐的第一杠杆

!!! info "基线：**vLLM 0.26.0** · 模型 `Qwen2.5-7B-Instruct` · 单张 RTX 4090 (24 GB)"
    本页所述 vLLM 旋钮——`max_num_seqs`（默认 **128**）、`max_num_batched_tokens`（默认 **2048**，引擎自动调）、以及 **continuous batching 恒开**（没有开关 flag，它*就是* V1 调度器）——均针对 vLLM 0.26.0 经 Context7 核实（ADR-0004）。§4 的 Python 仿真是**调度模型，不是 benchmark**（纯 Python、离线、不用 GPU）。加速数字为**示例 / 量级参考**，请在自己的 AutoDL 上实测。

---

## 1 · 直觉 & 为什么重要

你只有一张 GPU，请求随时间陆续到达，每个生成的 token 数各不相同。**吞吐**就是你每秒能完成多少请求（或 token）。撬动这个数字最大的杠杆——比量化大、比任何 kernel 都大——是*你如何随时间把请求塞进 GPU*。这就是 batching，做对了值**一个数量级**（示例）。

陷阱在这里。最直觉的 batching 做法是：凑齐 N 个请求，一起跑到**全部**结束，再取下一批 N 个。这是 **static batching**，它浪费了你大部分 GPU。一批里的请求结束长度天差地别——一个吐 20 个 token，邻居吐 500 个——但整批被锁在一起，直到*最长*那个跑完。每个提前结束的序列，都让它那个 GPU 槽位在这一批剩下的时间里**空转**。在真实流量上（输出长度差 10–50 倍），那意味着你大部分算力都在黑着。

**Continuous batching**（出自 Orca 论文，也是包括 vLLM 在内每个正经引擎的默认）通过改变*调度单位*来解决：不再是调度整个请求、跑到完成，而是**一次调度一个 decode 迭代**。每一步 forward 之后，引擎就驱逐刚结束的序列、释放它们的 [KV cache](../part0/kv-cache.md)，并立刻把等待中的请求塞进腾出的槽位。这一批永远不锁死——它是个每步都在增减成员的活集合。只要有活等着，就没有槽位空转。→ 术语见 [Glossary](../glossary.md) 的 *Static / Dynamic / Continuous batching*。

## 2 · 心智模型

把时间想成从左到右流动，GPU 的 batch 槽位竖着堆叠。每个 `█` 是该步做有用功的槽位；每个 `·` 是**空转**槽位（浪费的 GPU）。

```text
STATIC BATCHING（batch=4，跑到全部结束，再下一批）
        step→  1 2 3 4 5 6 7 8 9 …
  slot0  R0    █ █ █ ·  ·  ·  ·  ·          R0 在 step3 结束，槽位空到 step8
  slot1  R1    █ █ █ █ █ █ █ █              R1 是最长的——把整批扣为人质
  slot2  R2    █ █ ·  ·  ·  ·  ·  ·          R2 在 step2 结束
  slot3  R3    █ █ █ █ █ ·  ·  ·             R3 在 step5 结束
               └── 这批要等到 step8（R1 完）才能重填 ──┘
  利用率 ≈ 有色 / 总数  →  一堆 "·"（bubble 气泡）

CONTINUOUS BATCHING（迭代级：每步驱逐已完、塞入等待）
        step→  1 2 3 4 5 6 7 8 9 …
  slot0  R0→R4 █ █ █ █ █ █ █ █              R0 @3 完 → R4 @4 入，继续跑
  slot1  R1    █ █ █ █ █ █ █ █              R1（长）照跑，但不再堵住别人
  slot2  R2→R5 █ █ █ █ █ █ █ █              R2 @2 完 → R5 @3 入
  slot3  R3→R6 █ █ █ █ █ █ █ █              R3 @5 完 → R6 @6 入
               └── 腾出的槽位立刻回填，无需等待 ──┘
  利用率 ≈ 有色 / 总数  →  几乎没有 "·"
```

*（上图的 `R` 标号与步数是展示机制的示意草图——不是喂给 §4 仿真的那批具体请求。）*

三个要记的形状：

- **Static batching 按*请求*粒度调度；continuous batching 按*迭代*粒度调度。** 就这一个改动——*每步*决定谁在批里，而不是*每批*决定一次——就是全部思想。其余都是推论。
- **static batching 里的气泡就是全部问题。** 提前结束的序列在整批排空前无法让出槽位，所以短请求要为最长那个的尾巴买单。这还导致 **head-of-line blocking（队头阻塞）**：队列里等待的请求要等*一整批*空出才能开始，哪怕*此刻*就有槽位空着。
- **Continuous batching 把「空转到整批排空」变成「空转零步」。** 腾出的槽位在下一个迭代就被回填。批的构成不断变化；没有「批边界」可等。

## 3 · 原理——迭代级调度

### 3.1 引擎循环

continuous-batching 引擎跑一个循环。循环的每一转都是对当前*运行中*序列集合的**一次 forward**：

```text
loop forever:
    # 1. ADMIT 准入：只要有空间，就把等待请求拉进运行集
    while waiting and can_fit_next(waiting[0]):     # 空间 = KV-cache blocks + 槽位预算
        running.add(waiting.popleft())              # （新请求：它的 prefill 在这一步跑）

    # 2. STEP 步进：一次 forward——每个运行序列前进一个 token（decode），
    #          新准入的做它们的 prefill
    outputs = model.step(running)

    # 3. EVICT 驱逐：任何吐出 EOS 或达到 max_tokens 的序列就结束了
    for seq in running:
        if seq.done():
            free_kv_cache(seq)                      # 把它的 KV blocks 还给池子
            running.remove(seq); emit(seq)
```

三个阶段——**准入 → 步进 → 驱逐**——每个迭代重复。对比 static batching 的循环 `准入 N → 步进到全部完成 → 全部驱逐`——那个「步进到全部完成」正是制造气泡的部分。

### 3.2 什么在限制准入——是显存，不是算力

你会以为 batch 大小受算力限制。通常不是。Decode 是 **[memory-bound](../part0/inference-flow.md)**（Part 0）：每步把模型权重从 HBM 读一次，然后在批里*每个*序列上复用。所以往 decode 批里再加一个序列，**算力上几乎免费**——权重读取已经付过了——直到你撞上两堵墙之一：

- **KV-cache 容量。** 每个准入的序列都需要每 token 增长的 [KV cache](../part0/kv-cache.md) 存储。引擎只有在有空闲 KV block 装下上下文时才能准入。这是实践中*那道*绑定约束——也正是 [PagedAttention](paged-attention.md) 的意义所在：它消除碎片，让远多得多的序列能装下，于是批在撞墙前能长得更大。
- **compute ridge（算力屋脊）。** 堆足够多序列后，批处理的 GEMM 最终会打满 tensor core——你从 memory-bound 越过到 compute-bound（[roofline](../part2/roofline-analysis.md) 的屋脊）。过了它，更多序列只加延迟、不加吞吐。

两个 vLLM 旋钮直接给批设上限：

- **`max_num_seqs`**（默认 **128**）——运行集里的最大序列数。批*宽度*天花板。
- **`max_num_batched_tokens`**（默认 **2048**，但引擎会自动调）——一步内处理的最大 token 数（跨所有序列求和）。它约束每迭代的 prefill+decode 工作量（也是 chunked prefill 拧的那个旋钮——属 [Part 5 调度器](index.md) 话题）。

### 3.3 为什么这是*那个*吞吐杠杆

因为 decode 是 memory-bound，batch=1 几乎浪费了 GPU 全部算力：你读完所有权重只为产出一个 token。batch=32 读一次同样的权重却产出 32 个 token——*同样的*访存量换来约 32 倍有用功。Continuous batching 把批**在每一步都保持在 KV 容量允许的最满**，所以你总在那个摊薄甜点附近，而不是像 static 每轮尾巴那样排空到接近空批。这就是为什么它是每个推理引擎做的第一件事，也是面试官会探的第一件事。

## 4 · 完整可跑代码 + 逐行讲解

对*同一批*请求跑两种调度器的纯 Python 仿真，报告 GPU 槽位利用率与 makespan（完工时间）。它不用 GPU 就证明了机制——两个函数的唯一区别是*腾出的槽位何时被回填*。

```python title="batching_sim.py"
"""Static vs continuous batching —— 调度模型，不是 benchmark。
纯 Python、离线。每个请求需要固定的 decode 步数；
我们数每步有多少 GPU「槽位」在做有用功。"""
from collections import deque

# (request_id, num_steps_to_finish) —— 输出长度差异很大，像真实流量。
REQUESTS = [("R0", 2), ("R1", 12), ("R2", 3), ("R3", 2), ("R4", 10), ("R5", 2),
            ("R6", 4), ("R7", 2), ("R8", 8), ("R9", 3), ("R10", 2), ("R11", 6)]
SLOTS = 4                                             # 批宽度（类似 max_num_seqs）

def static_batching(requests, slots):
    """填满所有槽位，跑到批里每个序列都结束，才重填。"""
    q = deque(requests)
    busy_steps = total_slot_steps = step = 0
    while q:
        batch = [q.popleft() for _ in range(min(slots, len(q)))]   # 取一整批
        remaining = {rid: n for rid, n in batch}
        while any(r > 0 for r in remaining.values()):              # 步进到全部完成
            step += 1
            for rid in remaining:
                if remaining[rid] > 0:
                    remaining[rid] -= 1
                    busy_steps += 1                                # 这个槽位做了有用功
            total_slot_steps += slots                             # 全部 slots 都被占着
    return step, busy_steps, total_slot_steps

def continuous_batching(requests, slots):
    """迭代级：每步之后，驱逐已完序列、准入等待序列。"""
    waiting = deque(requests)
    running = {}                                                  # rid -> 剩余步数
    busy_steps = total_slot_steps = step = 0
    while waiting or running:
        while waiting and len(running) < slots:                   # 准入到空槽位
            rid, n = waiting.popleft(); running[rid] = n
        step += 1                                                 # 步进：一次 forward
        for rid in running:
            running[rid] -= 1; busy_steps += 1
        running = {rid: n for rid, n in running.items() if n > 0} # 驱逐已完成
        total_slot_steps += slots
    return step, busy_steps, total_slot_steps

if __name__ == "__main__":
    for name, fn in [("static", static_batching), ("continuous", continuous_batching)]:
        steps, busy, total = fn(REQUESTS, SLOTS)
        util = busy / total
        print(f"{name:>11}: makespan={steps:2d} steps | slot-utilization={util:5.1%} "
              f"({busy}/{total} slot-steps useful)")
```

**逐行讲解：**

- `REQUESTS`——12 个请求，输出长度**差异极大**（2–12 步）。这个跨度正是暴露差异的关键；如果每个请求一样长，static 和 continuous 会打平。
- `static_batching`——取一整批，然后内层 `while any(...)` 跑到**每个成员都结束**。每步 `total_slot_steps += slots` 按*所有*槽位被占计费，但 `busy_steps` 只数还在生成的槽位——差额就是空转气泡。这批要等内层循环退出（最长序列完成）才能重填。
- `continuous_batching`——同样的请求流，但循环在每步*之前***准入**任何空槽位、在每步*之后***驱逐**已完序列。第 *k* 步腾出的槽位在第 *k+1* 步被回填。字典推导式是驱逐阶段；`while waiting and len(running) < slots` 是准入阶段。
- 两个函数**只**在回填时机上不同。同样的请求、同样的槽位数、同样的每步工作——调度纪律就是全部差异。

预期输出（调度模型，不是 benchmark）：

```text
     static: makespan=30 steps | slot-utilization=46.7% (56/120 slot-steps useful)
 continuous: makespan=18 steps | slot-utilization=77.8% (56/72 slot-steps useful)
```

同样 56 个单位的有用功。Static 把它摊在 30 步、47% 利用率上；continuous 把它压进 18 步、78%——用约 1.7 倍更短的时间完成同样的请求。空转气泡——占着却不生成的槽位——是纯浪费，continuous batching 靠立刻回填把它收回来。（continuous 没到 100%，是因为在*尾部*剩下的等待请求太少、填不满每个腾出的槽位——一个真实效应。在稳态高负载流量下运行集始终满，利用率还会更高。）

## 5 · Lab——确认 vLLM 替你做了这件事（而且关不掉）

!!! gpu "GPU Lab（可选验证）"
    - **最低显存：** 读代码不需要；跑 `Qwen2.5-7B-Instruct`（INT4/AWQ）并观察批增减需 ~16 GB
    - **建议 AutoDL 卡型：** RTX 4090 (24 GB)
    - **预估耗时 / 花费：** 读 ~15 分钟（免费，无卡模式）· 可选运行 ~10 分钟 · ~¥1（示例）
    - **平台：** NVIDIA CUDA（默认）
    - **非 NVIDIA：** continuous batching 是*调度器*属性，与后端无关——AMD ROCm、TPU、CPU 版 vLLM 都这样调度；只有每步 kernel 不同。

要内化的关键：**你从不「开启」continuous batching——它就是调度器。** 只要并发发很多请求即可。

```python title="serve_and_load.py"
# 离线批：一次把很多 prompt 交给 vLLM；引擎会连续地调度它们。
# API 针对 vLLM 0.26.0 核实（LLM、SamplingParams）。在有 GPU 的 AutoDL 上跑。
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="awq",          # 把 7B 塞进 24 GB 的 4090（见 Part 4）
    max_num_seqs=128,            # 批宽度天花板（默认 128）——运行集上限
    # max_num_batched_tokens 由引擎自动调；除非在塑形 TTFT，否则别动
    gpu_memory_utilization=0.92, # 默认 0.92——引擎可用于 KV blocks 的显存占比
)
# 输出长度故意各异的 prompt——批每步都会增减成员。
prompts = ["Say hi.", "Write a 300-word essay on the sea.", "2+2?", "Explain PagedAttention."]
params  = [SamplingParams(max_tokens=n) for n in (8, 300, 4, 200)]

outputs = llm.generate(prompts, params)   # 引擎替你跨迭代准入/驱逐
for o in outputs:
    print(repr(o.outputs[0].text[:40]))
```

**观察什么：** 短 prompt（`"2+2?"`）远早于 300 词的 essay 完成并释放 KV block；有了 continuous batching，那些腾出的槽位在同一次 `generate` 调用内就被复用，而不是扣到 essay 完成。要*看到*它，用 `vllm serve Qwen/Qwen2.5-7B-Instruct --quantization awq` 起服务，观察日志里报 **"Running: N reqs, Waiting: M reqs"** 的那行——N 随请求逐步流过而一步步升降。没有 `--enable-continuous-batching` flag，因为没有东西可开。

## 6 · 常见坑 / 反直觉点

- **以为需要开启它。** Continuous batching *就是* vLLM 的调度器，没有 flag。错误在于找一个开关、找不到就假设 vLLM 做 static batching。它不做。
- **把 `max_num_seqs` 当成固定 batch 大小。** 它是运行集的*上限*，不是你填满再排空的目标。实际批浮动在它之下，受 KV 容量约束。
- **假设批越大越快。** 只在你撞上 KV-cache 容量或 [compute ridge](../part2/roofline-analysis.md) 之前成立。过了屋脊，更多序列只加延迟不加吞吐；在它之前（常见情形）decode 是 memory-bound，多加序列几乎免费。
- **无意中复现了 static batching。** 一个朴素循环，对固定列表调 `model.generate()` 并*等它全部完成*才发下一个列表——那就是你自己代码里的 static batching，你把引擎的连续调度扔掉了。请把请求流式送入；别在整批上设 barrier。
- **padding 浪费（HuggingFace `generate` 陷阱）。** Static batching 通常把所有序列 pad 到最长长度并计算这些 pad token——双重浪费（空转槽位*加上*在 padding 上浪费的算力）。paged KV cache 上的 continuous batching 没有 padding：每个序列只占它需要的 block。
- **把延迟归咎于 batching，其实是准入。** 若负载下 TTFT 飙升，原因通常是请求在*等* KV 空间（准入），不是 batching 纪律。修法是容量（[量化](../part4/index.md)、[KV-cache 量化](../part4/quantization-methods.md)、[PagedAttention](paged-attention.md)）或调小 `max_num_batched_tokens` 以优先新 prefill——而不是放弃 continuous batching。

## 7 · 面试连线

- [Static vs continuous batching：吞吐杠杆](../interview/continuous-batching.md)——本课为你准备的高频题：*为什么 static batching 浪费 GPU、「迭代级调度」是什么意思、以及到底什么在限制 batch 大小。*

## 8 · 小结 & 延伸阅读

**一句话：** Static batching 把一个固定批跑到最长成员结束，短请求因此留下空转气泡、排队请求遭队头阻塞；continuous batching（Orca 的迭代级调度）改为*每步 forward* 都决定批的成员——驱逐已完、准入等待——把批保持在 KV-cache 容量允许的最满，这是撬动推理吞吐最大的杠杆，因为 decode 是 memory-bound、多加序列几乎免费。

延伸阅读：

- Yu 等 —— *Orca: A Distributed Serving System for Transformer-Based Generative Models*（OSDI '22）—— 提出迭代级（continuous）batching 的论文。
- [PagedAttention 课](paged-attention.md) —— 为什么 KV-cache 容量（而非算力）通常限制准入，以及分页如何抬高那道天花板。
- [推理流程课](../part0/inference-flow.md) —— 为什么 decode 是 memory-bound，这是让 batching 近乎免费的前提。
- Part 5 下一课：[调度器](index.md)（chunked prefill、PD 分离）—— 引擎如何塑形*每步跑哪些* token，以平衡 TTFT 与吞吐。

## 9 · 自测小问

??? question "为什么 static batching 浪费 GPU？continuous batching 具体改了什么来修它？"
    Static batching 把一组固定序列锁在一起，跑到**最长**那个结束。因为输出长度差异很大，提前结束的序列在这批剩下的时间里让 GPU 槽位**空转**（气泡），排队请求要等整批排空才能开始（队头阻塞）。Continuous batching 把**调度粒度**从整个请求改成单个 decode **迭代**：每步 forward 之后驱逐已完序列（释放它们的 KV）、把等待序列准入腾出的槽位。批的构成每步都变，所以腾出的槽位在下个迭代就被回填，而不是空转到整批排空。

??? question "你往 decode 批里加序列，吞吐持续上升且几乎不加延迟——然后延迟突然跳升。你可能撞上的两堵墙是什么？哪个更常见？"
    （1）**KV-cache 容量**——每个序列需要每 token 增长的 KV 存储；空闲 KV block 用尽时，无法再准入序列，新请求就等（抬高 TTFT）。（2）**compute ridge**——足够多的批处理工作最终打满 tensor core，从 memory-bound 越过到 compute-bound，此后更多序列只加延迟不加吞吐。实践中 **KV-cache 容量是更常见的那堵墙**，这正是 PagedAttention（消除碎片 → 装下更多序列）与 KV-cache 量化对吞吐如此重要的原因。

??? question "同事说「我们应该在 vLLM 里开启 continuous batching 来提速」。这句话错在哪？你实际会去哪找吞吐问题？"
    没有东西可开——continuous batching **就是** vLLM 的调度器；它恒开、无 flag。若吞吐低，多半是批为 **KV-cache 空间**挨饿（序列在等准入），所以看容量：模型量化了吗（腾 VRAM 给 KV block）？`gpu_memory_utilization`（默认 0.92）是不是留了没用的余量？FP8 KV cache 或更大的有效 `max_num_seqs`（默认 128）能否装下更多序列？杠杆几乎总是**准入容量**，不是 batching 纪律——而容量正是 [PagedAttention](paged-attention.md) 与量化给你买来的东西。
