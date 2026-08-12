# Static vs continuous batching：吞吐杠杆

!!! info "基线：**vLLM 0.26.0** · flag 经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [从 Static 到 Continuous Batching](../part5/continuous-batching.md)

---

## Q：解释 static vs continuous batching。为什么 continuous batching 是推理吞吐最大的杠杆？「迭代级调度」是什么意思？到底什么在限制 batch 大小？

### 直接答案

**Static batching** 凑一组固定请求一起跑，直到**每个**成员都结束，再取下一组。因为输出长度差异很大，提前结束的序列让 GPU 槽位**空转**到最长那个跑完（气泡 bubble），排队请求要等整批排空才能开始（**队头阻塞 head-of-line blocking**）。

**Continuous batching**（Orca 的*迭代级调度*）把调度粒度从整个请求改成**单个 decode 迭代**：每步 forward 之后**驱逐**已完序列（释放它们的 [KV cache](../part0/kv-cache.md)）、把等待序列**准入**腾出的槽位。批的成员每步都变，所以没有槽位在有活等着时空转。

它是*那个*吞吐杠杆，因为 **decode 是 [memory-bound](../part0/inference-flow.md)**：每步把模型权重从 HBM 读一次，然后在批里每个序列上复用，所以多一个序列在算力上几乎免费。把批保持尽量满就最大化这种摊薄。限制批的几乎总是 **KV-cache 容量**（有空闲块才能再准入一个序列），不是算力——直到你最终越过 [roofline](../part2/roofline-analysis.md) 的 compute ridge（算力屋脊）。

### 深入原理

- **两个循环。** Static：`准入 N → 步进到全部完成 → 全部驱逐`。Continuous：`准入 → 步进（各一个 token）→ 驱逐 → 重复`。那个「步进到全部完成」正是制造气泡的部分。
- **为什么 memory-bound 让它成立。** batch=1 读完所有权重只产一个 token——几乎纯浪费。batch=32 读一次同样权重产 32 个 token——同样 HBM 访存换约 32 倍有用功。Continuous batching 让你每步都在那个甜点附近。
- **两堵墙。** （1）KV-cache 容量——常见那堵；序列越多需要越多 KV block，池子空了准入就停。这就是 [PagedAttention](../part5/paged-attention.md)（消除碎片 → 更多块）与 KV-cache 量化重要的原因。（2）compute ridge——足够的批处理 GEMM 打满 tensor core；过了它，延迟升、吞吐不升。
- **旋钮。** `max_num_seqs`（默认 **128**）限运行集宽度；`max_num_batched_tokens`（默认 **2048**，自动调）限每步处理的 token 数。两者都*不是* batch 大小——批浮动在它们之下，受 KV 空间约束。
- **你不「开启」它。** Continuous batching *就是* vLLM 的调度器——没有 flag。并发发请求即可。

### 代码

调度差异的纯 Python——只有回填时机不同：

```python
from collections import deque
def continuous(requests, slots):          # 迭代级：每步驱逐已完、准入等待
    waiting, running, step, busy = deque(requests), {}, 0, 0
    while waiting or running:
        while waiting and len(running) < slots:      # 准入到任意空槽位
            rid, n = waiting.popleft(); running[rid] = n
        step += 1
        for rid in running: running[rid] -= 1; busy += 1   # 步进：各一个 token
        running = {r: n for r, n in running.items() if n > 0}  # 驱逐已完
    return step, busy
# 对比 static：它会把同一批「步进到全部完成」才重填——留下气泡。
```

腾出的槽位在下个迭代就被回填，而不是扣到整批排空。

### 面试官追问

- *「吞吐低但 GPU 没打满——你看哪？」* → 看准入，不是 batching。批多半为 KV 空间挨饿：模型量化了吗（腾 VRAM 给块）？`gpu_memory_utilization`（0.92）留了余量？FP8 KV cache 能装更多序列吗？杠杆是容量（[PagedAttention](../part5/paged-attention.md)、量化）。
- *「批越大总是好吗？」* → 只到 KV 容量或 compute ridge 之前。屋脊前（memory-bound decode），多加序列几乎免费；过了它，只加延迟不加吞吐。
- *「你自己代码里怎么会不小心变成 static batching？」* → 一个循环把固定列表交给 vLLM 并*等它全部完成*才发下一个列表，就在整批上设了 barrier——你把连续调度扔了。应把请求流式送入。
- *「padding 差别是什么？」* → Static batching 通常把所有序列 pad 到最长并计算 pad token（双重浪费）；paged KV cache 上的 continuous batching 没有 padding——每个序列只用它需要的块。
- *「chunked prefill 放哪？」* → 它是那个旋钮（`max_num_batched_tokens`），让 prefill 与 decode 共享一步，平衡 TTFT vs 吞吐——continuous batching 之上的调度器精化（Part 5 下一话题）。

### 关联概念

- 课程：[从 Static 到 Continuous Batching](../part5/continuous-batching.md)
- 相关：[PagedAttention：KV cache 即虚拟内存](kv-cache-block-manager.md)（限制准入的容量）、[Prefill vs decode](prefill-vs-decode.md)（为何 decode memory-bound）、[KV 缓存与吞吐上限](kv-cache.md)、[显存预算与最大并发](vram-capacity-planning.md)、[算术强度](arithmetic-intensity.md)（compute ridge）
- 术语：[Static / Dynamic / Continuous batching](../glossary.md)
