# 抢占（Preemption）：KV 池耗尽时的 recompute vs swap

!!! info "基线：**vLLM 0.26.0** · flag 经 Context7 核实（ADR-0004）"

**模块：** Part 5 · 服务化与吞吐（vLLM 核心）   ·   **考察课程：** [PagedAttention：像虚拟内存一样管理 KV Cache](../part5/paged-attention.md)

---

## Q：当 vLLM 给运行中的 batch 找不到 KV block 时会怎么做？解释 preemption、recompute vs swap，以及 vLLM 的 V1 行为。

### 直接答案

当 KV 池无法为运行中的序列扩容（或无法容纳需要的序列）时，调度器**抢占（preempt）**：驱逐一个运行中的序列、回收其 KV block、把它重新入队，待有空间时恢复——相当于操作系统换出一页。这是**优雅降级**，不是崩溃。

恢复时有两种经典模式：

- **Recompute（重算）**——丢弃被驱逐的 KV；序列恢复时重跑 prefill、重建 KV。代价 = 重算 FLOPs；**无 CPU 传输**。
- **Swap（换出/换入）**——把被驱逐的 KV 拷到 CPU 内存（swap space），恢复时再拷回。代价 = PCIe **来回**传输；需要 `swap_space`。

**vLLM V1（0.26.0 的默认引擎）默认用 `RECOMPUTE`**——开销低于 swap。事实上 V1 **移除了 swap**：`--swap-space` flag 已去除，换出相关的指标（`vllm:num_requests_swapped`、`vllm:cpu_cache_usage_perc`）在 V1 已不再相关；**prefix caching** 通过复用已缓存的 block，给了重算一条近乎零开销的路径。你通过 Prometheus 计数器 **`vllm:num_preemptions`** 观测抢占（自启动累计；用 `disable_log_stats=false` 开启统计）：

```text
vllm:num_preemptions   # 自启动以来的累计抢占数（Prometheus 计数器）
```

（V1 不再逐次打印警告：V0 那条 `WARNING ... Sequence group ... PreemptionMode.RECOMPUTE ... total_cumulative_preemption_cnt` 已移除——别去 grep 它。）

### 深入原理

- **V1 为什么选重算：** 重算*就是* prefill——算力受限、在 GPU 上快——而 swap 要把数 MB 的 KV 搬过 PCIe，慢且带宽争用。配合 prefix caching，重算能命中缓存 block，往往近乎免费。
- **抢占是症状，不是旋钮。** 它意味着 KV 供给不足。解法是**容量**：↑ `gpu_memory_utilization`、量化（权重或 FP8 KV）、↓ `max_num_seqs` / `max_num_batched_tokens`，或 ↑ `tensor_parallel_size` / `pipeline_parallel_size` 把 KV 摊到更多卡。
- **运行集会缩，不只是涨。** 你以为「运行中」的序列会在压力下被驱逐再重新入场——稳定的低吞吐加上运行计数抖动就是信号（[连续批处理](continuous-batching.md)的 `schedule()` 抢占路径）。
- **抢占 ≠ OOM。** 抢占让服务存活、只是加延迟；OOM 是硬崩溃。抢占正是让 vLLM 在需求超过 KV 时仍稳健的机制。

### 代码

一个极简调度器模拟——固定 block 池；接纳请求；block 用尽时抢占最新的运行序列并重新入队（纯 Python，示意——计数为示例）：

```python
from collections import deque

def run(pool_blocks, arrivals, blocks_per_req):
    free, running, waiting, preemptions = pool_blocks, [], deque(arrivals), 0
    while waiting or running:
        # 有空间就接纳
        while waiting and free >= blocks_per_req:
            running.append(waiting.popleft()); free -= blocks_per_req
        # 某运行序列还要一个 block 但池已空 -> 抢占
        if running and free == 0:
            victim = running.pop()            # 驱逐最新的（LIFO）
            free += blocks_per_req            # 回收其 KV（RECOMPUTE：直接丢弃）
            waiting.append(victim)            # 重新入队，稍后恢复
            preemptions += 1
        # 前进一步：退休最老的运行序列
        if running:
            running.pop(0); free += blocks_per_req
    return preemptions

print(run(pool_blocks=4, arrivals=list(range(6)), blocks_per_req=2))  # 示例
```

### 面试官追问

- *「recompute vs swap 哪个便宜？」* → 重算（prefill 快、算力受限）；swap 要付来回 PCIe。V1 默认重算并彻底去掉了 swap。
- *「累计抢占计数在涨——你改什么？」* → 改容量，不是调度器：↑ `gpu_memory_utilization`、量化 / FP8 KV、↓ `max_num_seqs` / `max_num_batched_tokens`，或 ↑ TP/PP。
- *「抢占和 OOM 一样吗？」* → 不一样。抢占是优雅的（驱逐 + 恢复，代价是延迟）；OOM 会崩掉引擎。抢占正是防止在 KV 压力下崩溃。
- *「prefix caching 如何与重算配合？」* → 被重算的序列能命中 prefix cache，于是「重算」往往近乎免费——这也是 V1 能去掉 swap 的原因之一。
- *「什么指标能看出它在发生？」* → `vllm:num_preemptions` 这个 Prometheus 计数器在涨（或用 `disable_log_stats=false` 观测）——V1 已不再逐次打印警告。

### 关联概念

- 课程：[PagedAttention：像虚拟内存一样管理 KV Cache](../part5/paged-attention.md)——block manager 与压力下的抢占路径在此。
- 相关：[PagedAttention：block manager 与碎片](kv-cache-block-manager.md)、[静态 vs 连续批处理](continuous-batching.md)（`schedule()` 抢占路径）、[调参旋钮](tuning-knobs.md)（阻止它的容量旋钮）、[前缀缓存](prefix-caching.md)、[VRAM 预算与最大并发](vram-capacity-planning.md)
- 术语：[Preemption、PagedAttention](../glossary.md)
