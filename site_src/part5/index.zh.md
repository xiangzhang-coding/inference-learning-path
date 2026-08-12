# Part 5 · 服务化与吞吐（vLLM 核心）

> 整条路径的心脏。「最大化并发与后端吞吐」真正在这里拿下。

## 本 Part 覆盖

- 从 **static → continuous batching**：吞吐的第一杠杆
- **PagedAttention**：像虚拟内存一样管理 [KV 缓存](../part0/kv-cache.md)——vLLM 高吞吐的根源
- **调度器**：chunked prefill、PD 分离——调 TTFT/吞吐平衡
- **Prefix caching** 与 **speculative decoding**：合适场景下进一步提速
- vLLM **端到端架构地图**：engine / scheduler / block manager / worker
- 核心**调参旋钮**及各自如何移动吞吐/延迟曲线

## 课程

- **[从 Static 到 Continuous Batching](continuous-batching.md)** —— 为什么 static batching 让 GPU 空转（气泡、队头阻塞），Orca 的迭代级调度（每步驱逐已完、准入等待）如何把批保持满载，以及为何这是吞吐的第一杠杆——因为 decode 是 memory-bound。
- **[PagedAttention：KV Cache 即虚拟内存](paged-attention.md)** —— block manager 如何消除压低并发的内部碎片：池里的固定大小块（`num_gpu_blocks` 由 profiling 定）、每序列一张 block table、随长随分与用完即还、前缀块共享——以及收回的 VRAM 如何变成更大的 continuous batch。（*读*这些块的 [kernel](../part3/paged-attention-kernel.md) 在 Part 3。）

!!! note "脚手架状态"
    Batching + PagedAttention（票 #12）已落地，各配一道双向链接的面试题：[continuous batching](../interview/continuous-batching.md) 与 [block manager](../interview/kv-cache-block-manager.md)。仍待落地：**调度器**（chunked prefill、PD 分离）、**prefix caching** 与 **speculative decoding**、vLLM **架构地图**、**调参旋钮**（后续票）。所有 vLLM flag/API 经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**。见 **[术语表](../glossary.md)** 与 [面试题库](../interview/index.md)。
