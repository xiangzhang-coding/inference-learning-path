# Part 5 · 服务化与吞吐（vLLM 核心）

> 整条路径的心脏。「最大化并发与后端吞吐」真正在这里拿下。

## 本 Part 覆盖

- 从 **static → continuous batching**：吞吐的第一杠杆
- **PagedAttention**：像虚拟内存一样管理 [KV 缓存](../part0/kv-cache.md)——vLLM 高吞吐的根源
- **调度器**：chunked prefill、PD 分离——调 TTFT/吞吐平衡
- **Prefix caching** 与 **speculative decoding**：合适场景下进一步提速
- vLLM **端到端架构地图**：engine / scheduler / block manager / worker
- 核心**调参旋钮**及各自如何移动吞吐/延迟曲线

!!! note "脚手架状态"
    本 Part 课程在后续票落地。所有 vLLM flag/API 经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**。见 **[术语表](../glossary.md)**。
