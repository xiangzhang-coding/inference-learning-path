# 显存预算与最大并发

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 2 · 单卡推理性能   ·   **考察课程：** [KV 缓存显存数学：为部署做容量规划](../part2/kv-cache-math.md)

---

## Q：Qwen2.5-7B 在单张 24 GB RTX 4090、8k 上下文。走一遍完整显存预算并估最大并发序列。然后：你的 SLO 需要 ~60 条并发流——怎么达到？

### 直接答案

并发是个**剩余量**：可用显存减固定成本，除以每序列 KV。

$$
N_{\text{seq}} = \left\lfloor \frac{u\cdot V - W - A - O}{\kappa\, S} \right\rfloor
$$

其中 $u$ = `gpu_memory_utilization`（默认 0.92）、$V=24$ GiB、$W$ = 权重、$A$ = 激活、$O$ = CUDA/框架开销，$\kappa=56$ KiB/token 对 Qwen2.5-7B（见 [Part 0](../part0/kv-cache.md)）。取 $u=0.90$、$A+O\approx1.6$ GiB、$S=8192$（故 $\kappa S=0.44$ GiB/序列）：

- **BF16 权重（~14.2 GiB）：** 预算 $=0.9\cdot24-14.2-1.6=5.8$ GiB → $\approx$ **13** 并发。
- **AWQ 4-bit 权重（~5.5 GiB）：** 预算 $=14.5$ GiB → $\approx$ **33**。
- **AWQ 权重 + FP8 KV**（$\kappa\to28$ KiB）：$\approx$ **66**。

**要达到 ~60：**（1）量化**权重**（AWQ/GPTQ）——释放 ~8 GiB，单项最大的一块（~13→~33）；（2）量化 **KV**（`kv_cache_dtype=fp8`）——每序列字节减半（~33→~66，越过 60）；（3）若还不够，**封 `max_model_len`** 到真实工作负载上下文（并发 $\propto 1/S$）。先权重，因为它是最大的固定成本。

### 深入原理

- **为什么权重先于 KV。** 24 GB 上权重主导预算；量化它直接释放 ~8 GiB 进 KV，通常比把 KV 字节减半的并发收益更大。KV 量化是第二杠杆，不是第一。
- **省不掉的开销。** `gpu_memory_utilization` 把可用显存压到 24 GB 以下，CUDA context + 激活/workspace 在任何 KV 前吃 ~1–2 GiB。Part 0 朴素的「~22」忽略了这些；诚实的数更低。
- **`max_model_len` 是权衡。** 它界定每序列 KV，故为长上下文抬它会按比例砍并发。设成工作负载，而非模型 32k 上限。
- **PagedAttention vs 公式。** 分页按固定大小 block 分配；序列最后一块半空，故真实并发略低于公式。但它*消除外部碎片*，让剩余量比连续分配可用得多——这正是 vLLM 报告的容量能贴近算术的原因。

### 代码

规划器，以及逆问题：

```python
GIB = 1024**3
def kv_budget(util, vram, weight_gib, overhead=1.6):
    return util*vram - weight_gib - overhead
def max_conc(util, vram, weight_gib, kappa, S):
    return int(kv_budget(util, vram, weight_gib)*GIB / (kappa*S))

print(max_conc(0.90, 24, 14.2, 57344, 8192))   # ~13  BF16 权重，BF16 KV
print(max_conc(0.90, 24, 5.5,  57344, 8192))   # ~33  AWQ  权重，BF16 KV
print(max_conc(0.90, 24, 5.5,  28672, 8192))   # ~66  AWQ  权重，FP8  KV
```

拿 vLLM 自己的启动日志对账（v0.26.0）：`GPU KV cache size: N tokens`（$=$ 预算$/\kappa$）与 `Maximum concurrency for 8,192 tokens per request: Xx`（$=$ 它 $/$ `max_model_len`）。

### 面试官追问

- *"为什么不干脆设 `gpu_memory_utilization=0.99`？"* → 激活随并发 prefill 飙升；余量太薄启动正常、峰值 OOM。0.92 默认是故意留的余地。
- *"PagedAttention 怎么改变这个数？"* → 它不改每序列公式；它靠消除外部碎片让剩余量更多*可用*（减去小的最后一块补齐）。它抬的是有效容量，不是 $\kappa$。
- *"`kv_cache_dtype=fp8` 的代价？"* → 它把 KV 字节减半（更多并发）但可能移动输出——把质量差当成要在你评测集上*测量*的东西，别假设免费。
- *"在这张卡上服务 128k 上下文——可行吗？"* → 一条 128k 序列是 $\kappa\cdot128\text{k}\approx7$ GiB（BF16 KV）；几条就吃满预算。你得上 FP8 KV、激进权重量化、并接受极低并发——或跨 GPU 分片。

### 关联知识点

- 课程：[KV 缓存显存数学：为部署做容量规划](../part2/kv-cache-math.md)
- 相关：[KV 缓存与吞吐上限](kv-cache.md)（为什么是 KV 而非算力封住并发）、[GEMM 与 attention 的算术强度](arithmetic-intensity.md)
- 术语表：[KV cache、PagedAttention、GQA、SLO](../glossary.md)
