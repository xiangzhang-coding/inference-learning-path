# CUDA graphs 与 kernel fusion

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 2 · 单卡推理性能   ·   **考察课程：** [Kernel Fusion 与 CUDA Graphs：干掉 decode 的 launch overhead](../part2/kernel-fusion-cuda-graphs.md)

---

## Q：模型已加载，decode 步 GPU 利用率却很低。除了 memory-bound，decode 还付什么税？解释 kernel launch overhead、为何它伤 decode 而非 prefill、kernel fusion 与 CUDA graphs 如何应对，以及 `enforce_eager` 权衡了什么。

### 直接答案

额外的税是 **kernel launch overhead（启动开销）**。一个 decode 步跑*几百*个小 kernel（每层：norm、QKV/O projection、attention、gate/up/down、activation、残差加法——×~28 层）。每次启动 CPU 要花几微秒派发，而因为 decode kernel 很小（batch 1、memory-bound），GPU 算完每个就**空等 CPU 启动下一个**。把一步建模为 $T_{\text{eager}} \approx T_{\text{compute}} + N\tau$：$N\approx430$ 个 kernel、$\tau\approx5\,\mu s$，就是每步 ~2 ms 的纯启动税。

它**伤 decode 而非 prefill**，因为 prefill kernel 大（多 token、compute-bound），$N\tau \ll T_{\text{compute}}$——启动藏在真实工作后面。decode 每 kernel 的 GPU 工作太小，启动主导。

- **Kernel fusion** 把算子并成一个 kernel：更少启动（$N$ 降）*且*更少 HBM 往返（中间量留在 SRAM/寄存器）。
- **CUDA graphs** 把整串 kernel 序列录制一次、用*单*次 CPU 提交重放：$T_{\text{graph}} \approx T_{\text{compute}} + \tau$，塌缩 $N\tau$。

**`enforce_eager=True`** 禁用 CUDA-graph 捕获（与 torch.compile）：它**释放**捕获图会占的 VRAM，但每 decode 步**付启动税**——拿吞吐换内存。

### 深入原理

- **量化模型获益*更多*。** 启动税 $N\tau$ 固定，故缩小计算（AWQ 权重 → 更小 $T_{\text{compute}}$）让它占比更大：6 ms 步的 ~36% vs 15 ms 步的 ~14%。优化计算 → 图更重要，不是更不重要。
- **静态形状约束。** 图捕获固定形状与内存地址；vLLM 按 batch-size 桶各捕一张图，并把运行 batch **补齐**到它。新形状 / 数据相关控制流回退到 eager。
- **预热 + 静态 buffer。** 捕获前必须跑确切工作负载（让分配/autotune 稳定），每次重放把输入拷进同一组静态 buffer，因为图复用捕获的指针。
- **fusion ≠ graphs。** fusion 降 kernel *数量*；图降剩下 kernel 的*启动成本*。`torch.compile` 做 fusion；vLLM 两者都做。

### 代码

launch-overhead 模型——为什么量化 decode 获益更多（纯 CPU）：

```python
def step_ms(weight_gib, n_kernels=430, tau_us=5.0, bw=1e12):
    compute = weight_gib * 1024**3 / bw * 1e3      # 字节 / 带宽（ms）
    eager = compute + n_kernels * tau_us / 1e3
    graph = compute + tau_us / 1e3
    return compute, eager, graph, eager / graph
for w in (5.5, 14.2):                              # AWQ vs BF16 权重
    c, e, g, spd = step_ms(w)
    print(f"{w:>4} GiB: compute {c:4.1f}ms eager {e:4.1f}ms graph {g:4.1f}ms speedup {spd:.2f}x")
# 5.5 GiB: compute  5.9ms eager  8.1ms graph  5.9ms speedup 1.36x
# 14.2 GiB: compute 15.2ms eager 17.4ms graph 15.3ms speedup 1.14x
```

### 面试官追问

- *"为什么 CUDA graph 帮 decode 不帮 prefill？"* → 启动开销的影响 = $N\tau$ 相对 $T_{\text{compute}}$。prefill kernel 大/compute-bound（$N\tau$ 可忽略）；decode kernel 小/memory-bound（启动主导）。
- *"你开了图但吞吐没动——为什么？"* → 捕获可能没生效：动态形状落在捕获桶之外、不支持的算子、或回退到 eager。检查形状是否命中捕获的桶。
- *"什么时候你会*想*用 `enforce_eager`？"* → VRAM 紧张时（把图 buffer 收回给 [KV cache](vram-capacity-planning.md)）或调试时——接受更低的 decode 吞吐。
- *"光 fusion 能让每步只一次启动吗？"* → 不能——fusion 减少数量，但剩下每个 kernel 仍要启动。只有 CUDA graph 把整串序列塌缩成单次提交。

### 关联知识点

- 课程：[Kernel Fusion 与 CUDA Graphs：干掉 decode 的 launch overhead](../part2/kernel-fusion-cuda-graphs.md)
- 相关：[GEMM 与 attention 的算术强度](arithmetic-intensity.md)（为什么 decode 每 kernel 工作那么小）、[FlashAttention 与 IO-aware attention](flash-attention.md)、[显存预算与最大并发](vram-capacity-planning.md)（`enforce_eager` 释放的内存）
- 术语表：[CUDA graphs、Kernel fusion、Memory-bound / Compute-bound](../glossary.md)
