# GPU 内存层级与 roofline

!!! info "基线：**vLLM 0.26.0** · 经 Context7 核实（ADR-0004）"

**模块：** Part 0 · 基础   ·   **考察课程：** [GPU 硬件心智模型](../part0/gpu-hardware.md)

---

## Q：走一遍 GPU 内存层级与 SM/warp 执行模型，再用 roofline 模型解释为什么 LLM decode 是 memory-bound。拐点（ridge point）是什么，decode 与 prefill 相对它各在哪？

### 直接答案

GPU 有一座**内存金字塔**：片上的寄存器与 SRAM（L1/共享内存），~TB/s 到数十 TB/s；一层 L2 缓存；以及**片外的 HBM，~1 TB/s**——比 SRAM 慢约一个数量级。权重与 KV 缓存住在 HBM 里，所以每个 decode 步都要把它们拖过那条慢线。计算发生在 **SM**（流多处理器，4090 上约 128 个）上，每个 SM 锁步跑 32 线程的 **warp**；SM 靠切换 warp 隐藏内存延迟（**occupancy** = 有多少 warp 常驻可切换）。

**roofline** 说可达吞吐 $= \min(P,\ I\cdot B)$，其中 $P$ = 峰值 FLOP/s，$B$ = 带宽，$I$ = 算术强度（FLOP/字节）。**拐点** $I^{*} = P/B$ 是斜的带宽屋顶与平的算力屋顶相接处——对 4090（~165 TFLOP/s BF16 稠密，~1 TB/s）是 **≈165 FLOP/字节**。**Decode** 跑在 $I\approx1$，比拐点*低*约两个数量级，于是被钉在带宽屋顶上、~1 TFLOP/s——**< 1% 峰值，memory-bound**。**Prefill** 跑在 $I$ 数千，在拐点*之上*、平的算力屋顶上——**compute-bound**。同一张 GPU，相反的状态。

### 深入原理

- **为什么 HBM 是反派。** HBM 与 SRAM 之间的带宽差（一个数量级）意味着：一个本可留在 SRAM 却从 HBM 重读的算法，每次访问都要付这个罚。这就是 FlashAttention 这类 IO-aware kernel 的全部动机（把注意力工作集留在 SRAM、避开 HBM 往返）。
- **占用率隐藏延迟，而非带宽。** 一旦常驻 warp 多到足以让内存管子饱和，对 memory-bound kernel 而言再多也无益——你受限于字节/秒，而非空闲 SM。这就是为什么「提高占用率就行」修不了 decode。
- **可达 vs 峰值。** 真实 kernel 摸到规格书 HBM 带宽的 ~70–85%，而头条峰值 TFLOPS 常假设 2:4 稀疏或 FP8（4090 的「≈330 TFLOPS」是稀疏值；稠密 BF16 约一半）。永远补一句「稠密还是稀疏？什么 dtype？」。
- **杠杆如何挪动 roofline。** 量化削字节（更高 $I$ *且* 更少要搬的字节 → 直接的 decode 胜利）；批处理跨请求复用权重（更高 $I$）；IO-aware kernel 绕开 HBM（等效更多可用 $B$）。三者都攻*带宽*那侧，因为那是绑定约束。

### 代码

状态从 `min(P, I·B)` 里自然掉出——不需要 GPU：

```python
P, B = 165e12, 1.0e12          # 4090：~165 TFLOP/s BF16 稠密，~1 TB/s（示例）
ridge = P / B                  # I* = 165 FLOP/字节
for I in (1, 1000):            # decode ~1，prefill ~1000
    got = min(P, I * B)
    print(f"I={I:>4}: {got/1e12:6.1f} TFLOP/s ({got/P:5.1%} of peak)")
# I=   1:    1.0 TFLOP/s ( 0.6% of peak)   <- decode，memory-bound
# I=1000:  165.0 TFLOP/s (100.0% of peak)  <- prefill，compute-bound
```

### 面试官追问

- *「厂商把 TFLOPS 翻倍但带宽不变。decode 会提速吗？」* → 不会——decode 在拐点以下、被钉在 $I\cdot B$，而 $B$ 没变。Prefill 会（它骑上更高的 $P$ 屋顶）。decode 要带宽，prefill 要 FLOPs。
- *「FLOPs 一样，FlashAttention 为何更快？」* → 它 IO-aware：tiling + online softmax 把注意力工作集留在 SRAM，而非把巨大的中间分数矩阵在 HBM 里往返，削减了搬运字节——在带宽受限那侧取胜。
- *「量化把你在 roofline 上挪到哪？」* → 往右（更高 $I$、更少字节/参数）*且*降低必须跨 HBM 的字节——对 memory-bound 的 decode 阶段是直接的吞吐倍增。
- *「一个 100% 占用率却仍慢的 kernel 被什么限制？」* → 带宽。占用率已隐藏了它能隐藏的全部延迟；kernel 现在受限于 $B$（字节/秒），所以你要更少字节或更高强度，而非更多 warp。

### 关联知识点

- 课程：[GPU 硬件心智模型](../part0/gpu-hardware.md)
- 相关课程：[推理流程：Prefill 与 Decode](../part0/inference-flow.md)（decode 的 $I\approx1$ 从哪来）
- 术语：[SM / Warp / Occupancy、HBM / SRAM、Roofline、FlashAttention](../glossary.md)
