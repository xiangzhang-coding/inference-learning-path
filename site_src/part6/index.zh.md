# Part 6 · 进阶推理专题

> Part 5 基础扎实后，面试常问的那些专门服务形态——「一个用例一个模型」和「模型只吐纯文本」这两条，在这里都不够用了。

## 本 Part 覆盖

- **Multi-LoRA serving**：一份基座 + 多个低秩 adapter，按请求切换、甚至*在同一个 batch 内混用*——如何在一张 [24 GB 卡](../part0/gpu-hardware.md)上服务几十个微调，而不是每个都存一份完整副本。
- **Guided / structured decoding**：在每个 decode 步把不可能的 token 屏蔽掉，把输出约束成合法的 **JSON / 正则 / 语法 / 枚举**——把「prompt 后祈祷」变成「构造上就 schema 合法」。
- **长上下文推理** *(后续票)*：RoPE 外推、attention sink、KV 压缩，及长序列的显存/调度问题——它直接压在 [KV 缓存](../part0/kv-cache.md)的增长问题上。

这里两个专题都跑在 Part 5 同一套 [PagedAttention block 池](../part5/paged-attention.md)与 [continuous batching](../part5/continuous-batching.md) 机制之上——它们是吞吐搞定之后你*叠加*上去的东西。

## 课程

- **[Multi-LoRA Serving：一基座，多 adapter](multi-lora-serving.md)** — 为什么朴素地服务 N 个微调意味着 N 份完整模型副本（以及为什么这在一张卡上根本不可能），[LoRA](../glossary.md) 的低秩增量 $\Delta W = BA$ 如何把每个微调缩到几 MB，vLLM 如何保持一份冻结基座 + 一排 adapter、并通过 grouped GEMM kernel 给*同一个* batch 的*不同*行应用*不同*的 adapter，以及决定你能同服多少个的旋钮（`--max-lora-rank`、`max_loras`、动态加载）。
- **[Guided / Structured Decoding：让非法 token 不可能出现](structured-decoding.md)** — 为什么让模型输出 JSON 仍会有一定比例吐出坏 JSON，schema/正则/语法如何编译成一个在每步产出 **token 掩码**的有限状态机，vLLM 如何把不允许的 logits 设成 $-\infty$、使得只有 schema 合法的 token 能被采样（xgrammar/guidance 后端），以及每个面试官都会戳的那条锋利边界：它保证*形状*，从不保证*真值*。

!!! note "Part 6 状态"
    本票（#15，「Part 5A」）落地前两节课——**[multi-LoRA serving](multi-lora-serving.md)** 与 **[structured decoding](structured-decoding.md)**——各带一道双向链接的面试题（[multi-LoRA](../interview/multi-lora-serving.md)、[structured decoding](../interview/structured-decoding.md)）。**长上下文推理**在下一票（#16）跟进。所有 vLLM flag/API 均经 Context7 核实（ADR-0004）；基线为 **vLLM 0.26.0**，一切性能数字均为**示例 / 量级参考**。见 **[术语表](../glossary.md)**与[面试题库](../interview/index.md)。
