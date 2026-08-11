# Part 6 · 进阶推理专题

> 基础扎实后，面试常问的那些专门服务形态。

## 本 Part 覆盖

- **Multi-LoRA serving**：一份基座 + 多 adapter 动态切换
- **Guided / structured decoding**：受约束的 JSON / 正则 / 语法输出
- **长上下文推理**：RoPE 外推、attention sink、KV 压缩，及长序列的显存/调度问题

长上下文推理直接压在 [KV 缓存](../part0/kv-cache.md)的增长问题上。

!!! note "脚手架状态"
    本 Part 课程在后续票落地。见 **[术语表](../glossary.md)**。
