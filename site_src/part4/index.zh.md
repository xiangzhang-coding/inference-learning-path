# Part 4 · 量化

> 量化为何能提吞吐、代价是多少精度、真实场景怎么选型。

## 本 Part 覆盖

- 量化为何有助吞吐，以及**精度权衡**
- **Weight-only vs weight+activation**、粒度（per-tensor/channel/group）、对称/非对称
- 方法族：**GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8()** 与 **KV-cache 量化**
- 一段完整可跑路径：量化 `Qwen2.5-7B-Instruct` 并在 vLLM 里跑 INT4

KV-cache 量化直接连回 **[KV 缓存](../part0/kv-cache.md)**。

!!! note "脚手架状态"
    本 Part 课程在后续票落地。量化术语见 **[术语表](../glossary.md)**。
