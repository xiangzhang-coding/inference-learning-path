# Part 1 · Transformer（Infra 视角）

> 还是你训练时熟悉的那个 Transformer，但换一副**推理成本的眼镜**重读：每个架构选择都直接映射到显存与吞吐。

## 本 Part 覆盖

- **Q / K / V → KV cache**：缓存从哪来、注意力为何需要历史
- **MHA / MQA / GQA**：KV 头越少 → [KV 缓存](../part0/kv-cache.md)越小 → 吞吐上限越高
- **FFN / MLP**：大部分 FLOPs 与权重显存在哪
- **RoPE**：旋转位置编码，及其外推为何对长上下文重要

架构术语见 **[术语表](../glossary.md)**。

!!! note "脚手架状态"
    本 Part 课程在后续票落地。它们依赖的基础在 **[Part 0](../part0/index.md)**。
