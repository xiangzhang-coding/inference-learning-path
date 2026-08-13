# 5. 图表策略：按图类型三分（Mermaid / ASCII / SVG），标签两树皆英文

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-13

## Context（背景）

「深化」阶段要给全站 74 页（EN + ZH 双树）大量补图，尤其是 9 段骨架的 **§2 心智模型/图**。当前这些图是 **ASCII art（```text 围栏）**，有一个关键优点：**语言中立**——同一段 ASCII 在 EN/ZH 两棵树里逐字复用、零翻译成本。

若无策略地全面改用 Mermaid，会撞上三个真实约束：
1. **i18n 成本翻倍**——Mermaid 块内嵌在 `.md` 里，`.zh.md` 要么复制英文标签、要么翻译，形成双份维护。
2. **节点内无法渲染 KaTeX**——roofline（$I^{*}=P/B$）、KV 数学等公式在 Mermaid 节点标签里显示不出来（本站 KaTeX 作用于页面文本，不进 Mermaid SVG 节点）。
3. **空间/定量图布局弱**——带坐标轴的 roofline 图、显存层级金字塔、tiling 网格、KV block 布局，Mermaid 的自动布局做不好；它擅长的是流程/时序/状态/拓扑这类**结构图**。

## Decision（决策）

采用**按图类型三分**的混合策略，而非「全用 Mermaid」：

1. **Mermaid** —— 用于**流程 / 时序 / 状态 / 拓扑**图（请求生命周期 sequence、调度器状态机 stateDiagram、TP/PP 拓扑 graph、continuous-batching 时间线 gantt）。这类 Mermaid 明显优于 ASCII（干净、语义化、可维护）。
2. **ASCII art（```text）** —— 用于**语言中立的结构草图**；现有已画得好的图（如 roofline 算子分解、vLLM V1 管线图）保留/升级，不为改而改。
3. **SVG / 图片**（或保留富 ASCII）—— 用于**空间 + 带公式**的少数图（roofline 图、显存金字塔、tiling 网格），Mermaid 与朴素 ASCII 都力不从心处。

**图内标签在 EN/ZH 两棵树都保持英文**——契合全站「术语保留英文」约定，把双语图的维护成本减半，也绕开 Mermaid 的公式短板。

**启用 Mermaid** 需在 `mkdocs.yml` 的 `pymdownx.superfences` 下加 `custom_fences`（映射到 Material 的 mermaid 渲染）；具体写法在实现阶段经 **Context7** 对照 Material 官方文档核实后落地。

## Consequences（后果）

**正面：**
- 每类图用最合适的工具：结构图语义清晰、空间图控制精确、语言中立图零翻译。
- 图内标签统一英文 → 双语维护成本减半、无 Mermaid 公式盲区。

**权衡 / 负面：**
- 三种工具意味着作者每图都要**选型**（本 ADR 即给出选型规则以降低这一负担）。
- 需给 `mkdocs.yml` 加 Mermaid 的 `custom_fences` 配置。
- SVG 图的作者/维护成本高于 ASCII 与 Mermaid，故仅用于确有必要的空间/公式图。

**可逆性：** 中。单张图可随时换工具，但一旦大量页面按此策略成图，整体风格与 i18n 约定即固化，故立此 ADR 记录选型规则。
