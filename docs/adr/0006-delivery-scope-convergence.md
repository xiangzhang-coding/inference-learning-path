# 6. 交付边界收敛：题库精选（非「~100」）、Part 1 并入 Part 0、公开前清理「施工」话术

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-15
- **Amends（修订）**: spec（issue #1）Implementation Decisions 两条 ——「题库 schema：约 100 道高频题」与「课程结构：新增 **Transformer(infra 视角)**（作为独立 Part 1）」。本 ADR 就这两处的实际偏离作正式记录，优先级高于 spec 相应条款。

## Context（背景）

全部 tickets（#2–#23）落地后做了一次全仓质量审查 + grilling。站点定位得到澄清：**私人学习产物，同时可公开**；Cloudflare Pages 部署延后、不在本轮。在此定位下，三处与 spec 原始设想的偏离需要定夺：

1. **题库规模。** spec 定「~100 道」；实测 **38 道**（36 主题各 1 题 + `system-design` 3 题），且 `interview/index.md` 自述「a growing bank」「~100-question bank grows as parts land」。独立审计结论：现有 38 题**质量全部达标**（精准直答 + KaTeX 推导 + 可跑无 GPU 代码 + 面试官追问 + 双向链接，vLLM 0.26.0 细节正确），**但作为面向大厂推理岗的精选集，漏了几道近乎必问的开胃题**——采样参数、MoE 推理、preemption。
2. **Part 1。** spec 把「Transformer（infra 视角）」列为独立 Part；实现中它是 nav 顶层的一个**空壳占位页**（"lessons land in a later ticket"），而其四个子主题（Q/K/V→KV cache、MHA/MQA/GQA、FFN、RoPE）已在 `part0/transformer-infra.md` 完整覆盖（RoPE 的外推/长上下文深挖另在 `part6/long-context-inference.md`）。全仓无任何页面入链到 `part1/`。
3. **「施工」话术。** 多个 hub 的「脚手架状态」note 仍在讲施工进度（"being built part by part"、"Next up is Part 3"）、`interview/index.md` 暴露内部票号（"(tickets #2, #4, #5)…"）并自称未完成——对私人笔记无所谓，一旦公开即显半成品、且与「已完成」自相矛盾。

## Decision（决策）

1. **题库：精，不是多。** 放弃「~100」目标，正式定位为**精选高频集**。补 **3 道**当前缺失的必考题——**采样参数**（temperature / top-p / top-k、greedy vs sampling 及其对批处理/吞吐的影响）、**MoE 推理**（激活参数 vs 总参数、serving 期专家路由）、**preemption（recompute vs swap）**（KV 池耗尽时的调度行为）。每道**双向链接到最近的现有课**（分别 `tuning-knobs` / `parallelism-strategies` / `kv-cache-block-manager`），**不为此新写课页**；对应课的 §7「面试连线」补回链。首页文案改为「完成态精选集」，**不标题目总数**。难度档 / 频率标签 / 权重维持 backlog。
2. **Part 1 并入 Part 0。** 删除 Part 1 的 nav 条目与空壳页；其内容以 `part0/transformer-infra.md`（+ `part6` 的 RoPE 深挖）为准。课程曲目实为「**Part 0（含 Transformer infra 视角）、Part 2–8**」。
3. **公开前清理施工话术。** 作为可公开产物，全站移除施工旁白：hub 的「脚手架状态」note 改为成品描述或删除；移除公开页上暴露的内部票号；去掉「grows / ~100 / land in later tickets」等未完成暗示。

## Consequences（后果）

**正面：**
- 交付边界与站点实际一致；公开时读作「完成的 v1」，而非施工现场。
- 题库补齐 day-one 必考题后名副其实地「精」；曲目更诚实（不留悬空的空 Part）。

**权衡 / 负面：**
- 偏离 spec 字面（题数、独立 Part 1）——本 ADR 即该偏离的正式依据。
- 3 道新题链「最近课」而非专属课，接受「课仅触及该题」的**弱双向链接**（换取守住「精不是多」、不扩曲目）。
- 术语沉淀：**采样参数**与 **preemption** 应进 `CONTEXT.md`（本 ADR 落地时同步），站内 `glossary.md` 镜像随实现补齐；MoE 术语已在表内。

**明确不做：** 灌水到 100 题、难度/频率分级、为新题新写课页、Pages/CI、非 NVIDIA 实操——维持 backlog/延后。

**可逆性：** 中。题库文案与 Part 1 删除本身易改；但一旦公开、外链形成，曲目结构即趋固化，故立此 ADR 记录选型与依据。
