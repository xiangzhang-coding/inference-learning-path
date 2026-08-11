# Spec · 双语 AI 推理 Infra 教程站 (inference-learning-path)

> 本 spec 综合前期 grilling 共识、`CONTEXT.md`（术语表）与 `docs/adr/0001–0004` 而成。
> 术语一律沿用 `CONTEXT.md`；所有决策受 `docs/adr/` 约束。
> 本阶段不实现任何站点/课程代码。

---

## Problem Statement（问题陈述）

我有扎实的 PyTorch 基础，但不懂量化、性能优化、GPU 编程与并行编程。我想系统掌握 **LLM 推理 Infra**——尤其是**最大化 vLLM 的并发与后端服务器吞吐**——并达到能通过大厂推理 Infra 岗面试的水平。目前缺少一条**系统、完整、对新手友好**、且配有**真实工程/高频面试题 + 完整代码逐行讲解**的学习路径：现有资料要么零散、要么假设我已懂底层、要么只有理论没有可上手的 Lab 与可复现的成本约束。

## Solution（解决方案）

一个**双语教程站**（英文为默认/降级、中文为译文，按浏览器语言自动切换、顶部可一键切换），用 MkDocs Material 构建，按 **Part 0–8** 循序渐进：从「**为什么 LLM 推理是 memory-bound**」的心智模型出发（动机先行），依次覆盖 **Transformer(infra 视角)、单卡推理性能、GPU 编程(Triton)、量化、服务化与吞吐(vLLM 核心)、进阶推理专题、多卡与分布式、生产部署、容量规划与系统设计**。

每节课遵循统一的 **9 段骨架**（含完整可跑代码逐行讲解、GPU Lab、常见坑、自测小问、面试连线）；配一个 **~100 道高频面试题库**（按模块归类、每题含直接答案→原理→代码→面试官追问→关联链接）；并以一个 **Capstone**（在单张 RTX 4090、¥500 AutoDL 预算内把 `Qwen2.5-7B-Instruct` 的吞吐拉满，产出「优化前→后」报告）收束。所有代码/flag 经 **Context7 静态核实**；文中性能数字标为「示例/量级参考」，由学习者在自己的 AutoDL 环境实测。

## User Stories（用户故事）

*主角除特别说明外均为「有 PyTorch 基础的学习者」。*

1. 作为学习者，我想从「为什么 LLM 推理是 memory-bound」的心智模型入手，以便后续所有优化对我都是可推导的结论而非死记。
2. 作为学习者，我想理解 prefill 与 decode 两阶段的区别，以便知道每种优化作用在哪一阶段。
3. 作为学习者，我想从 infra 视角理解 Transformer 架构（Q/K/V → KV cache、MHA/MQA/GQA → KV 大小），以便把架构决策与推理成本直接挂钩。
4. 作为学习者，我想彻底搞懂 KV cache 是什么、为何存在、如何增长，以便理解它为何是吞吐上限的核心矛盾。
5. 作为学习者，我想建立 GPU 硬件心智模型（SM/warp、HBM vs SRAM、带宽 vs 算力），以便判断瓶颈在哪。
6. 作为学习者，我想掌握推理性能度量（TTFT、TPOT/ITL、throughput、goodput）及测量方法，以便量化任何优化的收益。
7. 作为学习者，我想弄清各数值格式（FP16/BF16/FP8/INT8/INT4），以便进入量化章节前无障碍。
8. 作为学习者，我想用 roofline 与 arithmetic intensity 分析 attention/GEMM，以便判断某算子受算力还是受带宽限制。
9. 作为学习者，我想理解 FlashAttention 的 IO-aware 思想（tiling、online softmax），以便读懂它为何更快。
10. 作为学习者，我想理解 kernel fusion 与 CUDA graphs，以便明白 decode 阶段 launch overhead 为何致命。
11. 作为不懂 GPU 编程的学习者，我想用心智模型（而非死记）理解 CUDA 执行模型与访存，以便不迷失在细节里。
12. 作为学习者，我想动手用 Triton 写几个简单 kernel，以便获得「能写一点」的底气。
13. 作为学习者，我想被带着导读 vLLM 的 PagedAttention kernel，以便建立读源码的能力。
14. 作为学习者，我想理解量化为何能提吞吐及其精度权衡，以便做出工程取舍。
15. 作为学习者，我想分清 weight-only vs weight+activation、量化粒度、对称/非对称，以便看懂各方法差异。
16. 作为学习者，我想了解 GPTQ/AWQ/SmoothQuant/FP8/LLM.int8()/KV 量化，以便在真实场景选型。
17. 作为学习者，我想有一段完整可跑代码把 `Qwen2.5-7B-Instruct` 量化并在 vLLM 里跑 INT4，以便亲手验证吞吐/精度变化。
18. 作为学习者，我想理解从 static 到 continuous batching 的演进，以便掌握吞吐的第一杠杆。
19. 作为学习者，我想深挖 PagedAttention 如何像虚拟内存一样管理 KV cache，以便理解 vLLM 高吞吐的根源。
20. 作为学习者，我想理解调度器（chunked prefill、PD 分离），以便调 TTFT/吞吐平衡。
21. 作为学习者，我想掌握 prefix caching 与 speculative decoding，以便在合适场景进一步提速。
22. 作为学习者，我想有一张 vLLM 端到端架构地图（engine/scheduler/block manager/worker），以便定位任何行为。
23. 作为学习者，我想逐一理解核心调参旋钮如何移动吞吐/延迟曲线，以便面对真实服务能调优。
24. 作为学习者，我想掌握 multi-LoRA serving，以便理解一基座多 adapter 的服务形态。
25. 作为学习者，我想掌握 guided/structured decoding，以便实现受约束的 JSON/正则输出。
26. 作为学习者，我想理解长上下文推理（RoPE 外推、attention sink、KV 压缩），以便应对长序列的显存与调度问题。
27. 作为学习者，我想理解 Tensor/Pipeline/Data/Expert parallelism 与 NCCL 集合通信，以便读懂并会配多卡。
28. 作为学习者，我想知道在 vLLM 里如何开 TP/PP、如何选 TP 度，以便扩到多卡。
29. 作为学习者，我想学会压测并找到并发拐点 (knee)，以便量出服务的并发天花板。
30. 作为学习者，我想了解路由/自动扩缩/KV 感知路由与可观测性 profiling，以便运维真实服务。
31. 作为求职者，我想了解 TensorRT-LLM/TGI/SGLang/LMDeploy 的取舍，以便在选型面试题上有观点。
32. 作为学习者，我想学会给定模型+硬件估显存/吞吐，以便快速做容量规划。
33. 作为求职者，我想练习「为 X QPS、Y 延迟设计推理服务」的系统设计题，以便通过终面。
34. 作为中文母语学习者，我想默认看到中文讲解、术语保留英文，并能顶部一键切到英文，以便兼顾理解与面试/源码语境。
35. 作为浏览器语言为中文的访客，我想站点自动展示中文（英文为降级），以便零操作进入合适语种。
36. 作为学习者，我想每节课都遵循统一的 9 段骨架，以便有稳定的学习节奏。
37. 作为预算有限的学习者，我想每个需 GPU 处都标明最低显存、AutoDL 卡型与预估花费（默认 NVIDIA，非 NVIDIA 兼容性另注），以便控制成本、不被卡住。
38. 作为在 AutoDL 上练习的学习者，我想被指引把环境安装/下载/调试放在无卡模式、GPU 只在真跑时开，以便把总花费压在预算内。
39. 作为面试准备者，我想有一个 ~100 道高频题的题库、按模块归类、每题含直接答案→原理→代码→面试官追问→关联链接，以便高效突击。
40. 作为学习者，我想每节课末尾能链到相关面试题、题库能反链回知识点，以便学练闭环。
41. 作为学习者，我想有一个 Capstone：在单 4090/预算内把 7B 模型吞吐拉满并产出「优化前后」报告，以便把所学串起来。
42. 作为学习者，我想站点内有一份镜像自术语表的 glossary 页，以便随时查双语术语。
43. 作为读者，我想每页标注所用 vLLM 版本、代码经 Context7 核实，以便信任并按标注版本复现。
44. 作为读者，我想文中性能数字标为「示例/量级参考」并由我自测，以便不被他人环境的数字误导。
45. 作为学习者，我想全站数学公式用 KaTeX 正确排版，以便推导清晰无歧义。
46. 作为学习者，我想站点支持中英文全文搜索，以便快速定位概念。

## Implementation Decisions（实现决策）

- **站点架构 / i18n / 数学 / 部署**：遵循 **ADR-0003**——MkDocs Material；`mkdocs-static-i18n`（suffix 结构，英文=默认/降级、中文=译文，顶部切换）；**英文先写、中文作译**；KaTeX（arithmatex generic）；`docs_dir = site_src`（仓库元信息不发布）；放弃 `navigation.instant`；客户端 JS 按浏览器语言跳转；Cloudflare Pages 部署；jieba 中文搜索。
- **硬件/模型基线**：遵循 **ADR-0001**——单张 RTX 4090(24G) + 量化；**主线基线模型 = `Qwen2.5-7B-Instruct`**，少数英文生态示例附 `Llama-3.1-8B-Instruct` 交叉引用；¥500 AutoDL 预算（属学习者）；非计算工作走无卡模式。
- **学习深度边界**：遵循 **ADR-0002**——读懂 + 会调 + 应用层（会写少量 Triton、能读 vLLM CUDA/Triton 源码）；手写 CUDA C++ 进 backlog。
- **vLLM 版本与验证**：遵循 **ADR-0004**——基线版本 + 章节级标注；**所有 API/flag/CLI 经 Context7 静态核实，作者不执行、不复现数字**。
- **课程结构**：Part 0–8 + 面试题库 + Capstone；排序**动机先行**；相较基础大纲新增 **Transformer(infra 视角)**、**multi-LoRA serving**、**guided/structured decoding**、**长上下文推理** 四块。
- **每节课契约（9 段骨架）**：① 直觉 & 为什么重要 → ② 心智模型/图 → ③ 原理与数学(KaTeX) → ④ 完整可跑代码 + 逐行讲解 → ⑤ Lab（含 GPU 标注 callout）→ ⑥ 常见坑/反直觉点 → ⑦ 面试连线（链到题库）→ ⑧ 一句话小结 + 延伸阅读 → ⑨ 自测小问（答案折叠）。
- **题库 schema**：约 100 道**高频**题；**按模块（Part 0–8）归类**；每题结构：直接答案 → 深入原理 → 代码（若适用）→ 面试官追问 → 关联知识点链接；保留若干系统设计长题。难度档/频率标签/跨模块分布权重 → backlog。
- **GPU 标注约定**：每个需 GPU 的页面/代码块顶部 callout，字段固定为——最低显存、建议 AutoDL 卡型、预估耗时/花费、平台（默认 NVIDIA CUDA）、非 NVIDIA（AMD ROCm / Intel / TPU / AWS Neuron / CPU）兼容性与差异（如适用）。
- **内容组织**：仓库元信息（`CONTEXT.md`、`docs/adr/`、`docs/spec/`）与站点内容（`site_src/`）物理分离；站内单独镜像一份 glossary 页。
- **评测资源**：内置一个 **~20 条小评测集**（量化等章节快速质量核对示意）与一个**更大的评测集**（更充分评测示意）；均作为学习者可运行的资源，作者不执行。
- **预算模型**：¥500 为学习者预算；主力实验在单 4090，仅 Part 6 多卡专题用 A100「开机即关」。因不再受"必须省 GPU 时"约束真实执行，可适度增加需 GPU 的示例代码量。

## Testing Decisions（测试决策）

- **什么是好测试**：只验证**外部可观察行为**，不测实现细节。对本内容站而言即：站点能构建；中英双语两棵树齐全；导航无缺项；内部链接/锚点可解析；i18n 结构合法；KaTeX 与代码高亮配置有效。
- **Seam A（主自动闸门）**：`mkdocs build --strict` 通过。strict 模式把坏链接、孤儿页、导航缺项、i18n 结构错误升级为构建失败——一条命令覆盖整站结构正确性（含中英双树、KaTeX、`site_src` 布局）。这是最高、最省的 seam。
- **Seam B（内容正确性 · 静态核实）**：所有 vLLM/相关库的 API/flag/CLI 通过 **Context7 MCP** 对照官方文档核实；**不执行代码、不复现数字**（对齐 ADR-0004）。文中性能数字标注为「示例/量级参考」。
- **可选补充**：对标注为「可离线运行」的**非 GPU** 代码块，抽出做 smoke import / 语法检查（不涉 GPU、不涉网络）。
- **受测模块**：站点构建产物（结构层）；内容中的 API/flag 断言（静态核实层）。
- **Prior art**：全新仓库，暂无既有测试。约定 `mkdocs build --strict` 为唯一结构闸门（未来接入 CI 时沿用；部署 CI 延后）。

## Out of Scope（不在范围内）

- 手写 / 优化 CUDA C++（backlog）。
- 题库的难度档 / 频率标签 / 跨模块分布权重（backlog）。
- 多模态推理、embedding / reranker 服务（backlog）。
- 训练 / 微调 infra（本课聚焦推理）。
- **实际执行 Lab、复现文中性能数字**——改由学习者在自己的 AutoDL 环境完成；文中数字为示例/量级参考。
- 非 NVIDIA 平台的动手 Lab——仅在相关处文字标注兼容性差异，不做实操。
- 公网部署自动化与 CI 流水线——Cloudflare Pages 部署延后到有实质内容后。

## Further Notes（补充说明）

- **实现阶段用 Context7 MCP** 核实 vLLM 及相关库真实 API/flag，杜绝幻觉参数。
- **部署目标 Cloudflare Pages**（根路径服务，利于语言跳转 JS），延后到有实质内容后。
- **仓库**：公网 `github.com/xiangzhang-coding/inference-learning-path`（public）；本地目录同名。
- **脱敏**：仓库不得包含任何雇主 / 公司内部信息；Git 提交身份使用 GitHub noreply 邮箱，避免泄漏企业邮箱。
- **基线 vLLM 版本号**在建站时锁定（Context7 确认当时稳定 release）。
