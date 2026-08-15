# LLM 推理 Infra 学习路径 · inference-learning-path

[English](README.md) · **简体中文**

> 一条**系统、完整、对新手友好**的 **LLM 推理基础设施**学习路径——围绕唯一的北极星：**最大化 vLLM 并发与后端服务器吞吐**。面向有扎实 PyTorch 基础、冲刺大厂推理 Infra 岗的人。

本仓库是一个用 **MkDocs Material** 构建的**双语教程站**源码：系统讲解 LLM 推理 Infra，覆盖**量化、GPU 编程（Triton）、多卡与分布式、vLLM 服务化与吞吐调优**，并配一个**精选高频面试题库**和一个**吞吐拉满 Capstone**。

## 你会得到什么

- **Part 0–8**，按「动机先行」排序——每个优化都是你能推导出的结论，而非需要死记的事实。
- **精选高频面试题库**，按模块归类，每题：直接答案 → 深入原理 → 代码 → 面试官追问 → 关联知识点。
- **吞吐拉满 Capstone**：在单张 RTX 4090、¥500 AutoDL 预算内把 `Qwen2.5-7B-Instruct` 的吞吐拉到极限，产出「优化前 → 后」报告。

## 课程结构

| 模块 | 主题 |
|---|---|
| **Part 0** | 基础与动机（推理流程 prefill/decode、Transformer infra 视角、KV cache、GPU 硬件、性能度量、数值格式） |
| **Part 2** | 单卡推理性能（roofline / arithmetic intensity、KV cache 显存数学、FlashAttention、kernel fusion / CUDA graphs） |
| **Part 3** | GPU 编程（CUDA 执行模型、访存、Triton kernel、PagedAttention kernel 导读） |
| **Part 4** | 量化（原理与方案、GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8() / KV 量化、`Qwen2.5-7B` → INT4 Lab） |
| **Part 5** | 服务化与吞吐 · vLLM 核心（continuous batching、PagedAttention、prefix caching、调度器 / chunked prefill / PD 分离、投机解码、调参旋钮、架构地图） |
| **Part 6** | 进阶推理专题（multi-LoRA serving、约束 / 结构化解码、长上下文推理） |
| **Part 7** | 多卡与分布式（TP / PP / DP / EP 策略、NCCL 集合通信与启动 TP/PP） |
| **Part 8** | 生产与系统设计（OpenAI 兼容服务、压测拐点、路由 / 自动扩缩、可观测性、SLO 调优、框架选型、容量规划、系统设计题） |

> 另有 **[面试题库](site_src/interview/)**、**[Capstone](site_src/capstone/)** 与 **[评测集](site_src/eval/)**（小集 ~20 + 大集 harness）。
> Transformer 的 infra 视角作为一个专题并入 Part 0。

## 每节课的 9 段骨架

① 直觉 & 为什么重要 → ② 心智模型 / 图 → ③ 原理与数学（KaTeX）→ ④ 完整可跑代码 + 逐行讲解 → ⑤ Lab（含 GPU 标注）→ ⑥ 常见坑 / 反直觉 → ⑦ 面试连线 → ⑧ 一句话小结 + 延伸阅读 → ⑨ 自测小问（答案折叠）。

样板课 **[KV cache](site_src/part0/kv-cache.zh.md)** 完整演示了这套骨架。

## 基线与约定

| 维度 | 基线 |
|---|---|
| GPU / 模型 | 单张 **RTX 4090（24 GB）** + `Qwen2.5-7B-Instruct`（量化） |
| vLLM | 基线 **v0.26.0**，每页页顶标注 |
| 深度 | 读懂 + 会调 + 应用层；会写少量 Triton、能读 vLLM CUDA/Triton 源码（不手写 CUDA C++） |

- 文中一切性能数字均为**示例 / 量级参考**；课程**经 Context7 静态核实、不执行代码**——真实数字由你在自己的 AutoDL 环境复现。
- 哪些东西被钉住、以及 vLLM 升级时如何刷新，详见 [`site_src/versioning.md`](site_src/versioning.md)。

## 本地构建与预览

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

mkdocs serve            # 本地预览，默认 http://127.0.0.1:8000
mkdocs build --strict   # 唯一结构闸门：坏链 / 孤儿页 / 导航缺项 / i18n 错误均判失败
```

中文全文搜索的分词依赖 `jieba`（已在 `requirements.txt` 中）。

## 双语 / i18n

站点用 `mkdocs-static-i18n` 的 **suffix 结构**：`page.md` = 英文（默认 + 降级），`page.zh.md` = 中文译文；**撰写顺序为英文先写、中文作译**（英文 = 站点默认语种）。中文浏览器首次访问自动跳转 `/zh/`，页首可一键切换语言。

## 仓库结构

```
site_src/            # 站点内容（mkdocs docs_dir）
  part0,2..8/        #   各 Part 课程（X.md 英文 / X.zh.md 中文）
  interview/         #   精选高频面试题库
  capstone/          #   吞吐拉满 Capstone
  eval/              #   评测集（小集 + 大集 harness）
  glossary*.md       #   站内术语表（镜像 CONTEXT.md）
  assets/            #   KaTeX / 语言跳转 JS，GPU callout CSS
docs/adr/            # 架构决策记录（ADR-0001…0006）
docs/spec/           # spec
CONTEXT.md           # 双语术语表（Ubiquitous Language）
mkdocs.yml           # 站点配置（nav、i18n、KaTeX、Mermaid）
requirements.txt     # 构建依赖（版本已设上界，见 versioning）
```

`CONTEXT.md` 与 `docs/` 是仓库元信息，**不发布**到站点。

## 许可证

暂未指定。
