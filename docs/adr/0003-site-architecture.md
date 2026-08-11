# 3. 站点架构：MkDocs Material + 双语 i18n + KaTeX + Cloudflare Pages

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-11

## Context（背景）

需要一个承载 100+ 页双语技术教程 + 面试题库的站点。硬需求：左侧导航、全文搜索（含中文）、数学公式、代码高亮、提示框、按浏览器语言自动切换且英文降级。同时仓库里已有非发布的元信息（`CONTEXT.md` glossary、`docs/adr/`），不能和课程内容混进导航。多种技术栈（VitePress / Docusaurus / MkDocs）与多个子选择（数学引擎、i18n 结构、部署平台）都需一次性定清，且这些选择互相牵连、日后不易零散更改。

## Decision（决策）

**框架**：MkDocs Material（Python 原生，贴合读者的 PyTorch/Python 世界，技术文档能力开箱即用）。

**双语 i18n**：`mkdocs-static-i18n`，**suffix 结构**——`page.md` = **英文（默认 + 降级）**、`page.zh.md` = 中文译文；顶部语言切换器由插件 `reconfigure_material` 自动生成（同页切换）。**撰写顺序：英文先写、中文作译**（技术表达以英文为原味，且天然对齐"英文=默认语言"）。`fallback_to_default: true`：缺失的中文页回落英文。

**自动语言**：静态站无服务端方案，用落地页一小段**客户端 JS** 读 `navigator.language`，中文浏览器跳 `/zh/`，其余留英文；`sessionStorage` 防抖、尊重手动切换。

**放弃 `navigation.instant`**：它与多语言切换器不兼容，且会使数学/JS 在换页时不重跑。

**数学**：`pymdownx.arithmatex(generic:true)` + **KaTeX**（轻、同步渲染），`document$.subscribe` 挂载。若日后公式量大到需要 KaTeX 不支持的扩展，再评估切 MathJax。

**内容根**：`docs_dir = site_src/`，使 `CONTEXT.md`、`docs/adr/` 作为纯仓库文档**不进站点**；术语表在站内单独镜像一份。

**搜索**：Material `search`，`lang:[en, zh]`，中文分词依赖 `jieba`。

**部署**：**Cloudflare Pages**（域名根路径服务，避开 GitHub 项目页 `user.github.io/repo/` 的子路径前缀问题，让语言跳转 JS 更简单）。部署本身延后到有实质内容后。

## Consequences（后果）

**正面：**
- 写作即 Markdown，读者与作者都低门槛；导航/搜索/公式/提示框全开箱即用。
- 元信息与课程内容物理分离，导航干净。
- Cloudflare 根路径部署使 i18n 跳转少一类 bug。

**权衡 / 负面：**
- 放弃 `navigation.instant` → 换页无 SPA 般顺滑（可接受）。
- 客户端 JS 跳转在首屏后执行，有极短闪烁；且需与手动切换做防抖。
- KaTeX 扩展不如 MathJax 全（当前推理数学够用）。
- 中文搜索强依赖 `jieba`，构建环境需装。

**可逆性：** 中。内容是 Markdown 可迁移，但 i18n 的 suffix 结构、`docs_dir` 布局、部署平台一旦落地，改动会牵动构建配置与所有页面路径，故立此 ADR 记录整簇选择。
