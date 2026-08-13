# 服务生态：选 vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy

!!! info "基线：**vLLM 0.26.0** · 跨框架说法是定位——对照各自当前文档核实（ADR-0004）"

**模块：** Part 8 · 生产部署与系统设计   ·   **对应课程：** [服务生态：vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](../part8/framework-comparison.md)

---

## Q：为什么用 vLLM 而不是 TensorRT-LLM / TGI / SGLang / LMDeploy——或何时不用？给出区分它们的轴、一个可辩护的默认与例外、以及你会怎么实际决定。

### 直接答案

**没有全局第一**——它们共享一个**基线**（continuous batching、paged KV、prefix caching、量化、OpenAI 兼容 API），在少数**轴**上分歧。主轴是**可移植性/速度 ↔ NVIDIA 峰值延迟**：

- **vLLM** —— 广度、速度、广硬件、庞大社区。合理**默认**。
- **TensorRT-LLM** —— **提前编译**引擎 → NVIDIA 峰值延迟，代价是灵活性（每模型/GPU 重建）+ NVIDIA 锁定。
- **TGI** —— HuggingFace 原生生产 server；HF 为中心的栈选它。
- **SGLang** —— **RadixAttention** 前缀缓存树；共享前缀 / 结构化 / agentic workload 选它。
- **LMDeploy** —— **TurboMind** 引擎 + 强 weight-only（INT4/AWQ）服务。

**用 *「默认 X，当约束 Z 时切到 Y」* 回答、不用排名列表。** 然后**靠压测决定**——都是 OpenAI 兼容，所以在你的 workload 上、你的 SLO 下跑*同一* `vllm bench serve --backend openai --base-url …`，比 **goodput**。

### 深入原理

- **基线是标配。** continuous batching / paged KV / prefix caching / OpenAI API 现在近乎通用——别为它们给任何框架加分。在边缘上争。
- **可移植 vs 峰值。** vLLM 今天就在多加速器上跑任意新模型（Python）；TensorRT-LLM 编译一个固定 NVIDIA 引擎求顶级延迟。多数其他差异是下游。
- **workload 契合。** SGLang 的 RadixAttention 在重共享前缀 / 分支上发光；LMDeploy 侧重 weight-only 量化吞吐；TGI 侧重 HF 集成。
- **收敛 + 漂移。** 它们每月互抄——你「知道」的差距可能已合上。核实当前文档；更好是测你会部署的版本。

### 代码

```bash
# 都暴露 /v1/* → 一套 harness 压测任意一个。每个后端服务同一模型，然后：
vllm bench serve --backend openai --base-url http://127.0.0.1:8001 \
    --model Qwen/Qwen2.5-7B-Instruct --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 128 \  # 匹配你的分布
    --num-prompts 500 --request-rate 16 --percentile-metrics "ttft,tpot,itl,e2el"
# 赢家 = 你 workload 上、你 SLO 下更高的 GOODPUT（不是博客排行榜）。
```

### 面试官追问

- *「固定 Llama、数百万 NVIDIA 用户、最低延迟——vLLM 还是 TensorRT-LLM？」* → 倾向 TensorRT-LLM（固定模型摊薄编译、反正仅 NVIDIA、延迟优先）——但点出构建/锁定成本、且仍两者都测。
- *「模型每周变、混合硬件？」* → vLLM——编译灵活性损失盖过峰值延迟。
- *「vLLM 的 PagedAttention 现在是差异化吗？」* → 弱——全生态标配。差异化在边缘（提前编译、RadixAttention、量化深度、硬件）。
- *「两个博客各称最快——怎么解决？」* → 不同模型/GPU/分布/版本；在你 workload 上 OpenAI 兼容地、SLO 下测，比 goodput。
- *「何时 SGLang？」* → 重共享前缀（RAG、分支）/ 结构化 / agentic——RadixAttention 复用。
- *「非延迟因素？」* → 模型覆盖、硬件、团队栈、量化支持、运维成本（TensorRT-LLM 的构建）。

### 关联知识点

- 课程：[服务生态：vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](../part8/framework-comparison.md)
- 相关：[SLO 驱动调优](slo-driven-tuning.md)（goodput，对比之分）、[压测与并发拐点](load-testing-knee.md)（OpenAI 兼容 harness）、[Static vs Continuous Batching](continuous-batching.md)（一项共享基线特性）
- 术语表：[SLO、Goodput](../glossary.md)
