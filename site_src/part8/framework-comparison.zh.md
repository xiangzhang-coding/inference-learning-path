# 服务生态：vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy

!!! info "基线：**vLLM 0.26.0** · 对比是*定位*、不是特性清单"
    已按 ADR-0004 用 Context7 对照 vLLM 0.26.0 核实：本节倚赖的一个 API 级事实是——这些引擎几乎都暴露 **OpenAI 兼容** HTTP endpoint，所以单个客户端——包括 **`vllm bench serve --backend openai --base-url …`**——能在*同一* workload 上给它们任何一个压测。vLLM 自己的差异化（PagedAttention、continuous batching、prefix caching、广泛的模型 + 硬件支持）见 Part 5–7。跨框架能力（TensorRT-LLM 的提前引擎编译、SGLang 的 RadixAttention、TGI 的 HF 集成、LMDeploy 的 TurboMind 引擎）是**逐版本变动的定位——在你投入前，对照各项目当前文档核实。** 本节所有数字均为**示例 / 量级参考**。

---

## 1 · 直觉 & 为什么重要

「为什么用 vLLM 而不是 TensorRT-LLM / TGI / SGLang / LMDeploy？」几乎是必考的选型题，而错误答案是一个排名列表（「vLLM 第一」）。没有全局第一——它们是**重叠的工具、各有甜区**，强答案会点出在*给定 workload 与约束*下作出决定的那个轴。

陷阱是拿 benchmark 博客论证。发布的数字是在别人的模型、硬件、prompt 分布、（常常）半年前的版本上测的。这些框架还**收敛得很快**：continuous batching、paged KV cache、prefix caching、量化、OpenAI 兼容 endpoint 现在几乎处处是标配。所以面试级技能是两件事：

1. **知道真正区分它们的*轴***——可移植性 vs 峰值、易用 vs 控制、各自为哪种 workload 调优——而非一份过时的特性网格。
2. **知道诚实的决胜法是在*你的* workload 上测它们**——而且因为它们 OpenAI 兼容，一套 harness（`vllm bench serve`）就能在你的 SLO 下压测全部。

所以：它们共同抵达的基线、少数分歧的轴、一个可辩护的默认与其例外、以及自己测的方法。→ 术语见 [术语表](../glossary.md) 的 *SLO、Goodput*。

## 2 · 心智模型

它们在基本面上多半一致；在少数轴上分歧。按*它优化什么*来放置每一个，别按标量排名。

```text
   共同基线（几乎处处是标配）：
     continuous batching · paged KV cache · prefix caching · 量化 · OpenAI 兼容 API

   真正分歧的轴：

   可移植性 / 速度  ◀───────────────────────────────────▶  NVIDIA 峰值延迟
   (Python、广硬件、                                        (提前编译的引擎、
    任意模型、更新快)                                         仅 NVIDIA)
        vLLM ─────────── SGLang ──── TGI ───────────────── LMDeploy ──── TensorRT-LLM

   各自主打：
     vLLM         广度 + 速度：新模型上得快、广硬件、庞大社区、易用默认
     TensorRT-LLM NVIDIA 峰值：编译每模型/每 GPU 的引擎 → 顶级延迟，灵活性更低
     TGI          HuggingFace 原生生产 server；与 HF 生态/Endpoints 紧密集成
     SGLang       共享前缀 / 结构化 / agentic workload（RadixAttention 前缀缓存树）
     LMDeploy     高性能 TurboMind（C++/CUDA）引擎 + 强的 weight-only（INT4/AWQ）服务

   按约束选、不按排名——再在你的 workload 上压测确认。
```

三个要记住的形状：

- **基线共享；在边缘上争。** 若一个候选人说不清什么是*标配*（continuous batching、paged KV、OpenAI API），他会为一个人人都有的特性过度加分。真正的差异在边缘：提前编译 vs 动态运行、前缀缓存策略、量化深度、硬件。
- **可移植性 ↔ 峰值是主轴。** vLLM 优化*广度与速度*（今天就在多种 GPU 上跑任意新模型、Python）；TensorRT-LLM 靠编译每模型/GPU 的固定引擎优化*NVIDIA 峰值*。多数其他权衡都是这个轴的下游。
- **决胜法是你自己的 benchmark。** 因为它们 OpenAI 兼容，你不用争——把同一模型在每个上服务，跑*同一* `vllm bench serve`、在你的 SLO 下。你 workload 上的 goodput 定胜负。

## 3 · 原理

### 3.1 共同基线

到 2025 年，本课程的获胜想法已扩散到整个生态：**continuous batching**（Part 5）、**paged KV cache**（Part 5）、**prefix caching**（Part 5）、**量化**（Part 4）、tensor/pipeline **并行**（Part 7）、以及 **OpenAI 兼容** server（本 Part）。所以「它有 continuous batching 吗」不再区分它们。区分它们的是某个轴*推得多远*。

### 3.2 分歧的轴

- **可移植性与速度（vLLM）。** Python 优先、最广的模型覆盖（新架构上得快）、广硬件（NVIDIA、AMD ROCm、更多）、最大社区。这就是它是合理**默认**的原因：今天你几乎能在任意加速器上服务几乎任意模型。
- **NVIDIA 峰值延迟（TensorRT-LLM）。** NVIDIA 的 TensorRT 工具箱为特定模型 + GPU + 精度**提前编译一个优化引擎**。那个编译步买来 NVIDIA 上顶级的延迟/吞吐——代价是灵活性（每模型/GPU/shape 重建）与仅 NVIDIA 锁定。当模型固定、硬件是 NVIDIA、且值得为最后 20% 延迟维护一条构建流水线时选它。
- **HuggingFace 原生生产（TGI）。** Text Generation Inference 是 HuggingFace 的 server（Rust router + Python worker），调优来嵌进 HF 生态与 Inference Endpoints。当你的栈已以 HF 为中心、想要他们支持的服务路径时选它。
- **共享前缀 / 结构化 / agentic（SGLang）。** 其 **RadixAttention** 把前缀缓存组织成基数树以激进地自动复用 KV，其前端瞄准结构化输出与多调用/agentic 程序。当你的 workload 由共享前缀主导（重 RAG、多分支 prompting）或程序化结构化生成时选它。
- **TurboMind + weight-only 量化（LMDeploy）。** 来自 InternLM/OpenMMLab 社区；其 **TurboMind** C++/CUDA 引擎与强的 **weight-only（INT4/AWQ）** 服务是招牌，在中文生态流行。当 NVIDIA 上量化 weight-only 吞吐是优先级时选它。

### 3.3 一个可辩护的默认 + 例外

面试就绪的立场：

- **默认 vLLM**——广度、速度、易用、硬件灵活、社区。对「我们服务很多模型 / 新模型 / 混合硬件」风险最低。
- **TensorRT-LLM**——模型稳定、硬件是 NVIDIA、需要绝对延迟下限且愿维护编译流水线时。
- **TGI**——深在 HuggingFace 栈里、想要他们的一方 server 时。
- **SGLang**——共享前缀复用或结构化/agentic 生成主导时。
- **LMDeploy**——NVIDIA 上 weight-only 量化吞吐（和/或 InternLM 生态）为核心时。

说成 *「默认 X，当约束 Z 时切到 Y」*——那是面试官奖励的形状，不是排行榜。

### 3.4 诚实的决胜：自己测

定位缩小了候选；**你的 workload 选出赢家。** 既然每个都暴露 OpenAI 兼容 endpoint，把同一模型在两个候选上服务，对每个 `--base-url` 跑*同一* `vllm bench serve`、在你的 SLO 下、你的 prompt 分布上。比 **goodput**（[SLO 调优](slo-driven-tuning.md)的分），不是厂商博客数字。这也中和了版本漂移——你测的是今天真会部署的东西。

## 4 · 完整可跑代码 + 逐行讲解

这里唯一重要的代码不是配置——是settled选择的**苹果对苹果 benchmark**。因为 harness 是 OpenAI 兼容客户端，它能指向任意后端。

```bash
# 把同一模型在两个候选上、以 OpenAI 兼容 endpoint、在不同端口服务：
#   终端 A (vLLM):        vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
#   终端 B (替代品):      <框架的 OpenAI 兼容 server> ... --port 8001
#   (TGI、SGLang、LMDeploy api_server、TensorRT-LLM 的 OpenAI 前端都暴露 /v1/*。
#    TensorRT-LLM 服务前还需一个提前的引擎 BUILD 步骤。)

# 用相同 harness + workload + SLO 压测两者，再比 goodput：
for PORT in 8000 8001; do
  echo "=== 端口 :$PORT 上的后端 ==="
  vllm bench serve \
      --backend openai \                        # 通用 OpenAI 兼容客户端——对它们任何一个都行
      --base-url "http://127.0.0.1:${PORT}" \
      --model Qwen/Qwen2.5-7B-Instruct \
      --endpoint /v1/completions \
      --dataset-name random \
      --random-input-len 512 --random-output-len 128 \   # ← 匹配你的生产 prompt 分布
      --num-prompts 500 --request-rate 16 \              # ← 你 SLO 的目标负载
      --percentile-metrics "ttft,tpot,itl,e2el"
done
# 赢家 = 在你的 workload 上、你的 SLO 下更高的 GOODPUT（不是博客排行榜）。
```

**逐行讲解：**

- **`--backend openai`** —— 整节要点浓缩成一个 flag：`vllm bench serve` 是 OpenAI 兼容*客户端*，所以同一 harness 能压测 vLLM、TGI、SGLang、LMDeploy 或 TensorRT-LLM 的 OpenAI 前端。没有每框架的 benchmark 脚本、没有苹果对橘子。
- **`--base-url http://127.0.0.1:${PORT}`** —— 只需重指到每个后端；`127.0.0.1`（非 `localhost`）避开 [knee 那节](load-testing-knee.md)提到的 IPv6 卡顿。
- **`--random-input-len / --random-output-len`** —— 设成*你的* prompt/输出分布。一个在 512/128 上赢的框架可能在 4k/1k 上输（prefill- vs decode-heavy）；对比只在你的分布上有效。
- **`--request-rate 16`** —— 在你 SLO 必须撑住的负载上压测、开环，于是你比的是*SLO 下的 goodput*、不是饱和吞吐。
- **TensorRT-LLM 注记** —— 它是唯一有额外**构建**步的候选（服务前按模型/GPU/精度编译引擎）；把那份运维成本算进选择、不只看结果延迟。

## 5 · Lab —— vLLM vs 一个替代品、同一 harness

!!! gpu "GPU Lab（单卡）"
    - **最低显存：** 两个 `Qwen2.5-7B-Instruct`（或更小模型）server 在一张 **24 GB 4090** 上并存有点挤——**顺序**跑（服务、压测、停；重复），或用更小模型同时跑。
    - **建议 AutoDL 卡型：** 单张 **RTX 4090 (24 GB)** 顺序做 vLLM-vs-一个替代品（ADR-0001）；想并发才要第二张 GPU。
    - **预估耗时 / 花费：** 含装第二个框架约 40–60 分钟 · **约 ¥3–8**（示例）。TensorRT-LLM 的引擎构建加时间——若那是你的替代品就预留。
    - **平台：** NVIDIA CUDA（默认）。**非 NVIDIA：** vLLM 与 TGI 有 AMD ROCm 路径；**TensorRT-LLM 与 TurboMind 仅 NVIDIA**——那个硬件约束本身就是一个选型轴。

步骤：

1. **按 §3.3 挑一个替代品** 对比 vLLM（例如若你的 workload 重共享前缀就选 SGLang）。
2. **把同一模型**在每个上以 `/v1/*` 服务、在 4090 上顺序进行。
3. **对每个 `--base-url` 跑相同 harness**（§4）、在镜像生产的 prompt 分布上、你 SLO 的负载下。
4. **比 goodput**、不是裸吞吐——并记下运维成本（TensorRT-LLM 的构建步、各自的模型覆盖）。**关机。**

## 6 · 常见坑 / 反直觉点

- **用排名列表回答。**「vLLM > TGI > …」表明你不懂权衡。用*默认 X，当约束 Z 时切到 Y* 回答。
- **拿别人的 benchmark 论证。** 博客数字用不同模型、GPU、prompt 分布、版本。它们在*他们的* workload 上排名、不是你的。自己测。
- **以为 TensorRT-LLM「就是更快」。** 它能在 NVIDIA 上赢峰值延迟，但它**编译一个固定引擎**（每模型/GPU/精度）——真实的构建/维护成本与 NVIDIA 锁定。对快速变化的模型或混合硬件，那份灵活性损失常盖过延迟收益。
- **为标配给框架加分。** continuous batching、paged KV、prefix caching、OpenAI API 现在近乎通用。差异化在边缘（提前编译、RadixAttention、量化深度、硬件），不在人人都有的特性。
- **无视版本漂移。** 这些项目每月在动；你「知道」的能力差距可能已合上。投入前对照*当前*文档核实——更好是测你会部署的版本。
- **低估运维契合。** 模型覆盖、硬件、你团队的栈（HF？仅 NVIDIA？）、量化支持，常比 10% 延迟差更重要。为整个系统选、不为一个数。

## 7 · 面试连线

- [服务生态：选 vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](../interview/framework-comparison.md) —— 本节为你准备的高频题：*共同基线 vs 分歧轴、一个可辩护的默认与其例外、为何「最快」取决于模型/硬件/workload、以及你会怎么实际决定（在你自己的 workload 上、你的 SLO 下、OpenAI 兼容地压测它们）。*

## 8 · 小结 & 延伸阅读

**一句话：** 没有全局第一——这些服务引擎共享一个**基线**（continuous batching、paged KV、prefix caching、量化、OpenAI 兼容 API），在少数**轴**上分歧：**vLLM** 主打广度/速度/硬件灵活（合理默认），**TensorRT-LLM** 靠提前编译引擎主打 NVIDIA 峰值延迟（代价是灵活性 + 锁定），**TGI** 主打 HuggingFace 原生生产，**SGLang** 主打共享前缀/结构化/agentic（RadixAttention），**LMDeploy** 主打 TurboMind + weight-only 量化吞吐；把选型答成 *「默认 X，当约束 Z 时切到 Y」*，并靠在**你自己 workload 上、你的 SLO 下、OpenAI 兼容地**压测候选（`vllm bench serve --backend openai --base-url …`）来定夺，而非信排行榜。

延伸阅读：

- 各项目自己的文档（核实当前能力）：vLLM、NVIDIA TensorRT-LLM、HuggingFace TGI、SGLang、LMDeploy。
- [SLO 调优那节](slo-driven-tuning.md) —— goodput，你对比框架用的分。
- [压测那节](load-testing-knee.md) —— 能压测任意 OpenAI 兼容后端的 `vllm bench serve` harness。
- Part [5](../part5/index.md)–[7](../part7/index.md) —— 这些引擎都实现的共同基线特性（continuous batching、PagedAttention、prefix caching、并行）。

## 9 · 自测小问

??? question "面试官问：『我们把一个固定的 Llama 模型服务给数百万 NVIDIA-GPU 用户、需要尽可能低的延迟。vLLM 还是 TensorRT-LLM？』你的答案与理由？"
    这里倾向 **TensorRT-LLM**——但要说*为什么*并点出代价。约束都利好它：**固定模型**（所以每模型/GPU/精度的提前**引擎编译**划算——你构建一次、摊到数百万请求上）、**仅 NVIDIA**（无可移植性损失，反正 TensorRT-LLM 就仅 NVIDIA）、**延迟是优先级**（其编译引擎瞄准 NVIDIA 峰值延迟）。要承认的代价：一条**构建/维护流水线**（模型或 GPU 变就重建）与**锁定**。我仍会在实际模型/GPU/prompt 分布上、SLO 下**测两者**再定，因为 vLLM 的差距已缩小、其运维简单可能值一点延迟差。对照：若模型*每周变*或跑*混合硬件*，我会默认 **vLLM**——编译灵活性损失会盖过延迟收益。

??? question "为什么 2025 年『vLLM 有 continuous batching 和 PagedAttention』是个弱差异化？真正区分框架的是什么？"
    因为那些想法已**扩散到整个生态**——continuous batching、paged/blocked KV cache、prefix caching、量化、OpenAI 兼容 endpoint 现在在 TGI、SGLang、LMDeploy、TensorRT-LLM 里也是**标配**。为它们给 vLLM 加分是高估了一个人人都有的特性。真正区分框架的住在**边缘**：*可移植性 vs 峰值*（Python + 广硬件 + 任意模型，vs 提前编译的仅 NVIDIA 引擎）、*前缀缓存策略*（如 SGLang 的 RadixAttention 树）、*量化深度*（如 LMDeploy 的 weight-only/AWQ 侧重）、*生态契合*（TGI ↔ HuggingFace）、*硬件支持*。强答案跳过共享基线、论证决定该案的那个轴。

??? question "两个工程师各引一个 benchmark 说自己的框架最快。你怎么可信地解决？"
    哪个博客都定不了——它们测的是不同**模型、GPU、prompt/输出长度分布、版本**，且这些项目每月在变。靠**在你自己 workload 上测候选**来解决：因为它们都暴露 **OpenAI 兼容** endpoint，把同一模型在每个上服务、跑*同一* harness（`vllm bench serve --backend openai --base-url <各>`）、在*你的* SLO 下、镜像生产的 prompt 分布上，比 **goodput**（满足 p99 SLO 的吞吐），不是裸吞吐或别人的图。这是苹果对苹果、用你会部署的版本、把争论变成测量。也权衡非延迟因素——模型覆盖、硬件、量化支持、运维成本（如 TensorRT-LLM 的构建步）——因为纸面最快的引擎可能在整体契合上输。
