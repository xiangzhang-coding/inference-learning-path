# The serving ecosystem: choosing vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy

!!! info "Baseline: **vLLM 0.26.0** · cross-framework claims are positioning — verify each against its current docs (ADR-0004)"

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [The Serving Ecosystem: vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](../part8/framework-comparison.md)

---

## Q: Why vLLM over TensorRT-LLM / TGI / SGLang / LMDeploy — or when not? Give the axes that separate them, a defensible default with exceptions, and how you'd actually decide.

### Direct answer

**There's no global #1** — they share a **baseline** (continuous batching, paged KV, prefix caching, quantization, OpenAI-compatible APIs) and diverge on a few **axes**. The master axis is **portability/velocity ↔ peak NVIDIA latency**:

- **vLLM** — breadth, velocity, wide hardware, huge community. The sensible **default**.
- **TensorRT-LLM** — **ahead-of-time compiled** engine → peak NVIDIA latency, at the cost of flexibility (rebuild per model/GPU) + NVIDIA lock-in.
- **TGI** — HuggingFace-native production server; pick it in an HF-centric stack.
- **SGLang** — **RadixAttention** prefix-cache tree; pick it for shared-prefix / structured / agentic workloads.
- **LMDeploy** — **TurboMind** engine + strong weight-only (INT4/AWQ) serving.

**Answer as *"default X, switch to Y when constraint Z,"* not a ranked list.** Then **decide by benchmarking** — all are OpenAI-compatible, so run the *same* `vllm bench serve --backend openai --base-url …` on your workload at your SLO and compare **goodput**.

### Deep dive

- **Baseline is table stakes.** Continuous batching / paged KV / prefix caching / OpenAI API are near-universal now — don't credit any framework for them. Fight over the edges.
- **Portability vs peak.** vLLM runs any new model on many accelerators today (Python); TensorRT-LLM compiles a fixed NVIDIA engine for top latency. Most other differences are downstream.
- **Workload fit.** SGLang's RadixAttention shines on heavy shared prefixes / branching; LMDeploy leans into weight-only-quant throughput; TGI into HF integration.
- **Convergence + drift.** They copy each other monthly — a "gap" you know may be closed. Verify current docs; better, measure the versions you'd deploy.

### Code

```bash
# All expose /v1/* → one harness benchmarks any of them. Serve same model per backend, then:
vllm bench serve --backend openai --base-url http://127.0.0.1:8001 \
    --model Qwen/Qwen2.5-7B-Instruct --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 128 \  # match YOUR mix
    --num-prompts 500 --request-rate 16 --percentile-metrics "ttft,tpot,itl,e2el"
# Winner = higher GOODPUT at your SLO on your workload (not a blog leaderboard).
```

### Interviewer follow-ups

- *"Fixed Llama, millions of NVIDIA users, lowest latency — vLLM or TensorRT-LLM?"* → Lean TensorRT-LLM (fixed model amortizes the compile, NVIDIA-only anyway, latency is priority) — but name the build/lock-in cost and still benchmark both.
- *"Model changes weekly, mixed hardware?"* → vLLM — the compile-flexibility loss outweighs peak latency.
- *"Is vLLM's PagedAttention a differentiator now?"* → Weak — table stakes ecosystem-wide. Differentiators are at the edges (compile-ahead, RadixAttention, quant depth, HW).
- *"Two blogs each claim fastest — resolve it?"* → Different model/GPU/mix/version; measure your own workload OpenAI-compatibly at your SLO, compare goodput.
- *"When SGLang?"* → shared-prefix-heavy (RAG, branching) / structured / agentic — RadixAttention reuse.
- *"Non-latency factors?"* → model coverage, hardware, team stack, quant support, operational cost (TensorRT-LLM's build).

### Linked concepts

- Lesson: [The Serving Ecosystem: vLLM vs TensorRT-LLM / TGI / SGLang / LMDeploy](../part8/framework-comparison.md)
- Related: [SLO-driven tuning](slo-driven-tuning.md) (goodput, the comparison score), [Load-testing & the concurrency knee](load-testing-knee.md) (the OpenAI-compatible harness), [Static vs continuous batching](continuous-batching.md) (a shared-baseline feature)
- Glossary: [SLO, Goodput](../glossary.md)
