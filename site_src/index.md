# Inference Infra Learning Path

> A **systematic, complete, beginner-friendly** path to **LLM inference infrastructure** — built around one north star: **maximizing vLLM concurrency and backend throughput**. For people with a solid PyTorch background who are aiming at big-tech inference-infra roles.

This site is **bilingual**. English is the default; a Simplified-Chinese translation lives under `/zh/`. Chinese browsers are auto-redirected on first visit, and you can switch languages anytime with the selector in the header.

## What you'll get

- **Parts 0–8**, ordered *motivation-first* — every optimization is a conclusion you can derive, not a fact to memorize.
- A **~100-question interview bank** organized by module, each entry: direct answer → deep dive → code → follow-up → linked concept.
- A **throughput-maxing Capstone**: push `Qwen2.5-7B-Instruct` as far as it goes on a single RTX 4090 within a ¥500 AutoDL budget, and write the "before → after" report.

## How each lesson is built

Every lesson follows the same **9-section skeleton**:

1. Intuition & why it matters
2. Mental model / diagram
3. Principle & math (KaTeX)
4. Complete runnable code + line-by-line
5. Lab (with a **GPU callout**)
6. Common pitfalls / counter-intuitive points
7. Interview links (to the bank)
8. One-line summary + further reading
9. Self-check questions (answers folded)

The sample lesson **[KV Cache](part0/kv-cache.md)** demonstrates the full skeleton end to end.

## Baselines (so numbers stay consistent)

| Dimension | Baseline | Source |
|---|---|---|
| GPU / model | **single RTX 4090 (24 GB) + `Qwen2.5-7B-Instruct` (quantized)** | ADR-0001 |
| Depth | read + tune + application layer; a little Triton; read vLLM CUDA/Triton source (no hand-written CUDA C++) | ADR-0002 |
| vLLM | **v0.26.0** baseline, annotated per page | ADR-0004 |

!!! warning "On the numbers in this site"
    All performance figures are **illustrative / order-of-magnitude references**. Following ADR-0004, lessons are **statically verified via Context7, not executed** — you reproduce the real numbers yourself on your own AutoDL box.

See **[Versioning & How to Refresh](versioning.md)** for what's pinned, why the numbers are illustrative, and the recipe for refreshing content when vLLM upgrades.

## GPU callout convention

Any page or code block that needs a GPU carries a callout like this:

!!! gpu "GPU Lab"
    - **Min VRAM:** e.g. 24 GB
    - **Suggested AutoDL card:** e.g. RTX 4090 (24 GB)
    - **Est. time / cost:** e.g. ~15 min · ~¥1 (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** AMD ROCm / Intel / TPU / AWS Neuron / CPU differences noted where relevant

## Start here

- New to inference infra? Begin with **[Part 0 · Foundations & Motivation](part0/index.md)**.
- Want the vocabulary first? See the **[Glossary](glossary.md)**.
