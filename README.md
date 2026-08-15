# Inference Infra Learning Path · inference-learning-path

**English** · [简体中文](README.zh.md)

> A **systematic, complete, beginner-friendly** path to **LLM inference infrastructure** — built around one north star: **maximizing vLLM concurrency and backend throughput**. For people with a solid PyTorch background aiming at big-tech inference-infra roles.

This repo is the source of a **bilingual tutorial site** built with **MkDocs Material**. It covers LLM inference infra end to end — **quantization, GPU programming (Triton), multi-GPU & distributed, and vLLM serving / throughput tuning** — plus a **curated interview bank** and a **throughput-maxing Capstone**.

## What you'll get

- **Parts 0–8**, ordered *motivation-first* — every optimization is a conclusion you can derive, not a fact to memorize.
- A **curated interview bank**, organized by module; each entry: direct answer → deep dive → code → interviewer follow-up → linked concept.
- A **throughput-maxing Capstone**: push `Qwen2.5-7B-Instruct` as far as it goes on a single RTX 4090 within a ¥500 AutoDL budget, and write the "before → after" report.

## Curriculum

| Part | Topics |
|---|---|
| **Part 0** | Foundations & motivation (prefill/decode flow, the Transformer infra view, KV cache, GPU hardware, perf metrics, number formats) |
| **Part 2** | Single-GPU performance (roofline / arithmetic intensity, KV-cache memory math, FlashAttention, kernel fusion / CUDA graphs) |
| **Part 3** | GPU programming (CUDA execution model, memory access, Triton kernels, a guided read of the PagedAttention kernel) |
| **Part 4** | Quantization (principles & schemes, GPTQ / AWQ / SmoothQuant / FP8 / LLM.int8() / KV quant, a `Qwen2.5-7B` → INT4 lab) |
| **Part 5** | Serving & throughput · vLLM core (continuous batching, PagedAttention, prefix caching, scheduler / chunked prefill / PD disaggregation, speculative decoding, tuning knobs, architecture map) |
| **Part 6** | Advanced inference topics (multi-LoRA serving, guided / structured decoding, long-context inference) |
| **Part 7** | Multi-GPU & distributed (TP / PP / DP / EP strategies, NCCL collective communication & launching TP/PP) |
| **Part 8** | Production & system design (OpenAI-compatible server, the load-testing knee, routing / autoscaling, observability, SLO-driven tuning, framework comparison, capacity planning, system-design questions) |

> Also: the **[interview bank](site_src/interview/)**, the **[Capstone](site_src/capstone/)**, and the **[eval sets](site_src/eval/)** (a small ~20-item set + a larger harness).
> The Transformer infra view is folded into Part 0 as a topic.

## How each lesson is built

Every lesson follows the same **9-section skeleton**: ① Intuition & why it matters → ② Mental model / diagram → ③ Principle & math (KaTeX) → ④ Complete runnable code + line-by-line → ⑤ Lab (with a GPU callout) → ⑥ Common pitfalls / counter-intuitive points → ⑦ Interview links → ⑧ One-line summary + further reading → ⑨ Self-check questions (answers folded).

The sample lesson **[KV cache](site_src/part0/kv-cache.md)** demonstrates the full skeleton end to end.

## Baselines & conventions

| Dimension | Baseline |
|---|---|
| GPU / model | a single **RTX 4090 (24 GB)** + `Qwen2.5-7B-Instruct` (quantized) |
| vLLM | **v0.26.0** baseline, annotated at the top of every page |
| Depth | read + tune + application layer; a little Triton, read vLLM CUDA/Triton source (no hand-written CUDA C++) |

- All performance figures are **illustrative / order-of-magnitude references**; lessons are **statically verified via Context7, not executed** — you reproduce the real numbers yourself on your own AutoDL box.
- For what's pinned and how to refresh when vLLM upgrades, see [`site_src/versioning.md`](site_src/versioning.md).

## Build & preview locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

mkdocs serve            # local preview, default http://127.0.0.1:8000
mkdocs build --strict   # the one structural gate: broken links / orphan pages / nav gaps / i18n errors all fail
```

Chinese full-text search relies on `jieba` for word segmentation (already in `requirements.txt`).

## Bilingual / i18n

The site uses the **suffix structure** of `mkdocs-static-i18n`: `page.md` = English (default + fallback), `page.zh.md` = the Chinese translation. **English is written first, Chinese is translated from it** (English = the site's default language). Chinese browsers are auto-redirected to `/zh/` on first visit; the header selector switches languages anytime.

## Repo layout

```
site_src/            # site content (mkdocs docs_dir)
  part0,2..8/        #   per-Part lessons (X.md English / X.zh.md Chinese)
  interview/         #   curated interview bank
  capstone/          #   throughput-maxing Capstone
  eval/              #   eval sets (small set + larger harness)
  glossary*.md       #   in-site glossary (mirrors CONTEXT.md)
  assets/            #   KaTeX / language-redirect JS, GPU-callout CSS
docs/adr/            # architecture decision records (ADR-0001…0006)
docs/spec/           # spec
CONTEXT.md           # bilingual glossary (ubiquitous language)
mkdocs.yml           # site config (nav, i18n, KaTeX, Mermaid)
requirements.txt     # build dependencies (version-capped; see versioning)
```

`CONTEXT.md` and `docs/` are repo metadata and are **not published** to the site.

## License

Not yet specified.
