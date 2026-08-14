# Versioning & How to Refresh

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    This page is the site's **maintenance stance**, not a lesson. It explains what version everything is pinned to, why the numbers are illustrative, and the exact recipe for refreshing content when vLLM moves on.

---

vLLM iterates fast — APIs, flags, CLI parameters, and even source-code line numbers change between releases. To keep code that runs **1:1 on your machine**, this site pins a single baseline version and annotates it at the top of every page (ADR-0004). This note tells you what that means as a reader, and gives a maintainer a checklist for keeping the site honest as vLLM advances.

## What is pinned

| Dimension | Baseline | Where it shows |
|---|---|---|
| vLLM | **v0.26.0** | callout at the top of every page (ADR-0004) |
| Model | `Qwen2.5-7B-Instruct` (quantized) | baseline callouts; a few English-ecosystem examples cross-reference `Llama-3.1-8B-Instruct` (ADR-0001) |
| GPU | single **RTX 4090 (24 GB)** | baseline callouts (ADR-0001) |
| Source read-along | file / symbol / line anchors align to the **v0.26.0** tag | Part 2 / 3 / 5 source-reading sections |

A handful of sections that discuss a newer feature may annotate a **higher** version and call out the difference explicitly (ADR-0004). Always trust the callout at the top of the page you are on.

## Why the numbers are illustrative

Following ADR-0004, verification here is **static**: the author does **not** execute Labs or reproduce the figures. Every performance number in the site is therefore an **illustrative / order-of-magnitude reference** — a shape to reason about, not a measurement to quote. You reproduce the real numbers yourself, on your own AutoDL box, for your own model and traffic.

Numbers that are *pure arithmetic* (KV-cache size, VRAM budgets, roofline arithmetic intensity) are **exact** — they are just multiplication, not benchmarks — and are labelled as such where they appear.

## How facts are verified (dual-channel)

Every API/flag/CLI claim and every conceptual claim is checked against a primary source before it lands — never written from memory:

- **Context7** → vLLM (and related libraries') **API, flags, CLI** — the concrete signatures you type.
- **Sonar** → **concepts, papers, and specifications** — algorithm provenance, number-format specs, hardware parameters.

This verification stays static: **no code is executed and no figures are reproduced** (ADR-0004). It catches hallucinated flags and misattributed algorithms at the source; it does not turn illustrative numbers into measured ones.

## When vLLM upgrades — the refresh recipe

A major vLLM release will gradually make the baseline examples stale. When you decide to bump the baseline (a deliberate, whole-site action — ADR-0004 keeps re-verification *deferred*, not automated), work through this checklist:

1. **Bump the version annotation** — the page-top baseline callout on affected pages, and the Baselines table on the [Home page](index.md).
2. **Re-verify every touched flag / API / CLI via Context7** against the *new* version's official docs. Assume nothing carried over.
3. **Re-verify concept / paper / spec claims via Sonar** for any section you deepen or change.
4. **Re-align source read-along anchors** — the file / symbol / line references in Part 2 / 3 / 5 must point at the new release tag, not the old line numbers.
5. **Re-run the Labs to recalibrate numbers you care about** — on your AutoDL box, following the [Capstone](capstone/index.md) measurement discipline (greedy, fixed seed, stated eval set). Update figures only where you re-measured; leave the rest labelled illustrative.
6. **Keep the build green** — `mkdocs build --strict` is the one structural gate; broken links, orphan pages, and i18n gaps all fail it.

The stance is **honest but lightweight**: annotate the version, verify what you touch, and defer a full re-verification sweep rather than standing up CI too early.

## Found drift? Open a refresh issue

If you hit a flag that no longer exists, code that won't run, or a number that feels off for the current release, open a **"Content refresh / vLLM version upgrade"** issue from the repository's [issue templates](https://github.com/xiangzhang-coding/inference-learning-path/issues/new/choose). The template walks through the same checklist above so nothing is missed.

## See also

- [Home · Baselines](index.md) — the one-glance baseline table and the "on the numbers" stance.
- [Capstone](capstone/index.md) — the measurement discipline to follow when you recalibrate numbers.
- [Glossary](glossary.md) — the bilingual terminology every page reuses.
