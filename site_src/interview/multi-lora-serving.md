# Multi-LoRA serving: one base, many adapters

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 6 · Advanced Inference Topics   ·   **Tests the lesson:** [Multi-LoRA Serving: One Base, Many Adapters](../part6/multi-lora-serving.md)

---

## Q: What is multi-LoRA serving, why does LoRA make an adapter tiny, how does vLLM run different adapters in one batch, and which knobs cap how many you can co-serve?

### Direct answer

Multi-LoRA serving hosts **one frozen base model plus many small [LoRA](../glossary.md) adapters**, choosing which adapter to apply **per request** — even mixing adapters across rows of the *same* [continuous batch](../part5/continuous-batching.md). It exists because serving N full fine-tunes means N full copies (N × ~15 GB for a 7B) — impossible on one [24 GB card](../part0/gpu-hardware.md).

**Why an adapter is tiny:** a fine-tune is a **low-rank** nudge, so LoRA freezes $W$ and learns $\Delta W = BA$ with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r \ll d$. Parameters drop from $d\cdot k$ to $r(d+k)$ — ~100× per matrix — so a whole-model adapter is single-digit-to-tens of MB. Load the base **once**, keep dozens of adapters on the shelf.

**How one batch serves many adapters:** the **base GEMM $Wx$ runs once** for the whole batch; each row's adapter delta $B(Ax)$ is added by a *grouped* kernel (SGMV/BGMV, from S-LoRA/Punica) that groups rows by adapter id. No per-adapter model replay.

**Knobs:** `--max-lora-rank` (size slots to your highest adapter rank — no higher), `max_loras` (distinct adapters active per batch), `max_cpu_loras` (CPU-side cache).

### Deep dive

- **Selection is data.** A request names its adapter by integer id (`LoRARequest(name, id, path)`) or, on the server, by putting the adapter **name in the `model` field**. Adding a fine-tune = load one file, not a new server.
- **Static vs dynamic.** Declare at startup (`--lora-modules name=path`) or hot-load at runtime via `POST /v1/load_lora_adapter`, gated by `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` — a security-sensitive, trusted-environments-only feature.
- **Memory trade.** Adapter slots share the VRAM that would otherwise be [KV-cache blocks](../part5/paged-attention.md); an oversized `--max-lora-rank` steals concurrency.
- **Batching economics.** A few hot adapters batch well; a long tail of cold adapters exceeding `max_loras` can't co-run in one step and drops throughput.

### Code

The verified 0.26.0 offline API — heterogeneous adapters in one `generate`:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True, max_lora_rank=64, max_loras=2)
outs = llm.generate(
    ["[user] SQL: airports in Malawi [/user] [assistant]", "What is a KV cache?"],
    SamplingParams(temperature=0, max_tokens=64),
    lora_request=[LoRARequest("sql", 1, sql_path), None],   # row 0 → adapter, row 1 → base
)
```

### Interviewer follow-ups

- *"Why not just merge the adapter into the base?"* → Merging gives a single dedicated model with zero per-request delta cost, but kills multi-tenancy. Merge when you serve exactly one fine-tune; keep runtime when you serve many.
- *"How can two rows use different adapters without re-running the model?"* → Shared base GEMM once; per-row $B(Ax)$ added by a grouped kernel. The base pass is the cost; deltas are cheap.
- *"You set `--max-lora-rank 256` for rank-16 adapters — what's wrong?"* → Wasted VRAM (vLLM docs' own "unnecessarily high" case) that could be KV cache = more concurrency. Set it to the true max rank.
- *"Adapter gives garbage — first check?"* → Base/adapter mismatch (wrong base model or version). An adapter is a delta on a *specific* base.
- *"Risk of exposing `/v1/load_lora_adapter`?"* → It loads weights from an arbitrary path = trusting arbitrary code/data. Keep it behind the trust boundary; trusted envs only.

### Linked concepts

- Lesson: [Multi-LoRA Serving](../part6/multi-lora-serving.md)
- Related: [Static vs continuous batching](continuous-batching.md) (the batch adapters ride in), [PagedAttention: block manager & fragmentation](kv-cache-block-manager.md) (the VRAM pool adapter slots share), [Tuning knobs](tuning-knobs.md) (LoRA knobs vs the throughput/latency curve)
- Glossary: [LoRA / Multi-LoRA serving](../glossary.md)
