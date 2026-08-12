# Prefix caching: reuse shared-prefix KV

!!! info "Baseline: **vLLM 0.26.0** · design verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [Prefix Caching: Reuse the KV of a Shared Prefix](../part5/prefix-caching.md)

---

## Q: What is prefix caching, how does vLLM guarantee it never reuses the wrong KV, when does it help, and why doesn't it change outputs?

### Direct answer

Prefix caching reuses the [KV cache](../part0/kv-cache.md) of a **shared prefix** so it's computed once instead of re-prefilled per request. Real traffic shares prefixes constantly — system prompts, few-shot examples, multi-turn chat history, RAG documents — and prefix caching turns that repetition into skipped prefill.

**Correctness comes from content-hashing the [blocks](../part5/paged-attention.md).** Each KV block's hash folds in **its tokens *and its parent block's hash*** (a chain), so a cached block only matches when the *entire* prefix leading to it is byte-identical — a match can't come from a different context. **Only full blocks cache** (a partial trailing block depends on future tokens). The reused KV is exactly what prefill would produce, so **outputs are unchanged** — it's a pure "don't redo work" optimization.

**When it helps:** high prefix-length × hit-rate. A 500-token system prompt on every request saves ~500 prefill tokens per request after the first (lower TTFT, freed capacity). Unique prompts with no shared prefix gain nothing.

### Deep dive

- **Serving a hit.** The block pool keeps a `cached_block_hash_to_block` map; on a match the manager `touch()`es the block (bumps `ref_cnt`, rescues it from the eviction queue) and points the new request's block table at it — prefill starts at the first *un*cached block.
- **No extra memory.** Cached-but-unreferenced blocks are just eviction candidates reclaimed under pressure (LRU via the free-queue order); steady-state memory is unchanged.
- **On by default (V1).** `enable_prefix_caching`; `prefix_caching_hash_algo` defaults to `"sha256"`.
- **Order matters.** Because the hash chain starts at token 0, variable content (timestamps, user ids, shuffled few-shot) *before* the shared prefix kills all hits. Stable prefix first, variable suffix last.

### Code

The savings as pure arithmetic — prefill tokens with vs without caching:

```python
BLOCK, PREFIX, SUFFIXES = 16, 512, [24, 40, 8, 32, 16, 48, 12, 20]
def prefill_tokens(prefix, suffixes, block, cache):
    cached = total = 0
    for s in suffixes:
        total += (prefix - (cached if cache else 0)) + s
        if cache: cached = (prefix // block) * block     # full prefix blocks now cached
    return total
print(prefill_tokens(PREFIX, SUFFIXES, BLOCK, False))    # 4296 — prefix recomputed 8x
print(prefill_tokens(PREFIX, SUFFIXES, BLOCK, True))     # 712  — prefix computed once (~83% fewer)
```

### Interviewer follow-ups

- *"How is a false match prevented?"* → The block hash chains in the parent's hash, so a block matches only if the whole preceding prefix is identical; one differing early token changes every downstream hash.
- *"Does it change outputs?"* → No — reused KV is byte-identical to recomputed KV. Different outputs = bug.
- *"Does it cost memory?"* → No net cost — it reuses the block pool; unreferenced cached blocks are eviction candidates.
- *"Why did my hit rate stay ~0 behind a round-robin LB?"* → Hits only help on the replica holding the blocks. Use **KV-cache-aware routing** to send shared-prefix requests to the same replica.
- *"What kills a hit silently?"* → Any prefix variation (timestamp, whitespace, reordered few-shot) or putting variable content first; also a partial (non-full) block.

### Linked concepts

- Lesson: [Prefix Caching](../part5/prefix-caching.md)
- Related: [PagedAttention: block manager & fragmentation](kv-cache-block-manager.md) (the blocks/`ref_cnt`/`touch()` it rides on), [Chunked prefill & PD](chunked-prefill-pd.md) (the sibling prefill lever), [Static vs continuous batching](continuous-batching.md) (what the freed capacity feeds)
- Glossary: [Prefix caching, KV-cache aware routing](../glossary.md)
