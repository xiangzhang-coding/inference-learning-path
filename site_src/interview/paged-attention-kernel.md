# PagedAttention kernel: paged KV cache & block tables

!!! info "Baseline: **vLLM 0.26.0** · kernel layout & `paged_attn.py` verified via Context7 (ADR-0004)"

**Module:** Part 3 · GPU Programming (Triton)   ·   **Tests the lesson:** [Reading vLLM's PagedAttention Kernel](../part3/paged-attention-kernel.md)

---

## Q: Explain PagedAttention. Why store the KV cache in blocks, what does a block table do, how does the kernel gather KV during attention, and is the result any different from dense attention?

### Direct answer

PagedAttention applies **virtual memory** to the [KV cache](../part0/kv-cache.md). A naive engine reserves one **contiguous** KV region per sequence, sized for the max length — so most of it sits empty (internal fragmentation), capping concurrency. PagedAttention instead:

- **Stores KV in fixed-size blocks** (pages, ~16 tokens each) in a shared pool. A sequence grows one block at a time (near-zero waste), blocks live anywhere, and identical blocks can be **shared** across sequences (prefix caching).
- **Keeps a per-sequence block table** mapping logical block index → physical block number — a page table for KV. `slot_mapping` is the write-side equivalent (token → flat physical slot).
- **The kernel gathers**: it walks the sequence's block table, loads each physical block's K/V, computes that block's $QK^\top$ scores, and folds them into a running **online-softmax** accumulator (the FlashAttention trick, per block).

Is the result different from dense attention? **No** — identical to machine precision. Paging changes *where* KV lives and *how* the kernel reaches it, never *what* attention computes. The win is memory efficiency (no fragmentation, block sharing), not a different output.

### Deep dive

- **Why blocks kill fragmentation.** Contiguous reservation wastes `max_len − actual_len` per sequence; block allocation wastes at most one partial block per sequence. That reclaimed VRAM becomes more KV cache → higher concurrency (tie to [VRAM budget](vram-capacity-planning.md)).
- **The K-cache's extra dimension.** vLLM lays the K cache out as `[num_blocks, num_kv_heads, head_size/x, block_size, x]` where `x = 16 // element_size` (8 for FP16). The trailing `x` packs `head_size` into 16-byte-aligned chunks so the kernel's loads are [coalesced](memory-coalescing.md); the V cache (`[num_blocks, num_kv_heads, head_size, block_size]`) is read along a different axis and needs no such packing.
- **v1 vs v2 (partitioning).** A `PARTITION_SIZE` template parameter splits a long KV sequence into partitions computed in parallel and combined after — the `max_num_partitions` dim in the output. Short contexts use the unpartitioned path; long ones use the partitioned one to raise occupancy (same idea as FlashDecoding).
- **Write vs read symmetry.** `write_to_paged_cache` → `ops.reshape_and_cache(..., slot_mapping, ...)` scatters new tokens into physical slots; the attention kernel gathers them back via the block table. One indirection, two directions.

### Code

Paged attention as a pure-Python model — the block-table walk + online-softmax fold, proven equal to dense:

```python
import math
BLOCK = 4                                          # KV tokens per block (vLLM: 16)

def paged_attention(q, block_table, k_pool, v_pool, seq_len):
    d = len(q); m, l, acc = -math.inf, 0.0, [0.0]*d; pos = 0
    for phys in block_table:                        # logical -> physical block ids
        for t in range(BLOCK):
            if pos >= seq_len: break                # last block may be partial
            k, v = k_pool[phys][t], v_pool[phys][t]
            s = sum(qi*ki for qi, ki in zip(q, k)) / math.sqrt(d)
            m_new = max(m, s); corr = math.exp(m - m_new) if m != -math.inf else 0.0
            p = math.exp(s - m_new); l = l*corr + p
            acc = [acc[j]*corr + p*v[j] for j in range(d)]; m = m_new; pos += 1
    return [a/l for a in acc]                        # == dense attention, to machine precision
```

`block_table = [3, 1]` (physical blocks out of order) still reproduces dense attention exactly — physical placement is arbitrary, the table restores logical order.

### Interviewer follow-ups

- *"What's the analogy to operating systems?"* → Virtual memory / paging: the block table is a page table mapping logical (virtual) KV blocks to physical blocks in a shared pool; growth = allocate a page, sharing = two page tables pointing at one physical page.
- *"How does prefix caching fall out of this?"* → If two sequences share a prompt prefix, their block tables point at the **same physical blocks** for that prefix — computed and stored once. Blocks are the unit of sharing.
- *"What's the cost of paging vs contiguous KV?"* → The kernel must **gather** KV block-by-block (indirection through the block table) instead of one contiguous read, plus block-table bookkeeping. That's the price for eliminating fragmentation and enabling sharing — and the gather is what the custom kernel optimizes.
- *"Why does the K cache have that weird `head_size/x … x` shape?"* → To make loads coalesced: `x` packs elements into 16-byte-aligned chunks so each thread reads an aligned vector. It's a memory-layout optimization, not a semantic one.
- *"Does PagedAttention change the attention output?"* → No — identical to dense to machine precision. It's purely a memory-management scheme; the kernel still computes standard attention with online softmax.

### Linked concepts

- Lesson: [Reading vLLM's PagedAttention Kernel](../part3/paged-attention-kernel.md)
- Related: [KV cache & throughput ceiling](kv-cache.md) (what's being paged), [VRAM budget & max concurrency](vram-capacity-planning.md) (the fragmentation paging reclaims), [FlashAttention & IO-aware attention](flash-attention.md) (the online-softmax fold), [Memory coalescing, shared memory & bank conflicts](memory-coalescing.md) (why the K-cache `x` packing exists)
- Glossary: [PagedAttention, KV cache, Block table](../glossary.md)
