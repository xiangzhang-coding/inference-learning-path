# Reading vLLM's PagedAttention Kernel

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The kernel signature, the paged KV-cache layouts (`k_cache`, `v_cache`), `PagedAttention.split_kv_cache`, and `write_to_paged_cache` → `ops.reshape_and_cache(..., slot_mapping, ...)` are quoted from vLLM's source and its `docs/design/paged_attention.md`, verified via Context7 (ADR-0004). The `block_tables` argument and the `v1`/`v2` op names are described at the **concept level** — read the design doc for the exact current argument list, which shifts across versions. This lesson builds *source-reading* skill (ADR-0002); the serving-side deep-dive (continuous batching, block manager, why paging raises throughput) is **Part 5**.

---

## 1 · Intuition & why it matters

You've written kernels ([Triton basics](triton-basics.md)); now read a real one that matters. PagedAttention is the mechanism behind vLLM's headline throughput, and "can you read the PagedAttention kernel and explain what it does" is a genuine senior-infra interview probe. Per ADR-0002, the goal is exactly this: **read + understand production kernels**, not author your own.

Here's the one idea. A naive engine stores each sequence's [KV cache](../part0/kv-cache.md) as one **contiguous** block sized for the *maximum* possible length — so a request that might reach 4k tokens reserves 4k worth of KV up front, and most of it sits empty. That internal fragmentation is what caps how many sequences fit in VRAM. PagedAttention borrows the operating-system trick of **virtual memory**: chop the KV cache into fixed-size **blocks** (pages), store them **anywhere** in a shared pool, and keep a per-sequence **block table** mapping logical block index → physical block number. A sequence grows one block at a time (near-zero waste), and blocks can even be **shared** across sequences (the basis of prefix caching). The cost: attention can no longer read one contiguous KV span — the kernel must **gather** KV block-by-block through the block table. That gather is what the PagedAttention kernel exists to do, efficiently. → see the [Glossary](../glossary.md) for *PagedAttention, KV cache, Block table*.

## 2 · Mental model

The virtual-memory analogy, and the kernel's loop:

```text
LOGICAL view (what attention wants)        PHYSICAL view (how vLLM stores it)
  seq A: [tok0 tok1 tok2 tok3 tok4 tok5]     KV block pool (fixed-size blocks, shared):
             │ block table maps            ┌──────┬──────┬──────┬──────┬──────┐
             ▼  logical→physical           │ blk0 │ blk1 │ blk2 │ blk3 │ blk4 │ ...
  A.block_table = [3, 1]  ───────────────► │ (B)  │ tok4 │ (C)  │ tok0 │      │
                                           │      │ tok5 │      │ tok1 │      │
      logical blk 0 ─► physical blk 3      │      │      │      │ tok2 │      │
      logical blk 1 ─► physical blk 1      │      │      │      │ tok3 │      │
                                           └──────┴──────┴──────┴──────┴──────┘
  (physical order is arbitrary; the block table restores logical order)

THE KERNEL (one query, its KV scattered across blocks):
  for logical_blk in seq.block_table:        # walk this sequence's blocks
      phys = block_table[logical_blk]         # logical -> physical
      K_blk, V_blk = k_cache[phys], v_cache[phys]
      s = Q · K_blkᵀ                          # scores for this block's tokens
      update running (m, l, acc) with ONLINE SOFTMAX   # same trick as FlashAttention
  out = acc / l
```

Three shapes to hold:

- **A block table is a page table for KV.** Physical placement is arbitrary; the block table is the indirection that turns scattered blocks back into a logical sequence. Growth appends one block; sharing points two tables at the same physical block.
- **The kernel gathers, then does ordinary attention.** Once a block's K/V is loaded, the math is the same $QK^\top$ → softmax → $\cdot V$ you already know — folded in with **online softmax** so the kernel never materializes the full score row (the [FlashAttention](../part2/flash-attention.md) idea, applied per block).
- **"Block" here is a KV page, not a thread block.** vLLM's default KV block size is 16 tokens. Don't confuse it with the CUDA/Triton thread block from the [execution-model](cuda-execution-model.md) lesson.

## 3 · Principle & reading the source

The files to open: the design doc `docs/design/paged_attention.md` (the narrative), the CUDA kernel in `csrc/attention/`, and the Python wrapper `vllm/.../attention/ops/paged_attn.py`. Here's the map.

### 3.1 The paged KV-cache layout

vLLM stores K and V caches as pools of blocks. The verified kernel signature shows the shapes:

```cpp
// k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
// v_cache: [num_blocks, num_kv_heads, head_size,   block_size]
//   num_blocks  = size of the physical pool     block_size = tokens per block (e.g. 16)
```

`num_blocks` (the pool) is the leading dimension — blocks are the unit of allocation. The odd part is the **`x`** in the K-cache: `split_kv_cache` computes `x = 16 // element_size` (so `x = 8` for FP16) and views the key cache as `[num_blocks, num_kv_heads, head_size // x, block_size, x]`. That splits `head_size` into groups of `x` contiguous elements so each thread reads a **16-byte-aligned** chunk — the [coalescing](memory-access.md) optimization from the last lesson, baked into the layout. The V cache doesn't need it (it's read along a different axis), hence the simpler shape. Reading tip: when a cache shape looks weird, the trailing packed dim is almost always there to make loads coalesced.

### 3.2 Writing into the cache

New KV is written by `PagedAttention.write_to_paged_cache`, which calls `ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping.flatten(), kv_cache_dtype, k_scale, v_scale)`. The key argument is **`slot_mapping`**: for each token, its flat physical slot (`physical_block × block_size + offset_in_block`). So the write path scatters tokens to physical slots via `slot_mapping`; the read path (the kernel) gathers them back via the block table. Same indirection, two directions.

### 3.3 The kernel's structure

Inputs (verified): the query `q [num_seqs, num_heads, head_size]`, the two caches, and the output `out [num_seqs, num_heads, max_num_partitions, head_size]`. The core is a loop over the sequence's blocks, and within each, an accumulation — the design doc's own pseudocode:

```cpp
float accs[NUM_ROWS_PER_THREAD];
for ... {                 // iterate over the sequence's blocks
    logits_vec = ...      //   scores for this block (Q·Kᵀ, then softmax weights)
    for ... {             //   iterate over rows
        v_vec = ...
        accs[i] += dot(logits_vec, v_vec);   // weighted sum into V — online-softmax accumulate
    }
}
```

That is exactly the online-softmax loop from FlashAttention, restricted to one physical block per outer step. Two template parameters worth noting: `BLOCK_SIZE` (KV tokens per block) and `PARTITION_SIZE`. When `PARTITION_SIZE > 0`, the kernel splits a long KV sequence into partitions computed in parallel and combined afterward — that's the `max_num_partitions` dimension in `out`, and the distinction between the two kernel variants (informally "v1" without partitioning for short contexts, "v2" with it for long ones — the split raises occupancy the same way FlashDecoding does).

### 3.4 What to trace when you read it

Follow one query through: (1) find its `block_table` row → the list of physical blocks; (2) for each, index `k_cache`/`v_cache` at that `num_blocks` slot; (3) watch the running max / sum / accumulator (online softmax); (4) see the final divide and the write to `out`. If you can narrate those four steps, you can read the kernel — the rest is CUDA thread-assignment detail that the layout (§3.1) exists to make coalesced.

## 4 · Complete runnable code + line-by-line

This is a pure-Python **paged attention** that mirrors the kernel's shape: KV lives in non-contiguous physical blocks, a block table restores logical order, and attention gathers block-by-block with online softmax. Comparing it to dense (contiguous) attention proves **paging is a storage scheme, not a math change** — pure CPU, offline-runnable.

```python title="paged_attention_ref.py"
"""Paged attention == dense attention, but KV lives in non-contiguous blocks.
Pure CPU, offline — mirrors the vLLM kernel's block-loop shape, not its threading."""
import math

BLOCK = 4                                             # KV tokens per block (vLLM default is 16)

def paged_attention(q, block_table, k_pool, v_pool, seq_len):
    """Walk the sequence's blocks via the block table; fold each in with online softmax."""
    d = len(q)
    m, l, acc = -math.inf, 0.0, [0.0] * d             # running max, normalizer, output
    pos = 0
    for phys in block_table:                          # logical block order -> physical block ids
        k_blk, v_blk = k_pool[phys], v_pool[phys]     # one physical block = BLOCK token slots
        for t in range(BLOCK):
            if pos >= seq_len:                        # the last block may be partially filled
                break
            k, v = k_blk[t], v_blk[t]
            s = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)
            m_new = max(m, s)
            corr = math.exp(m - m_new) if m != -math.inf else 0.0
            p = math.exp(s - m_new)
            l = l * corr + p                          # online-softmax update (rescale + add)
            acc = [acc[j] * corr + p * v[j] for j in range(d)]
            m = m_new
            pos += 1
    return [a / l for a in acc]

def dense_attention(q, K, V):                         # reference: one contiguous KV span
    d = len(q)
    s = [sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d) for k in K]
    m = max(s); e = [math.exp(x - m) for x in s]; Z = sum(e)
    return [sum(e[i] / Z * V[i][j] for i in range(len(V))) for j in range(d)]

if __name__ == "__main__":
    d, seq_len = 4, 6
    q = [0.5, -0.3, 0.8, 0.1]
    K = [[0.2, 0.1, -0.4, 0.6], [0.9, -0.2, 0.3, 0.0], [-0.5, 0.4, 0.7, -0.1],
         [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, -0.8, 0.2], [-0.3, 0.5, 0.2, 0.9]]
    V = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0],
         [0, 0, 0, 1.0], [.5, .5, .5, .5], [1.0, 1, 1, 1]]

    # Scatter 6 tokens into physical blocks (BLOCK=4) in NON-contiguous order, like a real allocator:
    # ceil(6/4)=2 logical blocks, placed at physical ids 3 and 1.  block_table restores order.
    block_table = [3, 1]
    k_pool, v_pool = {}, {}
    for i in range(seq_len):
        phys, slot = block_table[i // BLOCK], i % BLOCK
        k_pool.setdefault(phys, [[0.0] * d for _ in range(BLOCK)])
        v_pool.setdefault(phys, [[0.0] * d for _ in range(BLOCK)])
        k_pool[phys][slot], v_pool[phys][slot] = K[i], V[i]

    paged = paged_attention(q, block_table, k_pool, v_pool, seq_len)
    dense = dense_attention(q, K, V)
    diff = max(abs(a - b) for a, b in zip(paged, dense))
    print("paged :", [round(x, 6) for x in paged])
    print("dense :", [round(x, 6) for x in dense])
    print(f"max abs diff = {diff:.2e}   (paged == dense; blocks are just storage)")
```

**Line-by-line:**

- `paged_attention` — the kernel's shape in Python. The outer loop walks `block_table` (logical→physical); each `phys` indexes the block pool, exactly like indexing `k_cache`/`v_cache` at a `num_blocks` slot. The inner loop folds each token in with the **online-softmax** update (`corr` rescales the running normalizer and accumulator when the max moves) — the §3.3 `accs[i] += dot(...)` loop.
- The `pos >= seq_len` guard handles a **partially filled last block** — a real detail (a 6-token sequence uses 2 blocks of 4, the second half-empty), and why the kernel needs the true sequence length, not just the block count.
- The scatter loop places the sequence's two logical blocks at **physical ids 3 and 1** — deliberately out of order, to show physical placement is arbitrary; `block_table = [3, 1]` is the only thing that restores logical order. `slot_mapping` in vLLM is the flattened version of this placement.
- `dense_attention` — the same attention over one contiguous KV span, the ground truth.

Expected output (exact arithmetic, not a benchmark):

```text
paged : [0.363083, 0.449897, 0.392487, 0.386499]
dense : [0.363083, 0.449897, 0.392487, 0.386499]
max abs diff = 1.11e-16   (paged == dense; blocks are just storage)
```

The difference is machine epsilon. Paging changes *where* KV lives and *how* the kernel reaches it — never *what* attention computes. That is the whole license for PagedAttention: near-zero memory waste and block sharing, at the price of a gather the kernel absorbs.

## 5 · Lab — read the real kernel

!!! gpu "GPU Lab (optional verification)"
    - **Min VRAM:** none to *read*; ~16 GB to run vLLM and observe block allocation with `Qwen2.5-7B-Instruct` (INT4/AWQ) if you want to verify live
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** reading ~30 min (free, no-card mode) · optional run ~10 min · ~¥1 (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** the paged-KV *design* is backend-independent; the CUDA kernel is NVIDIA-specific, ROCm has its own port, and FlashInfer/FlashAttention backends implement the same paged-KV contract differently.

The core lab is **reading**, doable entirely in AutoDL no-card mode (free):

```text
Reading checklist — trace one query through the source:
1. Open docs/design/paged_attention.md — read the "Inputs" and layout sections.
2. In vllm/.../attention/ops/paged_attn.py, find split_kv_cache:
     - confirm x = 16 // element_size, and the [num_blocks, num_kv_heads, head_size//x, block_size, x] view.
3. In csrc/attention/, find the block loop:
     - locate where block_table maps logical -> physical block index,
     - locate the online-softmax running max / sum / accumulator,
     - locate the PARTITION_SIZE branch (the v1 vs v2 split) and the max_num_partitions output dim.
4. Map each to a line of §4's paged_attention(): block-table walk, per-block gather, online-softmax fold.
```

Optional GPU verification: launch vLLM with `Qwen2.5-7B-Instruct` and a small `--max-model-len`, send a few requests, and watch KV-block usage in the logs / metrics — you'll see blocks allocated on demand as sequences grow, not reserved up front. (The full serving picture is the Part 5 lesson; here it's just confirmation that the block pool behaves like §4's `k_pool`.)

## 6 · Common pitfalls / counter-intuitive points

- **Confusing KV "block" with thread "block".** vLLM's KV block is a page of ~16 token slots in the cache pool; the CUDA/Triton thread block is a scheduling unit. Same word, unrelated concept.
- **Thinking paging changes the attention result.** It doesn't — §4 proves paged == dense to machine precision. Paging is a memory-management scheme; the kernel gathers scattered KV but computes identical attention.
- **Missing why the K cache has that extra `x` dimension.** It's not arbitrary — `x` packs `head_size` into 16-byte-aligned chunks so the kernel's loads are coalesced. The V cache is read along a different axis and doesn't need it.
- **Assuming block tables are contiguous or ordered.** Physical blocks are allocated wherever there's room; the block table (and `slot_mapping`) is the only thing tying them to logical order. That freedom is the point — it's what kills fragmentation and enables sharing.
- **Reading the kernel expecting one contiguous KV read.** The gather (block loop) is the defining difference from a dense-KV attention kernel; if you're looking for a single strided load over the whole sequence, you'll miss the structure.
- **Over-claiming the serving story here.** *How* paging raises throughput (continuous batching, the block manager, prefix caching) is a systems topic — this lesson is about reading the kernel that makes it possible.

## 7 · Interview links

- [PagedAttention kernel: paged KV cache & block tables](../interview/paged-attention-kernel.md) — the high-frequency question this lesson prepares you for: *why store KV in blocks, what a block table does, how the kernel gathers KV, and why it's mathematically identical to dense attention.*

## 8 · Summary & further reading

**One line:** PagedAttention stores the KV cache as fixed-size blocks in a shared pool with a per-sequence block table (logical→physical), so sequences grow with near-zero waste and can share blocks; the kernel gathers KV block-by-block through that table and folds it in with online softmax — computing exactly the same attention as a contiguous-KV kernel, which is why reading it is mostly about following the indirection.

Further reading:

- Kwon et al. — *Efficient Memory Management for LLM Serving with PagedAttention* (the vLLM paper) — the virtual-memory framing and the fragmentation numbers.
- vLLM `docs/design/paged_attention.md` — the kernel walkthrough this lesson maps to; read it with §3 open.
- The [FlashAttention](../part2/flash-attention.md) lesson — the online-softmax accumulation the block loop reuses.
- Part 5 (serving) — where continuous batching, the block manager, and prefix caching turn this kernel into throughput.

## 9 · Self-check

??? question "Why does vLLM store the KV cache in fixed-size blocks instead of one contiguous span per sequence?"
    A contiguous per-sequence cache must be sized for the *maximum* possible length, so most of it sits empty for shorter or still-growing sequences — internal fragmentation that caps how many sequences fit in VRAM. Fixed-size blocks (pages) let a sequence grow one block at a time with near-zero waste, let the allocator place blocks anywhere in a shared pool, and let different sequences **share** identical blocks (e.g. a common prompt prefix). The trade is that attention can no longer read one contiguous KV span — the kernel must gather blocks via the block table.

??? question "What is a block table, and how does the kernel use it during attention?"
    A block table is a per-sequence list mapping each **logical** KV block index to a **physical** block number in the shared cache pool — a page table for the KV cache. The attention kernel walks the sequence's block table in logical order, and for each entry indexes `k_cache`/`v_cache` at that physical block's slot to load its K/V, computes that block's scores, and folds them into a running online-softmax accumulator. The write path uses the flattened equivalent (`slot_mapping`) to scatter new tokens into physical slots.

??? question "Is PagedAttention an approximation of attention? Justify your answer with what the kernel actually does."
    No — it computes exactly the same attention as a dense, contiguous-KV kernel (identical to machine precision, as the §4 reference shows). The only thing paging changes is *where* KV is stored (scattered fixed-size blocks) and *how* the kernel reaches it (gather via the block table instead of one contiguous read). Inside, it still does $QK^\top$ → softmax → $\cdot V$, accumulated with online softmax. Paging is a memory-management scheme, not a mathematical change — which is why the win is purely in memory efficiency (less fragmentation, block sharing), not in the result.
