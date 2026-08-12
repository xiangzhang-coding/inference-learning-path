# PagedAttention: KV cache as virtual memory (block manager & fragmentation)

!!! info "Baseline: **vLLM 0.26.0** · V1 block-manager internals verified via Context7 (ADR-0004)"

**Module:** Part 5 · Serving & Throughput (vLLM Core)   ·   **Tests the lesson:** [PagedAttention: Managing the KV Cache Like Virtual Memory](../part5/paged-attention.md)

---

## Q: Why does a contiguous KV cache fragment, what does PagedAttention's block manager do about it, how is the pool size (`num_gpu_blocks`) determined, and how does paging turn into throughput?

### Direct answer

A naive engine stores each sequence's [KV cache](../part0/kv-cache.md) as one **contiguous** region sized for the **maximum** length — it must, because the kernel reads it as one span and generation length is unknown up front. So a request that might reach 512 tokens reserves all 512 even if it emits 40: **internal fragmentation**. Variable-size holes from finished sequences add **external fragmentation**.

PagedAttention applies **virtual memory**: chop the KV cache into fixed-size **blocks** (pages, 16 tokens), keep them in a shared **pool**, and give each sequence a **block table** (logical block → physical block anywhere). The **block manager** allocates one block at a time as a sequence grows (waste ≤ one partial block), returns blocks to a free list on finish, and lets sequences with a shared prefix point at the **same** physical blocks. 

**How that becomes throughput:** killing fragmentation means far more sequences fit the same VRAM → the [continuous batch](../part5/continuous-batching.md) grows larger → the memory-bound decode amortizes better. Paging doesn't speed the math (the [kernel](../part3/paged-attention-kernel.md) computes identical attention); it raises **capacity**, and capacity is the concurrency ceiling.

### Deep dive

- **`num_gpu_blocks` is profiled at startup.** vLLM takes `gpu_memory_utilization × VRAM` (default **0.92**) as the budget, subtracts non-KV memory (weights + peak activations + CUDA-graph), and divides the rest by bytes-per-block: $\texttt{num\_gpu\_blocks}=\lfloor(\texttt{util}\cdot\text{VRAM}-\text{weights}-\text{act}-\text{cudagraph})/\text{bytes/block}\rfloor$. This is why quantization (smaller weights) and FP8 KV cache (smaller bytes/block) raise concurrency.
- **The V1 structures.** A `BlockPool` holds the `KVCacheBlock`s; a `free_block_queue` (doubly-linked list, eviction order) gives O(1) allocate (pop front) / free (push back); a `cached_block_hash_to_block` map backs prefix caching; each block has a `ref_cnt`. A per-request `SingleTypeKVCacheManager` (`req_to_blocks`) draws from the shared pool.
- **The lifecycle.** Admit → pop free blocks for the prompt; decode → pop one more when the last block fills; finish → push blocks back (decrement `ref_cnt`, free at 0); under pressure → preempt (evict/recompute), the OS page-out analogue.
- **Sharing → prefix caching.** A block is keyed by the hash of its tokens (+ parent hash). Identical prefixes → identical hashes → the second request's table points at existing blocks (`touch()` bumps `ref_cnt`); only **full** blocks cache, KV is byte-identical so **outputs are unchanged**; divergence triggers copy-on-write. (Tuning it is the next Part 5 topic.)
- **This is the kernel lesson's other half.** [Part 3](../part3/paged-attention-kernel.md) is how the kernel *gathers* scattered blocks; this is how the manager *allocates* them.

### Code

The fragmentation argument as arithmetic (pure Python) — same pool, two policies:

```python
import math
BLOCK, POOL, MAX_LEN = 16, 128, 512
lens = [40,128,300,64,210,96,180,48,150,80,60,420,33,256,90,110]

reserve = math.ceil(MAX_LEN/BLOCK)                       # contiguous: 32 blocks/seq, always
contig  = POOL // reserve                                 # -> 4 sequences fit
used = paged = 0
for L in lens:                                            # paged: ceil(actual/BLOCK) each
    need = math.ceil(L/BLOCK)
    if used + need <= POOL: used += need; paged += 1
print(contig, "vs", paged)   # 4 vs 13 — 3x+ concurrency from the same VRAM, no math change
```

### Interviewer follow-ups

- *"What's the OS analogy, exactly?"* → Virtual memory / demand paging: the block table is a page table (logical→physical), the pool is physical RAM, allocate-on-append is demand paging, preemption is page-out, and prefix sharing is two page tables mapping one physical page.
- *"KV block vs thread block?"* → Unrelated. KV block = a 16-token page of the cache pool; thread block = a CUDA scheduling unit. Same word.
- *"Does paging change the output or speed the kernel?"* → Neither the output (identical to dense, proved in [Part 3](../part3/paged-attention-kernel.md)) nor the math speed. The gain is capacity → bigger batch → throughput.
- *"How do you raise the concurrency ceiling when KV-bound?"* → Increase `num_gpu_blocks`: raise `gpu_memory_utilization` (carefully), quantize weights (frees budget), quantize KV to FP8 (halves bytes/block). All three map to terms in the formula.
- *"Why 16-token blocks — why not 1 or 256?"* → Trade-off: bigger blocks = coarser allocation (more partial-block waste) but fewer block-table entries / cheaper bookkeeping; smaller = less waste, more overhead. 16 is the tuned default.

### Linked concepts

- Lesson: [PagedAttention: Managing the KV Cache Like Virtual Memory](../part5/paged-attention.md)
- Related: [Static vs continuous batching](continuous-batching.md) (the batch this capacity feeds), [PagedAttention kernel & block tables](paged-attention-kernel.md) (the gather side), [KV cache & throughput ceiling](kv-cache.md), [VRAM budget & max concurrency](vram-capacity-planning.md)
- Glossary: [PagedAttention, KV cache, Block table](../glossary.md)
