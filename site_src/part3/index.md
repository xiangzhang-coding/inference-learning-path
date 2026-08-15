# Part 3 · GPU Programming (Triton)

> Enough GPU programming to *reason about* why kernels are fast and to *read* vLLM's kernels — not a CUDA C++ course (see ADR-0002 for the depth boundary).

## What this part covers

- The **[CUDA execution model](cuda-execution-model.md)** — grid/block/thread, warps, SIMT divergence, and occupancy — via mental models rather than rote detail
- **[Memory access](memory-access.md)** — coalescing, shared memory, and bank conflicts: how a warp *should* touch memory
- Writing a few simple **[Triton](triton-basics.md)** kernels — the "I can write a little" confidence
- A guided read of vLLM's **[PagedAttention kernel](paged-attention-kernel.md)** to build source-reading skill

See the **[Glossary](../glossary.md)** for GPU-programming vocabulary.

## Lessons

- **[The CUDA Execution Model: Threads, Warps, and Occupancy](cuda-execution-model.md)** — how the GPU hides memory latency by keeping many 32-thread warps resident; why SIMT divergence within a warp serializes; and why occupancy is *slack for hiding latency*, not a speed dial.
- **[Memory Access: Coalescing, Shared Memory, and Bank Conflicts](memory-access.md)** — why a warp's 32 lanes should touch contiguous addresses (uncoalesced access moves up to 32× the bytes), what shared memory buys through reuse, and the bank-conflict gotcha with its one-column padding fix.
- **[Triton: Writing Your First GPU Kernels](triton-basics.md)** — write vector add → fused softmax → simple matmul in Python-on-blocks: `program_id`, offsets, masks, on-chip reductions, `tl.dot`, and autotuning, with the compiler handling warps and coalescing.
- **[Reading vLLM's PagedAttention Kernel](paged-attention-kernel.md)** — the virtual-memory idea (KV blocks + block tables), how the kernel gathers KV block-by-block and folds it in with online softmax, and a pure-Python proof that paged == dense attention.

!!! note "Part 3 complete"
    All four lessons are written, each with a two-way-linked interview question — the GPU-programming mental models (execution model, memory access) and the hands-on half (Triton kernels, reading the PagedAttention kernel). Hand-written/optimized CUDA C++ stays out of scope (ADR-0002); the PagedAttention *serving* deep-dive (continuous batching, block manager) is covered in **Part 5**. All vLLM flags/APIs are verified via Context7 (ADR-0004); the baseline is **vLLM 0.26.0**. See the [Interview Bank](../interview/index.md) for the linked question set.
