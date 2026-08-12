# Part 3 · GPU Programming (Triton)

> Enough GPU programming to *reason about* why kernels are fast and to *read* vLLM's kernels — not a CUDA C++ course (see ADR-0002 for the depth boundary).

## What this part covers

- The **[CUDA execution model](cuda-execution-model.md)** — grid/block/thread, warps, SIMT divergence, and occupancy — via mental models rather than rote detail
- **[Memory access](memory-access.md)** — coalescing, shared memory, and bank conflicts: how a warp *should* touch memory
- Writing a few simple **Triton** kernels — the "I can write a little" confidence *(next ticket)*
- A guided read of vLLM's **PagedAttention** kernel to build source-reading skill *(next ticket)*

See the **[Glossary](../glossary.md)** for GPU-programming vocabulary.

## Lessons

- **[The CUDA Execution Model: Threads, Warps, and Occupancy](cuda-execution-model.md)** — how the GPU hides memory latency by keeping many 32-thread warps resident; why SIMT divergence within a warp serializes; and why occupancy is *slack for hiding latency*, not a speed dial.
- **[Memory Access: Coalescing, Shared Memory, and Bank Conflicts](memory-access.md)** — why a warp's 32 lanes should touch contiguous addresses (uncoalesced access moves up to 32× the bytes), what shared memory buys through reuse, and the bank-conflict gotcha with its one-column padding fix.

!!! note "Scaffolding status"
    The mental-model half of Part 3 is in (ticket #8): the [CUDA execution model](cuda-execution-model.md) and [memory access](memory-access.md), each with a two-way-linked interview question. Hands-on **Triton** and the **PagedAttention** kernel read land next. Hand-written/optimized CUDA C++ is intentionally out of scope (ADR-0002). See the [Interview Bank](../interview/index.md) for the linked question set.
