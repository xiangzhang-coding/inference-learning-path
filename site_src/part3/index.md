# Part 3 · GPU Programming (Triton)

> Enough GPU programming to *reason about* why kernels are fast and to *read* vLLM's kernels — not a CUDA C++ course (see ADR-0002 for the depth boundary).

## What this part covers

- The **CUDA execution model** and memory access, via mental models rather than rote detail
- Writing a few simple **Triton** kernels — the "I can write a little" confidence
- A guided read of vLLM's **PagedAttention** kernel to build source-reading skill

See the **[Glossary](../glossary.md)** for GPU-programming vocabulary.

!!! note "Scaffolding status"
    This part's lessons land in a later ticket. Hand-written/optimized CUDA C++ is intentionally out of scope (ADR-0002).
