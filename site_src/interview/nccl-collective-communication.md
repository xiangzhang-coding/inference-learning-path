# NCCL collectives, ring all-reduce & launching TP/PP

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 7 · Multi-GPU & Distributed   ·   **Tests the lesson:** [NCCL Collective Communication & Launching TP/PP in vLLM](../part7/nccl-and-launching-tp-pp.md)

---

## Q: What do all-reduce / all-gather / reduce-scatter each do, why is ring all-reduce ~2× the message independent of N, which collective does TP use and how often, and how does vLLM launch TP/PP single- vs multi-node (including debugging a hang)?

### Direct answer

**The collectives** (a *collective* = one op every rank runs together, not a loop of sends):

- **all-reduce** — combine (sum) a tensor every GPU holds; **all** GPUs end with the result. **TP's workhorse** — one after attention, one after the FFN = **2 per layer, every token**.
- **all-gather** — each GPU holds a shard → every GPU holds the full concatenation.
- **reduce-scatter** — dual of all-gather: sum across GPUs, each keeps only **its slice** of the sum.
- **point-to-point** send/recv — PP's stage-to-stage activation handoff.

**Ring all-reduce = reduce-scatter + all-gather**, each phase moving $\frac{N-1}{N}$ of the message per GPU → total $\approx 2\cdot\frac{N-1}{N}\cdot M \to 2M$, **independent of N** (bandwidth-optimal). So the cost driver isn't the GPU count — it's that TP's all-reduce fires **every layer, every token**, which is why it must ride **NVLink** and TP stays **within a node**.

**NCCL** runs these on NVIDIA GPUs (vLLM via `PyNcclCommunicator`); **GLOO** does CPU-side coordination.

**Launching in vLLM:**
- Single node → backend **`mp`**: `vllm serve M --tensor-parallel-size 4`.
- Multi-node → **`ray`** (default) or **`mp`**: the `mp` path launches on **every** node with `--nnodes` / `--node-rank` / `--master-addr` (+ `--headless` on workers); the `ray` path forms the cluster (`ray start`) then launches **once** on the head with `--distributed-executor-backend ray`. Rule **TP = GPUs/node, PP = nodes**.

**Debugging a hang** (multi-GPU fails by hanging at init, not crashing): run the `torch.distributed` NCCL `all_reduce` sanity test under `torchrun`; `NCCL_DEBUG=TRACE`; pin the interface with `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`; `VLLM_HOST_IP` for multi-homed nets; `NCCL_P2P_DISABLE=1` only as a temporary workaround.

### Deep dive

- **Why all-reduce for TP.** Megatron's column-then-row split leaves each GPU with a *partial sum* of the layer output; all-reduce turns partials into the full activation everyone carries forward. Two per transformer block.
- **Bandwidth-optimal.** Ring pipelines the data in $N$ chunks so every link is busy; per-GPU traffic → $2M$ regardless of $N$. Naive "gather to rank 0 and broadcast" would be $O(N)$ and bottleneck one link.
- **mp vs ray.** `mp` spawns one process per local GPU (single node). `ray` coordinates processes across nodes. Choose via `--distributed-executor-backend`.
- **TP degree.** Power of 2 dividing the (KV-)head count; ≤ GPUs/node (NVLink); smallest that fits/hits latency — more TP = more per-token all-reduce, sublinear return.
- **The hang.** `init_process_group` blocks until all ranks rendezvous; a wrong NIC (common on InfiniBand) stalls it silently — hence the interface env vars.

### Code

The minimal NCCL sanity test — one all-reduce of ones must sum to `world_size`:

```python
import torch, torch.distributed as dist
dist.init_process_group(backend="nccl")             # hangs here if the NIC is wrong
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
data = torch.ones(128, device="cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)          # sum across ranks; result on every GPU
assert data.mean().item() == dist.get_world_size()   # N ranks × 1.0 → N; else HW/driver/net fault
# launch: torchrun --nproc-per-node=2 nccl_sanity.py   (bare python → world_size=1 → tests nothing)
```

### Interviewer follow-ups

- *"Does all-reduce get N× slower with more GPUs?"* → No — ring all-reduce is ~$2M$ per GPU, independent of N. The cost is per-token/per-layer frequency × link bandwidth.
- *"Which collective does TP use, how often?"* → all-reduce, twice per layer (after attention, after FFN), on every token including decode.
- *"How is all-reduce built from simpler ops?"* → reduce-scatter then all-gather.
- *"Single-node vs multi-node backend?"* → `mp` (single) vs `ray` (multi); `--distributed-executor-backend`.
- *"Multi-node serve hangs at startup — first move?"* → Run the sanity test under `torchrun`; `NCCL_DEBUG=TRACE`; set `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`; check every node shares `--master-addr` and workers are `--headless`.
- *"Why keep TP within a node?"* → Its per-token all-reduce needs NVLink bandwidth; across nodes it crawls — use PP.
- *"Bare `python` on the sanity test passes — proof of health?"* → No; `world_size==1` makes all-reduce a no-op. Must use `torchrun`.

### Linked concepts

- Lesson: [NCCL Collective Communication & Launching TP/PP](../part7/nccl-and-launching-tp-pp.md)
- Related: [Parallelism: TP/PP/DP/EP & when to use each](parallelism-strategies.md) (the strategies these collectives implement), [CUDA graphs & kernel fusion](cuda-graphs-fusion.md) (vLLM's NCCL + CUDA-graph path), [VRAM budget & max concurrency](vram-capacity-planning.md) (TP frees VRAM for KV)
- Glossary: [Collective communication, TP degree](../glossary.md)
