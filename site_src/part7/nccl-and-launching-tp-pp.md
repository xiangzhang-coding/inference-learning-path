# NCCL Collective Communication & Launching TP/PP in vLLM

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` / `Llama-3.1-70B-Instruct` · A100 for the multi-GPU Lab"
    Verified against vLLM 0.26.0 via Context7 (ADR-0004): vLLM does GPU collectives over **NCCL** (via `vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`) and CPU collectives over **GLOO**; the distributed executor backend is **`mp`** (native multiprocessing, the single-node default) or **`ray`** (the multi-node default runtime) via `--distributed-executor-backend`; multi-node can also run over `mp` with `--nnodes` / `--node-rank` / `--master-addr` (+ `--headless` on workers); the GPU-comm **sanity test** uses `torch.distributed.init_process_group(backend="nccl")` + `all_reduce`, launched with `torchrun`; debug with **`NCCL_DEBUG=TRACE`**, and fix init hangs with **`NCCL_SOCKET_IFNAME`** / **`GLOO_SOCKET_IFNAME`** / **`VLLM_HOST_IP`** / **`NCCL_P2P_DISABLE`**. This is the hands-on companion to [Why Parallelize, and How](parallelism-strategies.md) — it opens the black box under TP's "all-reduce every layer" and actually launches TP/PP. All numbers are **illustrative / order-of-magnitude references**.

---

## 1 · Intuition & why it matters

The [previous lesson](parallelism-strategies.md) kept saying TP does "an all-reduce every layer, every token" and PP does "a point-to-point handoff." Those are **collective communication** operations, and the library that runs them on NVIDIA GPUs is **NCCL** (NVIDIA Collective Communications Library, "nickel"). This lesson opens that black box and then actually launches multi-GPU inference.

Two things an interviewer expects you to have touched, not just read about:

1. **The collective primitives.** When four GPUs each compute a partial result and need the sum on all of them, that's an **all-reduce** — not a `for` loop of sends. NCCL implements a small vocabulary — **all-reduce**, **all-gather**, **reduce-scatter**, plus point-to-point send/recv — as **topology-aware, bandwidth-optimal** algorithms (the ring all-reduce is the famous one). Knowing what each primitive moves, and that all-reduce = reduce-scatter + all-gather, is the difference between "TP is slow because... network?" and a real answer.
2. **How you actually turn it on.** TP/PP aren't magic flags that always work — they launch a **process group**, each rank binds a GPU, and NCCL discovers the interconnect. On one node that's `--tensor-parallel-size N` with the `mp` backend; across nodes it's `ray` + matching `--master-addr` on every node, and the #1 real-world failure is a **hang at init** because NCCL picked the wrong network interface. The engineer who can run the sanity test and read `NCCL_DEBUG=TRACE` is the one who ships multi-GPU.

So: the primitives (what moves on the wire), then the launch (how to make it run and how to debug it when it hangs). → see the [Glossary](../glossary.md) for *Collective communication, TP degree*.

## 2 · Mental model

Two layers: the **collective primitives** (what data moves) and the **launch topology** (what processes exist and how they find each other).

```text
THE COLLECTIVES — 4 GPUs, each starts with data; what ends up where:

ALL-REDUCE  (TP's workhorse: sum partials, everyone gets the total)
   in:  g0=[a]  g1=[b]  g2=[c]  g3=[d]
   out: every GPU = [a+b+c+d]                       ← this is the "all-reduce every layer"

ALL-GATHER  (each has a shard → everyone gets the full concatenation)
   in:  g0=[a]  g1=[b]  g2=[c]  g3=[d]
   out: every GPU = [a,b,c,d]

REDUCE-SCATTER  (sum across GPUs, but each keeps only its slice of the sum)
   in:  g0=[a0,a1,a2,a3]  g1=[b0..]  g2=[c0..]  g3=[d0..]
   out: g0=[Σ_0]  g1=[Σ_1]  g2=[Σ_2]  g3=[Σ_3]      (Σ_k = a_k+b_k+c_k+d_k)

RING ALL-REDUCE = REDUCE-SCATTER  then  ALL-GATHER
   cost per GPU ≈ 2·(N-1)/N · message   → ~2·message as N grows, INDEPENDENT of N (bandwidth-optimal)
   the reason a good all-reduce doesn't get N× slower with more GPUs

THE LAUNCH TOPOLOGY — processes, ranks, and how they rendezvous:

  SINGLE NODE (backend = mp):           MULTI-NODE (ray default, or mp):
    1 process/GPU, spawned locally        node0 (rank 0..3) ── network ── node1 (rank 4..7)
    NCCL over NVLink/PCIe                  --master-addr points every node at node0 (mp path)
    vllm serve --tensor-parallel-size 4    TP=4 within each node · PP=2 across the 2 nodes
                                           NCCL over NVLink in-node, over IB/Ethernet cross-node
```

Three shapes to keep:

- **A collective is one fused operation, not a loop of sends.** "Every GPU needs the sum" is *one* all-reduce that NCCL schedules as a ring/tree across the actual links — not N² point-to-point messages. That's why it's fast and why you never hand-roll it.
- **all-reduce = reduce-scatter + all-gather**, and its cost is ~$2\times$ the message *regardless of N*. More GPUs don't make a well-implemented all-reduce proportionally slower — but the message still fires every layer, every token, which is why it must ride the fast link.
- **Launch = a process group + a rendezvous.** Every rank must agree on `--master-addr` and find a working network interface, or NCCL **hangs at init** before a single token is generated. Single-node `mp` "just works"; multi-node `ray` is where the network bites.

## 3 · Principle

### 3.1 The collective primitives

A **collective** is an operation every rank in a group participates in together. The four that matter for inference:

- **all-reduce** — combine (usually sum) a tensor that every GPU holds, so **all** GPUs end with the reduced result. This is what TP's column-then-row split needs: each GPU computed a partial sum of the layer output, and all-reduce turns the partials into the full activation that every GPU carries forward. *One after attention, one after the FFN — two per layer.*
- **all-gather** — every GPU holds a **shard**; afterward every GPU holds the **full concatenation**. Used to reconstruct a tensor that was split across GPUs (e.g. sequence-parallel activations, gathering per-GPU outputs).
- **reduce-scatter** — the dual of all-gather: every GPU holds the full tensor, and afterward each GPU keeps only **its slice of the element-wise sum**. 
- **point-to-point** send/recv — one GPU to one GPU. This is PP's stage-to-stage handoff.

The key identity: **a ring all-reduce is a reduce-scatter followed by an all-gather.** Each phase moves $\frac{N-1}{N}$ of the message per GPU, so the total per-GPU traffic is

$$
\text{ring all-reduce cost} \approx 2\cdot\frac{N-1}{N}\cdot M
$$

where $M$ is the message size in bytes. As $N$ grows this approaches $2M$ — **independent of the GPU count**. That's the punchline: a good all-reduce is *bandwidth-optimal*, so adding GPUs doesn't make each all-reduce proportionally slower. The pain in TP isn't that all-reduce scales badly with $N$; it's that the all-reduce fires **every layer, every token**, so it must sit on a **high-bandwidth link**.

### 3.2 NCCL — what runs the collectives

**NCCL** is NVIDIA's library implementing these collectives with algorithms (ring, tree) tuned to the actual hardware topology — NVLink between GPUs in a node, PCIe as fallback, InfiniBand/Ethernet across nodes. vLLM drives it through a thin wrapper, **`PyNcclCommunicator`** (`vllm.distributed.device_communicators.pynccl`), and uses **GLOO** for the CPU-side coordination (rendezvous, metadata). You rarely call NCCL directly — but you *do* debug it, because when GPUs can't talk, NCCL is where it surfaces.

### 3.3 Launching TP/PP in vLLM

vLLM picks a **distributed executor backend**: **`mp`** (native Python multiprocessing) is the default for a **single node**; for **multi-node**, **`ray`** is the default runtime, but vLLM also supports multi-node over `mp`. Select it with `--distributed-executor-backend`.

**Single node** — one process per GPU, NCCL over NVLink/PCIe:

```bash
# 4 GPUs in one node, tensor-parallel
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4
```

**Multi-node over `mp`** — launch on *every* node with matching `--nnodes` / `--master-addr`; each node carries its own `--node-rank`, and workers add `--headless`:

```bash
# 2 nodes × 4 GPUs = 8 GPUs, TP=4 within a node, PP=2 across nodes
# on the head node (node 0):
vllm serve /models/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --nnodes 2 --node-rank 0 --master-addr $HEAD_NODE_IP
# on the worker node (node 1):
vllm serve /models/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --nnodes 2 --node-rank 1 --master-addr $HEAD_NODE_IP --headless
```

The **Ray** alternative forms the cluster first (`ray start` on each node), then launches vLLM **once** on the head with `--distributed-executor-backend ray` and the same TP/PP sizes (no per-node `--node-rank`). Either way, this is the [decision tree](parallelism-strategies.md) from the previous lesson made concrete: **TP = GPUs per node, PP = number of nodes.**

### 3.4 Choosing the TP degree

Three constraints, in order:

1. **It must divide the (KV-)head count** — TP splits attention heads, so the degree is (almost always) a **power of 2** that evenly divides the heads. Qwen2.5-7B has 28 attention / 4 KV heads; TP ∈ {1, 2, 4} keeps KV heads whole.
2. **Stay ≤ GPUs per node** — TP's per-layer all-reduce (§3.1) must ride **NVLink**, so TP shouldn't cross the node boundary; go to **PP** for that.
3. **Only as much as you need** — more TP means more all-reduce traffic per token and diminishing returns; use the smallest TP that makes the model fit (or hits your latency target), then add PP for capacity across nodes and DP for throughput.

### 3.5 Debugging the launch (the part that actually bites)

Multi-GPU rarely fails by crashing — it **hangs at initialization**, because NCCL/GLOO can't establish the process group. The verified toolkit:

- **`NCCL_DEBUG=TRACE`** — turn on verbose NCCL logging to see where init stalls.
- **`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`** — pin the **network interface** (e.g. `eth0`). On InfiniBand clusters especially, torch's group discovery can pick the wrong interface and hang; forcing Ethernet for the initial rendezvous fixes it.
- **`VLLM_HOST_IP`** — override the IP vLLM detects when the network is complex/multi-homed.
- **`NCCL_P2P_DISABLE=1`** — a *temporary* workaround if peer-to-peer (NVLink/PCIe P2P) has a hardware/driver fault; the real fix is the driver/topology.
- **The sanity test (§4)** — run it *before* blaming vLLM. If the bare PyTorch NCCL all-reduce hangs or crashes, the problem is hardware/driver/network, not the engine.

## 4 · Complete runnable code + line-by-line

The canonical **GPU-communication sanity test** — vLLM's own troubleshooting script, trimmed to the PyTorch-NCCL core. It does exactly one collective (an all-reduce of all-ones) and checks the math: after summing across `world_size` GPUs, every element must equal `world_size`. If this passes, your NCCL/driver/interconnect is healthy and any multi-GPU vLLM failure is config, not hardware.

```python title="nccl_sanity.py"
"""Minimal NCCL all-reduce sanity test (vLLM's troubleshooting script, trimmed).
Run on N GPUs:  torchrun --nproc-per-node=2 nccl_sanity.py
A correctness check, not a benchmark."""
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")            # form the process group over NCCL
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)                  # bind THIS rank to its own GPU

data = torch.FloatTensor([1.0] * 128).to("cuda")   # each rank starts with a vector of ones
dist.all_reduce(data, op=dist.ReduceOp.SUM)        # THE collective: sum across all ranks
torch.cuda.synchronize()                           # wait for the GPU to finish

value = data.mean().item()                         # every element should now be world_size
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"   # correctness gate
print(f"[rank {dist.get_rank()}] PyTorch NCCL all-reduce OK (value={value}, world={world_size})")

dist.destroy_process_group()
```

**Line-by-line:**

- `init_process_group(backend="nccl")` — every process launched by `torchrun` joins one **process group**; `nccl` selects the GPU collective backend. This is the step that **hangs** if the network interface is wrong (§3.5) — the rendezvous never completes.
- `local_rank = get_rank() % device_count()` then `set_device(local_rank)` — each rank **binds a distinct GPU**. Rank 0 → GPU 0, rank 1 → GPU 1, … Skip this and two ranks fight over one GPU.
- `data = [1.0]*128 .to("cuda")` — every rank puts the *same* all-ones vector on its GPU, so the expected sum is trivially `world_size`.
- `all_reduce(data, op=SUM)` — the one collective under test: NCCL sums `data` element-wise across all ranks (a ring reduce-scatter + all-gather, §3.1) and leaves the result on **every** GPU.
- `assert value == world_size` — with `world_size` ranks each contributing `1.0`, every element must be `world_size`. A wrong value or a hang here is a hardware/driver/network fault, *not* a vLLM bug — that's the whole diagnostic value.
- Launch with `torchrun --nproc-per-node=2 nccl_sanity.py`; **without** `torchrun`, `world_size==1`, the all-reduce is a no-op, and you've tested nothing.

Expected output on 2 GPUs (illustrative):

```text
[rank 0] PyTorch NCCL all-reduce OK (value=2.0, world=2)
[rank 1] PyTorch NCCL all-reduce OK (value=2.0, world=2)
```

Two ranks, each contributing `1.0`, sum to `2.0` on both GPUs — NCCL and the interconnect are healthy. (vLLM's full script continues with a GLOO/CPU test and a `PyNcclCommunicator` + CUDA-graph test — the same idea, exercising vLLM's own NCCL wrapper and the CUDA-graph replay path that decode uses.)

## 5 · Lab — run the collective, then launch TP on 2 GPUs

!!! gpu "GPU Lab (MULTI-GPU — rent, run, power off)"
    - **Min VRAM / GPUs:** **2 GPUs** required (the sanity test is a no-op on one). A 2× A100 (40/80 GB) node runs both the NCCL test and a real TP=2 serve of `Qwen2.5-7B-Instruct` comfortably.
    - **Suggested AutoDL card:** **2× A100** on a **"power-on-then-off" basis** (ADR-0001) — this is one of the rare topics that truly needs multiple GPUs. Do *not* leave it running. A 4090 has no NVLink peer, so it can't do this Lab.
    - **Est. time / cost:** ~30–40 min hands-on · **~¥8–20** for a short 2×A100 session (illustrative; A100 rental is far pricier than the single-4090 default — spin up, run, tear down).
    - **Platform:** NVIDIA CUDA + **NCCL** (default). **Non-NVIDIA:** AMD ROCm uses **RCCL** (a NCCL API-compatible reimplementation) with the same collectives; the launch flags are identical, the env vars are the ROCm equivalents.

Work up from the wire to the engine:

1. **Run the collective.** `torchrun --nproc-per-node=2 nccl_sanity.py` on the 2-GPU node — expect `value=2.0` on both ranks. This proves NCCL + interconnect *before* you touch vLLM.
2. **Watch it hang on purpose.** Re-run with a bogus interface: `NCCL_SOCKET_IFNAME=doesnotexist torchrun --nproc-per-node=2 nccl_sanity.py`. It stalls at `init_process_group` — the signature of the #1 multi-node failure. Kill it, unset the var, confirm it passes again.
3. **Launch real TP=2.** `vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2`. In the startup logs, confirm it spawns **2 workers** (the `mp` backend), initializes NCCL, and reports a larger KV-cache block pool than TP=1 (weights now split across 2 GPUs frees VRAM for KV). Send a request and confirm identical output to the single-GPU run — TP changes *where* compute happens, not *what* it computes.
4. **Read the traffic.** Re-launch with `NCCL_DEBUG=TRACE vllm serve … --tensor-parallel-size 2` and watch NCCL announce its ring/algorithm and channels — the all-reduce from §3.1, live. Then **power off the instance.**

## 6 · Common pitfalls / counter-intuitive points

- **Blaming vLLM for a hang that's the network.** If startup hangs at NCCL init, run the §4 sanity test first. If *that* hangs, it's hardware/driver/network — set `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to the right interface (e.g. `eth0`), or `VLLM_HOST_IP` in multi-homed networks. vLLM can't fix a bad rendezvous.
- **Running the sanity test without `torchrun`.** A bare `python nccl_sanity.py` has `world_size==1`, so the all-reduce is a no-op and the assert trivially passes — you've verified *nothing*. You must launch it with `torchrun --nproc-per-node=N`.
- **Forgetting `--headless` on worker nodes (or mismatching `--master-addr`).** Every node in a multi-node serve must point at the *same* head IP; the non-head nodes run `--headless`. A mismatch hangs the whole cluster at init.
- **Setting TP across the node boundary.** TP's all-reduce fires every layer, every token and must ride NVLink; a TP group that spans nodes crawls on the slower inter-node link. Keep **TP ≤ GPUs/node** and use **PP** across nodes.
- **Expecting TP=N to give N× throughput.** TP mainly buys **capacity** (fit the model) and some **latency**; it adds all-reduce traffic every token, so throughput scales sublinearly. Raw throughput comes from **DP replicas**, not more TP.
- **Assuming all-reduce gets N× slower with more GPUs.** A ring all-reduce is ~$2M$ per GPU *regardless of N* (§3.1) — bandwidth-optimal. The cost driver is the **per-token, per-layer frequency** and the **link bandwidth**, not the GPU count.
- **Leaving `NCCL_P2P_DISABLE=1` on as a "fix."** It's a diagnostic workaround that can tank performance; the real fix is the driver/topology. Don't ship with it.
- **Leaving the A100 instance running.** Per ADR-0001 this Lab is power-on-then-off. Multi-GPU rental burns budget fast — tear it down when the collective and TP serve are confirmed.

## 7 · Interview links

- [NCCL collectives, ring all-reduce & launching TP/PP](../interview/nccl-collective-communication.md) — the high-frequency question this lesson prepares you for: *what all-reduce / all-gather / reduce-scatter each move, why ring all-reduce is ~2× the message independent of N, which collective TP uses and how often, and how vLLM launches TP/PP single- vs multi-node (mp vs ray, the flags, and debugging an init hang).*

## 8 · Summary & further reading

**One line:** The "all-reduce every layer" from the [previous lesson](parallelism-strategies.md) is a **collective communication** primitive run by **NCCL** — the vocabulary is **all-reduce** (sum, everyone gets it — TP's workhorse, two per layer), **all-gather** (shards → full copy everywhere), **reduce-scatter** (sum, each keeps its slice), and point-to-point (PP's handoff); a **ring all-reduce = reduce-scatter + all-gather** and costs ~$2\cdot\frac{N-1}{N}\cdot M \to 2M$ per GPU **independent of N** (bandwidth-optimal, which is why the pain is the per-token frequency, not the GPU count) — and you launch it in vLLM with `--tensor-parallel-size` / `--pipeline-parallel-size` over the **`mp`** backend (single node, or multi-node via `--nnodes` / `--node-rank` / `--master-addr` + `--headless`) or a **`ray`** cluster across nodes, verifying the wire with a `torch.distributed` NCCL `all_reduce` test under `torchrun` and debugging init hangs with `NCCL_DEBUG=TRACE` + `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`VLLM_HOST_IP`.

Further reading:

- Baidu's *Bringing HPC Techniques to Deep Learning* (2017) — the ring all-reduce and why its cost is independent of GPU count.
- NVIDIA **NCCL** docs — the collective algorithms (ring/tree), and the `NCCL_*` environment variables quoted in §3.5/§6.
- vLLM `docs/usage/troubleshooting.md` — the exact GPU/CPU communication sanity script (§4) and the debugging env vars.
- vLLM `docs/serving/parallelism_scaling.md` and `docs/serving/expert_parallel_deployment.md` — the `mp`/`ray` backends, multi-node flags, and `GLOO_SOCKET_IFNAME` network setup quoted here.
- The [previous lesson](parallelism-strategies.md) — why you parallelize (the collectives here are *how* TP/PP pay for it).
- The [capacity-planning](../part8/capacity-planning.md) lesson (Part 8) — how TP/PP choices feed the VRAM and fleet-sizing math.

## 9 · Self-check

??? question "What does an all-reduce do, and why doesn't a good (ring) all-reduce get proportionally slower as you add GPUs?"
    An **all-reduce** takes a tensor that every GPU holds, combines it element-wise across all ranks (usually a sum), and leaves the **reduced result on every GPU** — it's exactly what TP needs to turn each GPU's partial layer output into the full activation (one all-reduce after attention, one after the FFN). A **ring all-reduce** is implemented as a **reduce-scatter** (each GPU ends with one summed slice) followed by an **all-gather** (broadcast the slices), each phase moving $\frac{N-1}{N}$ of the message per GPU, for a total of $\approx 2\cdot\frac{N-1}{N}\cdot M \to 2M$ as $N$ grows — **independent of the GPU count**. So it's *bandwidth-optimal*: more GPUs don't make each all-reduce proportionally slower. What hurts TP isn't $N$; it's that the all-reduce fires **every layer, every token**, so it must ride a high-bandwidth link (NVLink) — hence TP stays within a node.

??? question "You launch a 2-node × 4-GPU vLLM serve and it hangs at startup with no error. Walk through how you diagnose it."
    A silent hang at startup is almost always a **process-group / NCCL rendezvous** failure, not a model problem. (1) Run the **sanity test** (§4) with `torchrun` across the nodes — if the bare PyTorch NCCL `all_reduce` hangs, it's hardware/driver/**network**, not vLLM. (2) Turn on **`NCCL_DEBUG=TRACE`** to see where init stalls. (3) The usual culprit is the wrong **network interface**: set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to a real Ethernet interface (e.g. `eth0`) — critical on InfiniBand clusters where discovery picks the wrong one; set `VLLM_HOST_IP` if the detected IP is wrong. (4) Check every node points at the **same `--master-addr`** and that the workers run **`--headless`** (backend `ray` for multi-node). Only after the collective passes do you suspect config like a TP degree that doesn't divide the heads.

??? question "Your model fits on one A100 but latency is too high, so you serve it with `--tensor-parallel-size 4` and expect ~4× throughput. Why is that expectation wrong, and what would you check first?"
    TP mainly buys **capacity** (fit a model that's too big) and some **latency** reduction (more GPUs crunch each token) — it does **not** give linear throughput, because it adds an **all-reduce every layer, every token** (§3.1), pure communication overhead that grows with TP degree and yields diminishing returns. Raw throughput comes from **DP replicas**, not more TP. First checks: confirm **TP ≤ GPUs per node** so the all-reduce rides **NVLink** (across PCIe/nodes it crawls); confirm the degree **divides the head count** (power of 2); and if the goal is QPS rather than per-request latency, use **data-parallel replicas** instead. Measure with `NCCL_DEBUG` and a latency/throughput sweep rather than assuming.
