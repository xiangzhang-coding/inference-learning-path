# From Static to Continuous Batching: the First Lever on Throughput

!!! info "Baseline: **vLLM 0.26.0** · model `Qwen2.5-7B-Instruct` · single RTX 4090 (24 GB)"
    The vLLM knobs named here — `max_num_seqs` (default **128**), `max_num_batched_tokens` (default **2048**, auto-tuned by the engine), and the fact that **continuous batching is always on** (there is no flag to enable it; it *is* the V1 scheduler) — are verified against vLLM 0.26.0 via Context7 (ADR-0004). The Python simulation in §4 is a **scheduling model, not a benchmark** (pure-Python, offline, no GPU). Speedup figures are **illustrative / order-of-magnitude references**; measure on your own AutoDL box.

---

## 1 · Intuition & why it matters

You have one GPU and a stream of requests arriving over time, each generating a different number of tokens. **Throughput** is how many requests (or tokens) you finish per second. The single biggest lever on that number — bigger than quantization, bigger than any kernel — is *how you pack requests onto the GPU over time*. That's batching, and getting it right is worth **an order of magnitude** (illustrative).

Here's the trap. The obvious way to batch is: collect N requests, run them together until **all** finish, then take the next N. That's **static batching**, and it wastes most of your GPU. Requests in a batch finish at wildly different lengths — one emits 20 tokens, its neighbor emits 500 — but the whole batch is locked together until the *longest* one is done. Every sequence that finished early leaves its GPU slot **idle** for the rest of the batch. On real traffic (output lengths vary 10–50×), that's most of your compute sitting dark.

**Continuous batching** (from the Orca paper, and the default in every serious engine including vLLM) fixes this by changing the *unit of scheduling*: instead of scheduling a whole request and running it to completion, you schedule **one decode iteration at a time**. After every single forward step, the engine evicts sequences that just finished, frees their [KV cache](../part0/kv-cache.md), and immediately admits waiting requests into the freed slots. The batch is never locked — it's a living set that gains and loses members every step. No slot sits idle while work is waiting. → see the [Glossary](../glossary.md) for *Static / Dynamic / Continuous batching*.

## 2 · Mental model

Picture time flowing left→right and the GPU's batch slots stacked vertically. Each `█` is a slot doing useful work in that step; each `·` is an **idle** slot (wasted GPU).

```text
STATIC BATCHING (batch of 4, run until ALL finish, then next batch)
        step→  1 2 3 4 5 6 7 8 9 …
  slot0  R0    █ █ █ ·  ·  ·  ·  ·          R0 finished at step3, slot idle till step8
  slot1  R1    █ █ █ █ █ █ █ █              R1 is the long one — holds the batch hostage
  slot2  R2    █ █ ·  ·  ·  ·  ·  ·          R2 finished at step2
  slot3  R3    █ █ █ █ █ ·  ·  ·             R3 finished at step5
               └── the batch can't refill until step8 (R1 done) ──┘
  utilization ≈ shaded / total  →  lots of "·"  (bubbles)

CONTINUOUS BATCHING (iteration-level: evict done, admit waiting, every step)
        step→  1 2 3 4 5 6 7 8 9 …
  slot0  R0→R4 █ █ █ █ █ █ █ █              R0 done@3 → R4 admitted@4, keeps going
  slot1  R1    █ █ █ █ █ █ █ █              R1 (long) runs, but no longer blocks others
  slot2  R2→R5 █ █ █ █ █ █ █ █              R2 done@2 → R5 admitted@3
  slot3  R3→R6 █ █ █ █ █ █ █ █              R3 done@5 → R6 admitted@6
               └── freed slots backfilled immediately, no waiting ──┘
  utilization ≈ shaded / total  →  almost no "·"
```

*(The `R`-labels and step numbers above are a schematic sketch to show the mechanism — not the specific requests fed to the §4 simulation.)*

Three shapes to hold:

- **Static batching schedules at the *request* granularity; continuous batching schedules at the *iteration* granularity.** That one change — decide who's in the batch *every step* instead of *every batch* — is the whole idea. Everything else follows.
- **The bubbles in static batching are the whole problem.** A finished-early sequence can't yield its slot until the batch drains, so short requests pay for the longest one's tail. This also causes **head-of-line blocking**: a request waiting in the queue can't start until a *whole batch* frees up, even if slots are idle *right now*.
- **Continuous batching turns "idle until the batch drains" into "idle for zero steps."** The freed slot is backfilled on the very next iteration. The batch composition changes constantly; there's no "batch boundary" to wait for.

## 3 · Principle — iteration-level scheduling

### 3.1 The engine loop

A continuous-batching engine runs one loop. Each turn of the loop is **one forward pass** over the current *running* set of sequences:

```text
loop forever:
    # 1. ADMIT: pull waiting requests into the running set while there's room
    while waiting and can_fit_next(waiting[0]):     # room = KV-cache blocks + slot budget
        running.add(waiting.popleft())              # (new request: its prefill runs this step)

    # 2. STEP: one forward pass — every running sequence advances by one token (decode),
    #          newly admitted ones do their prefill
    outputs = model.step(running)

    # 3. EVICT: any sequence that emitted EOS or hit max_tokens is finished
    for seq in running:
        if seq.done():
            free_kv_cache(seq)                      # release its KV blocks back to the pool
            running.remove(seq); emit(seq)
```

The three phases — **admit → step → evict** — repeat every iteration. Contrast with static batching, whose loop is `admit N → step until all done → evict all` — the "step until all done" is exactly the part that creates bubbles.

### 3.2 What limits admission — it's memory, not compute

You'd think the batch size is limited by compute. It usually isn't. Decode is **[memory-bound](../part0/inference-flow.md)** (from Part 0): each step reads the model's weights from HBM once and reuses them across *every* sequence in the batch. So adding another sequence to a decode batch is **nearly free on compute** — the weight read is already paid — right up until you hit one of two walls:

- **KV-cache capacity.** Every admitted sequence needs [KV cache](../part0/kv-cache.md) storage that grows every token. The engine can only admit a request if there are free KV blocks to hold its context. This is *the* binding constraint in practice — and it's exactly why [PagedAttention](paged-attention.md) matters: by killing fragmentation, it lets far more sequences fit, so the batch grows larger before hitting the wall.
- **The compute ridge.** Pile on enough sequences and the batched GEMMs eventually saturate the tensor cores — you cross from memory-bound into compute-bound (the [roofline](../part2/roofline-analysis.md) ridge). Past that, more sequences add latency without adding throughput.

Two vLLM knobs cap the batch directly:

- **`max_num_seqs`** (default **128**) — the maximum number of sequences in the running set. The batch-*width* ceiling.
- **`max_num_batched_tokens`** (default **2048**, but the engine auto-tunes it) — the maximum tokens processed in one step, summed across all sequences. This bounds prefill+decode work per iteration (and is the dial that chunked prefill turns — a [Part 5 scheduler](index.md) topic).

### 3.3 Why this is *the* throughput lever

Because decode is memory-bound, a batch of 1 wastes almost all the GPU's compute: you read all the weights to produce a single token. A batch of 32 reads the same weights once and produces 32 tokens — ~32× the useful work for the *same* memory traffic. Continuous batching keeps the batch **as full as KV capacity allows, at every step**, so you're always near that amortization sweet spot instead of draining down to a near-empty batch at the tail of each static round. That's why it's the first thing every inference engine does, and the first thing an interviewer will probe.

## 4 · Complete runnable code + line-by-line

A pure-Python simulation of both schedulers over the *same* set of requests, reporting GPU-slot utilization and makespan. It proves the mechanism without a GPU — the only difference between the two functions is *when a freed slot gets refilled*.

```python title="batching_sim.py"
"""Static vs continuous batching — a scheduling model, not a benchmark.
Pure Python, offline. Each request needs a fixed number of decode steps;
we count how many GPU 'slots' do useful work each step."""
from collections import deque

# (request_id, num_steps_to_finish) — output lengths vary a lot, like real traffic.
REQUESTS = [("R0", 2), ("R1", 12), ("R2", 3), ("R3", 2), ("R4", 10), ("R5", 2),
            ("R6", 4), ("R7", 2), ("R8", 8), ("R9", 3), ("R10", 2), ("R11", 6)]
SLOTS = 4                                             # batch width (like max_num_seqs)

def static_batching(requests, slots):
    """Fill all slots, run until EVERY sequence in the batch finishes, then refill."""
    q = deque(requests)
    busy_steps = total_slot_steps = step = 0
    while q:
        batch = [q.popleft() for _ in range(min(slots, len(q)))]   # take a full batch
        remaining = {rid: n for rid, n in batch}
        while any(r > 0 for r in remaining.values()):              # step until ALL done
            step += 1
            for rid in remaining:
                if remaining[rid] > 0:
                    remaining[rid] -= 1
                    busy_steps += 1                                # this slot did useful work
            total_slot_steps += slots                             # all `slots` were reserved
    return step, busy_steps, total_slot_steps

def continuous_batching(requests, slots):
    """Iteration-level: after every step, evict finished seqs and admit waiting ones."""
    waiting = deque(requests)
    running = {}                                                  # rid -> steps remaining
    busy_steps = total_slot_steps = step = 0
    while waiting or running:
        while waiting and len(running) < slots:                   # ADMIT into free slots
            rid, n = waiting.popleft(); running[rid] = n
        step += 1                                                 # STEP: one forward pass
        for rid in running:
            running[rid] -= 1; busy_steps += 1
        running = {rid: n for rid, n in running.items() if n > 0} # EVICT finished
        total_slot_steps += slots
    return step, busy_steps, total_slot_steps

if __name__ == "__main__":
    for name, fn in [("static", static_batching), ("continuous", continuous_batching)]:
        steps, busy, total = fn(REQUESTS, SLOTS)
        util = busy / total
        print(f"{name:>11}: makespan={steps:2d} steps | slot-utilization={util:5.1%} "
              f"({busy}/{total} slot-steps useful)")
```

**Line-by-line:**

- `REQUESTS` — twelve requests with **very different output lengths** (2–12 steps). This spread is what exposes the difference; if every request were the same length, static and continuous would tie.
- `static_batching` — takes a full batch, then the inner `while any(...)` runs **until every member finishes**. `total_slot_steps += slots` on each step charges for *all* slots being reserved, but `busy_steps` only counts slots still generating — the gap is the idle bubbles. The batch can't refill until the inner loop exits (the longest sequence is done).
- `continuous_batching` — same request stream, but the loop **admits** into any free slot *before* each step and **evicts** finished sequences *after* each step. A slot freed at step *k* is refilled at step *k+1*. The dict comprehension is the evict phase; the `while waiting and len(running) < slots` is the admit phase.
- The two functions differ **only** in refill timing. Same requests, same slot count, same per-step work — the scheduling discipline is the entire delta.

Expected output (a scheduling model, not a benchmark):

```text
     static: makespan=30 steps | slot-utilization=46.7% (56/120 slot-steps useful)
 continuous: makespan=18 steps | slot-utilization=77.8% (56/72 slot-steps useful)
```

Same 56 units of useful work. Static spreads it over 30 steps at 47% utilization; continuous packs it into 18 steps at 78% — finishing the same requests in ~1.7× less time. The idle bubbles — slots reserved but not generating — are pure waste that continuous batching reclaims by refilling immediately. (Continuous doesn't hit 100% because near the *tail* there are too few waiting requests left to backfill every freed slot — a real effect. On steady high-load traffic the running set stays full and utilization climbs further.)

## 5 · Lab — confirm vLLM does this for you (and can't be turned off)

!!! gpu "GPU Lab (optional verification)"
    - **Min VRAM:** none to read; ~16 GB to run `Qwen2.5-7B-Instruct` (INT4/AWQ) and watch the batch grow/shrink
    - **Suggested AutoDL card:** RTX 4090 (24 GB)
    - **Est. time / cost:** reading ~15 min (free, no-card mode) · optional run ~10 min · ~¥1 (illustrative)
    - **Platform:** NVIDIA CUDA (default)
    - **Non-NVIDIA:** continuous batching is a *scheduler* property, backend-independent — AMD ROCm, TPU, and CPU builds of vLLM all schedule the same way; only the per-step kernels differ.

The key thing to internalize: **you never enable continuous batching — it's the scheduler.** Sending many concurrent requests is all it takes.

```python title="serve_and_load.py"
# Offline batch: hand vLLM many prompts at once; the engine schedules them continuously.
# API verified against vLLM 0.26.0 (LLM, SamplingParams). Run in AutoDL with a GPU.
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="awq",          # fit the 7B on a 24 GB 4090 (see Part 4)
    max_num_seqs=128,            # batch-width ceiling (default 128) — the running-set cap
    # max_num_batched_tokens auto-tuned by the engine; leave it unless you're shaping TTFT
    gpu_memory_utilization=0.92, # default 0.92 — fraction of VRAM the engine may use for KV blocks
)
# Prompts with deliberately different output lengths — the batch will lose/gain members every step.
prompts = ["Say hi.", "Write a 300-word essay on the sea.", "2+2?", "Explain PagedAttention."]
params  = [SamplingParams(max_tokens=n) for n in (8, 300, 4, 200)]

outputs = llm.generate(prompts, params)   # the engine admits/evicts across iterations for you
for o in outputs:
    print(repr(o.outputs[0].text[:40]))
```

**What to observe:** the short prompts (`"2+2?"`) finish and free their KV blocks long before the 300-word essay; with continuous batching those freed slots are reused within the same `generate` call, not held until the essay finishes. To *see* it, serve with `vllm serve Qwen/Qwen2.5-7B-Instruct --quantization awq` and watch the log line reporting **"Running: N reqs, Waiting: M reqs"** — N rises and falls step-by-step as requests flow through. There is no `--enable-continuous-batching` flag because there's nothing to enable.

## 6 · Common pitfalls / counter-intuitive points

- **Thinking you have to turn it on.** Continuous batching *is* vLLM's scheduler; there's no flag. The mistake is looking for one and, not finding it, assuming vLLM does static batching. It doesn't.
- **Confusing `max_num_seqs` with a fixed batch size.** It's a *ceiling* on the running set, not a target you fill and drain. The actual batch floats below it, bounded by KV capacity.
- **Assuming a bigger batch is always faster.** Only until you hit KV-cache capacity or the [compute ridge](../part2/roofline-analysis.md). Past the ridge, more sequences add latency without throughput; before it (the common case), decode is memory-bound and extra sequences are nearly free.
- **Reproducing static batching by accident.** A naive loop that calls `model.generate()` on a fixed list and *waits for all of it* before sending the next list is static batching in your own code — you've thrown away the engine's continuous scheduling. Stream requests in; don't barrier on whole batches.
- **Padding waste (the HuggingFace `generate` trap).** Static batching typically pads all sequences to the longest length and computes the pad tokens — double waste (idle slots *and* wasted compute on padding). Continuous batching over a paged KV cache has no padding: each sequence occupies exactly the blocks it needs.
- **Blaming latency on batching when it's admission.** If TTFT spikes under load, the cause is usually requests *waiting* for KV room (admission), not the batching discipline. The fix is capacity ([quantization](../part4/index.md), [KV-cache quant](../part4/quantization-methods.md), [PagedAttention](paged-attention.md)) or a smaller `max_num_batched_tokens` to prioritize new prefills — not abandoning continuous batching.

## 7 · Interview links

- [Static vs continuous batching: the throughput lever](../interview/continuous-batching.md) — the high-frequency question this lesson prepares you for: *why static batching wastes the GPU, what "iteration-level scheduling" means, and what actually limits the batch size.*

## 8 · Summary & further reading

**One line:** Static batching runs a fixed batch until its longest member finishes, so short requests leave idle bubbles and queued requests suffer head-of-line blocking; continuous batching (Orca's iteration-level scheduling) instead decides the batch membership *every forward step* — evicting finished sequences and admitting waiting ones — keeping the batch as full as KV-cache capacity allows, which is the single biggest lever on inference throughput because decode is memory-bound and extra sequences are nearly free.

Further reading:

- Yu et al. — *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22) — the paper that introduced iteration-level (continuous) batching.
- The [PagedAttention lesson](paged-attention.md) — why KV-cache capacity (not compute) usually limits admission, and how paging raises that ceiling.
- The [inference-flow lesson](../part0/inference-flow.md) — why decode is memory-bound, the premise that makes batching nearly free.
- Next in Part 5: the [scheduler](index.md) (chunked prefill, PD disaggregation) — how the engine shapes *which* tokens run each step to balance TTFT against throughput.

## 9 · Self-check

??? question "Why does static batching waste the GPU, and what specifically does continuous batching change to fix it?"
    Static batching locks a fixed set of sequences together and runs them until the **longest** one finishes. Because output lengths vary widely, sequences that finish early leave their GPU slots **idle** for the rest of the batch (bubbles), and queued requests can't start until the whole batch drains (head-of-line blocking). Continuous batching changes the **granularity of scheduling** from a whole request to a single decode **iteration**: after every forward step it evicts finished sequences (freeing their KV) and admits waiting ones into the freed slots. The batch composition changes every step, so a freed slot is refilled on the next iteration instead of sitting idle until the batch drains.

??? question "You add sequences to a decode batch and throughput keeps rising with almost no extra latency — then suddenly latency jumps. What are the two walls you might have hit, and which is more common?"
    (1) **KV-cache capacity** — every sequence needs KV storage that grows each token; when free KV blocks run out, no more sequences can be admitted and new requests wait (raising TTFT). (2) **The compute ridge** — enough batched work eventually saturates the tensor cores, crossing from memory-bound into compute-bound, after which more sequences add latency without throughput. In practice **KV-cache capacity is the more common wall**, which is exactly why PagedAttention (kill fragmentation → fit more sequences) and KV-cache quantization matter so much for throughput.

??? question "A teammate says 'we should enable continuous batching in vLLM to go faster.' What's wrong with the statement, and where would you actually look for a throughput problem?"
    There's nothing to enable — continuous batching **is** vLLM's scheduler; it's always on, with no flag. If throughput is low, the batch is probably starved for **KV-cache room** (sequences waiting for admission), so look at capacity: is the model quantized to free VRAM for KV blocks? Is `gpu_memory_utilization` (default 0.92) leaving headroom unused? Would FP8 KV cache or a larger effective `max_num_seqs` (default 128) fit more sequences? The lever is almost always **admission capacity**, not the batching discipline — and capacity is what [PagedAttention](paged-attention.md) and quantization buy you.
