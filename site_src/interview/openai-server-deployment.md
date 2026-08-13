# Serving over HTTP: the OpenAI-compatible server & its endpoints

!!! info "Baseline: **vLLM 0.26.0** · APIs verified via Context7 (ADR-0004)"

**Module:** Part 8 · Production & System Design   ·   **Tests the lesson:** [Serving vLLM over HTTP: the OpenAI-Compatible Server](../part8/openai-server.md)

---

## Q: How do you serve a vLLM model over HTTP? Walk through `vllm serve`, its main endpoints, how `/v1/chat/completions` differs from `/v1/completions`, what `/health` does and doesn't guarantee, how auth works, and which flags shape the interface vs the capacity.

### Direct answer

**Launch:** `vllm serve <model>` boots the engine and starts a **FastAPI/uvicorn** frontend speaking the **OpenAI API** — so any OpenAI client (the `openai` SDK, LangChain, a chat UI) retargets by changing one line, `base_url="http://host:8000/v1"`.

**Endpoints:**

- **`/v1/chat/completions`** — role messages; the server **applies the model's chat template**. The main endpoint for instruct models.
- **`/v1/completions`** — **raw** text-in/text-out; **no** template.
- **`/v1/models`** — the served id (`--served-model-name`) + any loaded **LoRA adapters**.
- **`/health`** — **liveness**: 200 if the engine is alive, 503 if it died. *Not* readiness, *not* a load signal.
- **`/metrics`** — Prometheus feed (`vllm:num_requests_running` / `num_requests_waiting`, KV usage, latency histograms).
- Utility: `/ping` (SageMaker), `/version`, `/load`, `/tokenize`, `/detokenize`.

**Auth:** `--api-key KEY` (or `VLLM_API_KEY`); pass it **multiple times** for key rotation. With no key set, the server is **open** (any non-empty key passes).

**Interface knobs** (`--host`/`--port`/`--uds`, `--api-key`, `--served-model-name`) shape *how clients talk to you*; **capacity knobs** (`--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--gpu-memory-utilization`) shape *how much you can serve* — the ceiling you go [measure](load-testing-knee.md).

### Deep dive

- **Frontend vs engine core.** The HTTP layer does auth, JSON validation, chat-template rendering, and SSE streaming; the [engine core](../part5/vllm-architecture-map.md) (scheduler + workers) does batching and GPU work. Latency problems live in the engine queue, not FastAPI.
- **Chat template.** `/v1/chat/completions` applies the exact special-token formatting the instruct model was tuned on. Sending raw text to `/v1/completions` for an instruct model skips it → out-of-distribution prompt → quiet quality loss.
- **Streaming.** `"stream": true` switches to **Server-Sent Events**: the first `data:` chunk lands at ~**TTFT**, the rest paced by **TPOT**. This is why streaming makes TTFT user-visible.
- **`--served-model-name`.** Decouples the public id from the checkpoint path; clients must send that id or get *model not found*.
- **Liveness ≠ readiness.** `/health` says the process is alive, not that weights are loaded/warm or that there's spare capacity — route readiness and autoscaling off `/metrics`.

### Code

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-demo-key")  # the one line
# chat: server applies the chat template to role messages
r = client.chat.completions.create(model="qwen2.5-7b",                       # = --served-model-name
    messages=[{"role": "user", "content": "hi"}], stream=True)               # SSE: first chunk ≈ TTFT
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
# ops: curl -s localhost:8000/health          -> 200 (alive) / 503 (dead)
#      curl -s localhost:8000/v1/models -H "Authorization: Bearer sk-demo-key"
```

### Interviewer follow-ups

- *"Client sends the HF path as `model` and gets *not found* — why?"* → `--served-model-name` set the public id to something else; the `model` field must equal the served name.
- *"Is `/health` == ready for traffic?"* → No — liveness only (200 alive / 503 dead). Readiness/load come from `/metrics` (`num_requests_waiting`).
- *"`/v1/chat/completions` vs `/v1/completions`?"* → chat applies the template (role messages); completions is raw text. Wrong one for an instruct model quietly hurts quality.
- *"How do you rotate an API key with no downtime?"* → pass `--api-key` multiple times so old and new keys are both valid during the switch.
- *"No `--api-key` set — is the server secure?"* → No, it's open; any non-empty key passes. Bind `127.0.0.1` or set a key + firewall before exposing `0.0.0.0`.
- *"Streaming looks batched, not token-by-token — why?"* → an intermediate proxy/LB is buffering the SSE response; disable buffering on that route.

### Linked concepts

- Lesson: [Serving vLLM over HTTP: the OpenAI-Compatible Server](../part8/openai-server.md)
- Related: [Trace a request through vLLM's architecture](vllm-architecture.md) (the engine core behind the frontend), [Load-testing & the concurrency knee](load-testing-knee.md) (measuring the capacity the server exposes), [Multi-LoRA serving](multi-lora-serving.md) (the adapters `/v1/models` lists)
- Glossary: [SLO, Goodput](../glossary.md)
