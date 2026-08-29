# memorysync-pipecat

[MemorySync](https://memorysync.io) for [Pipecat](https://github.com/pipecat-ai/pipecat) —
long-term memory for voice pipelines that never stalls a reply and never
re-stores what it already knows. PyPI package: **`pipecat-memorysync`**.

Built and maintained by the [MemorySync](https://memorysync.io) team —
MemorySync is our product, and this integration is actively maintained
alongside it.

**Tested with Pipecat v1.8.1** (`pipecat-ai>=1.0.0,<2`).

```bash
pip install pipecat-memorysync
# or
uv add pipecat-memorysync
```

## Where it sits

`MemorySyncMemoryService` is a `FrameProcessor`. Place it **between your
context aggregator and your LLM service**:

```
transport.input() → stt → context_aggregator.user()
    → MemorySyncMemoryService        ← enriches + captures here
    → llm → tts → transport.output() → context_aggregator.assistant()
```

```python
from pipecat_memorysync import MemorySyncMemoryService

memory = MemorySyncMemoryService(
    api_key="ms_...",                 # or MEMORYSYNC_API_KEY env var
    user_id="caller-42",              # stable end-user id
    session_id="call-123",            # optional: scope to this call
)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    memory,
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

Every `LLMContextFrame` that flows through is enriched with relevant memories
(as a system message) and mined for **new** turns to persist — then pushed on,
enriched or not, on time.

## Design guarantees

- **Budgeted recall.** Enrichment runs under a hard timeout (default
  **1.2 s**). A slow or dead memory backend means an unenriched frame, never a
  stalled voice reply.
- **Delta-only capture.** Only messages *not seen before* are stored, tracked
  by deterministic idempotency seeds. Growing a 50-message context does not
  re-store 50 messages per turn.
- **Injection exclusion.** The memory block this service adds is never captured
  back as a new memory.
- **Graceful end, salvaged abort.** On `EndFrame`, queued writes get a bounded
  window (3 s) to land before the pipeline stops — the call's final exchange is
  not lost. On `CancelFrame`, the frame is pushed first and writes get a brief
  salvage window.
- **Failure-proof.** HTTP errors, quota limits, and timeouts all degrade to
  "no memories this turn". Nothing propagates into the pipeline.

## Running the example

A single-file voice agent that remembers callers across calls lives in
[`examples/foundational.py`](examples/foundational.py):

```bash
uv add pipecat-memorysync "pipecat-ai[deepgram,cartesia,openai,silero,runner,webrtc]"

export MEMORYSYNC_API_KEY=ms_...   # https://app.memorysync.io
export DEEPGRAM_API_KEY=...
export CARTESIA_API_KEY=...
export OPENAI_API_KEY=...

python examples/foundational.py
```

Open `http://localhost:7860/client`, tell the bot your name and a preference,
hang up, and connect again — it remembers.

## Configuration (`InputParams`)

```python
from pipecat_memorysync import MemorySyncMemoryService

memory = MemorySyncMemoryService(
    api_key="ms_...",
    user_id="caller-42",
    params=MemorySyncMemoryService.InputParams(
        top_k=5,                  # memories injected per turn
        recall_timeout=1.2,       # hard budget, seconds
        add_as_system_message=True,
        position="end",           # where the memory block lands in the context
        min_prompt_chars=8,       # skip enrichment for shorter user prompts
    ),
)
```

| Param | Default | Meaning |
| --- | --- | --- |
| `top_k` | `5` | Memories injected per turn |
| `recall_timeout` | `1.2` | Hard recall budget in seconds |
| `system_prompt` | (guarded header) | Prefix line for the injected block; also the capture-exclusion marker |
| `add_as_system_message` | `True` | Inject as `system` (else appended to the latest user message) |
| `position` | `"end"` | `"start"` or `"end"` of the message list |
| `min_prompt_chars` | `8` | Skip recall for trivial prompts |

## Semantics worth knowing

- Both **user and assistant** turns are persisted, with role fidelity.
- Idempotency seeds make retries/reconnects duplicate-free server-side.
- Free-tier quota exhaustion is silent by design (empty recall, accepted-but-
  dropped writes); evaluation keys surface strict `429`s instead.
- The service is reusable across pipeline runs; call `await memory.aclose()`
  from application shutdown if you want an explicit flush + client close.

## Development

```bash
python -m venv venv && venv/Scripts/pip install -e . pipecat-ai pytest pytest-asyncio
venv/Scripts/python -m pytest tests -q    # 14 tests, run via pipecat's official test harness
```

## Docs

Full guide: [docs.memorysync.io/guides/pipecat](https://docs.memorysync.io/guides/pipecat)

## License

MIT
