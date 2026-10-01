# memorysync-pipecat

[MemorySync](https://memorysync.io) for [Pipecat](https://github.com/pipecat-ai/pipecat) —
long-term memory for voice pipelines that never stalls a reply and never
re-sends a turn it already sent. PyPI package: **`pipecat-memorysync`**.

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
(as a system message) and its **new** user turns are sent to fact extraction —
then pushed on, enriched or not, on time.

## Design guarantees

- **Budgeted recall.** Enrichment runs under a hard timeout (default
  **1.2 s**). A slow or dead memory backend means an unenriched frame, never a
  stalled voice reply.
- **Immediate, loss-proof capture.** Every new user message is sent the
  moment its frame passes — the current utterance is in flight BEFORE the
  LLM replies, so a disconnect can never lose it. Filler filtering and fact
  extraction happen server-side: only the durable facts in what the caller
  said are stored as memories, not the turn text.
- **Delta-only capture.** Only user messages *not seen before* are sent,
  tracked by deterministic idempotency seeds; a retried turn is recognised
  server-side and not extracted twice. Growing a 50-message context does not
  re-send 50 messages per turn.
- **Injection exclusion.** The memory block this service adds is never sent
  back as a new turn.
- **Graceful end, salvaged abort.** On `EndFrame`, queued sends get a bounded
  window (3 s) to land before the pipeline stops — the caller's last words are
  not lost. On `CancelFrame`, the frame is pushed first and sends get a brief
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
        top_k=6,                  # memories injected per turn
        recall_timeout=1.2,       # hard budget, seconds
        add_as_system_message=True,
        position=1,               # index where the memory block is inserted
        min_prompt_chars=8,       # skip enrichment for shorter user prompts
    ),
)
```

| Param | Default | Meaning |
| --- | --- | --- |
| `top_k` | `6` | Memories injected per turn |
| `recall_timeout` | `1.2` | Hard recall budget in seconds |
| `system_prompt` | (guarded header) | Prefix line for the injected block; also the capture-exclusion marker |
| `add_as_system_message` | `True` | Inject as `system` (else as a `user` message) |
| `position` | `1` | Index in the message list where the block is inserted (`0` = first; the default lands right after your system prompt) |
| `min_prompt_chars` | `8` | Skip recall for trivial prompts |

## Semantics worth knowing

- Only the caller's (**user**) turns are sent, as plain text with
  `role: "user"`. The server extracts the durable facts they contain and
  stores only those, tagged with `session_id: "pipecat::<session>"`.
  Assistant replies are not sent — they are not stored as memories — and
  filler such as "ok" or "thanks" stores nothing.
- Idempotency seeds let the server recognise retries and reconnects: the same
  turn is not extracted twice.
- Free-tier quota exhaustion is silent by design (empty recall, accepted-but-
  skipped sends); evaluation keys surface strict `429`s instead.
- The service is reusable across pipeline runs; call `await memory.aclose()`
  from application shutdown if you want an explicit flush + client close.

## Development

```bash
python -m venv venv && venv/Scripts/pip install -e . pipecat-ai pytest pytest-asyncio
venv/Scripts/python -m pytest tests -q    # 19 tests, run via pipecat's official test harness
```

## Docs

Full guide: [docs.memorysync.io/guides/pipecat](https://docs.memorysync.io/guides/pipecat)

## License

MIT
