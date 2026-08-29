# Changelog

All notable changes to `pipecat-memorysync` are documented here.

## 1.1.0 — 2026-08-29

- **Turn-complete capture.** Voice aggregators split one utterance across
  several context messages at speech pauses; those fragments previously
  stored as separate rows ("and", "dinner", …). Consecutive new user
  fragments now merge into ONE verbatim turn, stored when the assistant
  reply completes the turn; an in-progress utterance is flushed as one
  merged turn at end of call. Assistant turns still store immediately,
  and a failed store releases its fragments for retry on the next frame.
- Delta-only and idempotency guarantees unchanged; suite grows to 17
  checks on Pipecat v1.8.1 (fragment merging, end-of-call tail flush,
  failed-store retry).

## 1.0.1 — 2026-08-29

- Metadata only: the source repository moved to
  `https://github.com/memorysyncio/memorysync-pipecat`; project URLs updated.
  No code changes.

## 1.0.0 — 2026-08-29

Initial release.

- `MemorySyncMemoryService`, a `FrameProcessor` for the position between the
  user context aggregator and the LLM service.
- Budgeted memory recall (default 1.2 s hard timeout) injected into every
  `LLMContextFrame` as a guarded system message — a slow or unreachable
  backend degrades to an unenriched frame, never a stalled reply.
- Delta-only capture with deterministic idempotency seeds: only turns not
  seen before are persisted, with role fidelity for user and assistant.
- Injection exclusion: the injected memory block is never re-captured.
- Graceful `EndFrame` flush (bounded 3 s window) and `CancelFrame` salvage.
- Tested with Pipecat v1.8.1 through `pipecat.tests.utils.run_test`
  (14-test suite).
