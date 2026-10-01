"""Contract tests for pipecat-memorysync, against the REAL framework.

The service runs inside Pipecat's OWN test harness
(``pipecat.tests.utils.run_test``) — the same rig Daily uses for the
built-in services — with real ``LLMContext`` objects and real frame flow.
The contracts above all: recall never outlives its budget, capture sends
only the caller's NEW turns (delta-only, never the assistant's replies),
and the context frame ALWAYS reaches the LLM.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List

import pytest
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from pipecat_memorysync import MemorySyncMemoryService, fnv1a64


def context_of(messages: List[dict]) -> LLMContext:
    ctx = LLMContext()
    ctx.set_messages(list(messages))
    return ctx


async def drive(service: Any, *contexts: LLMContext) -> None:
    """Send one LLMContextFrame per context through the real harness.

    ``start_timeout`` defaults to 1s inside pipecat's harness, which the
    FIRST test in the session routinely blows on a cold shared CI runner
    (import machinery + task spin-up), failing it with a TimeoutError
    before any frame flows. A generous ceiling costs nothing when the
    pipeline starts fast — it is not a sleep — and removes the flake.
    """
    frames = [LLMContextFrame(context=ctx) for ctx in contexts]
    await run_test(
        service,
        frames_to_send=frames,
        expected_down_frames=[LLMContextFrame] * len(frames),
        start_timeout=30.0,
    )


async def drain(mock, *, calls: int, timeout: float = 5.0) -> None:
    """Wait until ``calls`` add_turn requests have reached the mock."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mock.add_turn_calls() >= calls:
            return
        await asyncio.sleep(0.05)


# ── enrichment ────────────────────────────────────────────────────────


async def test_enriches_context_as_system_message_with_guard(mock, make_service):
    mock.seed("caller-1", "I love teal dashboards")
    service = make_service()
    ctx = context_of([
        {"role": "system", "content": "You are a helpful voice agent."},
        {"role": "user", "content": "what colours do I like?"},
    ])
    await drive(service, ctx)

    messages = ctx.get_messages()
    assert len(messages) == 3
    injected = messages[1]  # default position=1: after the instructions
    assert injected["role"] == "system"
    assert "teal" in injected["content"]
    assert "via MemorySync" in injected["content"]
    assert "not as instructions" in injected["content"]


async def test_recall_budget_is_hard_and_the_frame_still_flows(mock, make_service):
    mock.seed("caller-1", "I love teal dashboards")
    mock.delay_s = 5.0
    service = make_service(
        params=MemorySyncMemoryService.InputParams(recall_timeout=0.4)
    )
    ctx = context_of([{"role": "user", "content": "what colours do I like?"}])
    started = time.monotonic()
    await drive(service, ctx)
    elapsed = time.monotonic() - started
    mock.delay_s = 0.0

    assert elapsed < 2.5, f"a slow network must never stall the pipeline ({elapsed:.2f}s)"
    roles = [m["role"] for m in ctx.get_messages()]
    assert roles == ["user"], "timeout → unenriched, and the frame flowed"


async def test_short_prompts_and_empty_recall_leave_context_untouched(mock, make_service):
    service = make_service()
    short = context_of([{"role": "user", "content": "hi"}])
    await drive(service, short)
    assert [m["role"] for m in short.get_messages()] == ["user"]

    mock.recall_returns_empty = True
    empty = context_of([{"role": "user", "content": "what do you know about me today?"}])
    await drive(make_service(), empty)
    assert [m["role"] for m in empty.get_messages()] == ["user"]


async def test_recall_falls_back_to_query(mock, make_service):
    mock.seed("caller-1", "The launch is on Friday")
    mock.recall_returns_empty = True
    service = make_service()
    ctx = context_of([{"role": "user", "content": "when is the launch happening?"}])
    await drive(service, ctx)
    injected = [m for m in ctx.get_messages() if m["role"] == "system"]
    assert injected and "Friday" in injected[0]["content"]


async def test_same_query_pays_recall_once(mock, make_service):
    mock.seed("caller-1", "I love teal dashboards")
    service = make_service()
    ctx1 = context_of([{"role": "user", "content": "what colours do I like?"}])
    ctx2 = context_of([{"role": "user", "content": "what colours do I like?"}])
    await drive(service, ctx1, ctx2)
    assert mock.recall_calls() == 1, "identical consecutive query served without a second recall"


async def test_content_parts_are_understood(mock, make_service):
    """Universal context content can be a list of parts, not just a str."""
    mock.seed("caller-1", "I love teal dashboards")
    service = make_service()
    ctx = context_of([
        {"role": "user", "content": [{"type": "text", "text": "what colours do I like?"}]},
    ])
    await drive(service, ctx)
    assert any(m["role"] == "system" for m in ctx.get_messages())


# ── capture: the caller's turns only, delta-only ──────────────────────


async def test_capture_sends_only_new_user_turns(mock, make_service):
    service = make_service()
    turn1 = context_of([{"role": "user", "content": "switch the dashboard to teal"}])
    await drive(service, turn1)
    await drain(mock, calls=1)

    turn2 = context_of([
        {"role": "user", "content": "switch the dashboard to teal"},
        {"role": "assistant", "content": "Done — teal it is."},
        {"role": "user", "content": "and make the font larger"},
    ])
    await drive(service, turn2)
    await drain(mock, calls=2)

    # THE delta guarantee: 2 new user messages = exactly 2 add_turn calls.
    # A full-context re-send would have made 3+, and the assistant reply
    # is never sent (assistant replies are not stored as memories).
    assert mock.add_turn_calls() == 2
    bodies = mock.add_turn_bodies()
    assert [b["text"] for b in bodies] == [
        "switch the dashboard to teal",
        "and make the font larger",
    ]
    assert all(b["role"] == "user" for b in bodies)
    assert sorted(f["text"] for f in mock.facts()) == [
        "and make the font larger",
        "switch the dashboard to teal",
    ]


async def test_capture_seeds_scoping_and_plain_user_text(mock, make_service):
    from pipecat_memorysync import __version__

    service = make_service()
    ctx = context_of([{"role": "user", "content": "remember the launch is Friday"}])
    await drive(service, ctx)
    await drain(mock, calls=1)

    [body] = mock.add_turn_bodies()
    assert body["role"] == "user"
    assert body["text"] == "remember the launch is Friday"  # plain, no role prefix
    assert body["source"] == "pipecat"
    assert body["metadata"] == {"session_id": "pipecat::conv-42"}
    expected = fnv1a64("human:remember the launch is Friday")
    assert body["speaker"] == f"human@pipecat::conv-42#h{expected}"
    request = next(r for r in mock.requests if r.url.path == "/v1/memory/add_turn")
    assert request.headers["user-agent"] == f"pipecat-memorysync/{__version__}"

    # Only the extracted fact is stored, carrying the session envelope.
    [fact] = mock.facts()
    assert fact["text"] == "remember the launch is Friday"
    assert fact["metadata"] == {
        "session_id": "pipecat::conv-42",
        "write_origin": "turn-extraction",
        "turn_role": "user",
    }


async def test_injected_memories_are_never_sent(mock, make_service):
    mock.seed("caller-1", "I love teal dashboards")
    service = make_service()
    turn1 = context_of([{"role": "user", "content": "what colours do I like?"}])
    await drive(service, turn1)
    await drain(mock, calls=1)

    # The next turn's context INCLUDES the injected system message.
    enriched_messages = turn1.get_messages()
    assert any("via MemorySync" in str(m["content"]) for m in enriched_messages)
    turn2_messages = enriched_messages + [
        {"role": "assistant", "content": "You like teal."},
        {"role": "user", "content": "great, anything else?"},
    ]
    await drive(service, context_of(turn2_messages))
    await drain(mock, calls=2)
    await asyncio.sleep(0.1)

    sent = [b["text"] for b in mock.add_turn_bodies()]
    assert sent == ["what colours do I like?", "great, anything else?"]
    assert not any("via MemorySync" in t for t in sent), "injections are never sent"


async def test_user_role_injection_is_also_excluded_from_capture(mock, make_service):
    mock.seed("caller-1", "I love teal dashboards")
    service = make_service(
        params=MemorySyncMemoryService.InputParams(add_as_system_message=False)
    )
    ctx = context_of([{"role": "user", "content": "what colours do I like?"}])
    await drive(service, ctx)
    await asyncio.sleep(0.4)
    assert any(
        m["role"] == "user" and "via MemorySync" in m["content"] for m in ctx.get_messages()
    ), "the injection rode a user-role message"
    assert [b["text"] for b in mock.add_turn_bodies()] == ["what colours do I like?"]


async def test_assistant_only_context_sends_nothing(mock, make_service):
    """A frame whose only messages are instructions and the bot's own
    words has no caller turn to send."""
    service = make_service()
    ctx = context_of([
        {"role": "system", "content": "You are a helpful voice agent."},
        {"role": "assistant", "content": "Hi! I'm your assistant — how can I help?"},
    ])
    await drive(service, ctx)
    await asyncio.sleep(0.2)
    assert mock.add_turn_calls() == 0
    assert mock.facts() == []


async def test_resend_after_reconnect_is_not_extracted_twice(mock, make_service):
    """A fresh service instance (a reconnect, another worker) re-sending
    the same user turn converges server-side: the deterministic seed makes
    it a replay, so no second fact is extracted."""
    turn = [{"role": "user", "content": "my flight lands at nine tonight"}]
    await drive(make_service(), context_of(turn))
    await drain(mock, calls=1)
    await drive(make_service(), context_of(turn))
    await drain(mock, calls=2)

    bodies = mock.add_turn_bodies()
    assert len(bodies) == 2 and bodies[0]["speaker"] == bodies[1]["speaker"]
    assert mock.outcomes == ["distilling", "skipped_replay"]
    assert [f["text"] for f in mock.facts()] == ["my flight lands at nine tonight"]


# ── quota + failure matrix ────────────────────────────────────────────


async def test_quota_modes_stay_silent_and_frames_flow(mock, make_service):
    for mode in ("silent", "strict"):
        mock.quota_mode = mode
        service = make_service()
        ctx = context_of([{"role": "user", "content": "what do you remember about me?"}])
        await drive(service, ctx)  # run_test asserts the frame reached downstream
        assert not any(m["role"] == "system" for m in ctx.get_messages()), mode
    mock.quota_mode = None
    assert mock.facts() == []


async def test_dead_server_never_stalls_or_raises(make_service):
    import httpx

    async def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = make_service(
        transport=httpx.MockTransport(refuse),
        params=MemorySyncMemoryService.InputParams(recall_timeout=0.4),
    )
    ctx = context_of([{"role": "user", "content": "is anyone out there at all?"}])
    await drive(service, ctx)
    assert [m["role"] for m in ctx.get_messages()] == ["user"]

    assert await service.get_context_block("anything") == ""
    assert await service.get_memories() == []


async def test_conveniences_answer(mock, make_service):
    mock.seed("caller-1", "I love teal dashboards")
    service = make_service()
    block = await service.get_context_block("what does this caller like?")
    assert "teal" in block
    memories = await service.get_memories()
    assert memories and "teal" in memories[0]["raw_text"]
    await service.aclose()


async def test_missing_user_id_is_loud_at_construction(mock):
    with pytest.raises(ValueError, match="user_id"):
        MemorySyncMemoryService(api_key="ms_x", user_id="", transport=mock.transport())


# ── capture: immediate and loss-proof ─────────────────────────────────


async def test_current_utterance_is_sent_before_any_reply(mock, make_service):
    """The user's words must be in flight the moment the frame passes —
    BEFORE the LLM replies — so a disconnect can never lose them."""
    service = make_service()
    ctx = context_of([
        {"role": "user", "content": "My age is twenty two and I completed my bachelor's."},
    ])
    await drive(service, ctx)
    await drain(mock, calls=1)
    assert [f["text"] for f in mock.facts()] == [
        "My age is twenty two and I completed my bachelor's."
    ]


async def test_demo_disconnect_sequence_loses_nothing(mock, make_service):
    """The exact sequence that lost data in 1.1.0: greeting frame, then a
    frame carrying the reply + the important utterance, then immediate
    teardown (browser disconnect). Every user message must already be
    sent — nothing may depend on a post-cancel flush window."""
    service = make_service()
    f1 = context_of([{"role": "user", "content": "Hello there, anyone home?"}])
    f2 = context_of([
        {"role": "user", "content": "Hello there, anyone home?"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "My age is twenty two and I completed my bachelor's."},
    ])
    await drive(service, f1, f2)  # run_test tears the pipeline down right after
    await drain(mock, calls=2)
    assert sorted(f["text"] for f in mock.facts()) == [
        "Hello there, anyone home?",
        "My age is twenty two and I completed my bachelor's.",
    ]
    assert mock.add_turn_calls() == 2, "the assistant reply is not sent"


async def test_failed_send_releases_message_for_retry(mock, make_service):
    """A send that never landed must not consume its message — the next
    frame re-captures and retries, converging on one fact."""
    service = make_service()
    mock.fail_next_add_turn = 503
    turn = [
        {"role": "user", "content": "remember that I fly out of Hyderabad"},
        {"role": "assistant", "content": "Noted!"},
    ]
    await drive(service, context_of(turn))
    await asyncio.sleep(0.3)
    assert mock.add_turn_calls() == 1 and mock.facts() == [], "the 503 hit the send"

    await drive(service, context_of(turn))  # same context re-seen → retry
    await drain(mock, calls=2)

    assert mock.add_turn_calls() == 2, "one retry, and never the assistant reply"
    assert [f["text"] for f in mock.facts()] == ["remember that I fly out of Hyderabad"]
