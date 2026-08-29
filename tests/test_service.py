"""Contract tests for pipecat-memorysync, against the REAL framework.

The service runs inside Pipecat's OWN test harness
(``pipecat.tests.utils.run_test``) — the same rig Daily uses for the
built-in services — with real ``LLMContext`` objects and real frame flow.
The contracts above all: recall never outlives its budget, capture is
delta-only, and the context frame ALWAYS reaches the LLM.
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


async def drain(mock, expected_rows: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(mock.rows) >= expected_rows:
            return
        await asyncio.sleep(0.05)


# ── enrichment ────────────────────────────────────────────────────────


async def test_enriches_context_as_system_message_with_guard(mock, make_service):
    mock.seed("caller-1", "human: I love teal dashboards")
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
    mock.seed("caller-1", "human: I love teal dashboards")
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


async def test_recall_falls_back_to_query_for_verbatim_turns(mock, make_service):
    mock.seed("caller-1", "human: the launch is on Friday")
    mock.recall_returns_empty = True
    service = make_service()
    ctx = context_of([{"role": "user", "content": "when is the launch happening?"}])
    await drive(service, ctx)
    injected = [m for m in ctx.get_messages() if m["role"] == "system"]
    assert injected and "Friday" in injected[0]["content"]


async def test_same_query_pays_recall_once(mock, make_service):
    mock.seed("caller-1", "human: I love teal dashboards")
    service = make_service()
    ctx1 = context_of([{"role": "user", "content": "what colours do I like?"}])
    ctx2 = context_of([{"role": "user", "content": "what colours do I like?"}])
    await drive(service, ctx1, ctx2)
    assert mock.recall_calls() == 1, "identical consecutive query served without a second recall"


async def test_content_parts_are_understood(mock, make_service):
    """Universal context content can be a list of parts, not just a str."""
    mock.seed("caller-1", "human: I love teal dashboards")
    service = make_service()
    ctx = context_of([
        {"role": "user", "content": [{"type": "text", "text": "what colours do I like?"}]},
    ])
    await drive(service, ctx)
    assert any(m["role"] == "system" for m in ctx.get_messages())


# ── capture: delta-only ───────────────────────────────────────────────


async def test_capture_is_delta_only_never_o_n_squared(mock, make_service):
    service = make_service()
    turn1 = context_of([{"role": "user", "content": "switch the dashboard to teal"}])
    await drive(service, turn1)
    await drain(mock, 1)

    turn2 = context_of([
        {"role": "user", "content": "switch the dashboard to teal"},
        {"role": "assistant", "content": "Done — teal it is."},
        {"role": "user", "content": "and make the font larger"},
    ])
    await drive(service, turn2)
    await drain(mock, 3)

    texts = sorted(r["text"] for r in mock.rows)
    assert texts == [
        "ai: Done — teal it is.",
        "human: and make the font larger",
        "human: switch the dashboard to teal",
    ]
    # THE delta guarantee: 3 stored messages = exactly 3 add_turn calls.
    # A full-context re-store (the competitor pattern) would have made 4+.
    assert mock.add_turn_calls() == 3


async def test_capture_seeds_and_scoping(mock, make_service):
    service = make_service()
    ctx = context_of([{"role": "user", "content": "remember the launch is Friday"}])
    await drive(service, ctx)
    await drain(mock, 1)

    row = mock.rows[0]
    assert row["source"] == "pipecat"
    assert row["metadata"]["session_id"] == "pipecat::conv-42"
    expected = fnv1a64("human:remember the launch is Friday")
    assert row["speaker"] == f"human@pipecat::conv-42#h{expected}"


async def test_injected_memories_never_reenter_storage(mock, make_service):
    mock.seed("caller-1", "human: I love teal dashboards")
    service = make_service()
    turn1 = context_of([{"role": "user", "content": "what colours do I like?"}])
    await drive(service, turn1)
    await drain(mock, 2)  # seed + the user turn

    # The next turn's context INCLUDES the injected system message.
    enriched_messages = turn1.get_messages()
    turn2_messages = enriched_messages + [
        {"role": "assistant", "content": "You like teal."},
        {"role": "user", "content": "great, anything else?"},
    ]
    await drive(service, context_of(turn2_messages))
    await asyncio.sleep(0.3)

    stored = [r["text"] for r in mock.rows]
    assert not any("via MemorySync" in t for t in stored), "injections never re-enter memory"


async def test_user_role_injection_is_also_excluded_from_capture(mock, make_service):
    mock.seed("caller-1", "human: I love teal dashboards")
    service = make_service(
        params=MemorySyncMemoryService.InputParams(add_as_system_message=False)
    )
    ctx = context_of([{"role": "user", "content": "what colours do I like?"}])
    await drive(service, ctx)
    await asyncio.sleep(0.4)
    stored = [r["text"] for r in mock.rows]
    assert not any("via MemorySync" in t for t in stored)


# ── quota + failure matrix ────────────────────────────────────────────


async def test_quota_modes_stay_silent_and_frames_flow(mock, make_service):
    for mode in ("silent", "strict"):
        mock.quota_mode = mode
        service = make_service()
        ctx = context_of([{"role": "user", "content": "what do you remember about me?"}])
        await drive(service, ctx)  # run_test asserts the frame reached downstream
        assert not any(m["role"] == "system" for m in ctx.get_messages()), mode
    mock.quota_mode = None


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
    mock.seed("caller-1", "human: I love teal dashboards")
    service = make_service()
    block = await service.get_context_block("what does this caller like?")
    assert "teal" in block
    memories = await service.get_memories()
    assert memories and "teal" in memories[0]["raw_text"]
    await service.aclose()


async def test_missing_user_id_is_loud_at_construction(mock):
    with pytest.raises(ValueError, match="user_id"):
        MemorySyncMemoryService(api_key="ms_x", user_id="", transport=mock.transport())
