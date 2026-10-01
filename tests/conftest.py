"""Shared plumbing: the same stateful httpx-mock MemorySync as the
LiveKit adapter's suite, for the Pipecat service.

``POST /v1/memory/add_turn`` follows the server contract: a turn is never
stored as a row. A USER turn stores one "fact" — the turn text with any
role prefix removed, the caller's scalar metadata plus ``write_origin:
"turn-extraction"`` and ``turn_role: "user"``; assistant/system/tool
turns store nothing (``skipped_non_user_turn``); filler stores nothing
(``skipped_low_value``); a replay of the same (speaker, occurred_at,
text) stores nothing and answers ``already_exists: true``. ``memory_id``
is always null.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

_HUMAN = {"human", "user", "customer", "caller"}
_PREFIX = re.compile(r"^\s*(human|user|ai|assistant)\s*:\s*", re.IGNORECASE)
_RESERVED = {"write_origin", "distilled_from", "turn_role", "history_id", "episodic"}
_FILLER = {"ok", "okay", "yes", "no", "sure", "thanks", "thank", "you", "hi", "hello", "hey", "bye"}


def _norm(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return "user" if value.strip().lower() in _HUMAN else "assistant"


def _is_filler(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) <= 1 or (len(words) == 2 and all(w in _FILLER for w in words))


class MockMemorySync:
    def __init__(self, tenant_id: str = "org_1") -> None:
        self.tenant_id = tenant_id
        #: Memory rows: seeded facts plus the facts extracted from user turns.
        self.rows: List[Dict[str, Any]] = []
        self.requests: List[httpx.Request] = []
        #: (user_id, speaker, occurred_at, text) of every user turn extracted.
        self.receipts: set = set()
        #: processing_status of every add_turn answer, in order.
        self.outcomes: List[str] = []
        self._next_id = 1
        self.fail_next: Optional[int] = None
        #: Fail only the next add_turn call (tenant/recall calls unaffected).
        self.fail_next_add_turn: Optional[int] = None
        self.recall_returns_empty = False
        self.quota_mode: Optional[str] = None
        self.delay_s = 0.0

    def seed(self, user_id: str, text: str) -> None:
        """A fact already in memory (as if extracted from an earlier call)."""
        self.rows.append(
            {"id": self._alloc(), "user_id": user_id, "text": text, "metadata": None}
        )

    def _alloc(self) -> int:
        mid = self._next_id
        self._next_id += 1
        return mid

    def recall_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path == "/v1/memory/recall")

    def add_turn_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path == "/v1/memory/add_turn")

    def add_turn_bodies(self) -> List[Dict[str, Any]]:
        return [
            json.loads(r.content.decode("utf-8"))
            for r in self.requests
            if r.url.path == "/v1/memory/add_turn"
        ]

    def facts(self) -> List[Dict[str, Any]]:
        """Rows extracted from add_turn (seeded rows excluded)."""
        return [
            r for r in self.rows
            if (r.get("metadata") or {}).get("write_origin") == "turn-extraction"
        ]

    _METERED_ADDS = {("POST", "/v1/memory/add_turn")}
    _METERED_READS = {("POST", "/v1/memory/recall"), ("POST", "/v1/memory/query")}

    def _quota(self, method: str, path: str) -> Optional[httpx.Response]:
        if self.quota_mode is None:
            return None
        is_add = (method, path) in self._METERED_ADDS
        is_read = (method, path) in self._METERED_READS
        if not (is_add or is_read):
            return None
        if self.quota_mode == "strict":
            return httpx.Response(
                429,
                json={
                    "detail": {
                        "error": "limit_exceeded",
                        "message": "You have reached your monthly limit. Upgrade your plan.",
                    }
                },
            )
        if is_add:
            self.outcomes.append("skipped")
            return httpx.Response(
                201,
                json={
                    "memory_id": None,
                    "status": "ok",
                    "processing_status": "skipped",
                    "embed_mode": "none",
                    "already_exists": False,
                    "request_id": None,
                },
            )
        return httpx.Response(200, json={"memories": []})

    def _add_turn(self, body: Dict[str, Any]) -> httpx.Response:
        def answer(processing_status: str, *, already: bool = False, request_id=None):
            self.outcomes.append(processing_status)
            return httpx.Response(
                201,
                json={
                    "memory_id": None,
                    "status": "processing" if request_id else "ok",
                    "processing_status": processing_status,
                    "embed_mode": "none",
                    "already_exists": already,
                    "request_id": request_id,
                },
            )

        text = str(body.get("text") or "")
        speaker = body.get("speaker")
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        match = _PREFIX.match(text)
        content = (text[match.end():] if match else text).strip()
        head = speaker.split("@", 1)[0] if isinstance(speaker, str) and "@" in speaker else None
        role = (
            _norm(body.get("role"))
            or _norm(match.group(1) if match else None)
            or _norm(head)
            or _norm(metadata.get("role"))
            or "user"
        )
        if role != "user":
            return answer("skipped_non_user_turn")
        if _is_filler(content):
            return answer("skipped_low_value")
        receipt = (body.get("user_id"), speaker, body.get("occurred_at"), text)
        if receipt in self.receipts:
            return answer("skipped_replay", already=True)
        self.receipts.add(receipt)
        fact_md = {
            k: v
            for k, v in metadata.items()
            if k not in _RESERVED and (v is None or isinstance(v, (str, int, float, bool)))
        }
        fact_md.update({"write_origin": "turn-extraction", "turn_role": "user"})
        self.rows.append(
            {
                "id": self._alloc(),
                "user_id": body.get("user_id"),
                "text": content,
                "source": body.get("source"),
                "metadata": fact_md,
            }
        )
        return answer("distilling", request_id=uuid.uuid4().hex)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method, path = request.method, request.url.path
        # The latency knob models a SLOW RECALL — the voice-critical read
        # path. Writes stay fast so flush-at-end assertions stay sharp.
        if self.delay_s and (method, path) in self._METERED_READS:
            await asyncio.sleep(self.delay_s)
        if self.fail_next is not None:
            status = self.fail_next
            self.fail_next = None
            return httpx.Response(status, json={"detail": "injected failure"})
        if self.fail_next_add_turn is not None and path == "/v1/memory/add_turn":
            status = self.fail_next_add_turn
            self.fail_next_add_turn = None
            return httpx.Response(status, json={"detail": "injected failure"})
        quota = self._quota(method, path)
        if quota is not None:
            return quota

        if method == "GET" and path == "/org/projects":
            return httpx.Response(200, json=[{"id": "proj_1", "tenant_id": self.tenant_id}])

        body = json.loads(request.content.decode("utf-8") or "{}") if request.content else {}

        if method == "POST" and path == "/v1/memory/add_turn":
            return self._add_turn(body)

        if method == "POST" and path == "/v1/memory/recall":
            if self.recall_returns_empty:
                return httpx.Response(200, json={"context": "", "memories": []})
            mine = [r for r in self.rows if r.get("user_id") == body.get("user_id")]
            context = "\n".join(f"- {r['text']}" for r in mine)
            return httpx.Response(200, json={"context": context, "memories": []})

        if method == "POST" and path == "/v1/memory/query":
            words = [w for w in str(body.get("prompt") or "").lower().split() if w]
            hits = [
                {"memory_id": f"m_{r['id']}", "raw_text": r["text"], "score": 0.9}
                for r in self.rows
                if r.get("user_id") == body.get("user_id")
                and any(w in str(r["text"]).lower() for w in words)
            ]
            return httpx.Response(200, json={"memories": hits[: int(body.get("k") or 8)]})

        if method == "GET" and "/list" in path:
            parts = path.split("/")
            user = parts[-2] if len(parts) >= 3 else ""
            mine = [r for r in self.rows if r.get("user_id") == user]
            return httpx.Response(
                200,
                json={"memories": [{"memory_id": f"m_{r['id']}", "raw_text": r["text"]} for r in mine]},
            )

        return httpx.Response(404, json={"detail": f"no mock for {method} {path}"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture()
def mock() -> MockMemorySync:
    return MockMemorySync()


@pytest.fixture()
def make_service(mock):
    from pipecat_memorysync import MemorySyncMemoryService

    def _make(**overrides):
        params = overrides.pop("params", None)
        kwargs = dict(
            api_key="ms_test_key_1",
            user_id="caller-1",
            session_id="conv-42",
            transport=mock.transport(),
        )
        kwargs.update(overrides)
        if params is not None:
            kwargs["params"] = params
        return MemorySyncMemoryService(**kwargs)

    return _make
