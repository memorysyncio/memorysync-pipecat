"""Shared plumbing: the same stateful httpx-mock MemorySync as the
LiveKit adapter's suite, for the Pipecat service."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))


class MockMemorySync:
    def __init__(self, tenant_id: str = "org_1") -> None:
        self.tenant_id = tenant_id
        self.rows: List[Dict[str, Any]] = []
        self.requests: List[httpx.Request] = []
        self._next_id = 1
        self.fail_next: Optional[int] = None
        self.recall_returns_empty = False
        self.quota_mode: Optional[str] = None
        self.delay_s = 0.0

    def seed(self, user_id: str, text: str) -> None:
        self.rows.append(
            {
                "id": self._alloc(),
                "user_id": user_id,
                "text": text,
                "speaker": f"seed#{self._next_id}",
                "metadata": None,
                "_seed": f"seed-{self._next_id}",
            }
        )

    def _alloc(self) -> int:
        mid = self._next_id
        self._next_id += 1
        return mid

    def recall_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path == "/v1/memory/recall")

    def add_turn_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path == "/v1/memory/add_turn")

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
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"memories": []})

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
        quota = self._quota(method, path)
        if quota is not None:
            return quota

        if method == "GET" and path == "/org/projects":
            return httpx.Response(200, json=[{"id": "proj_1", "tenant_id": self.tenant_id}])

        body = json.loads(request.content.decode("utf-8") or "{}") if request.content else {}

        if method == "POST" and path == "/v1/memory/add_turn":
            seed = f"{body.get('speaker')}:{body.get('text')}"
            for row in self.rows:
                if row.get("_seed") == seed:
                    return httpx.Response(
                        201,
                        json={"memory_id": f"m_{row['id']}", "status": "exists", "already_exists": True},
                    )
            row = {
                "id": self._alloc(),
                "user_id": body.get("user_id"),
                "text": body.get("text"),
                "speaker": body.get("speaker"),
                "source": body.get("source"),
                "metadata": body.get("metadata"),
                "_seed": seed,
            }
            self.rows.append(row)
            return httpx.Response(201, json={"memory_id": f"m_{row['id']}", "status": "created"})

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
