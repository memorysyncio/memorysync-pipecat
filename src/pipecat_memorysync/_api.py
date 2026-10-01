"""Async client for the MemorySync v1 data plane used by this integration.

The caller's turns go to ``POST /v1/memory/add_turn``, where the server
extracts the durable facts a user turn contains (preferences, plans,
details) and stores only those as memories, asynchronously, each with its
own id. The turn text itself is not stored, assistant replies are not
stored as memories, and filler such as "ok" or "thanks" stores nothing.

Everything here is async-native because Pipecat drives every processor
from the event loop — a blocking HTTP client inside ``process_frame``
would stall the whole pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from ._version import __version__

DEFAULT_BASE_URL = "https://api.memorysync.io"
_USER_AGENT = f"pipecat-memorysync/{__version__}"

#: Namespace used when the key cannot list projects (see resolve_tenant_id).
FALLBACK_TENANT = "default"


class MemorySyncAPIError(Exception):
    """A MemorySync call failed. Carries the status code and server detail."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("MEMORYSYNC_API_KEY", "")
    if not key or not key.strip():
        raise ValueError(
            "A MemorySync API key is required. Pass api_key=... or set the "
            "MEMORYSYNC_API_KEY environment variable."
        )
    return key.strip()


def resolve_base_url(base_url: Optional[str]) -> str:
    url = base_url or os.environ.get("MEMORYSYNC_BASE_URL", "") or DEFAULT_BASE_URL
    return url.rstrip("/")


def fnv1a64(value: str) -> str:
    """FNV-1a 64-bit over UTF-16 code units, as a fixed-width hex string.

    Over UTF-16 code units — not code points, not UTF-8 bytes — so the
    output matches the JavaScript adapters character for character.
    Identical seeds across languages mean a turn sent by a Python surface
    and again by a JS surface is recognised server-side as the same turn
    and extracted once.
    """
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    h = 0xCBF29CE484222325
    data = value.encode("utf-16-le")
    for i in range(0, len(data), 2):
        unit = data[i] | (data[i + 1] << 8)
        h ^= unit
        h = (h * prime) & mask
    return format(h, "016x")


class AsyncV1Api:
    """Minimal asynchronous v1 client: add_turn, recall, query, list."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        project_id: Optional[str] = None,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._project_id = project_id
        self._timeout = timeout
        self._transport = transport
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._tenant_id: Optional[str] = None
        self._tenant_is_fallback = False

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def project_id(self) -> Optional[str]:
        return self._project_id

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def transport(self) -> Optional[httpx.AsyncBaseTransport]:
        return self._transport

    # ── plumbing ─────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        h = {
            "X-API-Key": self._api_key,
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if self._project_id:
            h["X-Project-ID"] = self._project_id
        return h

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = await self._http.request(
                method, url, headers=self._headers(), json=json, params=params
            )
        except httpx.TimeoutException as e:
            raise MemorySyncAPIError(f"Request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise MemorySyncAPIError(f"Network error: {e}") from e

        if response.status_code == 204:
            return None
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text or None
        if response.status_code >= 400:
            detail = body.get("detail") if isinstance(body, dict) else body
            raise MemorySyncAPIError(
                f"{method} {path} failed with HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
            )
        return body

    # ── calls ────────────────────────────────────────────────────────

    async def resolve_tenant_id(self) -> str:
        """The tenant id, which the v1 routes need in path or body.

        Derived from the project listing rather than asked for. Cached for
        the lifetime of this client. Keys without the ``projects:read``
        scope (evaluation keys) fall back to the fixed namespace
        ``"default"`` — deterministic. Only a definite 401/403 triggers the
        fallback; a transient server error re-raises rather than silently
        switching namespaces.
        """
        if self._tenant_id:
            return self._tenant_id
        try:
            projects = await self._request("GET", "/org/projects")
        except MemorySyncAPIError as exc:
            if exc.status_code in (401, 403):
                self._tenant_id = FALLBACK_TENANT
                self._tenant_is_fallback = True
                return self._tenant_id
            raise
        first = projects[0] if isinstance(projects, list) and projects else None
        tenant = first.get("tenant_id") if isinstance(first, dict) else None
        if not tenant:
            raise MemorySyncAPIError(
                "Could not determine the tenant for this API key."
            )
        self._tenant_id = str(tenant)
        return self._tenant_id

    @property
    def tenant_is_fallback(self) -> bool:
        """True when the namespace came from the 401/403 fallback."""
        return self._tenant_is_fallback

    def set_tenant_id(self, tenant_id: str) -> None:
        self._tenant_id = tenant_id

    async def add_turn(
        self,
        *,
        tenant_id: str,
        user_id: str,
        text: str,
        role: Optional[str] = None,
        speaker: Optional[str] = None,
        occurred_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "pipecat",
        sync_embed: bool = False,
    ) -> Dict[str, Any]:
        """Send one turn to fact extraction.

        ``role`` (``"user"`` or ``"assistant"``) says who spoke; the
        memory service only ever sends ``"user"``. Only user turns are
        extracted, and only the durable facts in them are stored as
        memories, carrying the scalar values of ``metadata``. There is no
        memory per turn (``memory_id`` is always ``None``);
        ``processing_status`` says what happened: ``"distilling"`` (facts
        are being extracted; ``request_id`` correlates the job),
        ``"skipped_low_value"`` (filler), ``"skipped_non_user_turn"``
        (assistant turns are not stored as memories), ``"skipped_replay"``
        (already sent) or ``"skipped"`` (monthly quota reached).

        ``speaker`` + ``occurred_at`` + ``text`` identify the turn:
        sending the same turn again is recognised (``already_exists:
        true``) and not extracted twice.
        """
        body: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "source": source,
            "text": text,
            "sync_embed": sync_embed,
        }
        if role is not None:
            body["role"] = role
        if speaker is not None:
            body["speaker"] = speaker
        if occurred_at is not None:
            body["occurred_at"] = occurred_at
        if metadata is not None:
            body["metadata"] = metadata
        return await self._request("POST", "/v1/memory/add_turn", json=body) or {}

    async def recall(
        self,
        *,
        tenant_id: str,
        user_id: str,
        prompt: str,
        k: Optional[int] = None,
        types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Hierarchical recall: grouped, prompt-ready context block."""
        body: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "prompt": prompt,
        }
        if k is not None:
            body["k"] = k
        if types is not None:
            body["types"] = types
        return await self._request("POST", "/v1/memory/recall", json=body) or {}

    async def query(
        self,
        *,
        tenant_id: str,
        user_id: str,
        prompt: str,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Plain semantic search over the pair's memories."""
        body: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "prompt": prompt,
        }
        if k is not None:
            body["k"] = k
        return await self._request("POST", "/v1/memory/query", json=body) or {}

    async def list_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """Every memory for the tenant/user pair, newest first."""
        from urllib.parse import quote

        raw = await self._request(
            "GET",
            f"/v1/memory/{quote(tenant_id, safe='')}/{quote(user_id, safe='')}/list",
            params={"limit": limit},
        )
        memories = raw.get("memories") if isinstance(raw, dict) else None
        return list(memories) if isinstance(memories, list) else []
