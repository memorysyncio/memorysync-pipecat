"""MemorySync for Pipecat — voice pipelines that remember callers.

One processor between your user context aggregator and the LLM::

    from pipecat_memorysync import MemorySyncMemoryService

    memory = MemorySyncMemoryService(
        api_key=os.getenv("MEMORYSYNC_API_KEY"),
        user_id="caller-123",
        session_id="conversation-456",
    )

The voice contract: **recall runs under a hard time budget** (default
1.2 s — a slow network passes the frame through unenriched, never
stalling a spoken reply), **capture is delta-only** (each frame sends only
the caller's new messages to fact extraction, with idempotency seeds — not
the whole conversation re-sent every turn; only the durable facts they
contain are stored, and assistant replies are not sent), and **nothing
ever raises into the pipeline**.
"""

from ._api import AsyncV1Api, MemorySyncAPIError, fnv1a64
from ._version import __version__
from .service import MemorySyncMemoryService

__all__ = [
    "MemorySyncMemoryService",
    "AsyncV1Api",
    "MemorySyncAPIError",
    "fnv1a64",
    "__version__",
]
