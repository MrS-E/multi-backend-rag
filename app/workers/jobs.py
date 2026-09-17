"""arq jobs: the RAG-core side of the broker.

``ingest_events`` is the single entry point the API enqueues. It receives
JSON-serializable event payloads (see :mod:`app.domain.events`) plus the target
KB id, then drives the ingestion pipeline with a fresh DB session and the
process-wide provider/Qdrant/storage singletons.

Provider errors are retried with exponential backoff by raising arq's
:class:`~arq.worker.Retry`; any other failure propagates and marks the job
``failed`` in arq's Redis metadata, where ``GET /jobs/{id}`` surfaces it.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from arq.worker import Retry

from app.core.settings import get_settings
from app.domain.errors import ProviderError
from app.domain.events import event_from_payload
from app.persistence.db import SessionLocal
from app.persistence.file_storage import FileStorage
from app.persistence.qdrant_store import QdrantStore
from app.providers.registry import get_registry
from app.rag.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

#: Base seconds for the exponential backoff applied to provider errors.
RETRY_BACKOFF_BASE_S = 5

#: Cap on the backoff delay so a long outage still retries promptly.
RETRY_BACKOFF_MAX_S = 300


def retry_delay(job_try: int) -> int:
    """Return the backoff delay (seconds) for the given 1-based attempt number."""
    return min(RETRY_BACKOFF_BASE_S * 2 ** max(job_try - 1, 0), RETRY_BACKOFF_MAX_S)


async def ingest_events(ctx: dict[str, Any], kb_id: str, events: list[dict[str, Any]]) -> int:
    """Ingest a batch of serialized source events for one knowledge base.

    Returns the number of events processed. A provider outage raises arq's
    ``Retry`` with exponential backoff so the job is re-run once the provider
    recovers instead of failing permanently.
    """
    parsed = [event_from_payload(payload) for payload in events]
    pipeline = IngestionPipeline(
        session=SessionLocal(),
        settings=get_settings(),
        registry=get_registry(),
        qdrant=QdrantStore(),
        storage=FileStorage(),
    )
    job_try = int(ctx.get("job_try", 1))
    try:
        pipeline.process_events(uuid.UUID(kb_id), parsed)
    except ProviderError as exc:
        delay = retry_delay(job_try)
        logger.warning(
            "provider error during ingestion; retrying",
            extra={"kb_id": kb_id, "job_try": job_try, "defer_s": delay, "error": str(exc)},
        )
        raise Retry(defer=delay) from exc
    return len(parsed)
