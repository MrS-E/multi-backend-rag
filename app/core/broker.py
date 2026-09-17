"""Redis broker helpers: the API's arq connection pool and job queue.

The API layer only ever *enqueues* jobs and *reads* their ephemeral status; it
never imports the worker or RAG core. This module keeps that boundary: it knows
how to build an :class:`~arq.connections.ArqRedis` pool from the configured
``redis_url``, how to enqueue the ``ingest_events`` job, and nothing about what
the job does.
"""

from __future__ import annotations

import logging
import uuid

from arq.connections import ArqRedis, RedisSettings, create_pool

from app.domain.events import SourceEvent

logger = logging.getLogger(__name__)

#: Name of the arq function the API enqueues for ingestion.
INGEST_EVENTS_JOB = "ingest_events"


def redis_settings_from_url(redis_url: str) -> RedisSettings:
    """Build arq :class:`RedisSettings` from a ``redis://`` DSN."""
    return RedisSettings.from_dsn(redis_url)


async def create_arq_pool(redis_url: str) -> ArqRedis:
    """Create a Redis pool for enqueuing jobs and reading job status.

    Connection retries are disabled so a request (or startup) fails fast when
    Redis is down rather than blocking on arq's default backoff.
    """
    settings = redis_settings_from_url(redis_url)
    settings.conn_retries = 0
    return await create_pool(settings)


class JobQueue:
    """Thin wrapper over the arq pool: enqueue ingestion events, return the job id.

    Wrapping (rather than passing the raw pool around) gives tests a single,
    small seam — a fake needs only :meth:`enqueue_ingest`.
    """

    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_ingest(self, kb_id: uuid.UUID, events: list[SourceEvent]) -> str:
        """Enqueue ``ingest_events`` for ``kb_id`` and return the new job id."""
        job = await self._pool.enqueue_job(
            INGEST_EVENTS_JOB,
            str(kb_id),
            [event.to_payload() for event in events],
        )
        if job is None:
            raise RuntimeError("arq refused to enqueue the ingestion job")
        return job.job_id
