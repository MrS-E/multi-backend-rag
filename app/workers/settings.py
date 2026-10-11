"""arq worker configuration.

The worker is the RAG-core side of the broker: it consumes ``ingest_events``
jobs enqueued by the API. Job status is not persisted anywhere — arq's Redis
job metadata is the single source of truth, retained for ``jobs.keep_result_s``
(the API's ``GET /jobs/{id}`` proxies it and 404s once it expires).

Retries with backoff are handled inside the job itself (see
:mod:`app.workers.jobs`): a provider outage raises arq's ``Retry`` with an
exponential defer, so a transient Ollama/Qdrant failure does not lose the job.
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings

from app.core.settings import get_settings
from app.workers.jobs import ingest_events, reindex_kb, retry_raw_file_cleanups, sync_source
from app.workers.scheduler import poll_tick


def redis_settings_from_url(url: str) -> RedisSettings:
    """Build arq :class:`RedisSettings` from a ``redis://`` DSN."""
    return RedisSettings.from_dsn(url)


class WorkerSettings:
    """arq entry point: ``uv run arq app.workers.settings.WorkerSettings``."""

    functions: ClassVar[list[Any]] = [
        ingest_events,
        sync_source,
        reindex_kb,
        retry_raw_file_cleanups,
    ]
    cron_jobs: ClassVar[list[Any]] = [
        cron(poll_tick, minute=set(range(60)), second=0),
        cron(retry_raw_file_cleanups, minute=set(range(60)), second=30),
    ]
    redis_settings: ClassVar[RedisSettings] = redis_settings_from_url(get_settings().redis_url)
    max_jobs: ClassVar[int] = 2
    keep_result: ClassVar[int] = get_settings().jobs.keep_result_s
    #: arq's own retry budget; the job additionally defers with backoff on
    #: provider errors so repeated failures do not hammer a downed provider.
    max_tries: ClassVar[int] = 5
