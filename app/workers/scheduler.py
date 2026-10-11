"""The minute poller that enqueues due pull-source synchronizations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import or_, select, update

from app.core.coordination import SOURCE_LOCK_TIMEOUT_S
from app.persistence.db import SessionLocal
from app.persistence.models import Source, SourceMode

SYNC_CLAIM_TIMEOUT_S = 900


async def poll_tick(ctx: dict[str, Any]) -> int:
    """Enqueue each enabled pull source whose configured interval has elapsed."""
    redis = ctx.get("redis")
    if redis is None:
        return 0
    now = datetime.now(UTC)
    enqueued = 0
    with SessionLocal() as session:
        sources = list(
            session.scalars(
                select(Source).where(Source.enabled.is_(True), Source.mode == SourceMode.pull)
            )
        )
        for source in sources:
            if source.sync_interval_s <= 0:
                continue
            last_sync = source.last_sync_at
            if last_sync is not None:
                if last_sync.tzinfo is None:
                    last_sync = last_sync.replace(tzinfo=UTC)
                if now < last_sync + timedelta(seconds=source.sync_interval_s):
                    continue
            # This lock only deduplicates concurrent scheduler ticks.  The worker
            # owns the execution lock; sharing it here lets the queued job start
            # while the scheduler is still releasing its enqueue lock.
            lock = redis.lock(
                f"lock:schedule:sync:{source.id}",
                timeout=SOURCE_LOCK_TIMEOUT_S,
                blocking=False,
            )
            if not await lock.acquire():
                continue
            try:
                claim_token = str(uuid.uuid4())
                claimed_at = now
                claimed = session.execute(
                    update(Source)
                    .where(
                        Source.id == source.id,
                        Source.enabled.is_(True),
                        Source.mode == SourceMode.pull,
                        or_(
                            Source.sync_claim_token.is_(None),
                            Source.sync_claimed_at.is_(None),
                            Source.sync_claimed_at
                            < claimed_at - timedelta(seconds=SYNC_CLAIM_TIMEOUT_S),
                        ),
                    )
                    .values(sync_claim_token=claim_token, sync_claimed_at=claimed_at)
                    .execution_options(synchronize_session=False)
                )
                claimed_count = cast(Any, claimed).rowcount
                if claimed_count != 1:
                    session.rollback()
                    continue
                session.commit()
                try:
                    job = await redis.enqueue_job("sync_source", str(source.id), claim_token)
                except Exception:
                    session.rollback()
                    session.execute(
                        update(Source)
                        .where(Source.id == source.id, Source.sync_claim_token == claim_token)
                        .values(sync_claim_token=None, sync_claimed_at=None)
                    )
                    session.commit()
                    raise
                if job is not None:
                    enqueued += 1
                else:
                    session.execute(
                        update(Source)
                        .where(Source.id == source.id, Source.sync_claim_token == claim_token)
                        .values(sync_claim_token=None, sync_claimed_at=None)
                    )
                    session.commit()
            finally:
                await lock.release()
    return enqueued