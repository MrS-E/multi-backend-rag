"""Focused source-service coordination tests."""

from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.coordination import LeaseLostError, LockLease
from app.persistence.models import KnowledgeBase, Source
from app.services.source_service import SourceService


@pytest.mark.asyncio
async def test_given_redis_unavailable_when_creating_source_then_process_local_lock_is_used(
    session_factory: sessionmaker[Session],
) -> None:
    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Sources"))
    session.commit()
    try:
        source = await SourceService(session, redis=None).create_source(
            kb_id,
            "webhook",
            "event",
            {},
            0,
            True,
        )
        assert source.kb_id == kb_id
    finally:
        session.close()


@pytest.mark.asyncio
async def test_given_source_crud_when_using_local_coordination_then_orm_runs_off_event_loop(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Sources"))
    session.commit()
    commit_threads: list[int] = []
    original_commit = session.commit

    def record_commit() -> None:
        commit_threads.append(threading.get_ident())
        original_commit()

    monkeypatch.setattr(session, "commit", record_commit)
    event_loop_thread = threading.get_ident()
    service = SourceService(session, redis=None)
    try:
        source = await service.create_source(kb_id, "webhook", "event", {}, 0, True)
        await service.update_source(kb_id, source.id, enabled=False)
        await service.delete_source(kb_id, source.id)
    finally:
        session.close()

    assert len(commit_threads) == 3
    assert all(thread_id != event_loop_thread for thread_id in commit_threads)


@pytest.mark.asyncio
async def test_given_lease_loss_before_source_commit_then_create_is_not_persisted(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Sources"))
    session.commit()

    def reject_commit(_: LockLease) -> None:
        raise LeaseLostError("lease expired")

    monkeypatch.setattr(LockLease, "ensure_valid", reject_commit)
    try:
        with pytest.raises(LeaseLostError):
            await SourceService(session, redis=None).create_source(
                kb_id, "webhook", "event", {}, 0, True
            )
        assert session.scalars(select(Source).where(Source.kb_id == kb_id)).all() == []
    finally:
        session.close()