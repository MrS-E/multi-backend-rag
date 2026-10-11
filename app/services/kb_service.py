"""Knowledge-base application service: CRUD, collection lifecycle, config merge.

Sits between the KB router and persistence/providers. It owns the two rules the
plan calls out explicitly:

* **effective config** = ``override ?? global`` (delegated to
  :func:`app.domain.config_resolution.resolve_effective_config`);
* **embedding-model change** marks the KB ``reindex_required`` (the reindex job
  itself lands in Step 5).

Collection creation is *best-effort*: the vector dimension is obtained by
embedding a probe string with the KB's effective embed model. If the provider or
Qdrant is unreachable at create time the KB row is still created, with status
``collection_pending`` so the caller can see that the collection has not been
provisioned yet; Step 3's ingestion pipeline will ensure it before indexing.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.broker import JobQueue
from app.core.coordination import (
    LeaseLostError,
    LockLease,
    acquire_kb_coordination,
    release_kb_coordination,
)
from app.core.settings import Settings
from app.domain.config_resolution import EffectiveConfig, resolve_effective_config
from app.domain.errors import BrokerUnavailableError, ConflictError, NotFoundError
from app.domain.models import (
    KB_STATUS_ACTIVE,
    KB_STATUS_COLLECTION_PENDING,
    KB_STATUS_REINDEX_REQUIRED,
)
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document, KnowledgeBase, RawFileCleanup
from app.persistence.qdrant_store import QdrantStore
from app.persistence.raw_cleanup import ensure_raw_file_cleanup_marker
from app.providers.registry import ProviderRegistry, parse_model_str

logger = logging.getLogger(__name__)

#: Short string embedded to discover the model's vector dimension.
_DIMENSION_PROBE = "dimension probe"


class KBService:
    """Create, read, update and delete knowledge bases."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        registry: ProviderRegistry,
        qdrant: QdrantStore,
        storage: FileStorage,
        queue: JobQueue | None = None,
        redis: Any | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._registry = registry
        self._qdrant = qdrant
        self._storage = storage
        self._queue = queue
        self._redis = redis

    def list_kbs(self) -> list[KnowledgeBase]:
        """Return all knowledge bases, oldest first."""
        return list(self._session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at)))

    def get_kb(self, kb_id: uuid.UUID) -> KnowledgeBase:
        """Return a KB or raise :class:`NotFoundError`."""
        kb = self._session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise NotFoundError(f"knowledge base {kb_id} not found")
        return kb

    def effective_config(self, kb: KnowledgeBase | None) -> EffectiveConfig:
        """Resolve the effective config for ``kb`` (``None`` → global defaults)."""
        overrides = kb.overrides if kb is not None else None
        return resolve_effective_config(self._settings.defaults, overrides)

    def create_kb(
        self,
        name: str,
        description: str,
        overrides: dict[str, Any] | None,
        owner_id: str | None,
    ) -> KnowledgeBase:
        """Create a KB row and provision its Qdrant collection (best-effort)."""
        cleaned = {key: value for key, value in (overrides or {}).items() if value is not None}
        kb = KnowledgeBase(
            name=name,
            description=description,
            owner_id=owner_id,
            overrides=cleaned,
            status=KB_STATUS_ACTIVE,
        )
        self._session.add(kb)
        self._session.flush()

        if self._provision_collection(kb):
            kb.status = KB_STATUS_ACTIVE
        else:
            kb.status = KB_STATUS_COLLECTION_PENDING
        self._session.commit()
        self._session.refresh(kb)
        return kb

    async def update_kb(
        self,
        kb_id: uuid.UUID,
        name: str | None,
        description: str | None,
        overrides_update: dict[str, Any] | None,
    ) -> KnowledgeBase:
        """Apply a partial update; flag ``reindex_required`` on embed-model change."""
        try:
            lock = (
                await acquire_kb_coordination(self._redis, str(kb_id))
            )
        except Exception as exc:  # noqa: BLE001 - Redis failures are API 503s
            raise BrokerUnavailableError("job broker unavailable") from exc
        if self._redis is not None and lock is None:
            raise ConflictError("knowledge base is busy")
        lease = LockLease(lock) if lock is not None else None
        if lease is not None:
            await lease.__aenter__()
        try:
            if lease is None:
                kb, embedding_changed = await asyncio.to_thread(
                    self._prepare_update_without_lease,
                    kb_id,
                    name,
                    description,
                    overrides_update,
                )
            else:
                kb, embedding_changed = await lease.run_blocking(
                    self._prepare_update_sync,
                    kb_id,
                    name,
                    description,
                    overrides_update,
                    lease,
                )
            if embedding_changed:
                if self._queue is None:
                    await asyncio.to_thread(self._session.rollback)
                    raise BrokerUnavailableError("job broker unavailable")
                if lease is not None:
                    lease.ensure_valid()
                try:
                    await self._queue.enqueue_reindex(kb.id)
                except BrokerUnavailableError:
                    await asyncio.to_thread(self._session.rollback)
                    raise
                except Exception as exc:  # noqa: BLE001 - enqueue failure must roll back state
                    await asyncio.to_thread(self._session.rollback)
                    raise BrokerUnavailableError("job broker unavailable") from exc

            if lease is None:
                return await asyncio.to_thread(self._commit_update_without_lease, kb)
            return await lease.run_blocking(self._commit_update_sync, kb, lease)
        except LeaseLostError:
            await asyncio.to_thread(self._session.rollback)
            raise
        finally:
            if lease is not None:
                await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(self._redis, lock)

    def _prepare_update_sync(
        self,
        kb_id: uuid.UUID,
        name: str | None,
        description: str | None,
        overrides_update: dict[str, Any] | None,
        lease: LockLease,
    ) -> tuple[KnowledgeBase, bool]:
        kb, embedding_changed = self._prepare_update_without_lease(
            kb_id, name, description, overrides_update
        )
        if embedding_changed:
            kb.config_version += 1
            kb.status = KB_STATUS_REINDEX_REQUIRED
        lease.ensure_valid()
        return kb, embedding_changed

    def _prepare_update_without_lease(
        self,
        kb_id: uuid.UUID,
        name: str | None,
        description: str | None,
        overrides_update: dict[str, Any] | None,
    ) -> tuple[KnowledgeBase, bool]:
        kb = self.get_kb(kb_id)
        previous_embed_model = self.effective_config(kb).embed_model
        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description
        if overrides_update is not None:
            kb.overrides = self._merge_overrides(kb.overrides, overrides_update)
        embedding_changed = self.effective_config(kb).embed_model != previous_embed_model
        if embedding_changed:
            kb.config_version += 1
            kb.status = KB_STATUS_REINDEX_REQUIRED
        return kb, embedding_changed

    def _commit_update_sync(self, kb: KnowledgeBase, lease: LockLease) -> KnowledgeBase:
        lease.ensure_valid()
        self._session.commit()
        self._session.refresh(kb)
        return kb

    def _commit_update_without_lease(self, kb: KnowledgeBase) -> KnowledgeBase:
        self._session.commit()
        self._session.refresh(kb)
        return kb

    async def delete_kb(self, kb_id: uuid.UUID) -> None:
        """Delete a KB: Qdrant collection, stored files, then DB rows (cascade)."""
        try:
            lock = (
                await acquire_kb_coordination(self._redis, str(kb_id))
            )
        except Exception as exc:  # noqa: BLE001 - Redis failures are API 503s
            raise BrokerUnavailableError("job broker unavailable") from exc
        if self._redis is not None and lock is None:
            raise ConflictError("knowledge base is busy")
        lease = LockLease(lock) if lock is not None else None
        if lease is not None:
            await lease.__aenter__()
        try:
            if lease is None:
                self._delete_kb_sync(kb_id, None)
            else:
                await lease.run_blocking(self._delete_kb_sync, kb_id, lease)
        except LeaseLostError:
            self._session.rollback()
            raise
        finally:
            if lease is not None:
                await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(self._redis, lock)

    def _delete_kb_sync(self, kb_id: uuid.UUID, lease: LockLease | None) -> None:
        """Delete Qdrant, rows, and raw files without blocking the event loop."""
        kb = self.get_kb(kb_id)
        raw_paths = {document.raw_path for document in kb.documents}

        if lease is not None:
            lease.ensure_valid()
        for raw_path in raw_paths:
            ensure_raw_file_cleanup_marker(self._session, kb_id, raw_path)
        # Make the durable recovery record exist before deleting the remote
        # collection.  If a later DB transaction fails, a retry can safely
        # repeat the idempotent Qdrant deletion and finish the DB cleanup.
        self._session.commit()
        if lease is not None:
            lease.ensure_valid()
        try:
            self._qdrant.delete_collection(kb_id)
        except LeaseLostError:
            raise
        except Exception as exc:  # noqa: BLE001 - retain the DB row on Qdrant failure
            self._session.rollback()
            logger.warning(
                "failed to delete qdrant collection for kb",
                extra={"kb_id": str(kb_id), "error": str(exc)},
            )
            raise RuntimeError("knowledge base vector collection could not be deleted") from exc
        self._session.delete(kb)
        if lease is not None:
            lease.ensure_valid()
        self._session.commit()

        for raw_path in raw_paths:
            try:
                if lease is not None:
                    lease.ensure_valid()
                referenced = self._session.scalar(
                    select(Document.id).where(Document.raw_path == raw_path)
                )
                if referenced is not None:
                    self._session.query(RawFileCleanup).filter(
                        RawFileCleanup.raw_path == raw_path
                    ).delete(synchronize_session=False)
                    self._session.commit()
                    continue
                if lease is not None:
                    lease.ensure_valid()
                self._storage.delete(raw_path)
                self._session.query(RawFileCleanup).filter(
                    RawFileCleanup.raw_path == raw_path
                ).delete(synchronize_session=False)
                self._session.commit()
            except LeaseLostError:
                raise
            except Exception as exc:  # noqa: BLE001 - best-effort file cleanup
                self._session.rollback()
                logger.warning(
                    "failed to delete stored file",
                    extra={"kb_id": str(kb_id), "raw_path": raw_path, "error": str(exc)},
                )

    async def retry_reindex(self, kb_id: uuid.UUID) -> str:
        """Safely enqueue a reindex after a failed job left the KB recoverable."""
        if self._queue is None:
            raise BrokerUnavailableError("job broker unavailable")
        for attempt in range(2):
            try:
                return await self._retry_reindex_once(kb_id)
            except LeaseLostError:
                if attempt == 1:
                    raise
        raise LeaseLostError("reindex retry exhausted")

    async def _retry_reindex_once(self, kb_id: uuid.UUID) -> str:
        """Perform one fenced retry attempt, reacquiring after lease loss."""
        queue = self._queue
        if queue is None:
            raise BrokerUnavailableError("job broker unavailable")
        try:
            lock = (
                await acquire_kb_coordination(self._redis, str(kb_id))
            )
        except Exception as exc:  # noqa: BLE001 - Redis failures are API 503s
            raise BrokerUnavailableError("job broker unavailable") from exc
        if self._redis is not None and lock is None:
            raise ConflictError("knowledge base is busy")
        lease = LockLease(lock) if lock is not None else None
        if lease is not None:
            await lease.__aenter__()
        try:
            kb = self.get_kb(kb_id)
            if kb.status == "reindexing" and not self._reindex_is_stale(kb):
                raise ConflictError("knowledge base reindex is already running")
            kb.status = KB_STATUS_REINDEX_REQUIRED
            kb.reindex_started_at = None
            if lease is not None:
                lease.ensure_valid()
            try:
                job_id = await queue.enqueue_reindex(kb.id)
                if lease is not None:
                    lease.ensure_valid()
            except BrokerUnavailableError:
                self._session.rollback()
                raise
            except LeaseLostError:
                self._session.rollback()
                raise
            except Exception as exc:  # noqa: BLE001 - enqueue failures are API 503s
                self._session.rollback()
                raise BrokerUnavailableError("job broker unavailable") from exc
            if lease is not None:
                lease.ensure_valid()
            self._session.commit()
            return job_id
        except LeaseLostError:
            self._session.rollback()
            raise
        finally:
            if lease is not None:
                await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(self._redis, lock)

    def retry_raw_file_cleanups(self) -> int:
        """Consume persisted raw-file deletion markers safely and idempotently."""
        processed = 0
        markers = list(
            self._session.scalars(
                select(RawFileCleanup).order_by(RawFileCleanup.created_at)
            )
        )
        for marker in markers:
            try:
                referenced = self._session.scalar(
                    select(Document.id).where(Document.raw_path == marker.raw_path)
                )
                if referenced is not None:
                    self._session.delete(marker)
                    self._session.commit()
                    processed += 1
                    continue
                self._storage.delete(marker.raw_path)
                self._session.delete(marker)
                self._session.commit()
                processed += 1
            except Exception:  # noqa: BLE001 - retain the marker for a later retry
                self._session.rollback()
                logger.warning(
                    "failed to retry raw file cleanup",
                    extra={"raw_path": marker.raw_path, "kb_id": str(marker.kb_id)},
                    exc_info=True,
                )
        return processed

    @staticmethod
    def _reindex_is_stale(kb: KnowledgeBase) -> bool:
        """Return whether a persisted running state can be taken over."""
        if kb.reindex_started_at is None:
            return True
        started_at = kb.reindex_started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        return datetime.now(UTC) - started_at >= timedelta(seconds=3600)

    def _merge_overrides(
        self, existing: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a PATCH's override fields; explicit ``None`` clears an override."""
        merged = dict(existing)
        for key, value in update.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return merged

    def _provision_collection(self, kb: KnowledgeBase) -> bool:
        """Create the KB's Qdrant collection; return ``False`` if it could not be done."""
        config = self.effective_config(kb)
        try:
            _, model = parse_model_str(config.embed_model)
            embedder = self._registry.get_embedder(config.embed_model)
            vectors = embedder.embed([_DIMENSION_PROBE], model)
            dimension = len(vectors[0])
            self._qdrant.create_collection(kb.id, dimension)
            return True
        except Exception as exc:  # noqa: BLE001 - provider/Qdrant outage must not block KB creation
            logger.warning(
                "could not provision qdrant collection; kb marked collection_pending",
                extra={"kb_id": str(kb.id), "embed_model": config.embed_model, "error": str(exc)},
            )
            return False
