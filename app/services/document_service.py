"""Document application service: upload, list and delete.

Sits between the documents router and persistence/broker. The API edge validates
the upload (extension, size, magic bytes), stores the raw file content-addressed
and creates the ``documents`` row, then hands the actual indexing to the worker
as an ``ingest_events`` job — the API never imports the RAG core.

Deletion follows the same brokered path: a ``DeleteEvent`` job removes the
vectors, the stored file and the row in the worker.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.core.broker import JobQueue
from app.core.coordination import (
    LockLease,
    acquire_kb_coordination,
    release_kb_coordination,
)
from app.domain.errors import ConflictError, NotFoundError
from app.domain.events import DeleteEvent, UpsertEvent
from app.domain.filetypes import is_allowed_uri, mime_for_uri, validate_content
from app.domain.models import DOC_STATUS_PENDING
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document, KnowledgeBase
from app.persistence.raw_cleanup import ensure_raw_file_cleanup_marker


class UnsupportedFileTypeError(ValueError):
    """Raised when an upload's extension is not one the pipeline handles."""


class DocumentTooLargeError(ValueError):
    """Raised when an upload exceeds the configured size limit."""


class DocumentService:
    """Upload, list and delete documents for a knowledge base."""

    def __init__(
        self, session: Session, storage: FileStorage, queue: JobQueue, redis: Any | None = None
    ) -> None:
        self._session = session
        self._storage = storage
        self._queue = queue
        self._redis = redis

    def list_documents(self, kb_id: uuid.UUID) -> list[Document]:
        """Return a KB's documents, oldest first."""
        statement = select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at)
        return list(self._session.scalars(statement))

    def get_document(self, kb_id: uuid.UUID, doc_id: uuid.UUID) -> Document:
        """Return a document scoped to its KB or raise :class:`NotFoundError`."""
        document = self._session.get(Document, doc_id)
        if document is None or document.kb_id != kb_id:
            raise NotFoundError(f"document {doc_id} not found in knowledge base {kb_id}")
        return document

    async def upload(
        self,
        kb_id: uuid.UUID,
        uri: str,
        data: bytes,
        max_upload_bytes: int,
    ) -> tuple[Document, str]:
        """Store and enqueue indexing for one upload; return ``(document, job_id)``.

        Raises :class:`UnsupportedFileTypeError`, :class:`DocumentTooLargeError`
        or :class:`ValueError` (content does not match its extension) on invalid
        input, before anything is persisted.
        """
        lock = await acquire_kb_coordination(self._redis, str(kb_id))
        if self._redis is not None and lock is None:
            raise ConflictError("knowledge base is busy")
        lease = LockLease(lock) if lock is not None else None
        if lease is not None:
            await lease.__aenter__()
        try:
            document, event = await self._run_blocking(
                lease, self._prepare_upload, kb_id, uri, data, max_upload_bytes, lease
            )
            if lease is not None:
                lease.ensure_valid()

            try:
                job_id = await self._queue.enqueue_ingest(kb_id, [event])
                if lease is not None:
                    lease.ensure_valid()
            except BaseException:
                await self._run_blocking(
                    lease, self._rollback_upload, kb_id, document, lease
                )
                raise
            return document, job_id
        finally:
            if lease is not None:
                await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(self._redis, lock)

    async def _run_blocking(self, lease: LockLease | None, function: Any, *args: Any) -> Any:
        """Run a synchronous mutation under the held lease, including cancellation cleanup."""
        if lease is None:
            return function(*args)
        return await lease.run_blocking(function, *args)

    def _prepare_upload(
        self,
        kb_id: uuid.UUID,
        uri: str,
        data: bytes,
        max_upload_bytes: int,
        lease: LockLease | None,
    ) -> tuple[Document, UpsertEvent]:
        """Validate and persist an upload without blocking the event loop."""
        if self._session.get(KnowledgeBase, kb_id) is None:
            raise NotFoundError(f"knowledge base {kb_id} not found")
        if not is_allowed_uri(uri):
            raise UnsupportedFileTypeError(f"unsupported file type: {uri!r}")
        if len(data) > max_upload_bytes:
            raise DocumentTooLargeError(
                f"upload exceeds the {max_upload_bytes // (1024 * 1024)} MB limit"
            )
        validate_content(data, uri)

        if lease is not None:
            lease.ensure_valid()
        doc_id = uuid.uuid4()
        raw_path, digest = self._storage.save(kb_id, data)
        if lease is not None:
            lease.ensure_valid()
        document = Document(
            id=doc_id,
            kb_id=kb_id,
            uri=uri,
            content_hash=digest,
            mime=mime_for_uri(uri),
            raw_path=raw_path,
            status=DOC_STATUS_PENDING,
        )
        self._session.add(document)
        if lease is not None:
            lease.ensure_valid()
        self._session.commit()
        self._session.refresh(document)
        return document, UpsertEvent(
            doc_id=str(doc_id),
            uri=uri,
            mime=document.mime,
            content_hash=digest,
            path=raw_path,
        )

    def _rollback_upload(
        self, kb_id: uuid.UUID, document: Document, lease: LockLease | None
    ) -> None:
        """Undo a persisted upload after enqueue failure, retaining durable cleanup."""
        if lease is not None:
            lease.ensure_valid()
        raw_path = document.raw_path
        self._session.delete(document)
        self._session.commit()
        try:
            still_referenced = self._session.scalar(
                select(exists().where(Document.raw_path == raw_path))
            )
            if still_referenced:
                return
            if lease is not None:
                lease.ensure_valid()
            self._storage.delete(raw_path)
        except Exception:  # noqa: BLE001 - never mask the original enqueue failure
            try:
                ensure_raw_file_cleanup_marker(self._session, kb_id, raw_path)
                self._session.commit()
            except Exception:  # noqa: BLE001 - never mask the original enqueue failure
                self._session.rollback()
            return

    async def delete(self, kb_id: uuid.UUID, doc_id: uuid.UUID) -> str:
        """Enqueue removal of a document's vectors, stored file and row."""
        lock = await acquire_kb_coordination(self._redis, str(kb_id))
        if self._redis is not None and lock is None:
            raise ConflictError("knowledge base is busy")
        lease = LockLease(lock) if lock is not None else None
        if lease is not None:
            await lease.__aenter__()
        try:
            document = self.get_document(kb_id, doc_id)
            event = DeleteEvent(doc_id=str(document.id), uri=document.uri)
            if lease is not None:
                lease.ensure_valid()
            job_id = await self._queue.enqueue_ingest(kb_id, [event])
            if lease is not None:
                lease.ensure_valid()
            return job_id
        finally:
            if lease is not None:
                await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(self._redis, lock)
