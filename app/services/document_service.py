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

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.broker import JobQueue
from app.domain.errors import NotFoundError
from app.domain.events import DeleteEvent, UpsertEvent
from app.domain.filetypes import is_allowed_uri, mime_for_uri, validate_content
from app.domain.models import DOC_STATUS_PENDING
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document


class UnsupportedFileTypeError(ValueError):
    """Raised when an upload's extension is not one the pipeline handles."""


class DocumentTooLargeError(ValueError):
    """Raised when an upload exceeds the configured size limit."""


class DocumentService:
    """Upload, list and delete documents for a knowledge base."""

    def __init__(self, session: Session, storage: FileStorage, queue: JobQueue) -> None:
        self._session = session
        self._storage = storage
        self._queue = queue

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
        if not is_allowed_uri(uri):
            raise UnsupportedFileTypeError(f"unsupported file type: {uri!r}")
        if len(data) > max_upload_bytes:
            raise DocumentTooLargeError(
                f"upload exceeds the {max_upload_bytes // (1024 * 1024)} MB limit"
            )
        validate_content(data, uri)

        doc_id = uuid.uuid4()
        raw_path, digest = self._storage.save(kb_id, data)
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
        self._session.commit()
        self._session.refresh(document)

        event = UpsertEvent(
            doc_id=str(doc_id),
            uri=uri,
            mime=document.mime,
            content_hash=digest,
            path=raw_path,
        )
        job_id = await self._queue.enqueue_ingest(kb_id, [event])
        return document, job_id

    async def delete(self, kb_id: uuid.UUID, doc_id: uuid.UUID) -> str:
        """Enqueue removal of a document's vectors, stored file and row."""
        document = self.get_document(kb_id, doc_id)
        event = DeleteEvent(doc_id=str(document.id), uri=document.uri)
        return await self._queue.enqueue_ingest(kb_id, [event])
