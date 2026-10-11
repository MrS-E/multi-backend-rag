"""Ingestion pipeline: convert → caption → chunk → embed → index.

One document event is processed end to end, with hash-based change detection so
re-uploading unchanged content never re-embeds:

* the event's content hash (or the sha256 of its bytes) is compared with the
  stored ``documents.content_hash``; an already-indexed document with the same
  hash is skipped;
* a changed hash deletes the document's old vectors before the new ones are
  written, so a document never leaves stale points behind;
* a :class:`DeleteEvent` removes vectors, the stored raw file and the DB row.

The pipeline is deliberately synchronous and session-scoped: the arq worker
builds one instance per job. Providers and Qdrant are reached only through the
existing ports/stores, so the whole flow is testable with in-memory fakes.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable, Sequence
from pathlib import PurePosixPath

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.core.coordination import LeaseLostError
from app.core.settings import Settings
from app.domain.config_resolution import EffectiveConfig, resolve_effective_config
from app.domain.events import DeleteEvent, SourceEvent, UpsertEvent, normalize_sha256
from app.domain.models import DOC_STATUS_FAILED, DOC_STATUS_INDEXED
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document, KnowledgeBase, Source
from app.persistence.qdrant_store import QdrantStore
from app.persistence.raw_cleanup import (
    ensure_raw_file_cleanup_marker,
    remove_raw_file_cleanup_marker,
)
from app.providers.registry import ProviderRegistry, parse_model_str
from app.rag.caption import caption_image
from app.rag.chunk import Chunk, split_into_chunks
from app.rag.convert import convert_document
from app.rag.convert.base import KIND_IMAGE_CAPTION, ConvertedDocument, ExtractedImage
from app.rag.embed import embed_texts
from app.rag.index import IndexedChunk, delete_document, index_chunks

logger = logging.getLogger(__name__)

#: Metadata key a connector uses to attribute a document to a source row.
SOURCE_ID_KEY = "source_id"


def sha256_hex(data: bytes) -> str:
    """Return the lower-case hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def document_uuid(doc_id: str) -> uuid.UUID:
    """Parse an event ``doc_id`` into the ``documents.id`` UUID it references.

    Events carry the document's primary key as a string; connectors are expected
    to mint UUIDs so upsert and delete always address the same row.
    """
    try:
        return uuid.UUID(doc_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"event doc_id must be a UUID, got {doc_id!r}") from exc


def image_chunk_text(image: ExtractedImage, caption: str) -> str:
    """Compose the retrievable text for one captioned image (PoC-style layout)."""
    label = image.alt.strip()
    description = caption.strip()
    if label and description:
        body = f"[Image] {label} - {description}"
    elif description:
        body = f"[Image] {description}"
    elif label:
        body = f"[Image] {label}"
    else:
        body = "[Image]"
    if image.uri:
        body += f"\n(Image path: {image.uri})"
    return body


class IngestionPipeline:
    """Apply source events to one KB's index and document rows."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        registry: ProviderRegistry,
        qdrant: QdrantStore,
        storage: FileStorage,
        lease_check: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._registry = registry
        self._qdrant = qdrant
        self._storage = storage
        self._lease_check = lease_check

    def _fence(self) -> None:
        """Stop before a backend mutation when the coordinating lease is lost."""
        if self._lease_check is not None:
            self._lease_check()

    def process_events(
        self, kb_id: uuid.UUID, events: Sequence[SourceEvent], force: bool = False
    ) -> None:
        """Process every event for ``kb_id`` in order."""
        for event in events:
            if isinstance(event, DeleteEvent):
                self.process_delete(kb_id, event)
            elif isinstance(event, UpsertEvent):
                self.process_upsert(kb_id, event, force=force)
            else:  # pragma: no cover - exhaustive over the SourceEvent union
                raise ValueError(f"unsupported source event {event!r}")

    def process_upsert(self, kb_id: uuid.UUID, event: UpsertEvent, force: bool = False) -> None:
        """Index (or re-index) a document, skipping unchanged content."""
        kb = self._require_kb(kb_id)
        self._validate_source_id(kb_id, event)
        config = resolve_effective_config(self._settings.defaults, kb.overrides)
        doc_uuid = document_uuid(event.doc_id)
        existing = self._session.get(Document, doc_uuid)
        self._reject_cross_kb_document(existing, kb_id, doc_uuid)
        data = self._load_bytes(event, kb_id)
        actual_hash = sha256_hex(data)
        event_hash = normalize_sha256(event.content_hash)
        if event_hash is not None and event_hash != actual_hash:
            raise ValueError(f"content_hash does not match event data for {event.doc_id!r}")
        content_hash = actual_hash
        if (
            not force
            and existing is not None
            and existing.status == DOC_STATUS_INDEXED
            and existing.content_hash == content_hash
        ):
            # Content defines a document's identity: identical bytes under a new
            # filename keep the original ``uri`` rather than repointing the row,
            # so a rename alone never triggers a re-embed.
            logger.info(
                "skipping unchanged document",
                extra={"kb_id": str(kb_id), "doc_id": event.doc_id},
            )
            return

        old_raw_path = existing.raw_path if existing is not None else None
        raw_path = self._store(event, kb_id, data)
        if existing is not None and existing.content_hash != content_hash:
            self._delete_vectors(kb_id, event.doc_id)

        try:
            indexed, vectors = self._prepare(kb_id, event, data, content_hash, config)
            self._ensure_collection(kb_id, vectors)
            self._fence()
            index_chunks(kb_id, event.doc_id, indexed, vectors, self._qdrant)
        except Exception:
            self._update_row(kb_id, doc_uuid, event, raw_path, content_hash, DOC_STATUS_FAILED)
            self._remove_replaced_file(existing, old_raw_path, raw_path)
            raise

        self._update_row(kb_id, doc_uuid, event, raw_path, content_hash, DOC_STATUS_INDEXED)
        self._remove_replaced_file(existing, old_raw_path, raw_path)

    def process_delete(self, kb_id: uuid.UUID, event: DeleteEvent) -> None:
        """Remove a document's vectors, stored file and DB row."""
        self._require_kb(kb_id)
        doc_uuid = document_uuid(event.doc_id)
        document = self._session.get(Document, doc_uuid)
        self._reject_cross_kb_document(document, kb_id, doc_uuid)
        if document is None:
            self._delete_vectors(kb_id, event.doc_id)
            return
        raw_path = document.raw_path
        self._delete_vectors(kb_id, event.doc_id)
        self._fence()
        self._session.delete(document)
        self._fence()
        self._session.commit()
        self._delete_raw_if_unreferenced(raw_path, kb_id=kb_id, doc_id=doc_uuid)

    def _require_kb(self, kb_id: uuid.UUID) -> KnowledgeBase:
        kb = self._session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise ValueError(f"knowledge base {kb_id} not found")
        return kb

    def _load_bytes(self, event: UpsertEvent, kb_id: uuid.UUID) -> bytes:
        """Return the event's content, reading the stored file when only a path is given."""
        max_bytes = self._settings.server.max_upload_mb * 1024 * 1024
        if event.data is not None:
            if len(event.data) > max_bytes:
                raise ValueError("event data exceeds the configured upload limit")
            return event.data
        if event.path is not None:
            path = PurePosixPath(event.path)
            if (
                path.is_absolute()
                or len(path.parts) != 2
                or path.parts[0] != str(kb_id)
                or len(path.parts[1]) != 64
                or any(character not in "0123456789abcdef" for character in path.parts[1])
            ):
                raise ValueError("stored event path must be a content-addressed path in its KB")
            data = self._storage.read(event.path, max_bytes=max_bytes)
            if sha256_hex(data) != path.parts[1]:
                raise ValueError("stored event path does not match its content hash")
            return data
        raise ValueError(f"upsert event for {event.doc_id!r} carries neither data nor path")

    def _store(self, event: UpsertEvent, kb_id: uuid.UUID, data: bytes) -> str:
        """Persist inline bytes (content-addressed) or reuse the event's stored path."""
        if event.path is not None:
            return event.path
        self._fence()
        rel_path, _ = self._storage.save(kb_id, data)
        return rel_path

    def _remove_replaced_file(
        self, existing: Document | None, old_raw_path: str | None, raw_path: str
    ) -> None:
        """Delete the previous raw file when a changed document moved to a new path."""
        if existing is None or old_raw_path is None or old_raw_path == raw_path:
            return
        self._delete_raw_if_unreferenced(old_raw_path, kb_id=existing.kb_id, doc_id=existing.id)

    def _delete_raw_if_unreferenced(
        self, raw_path: str, kb_id: uuid.UUID, doc_id: uuid.UUID | None = None
    ) -> None:
        """Delete a raw file only after confirming no document still references it."""
        statement = select(exists().where(Document.raw_path == raw_path))
        if doc_id is not None:
            statement = select(
                exists().where(Document.raw_path == raw_path, Document.id != doc_id)
            )
        if self._session.scalar(statement):
            return
        try:
            self._fence()
            self._storage.delete(raw_path)
            remove_raw_file_cleanup_marker(self._session, raw_path)
            self._session.commit()
        except LeaseLostError:
            self._session.rollback()
            raise
        except Exception as exc:  # noqa: BLE001 - best-effort file cleanup
            self._session.rollback()
            try:
                ensure_raw_file_cleanup_marker(self._session, kb_id, raw_path)
                self._session.commit()
            except Exception:  # noqa: BLE001 - preserve the indexing outcome
                self._session.rollback()
            logger.warning(
                "failed to delete unreferenced stored file",
                extra={
                    "kb_id": str(kb_id) if kb_id is not None else None,
                    "raw_path": raw_path,
                    "error": str(exc),
                },
            )

    def _reject_cross_kb_document(
        self, document: Document | None, kb_id: uuid.UUID, doc_uuid: uuid.UUID
    ) -> None:
        """Reject an event that tries to address a document owned by another KB."""
        if document is not None and document.kb_id != kb_id:
            raise ValueError(f"document {doc_uuid} does not belong to knowledge base {kb_id}")

    def _validate_source_id(self, kb_id: uuid.UUID, event: UpsertEvent) -> None:
        """Reject UUID source metadata that is not owned by the target KB."""
        source_id = event.metadata.get(SOURCE_ID_KEY)
        if source_id is None:
            return
        try:
            source_uuid = uuid.UUID(str(source_id))
        except (ValueError, AttributeError, TypeError):
            return
        source = self._session.get(Source, source_uuid)
        if source is None or source.kb_id != kb_id:
            raise ValueError(f"source {source_uuid} does not belong to knowledge base {kb_id}")

    def _delete_vectors(self, kb_id: uuid.UUID, doc_id: str) -> None:
        """Delete a document's vectors, no-op when its KB has no collection yet."""
        if not self._qdrant.collection_exists(kb_id):
            return
        self._fence()
        delete_document(kb_id, doc_id, self._qdrant)

    def _prepare(
        self,
        kb_id: uuid.UUID,
        event: UpsertEvent,
        data: bytes,
        content_hash: str,
        config: EffectiveConfig,
    ) -> tuple[list[IndexedChunk], list[list[float]]]:
        """Convert, caption and chunk the document, then embed every chunk.

        Content is stored faithfully: sanitization happens at prompt-assembly
        time (plan Key Decision 8), not during ingest.
        """
        converted = convert_document(data, event.uri, resolve_image=self._resolve_image)
        items = self._collect_chunks(converted, config, event)

        source_id = event.metadata.get(SOURCE_ID_KEY)
        indexed = [
            IndexedChunk(
                chunk=chunk,
                uri=uri,
                content_hash=content_hash,
                source_id=str(source_id) if source_id is not None else None,
            )
            for chunk, uri in items
        ]
        texts = [item.chunk.text for item in indexed]
        _, embed_model = parse_model_str(config.embed_model)
        embedder = self._registry.get_embedder(config.embed_model)
        vectors = embed_texts(texts, embedder, embed_model)
        return indexed, vectors

    def _collect_chunks(
        self, converted: ConvertedDocument, config: EffectiveConfig, event: UpsertEvent
    ) -> list[tuple[Chunk, str]]:
        """Build the (chunk, uri) pairs for the text body and every captioned image."""
        items: list[tuple[Chunk, str]] = []
        for chunk in split_into_chunks(
            converted.text, config.chunk_size, config.chunk_overlap, converted.kind
        ):
            items.append((chunk, event.uri))

        if not converted.images:
            return items

        _, vision_model = parse_model_str(config.vision_model)
        vision = self._registry.get_vision(config.vision_model)
        for index, image in enumerate(converted.images):
            caption = caption_image(image.data, vision, vision_model)
            body = image_chunk_text(image, caption)
            items.append(
                (
                    Chunk(text=body, kind=KIND_IMAGE_CAPTION, chunk_index=index),
                    image.uri or event.uri,
                )
            )
        return items

    def _ensure_collection(self, kb_id: uuid.UUID, vectors: Sequence[Sequence[float]]) -> None:
        """Create the KB's collection when it does not exist yet (dimension from vectors)."""
        if not vectors:
            return
        if not self._qdrant.collection_exists(kb_id):
            self._fence()
            self._qdrant.create_collection(kb_id, len(vectors[0]))

    def _resolve_image(self, path: str) -> bytes | None:
        """Resolve a markdown image reference against the storage root, if present."""
        try:
            return self._storage.read(path)
        except (OSError, ValueError):
            return None

    def _update_row(
        self,
        kb_id: uuid.UUID,
        doc_uuid: uuid.UUID,
        event: UpsertEvent,
        raw_path: str,
        content_hash: str,
        status: str,
    ) -> None:
        """Create or update the ``documents`` row for an upsert."""
        source_id = event.metadata.get(SOURCE_ID_KEY)
        try:
            source_uuid = uuid.UUID(str(source_id)) if source_id is not None else None
        except (ValueError, AttributeError, TypeError):
            source_uuid = None
        document = self._session.scalar(
            select(Document).where(Document.id == doc_uuid, Document.kb_id == kb_id)
        )
        self._fence()
        if document is None:
            document = Document(
                id=doc_uuid,
                kb_id=kb_id,
                source_id=source_uuid,
                uri=event.uri,
                content_hash=content_hash,
                mime=event.mime,
                raw_path=raw_path,
                status=status,
            )
            self._session.add(document)
        else:
            document.source_id = source_uuid
            document.uri = event.uri
            document.content_hash = content_hash
            document.mime = event.mime
            document.raw_path = raw_path
            document.status = status
        self._fence()
        self._session.commit()
