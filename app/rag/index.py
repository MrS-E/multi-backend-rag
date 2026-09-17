"""Indexing stage: upsert sanitized chunks into the KB's Qdrant collection.

Owns the Qdrant payload contract from the plan:
``{kb_id, doc_id, source_id, uri, kind, chunk_index, content_hash, text}``.

Point ids are deterministic (UUID5 over KB/doc/kind/index), so re-indexing the
same document replaces its points instead of duplicating them even when the
caller skips the delete-by-doc_id step.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from app.persistence.qdrant_store import QdrantStore
from app.rag.chunk import Chunk

#: Namespace for deterministic point ids; stable across processes and restarts.
POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A chunk ready to be embedded and stored, with its full provenance."""

    chunk: Chunk
    uri: str
    content_hash: str
    source_id: str | None = None


def point_id(kb_id: uuid.UUID | str, doc_id: str, kind: str, chunk_index: int) -> str:
    """Return the deterministic Qdrant point id for one chunk."""
    name = f"{kb_id}:{doc_id}:{kind}:{chunk_index}"
    return str(uuid.uuid5(POINT_ID_NAMESPACE, name))


def build_payload(kb_id: uuid.UUID | str, doc_id: str, indexed: IndexedChunk) -> dict[str, object]:
    """Build the Qdrant payload for one chunk per the plan's payload contract."""
    return {
        "kb_id": str(kb_id),
        "doc_id": str(doc_id),
        "source_id": str(indexed.source_id) if indexed.source_id is not None else None,
        "uri": indexed.uri,
        "kind": indexed.chunk.kind,
        "chunk_index": indexed.chunk.chunk_index,
        "content_hash": indexed.content_hash,
        "text": indexed.chunk.text,
    }


def index_chunks(
    kb_id: uuid.UUID | str,
    doc_id: str,
    indexed_chunks: Sequence[IndexedChunk],
    vectors: Sequence[Sequence[float]],
    store: QdrantStore,
) -> int:
    """Upsert one point per chunk; return the number of points written."""
    if not indexed_chunks:
        return 0
    ids = [
        point_id(kb_id, doc_id, item.chunk.kind, item.chunk.chunk_index)
        for item in indexed_chunks
    ]
    payloads = [build_payload(kb_id, doc_id, item) for item in indexed_chunks]
    store.upsert(kb_id, ids, vectors, payloads)
    return len(ids)


def delete_document(kb_id: uuid.UUID | str, doc_id: str, store: QdrantStore) -> None:
    """Remove every vector belonging to ``doc_id`` from the KB's collection."""
    store.delete_by_doc_id(kb_id, doc_id)
