"""Thin Qdrant client wrapper for per-KB collection lifecycle.

Only collection create/delete/exists are implemented at this step; upsert,
delete-by-doc_id and search arrive with the ingestion pipeline. The module must
be importable without a running Qdrant server — the client is lazy.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.settings import get_settings


def collection_name(kb_id: uuid.UUID | str) -> str:
    """Return the Qdrant collection name for a knowledge base."""
    return f"kb_{kb_id}"


class QdrantStore:
    """Per-KB Qdrant collection management."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            settings = get_settings()
            self._client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)

    def collection_exists(self, kb_id: uuid.UUID | str) -> bool:
        """Return whether the KB's collection exists."""
        return self._client.collection_exists(collection_name(kb_id))

    def create_collection(self, kb_id: uuid.UUID | str, vector_size: int) -> None:
        """Create the KB's collection with cosine distance and the given dimension."""
        self._client.create_collection(
            collection_name=collection_name(kb_id),
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    def delete_collection(self, kb_id: uuid.UUID | str) -> None:
        """Delete the KB's collection if it exists."""
        self._client.delete_collection(collection_name(kb_id))
