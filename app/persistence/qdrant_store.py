"""Thin Qdrant client wrapper for per-KB collections.

Covers collection lifecycle plus point upsert, payload-filtered delete by
``doc_id`` and vector search. The module must be importable without a running
Qdrant server — the client is lazy.

The store deliberately takes plain Python values (ids, vectors, payload dicts)
rather than Qdrant model objects, so the ingestion and retrieval layers stay
testable with simple in-memory fakes.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.settings import get_settings


def collection_name(kb_id: uuid.UUID | str) -> str:
    """Return the Qdrant collection name for a knowledge base."""
    return f"kb_{kb_id}"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A single search result: its payload and similarity score."""

    payload: dict[str, Any]
    score: float


class QdrantStore:
    """Per-KB Qdrant collection management and point operations."""

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
        try:
            self._client.delete_collection(collection_name(kb_id))
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) != 404:
                raise

    def upsert(
        self,
        kb_id: uuid.UUID | str,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """Upsert points by id; re-upserting an id replaces its vector and payload."""
        if not ids:
            return
        points = [
            qmodels.PointStruct(id=point_id, vector=list(vector), payload=dict(payload))
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=collection_name(kb_id), points=points)

    def delete_by_doc_id(self, kb_id: uuid.UUID | str, doc_id: str) -> None:
        """Delete every point belonging to ``doc_id`` via a payload filter."""
        self._client.delete(
            collection_name=collection_name(kb_id),
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="doc_id", match=qmodels.MatchValue(value=str(doc_id))
                        )
                    ]
                )
            ),
        )

    def search(
        self,
        kb_id: uuid.UUID | str,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        """Return the ``limit`` nearest points to ``vector``, best first."""
        response = self._client.query_points(
            collection_name=collection_name(kb_id),
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(payload=dict(point.payload or {}), score=float(point.score))
            for point in response.points
        ]

    def count(self, kb_id: uuid.UUID | str) -> int:
        """Return the number of points in the KB's collection."""
        result = self._client.count(collection_name=collection_name(kb_id), exact=True)
        return int(result.count)
