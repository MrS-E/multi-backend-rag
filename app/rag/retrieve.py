"""Retrieval stage: embed the question and search the KB's Qdrant collection.

The question is embedded with the KB's *effective* embed model (``override ??
global``, resolved upstream via
:func:`app.domain.config_resolution.resolve_effective_config`) through the
provider registry, so retrieval always uses the same model the collection was
indexed with. Hits carry their full Qdrant payload plus the similarity score;
the prompt layer turns them into citation blocks.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app.domain.config_resolution import EffectiveConfig
from app.persistence.qdrant_store import QdrantStore, SearchHit
from app.providers.registry import ProviderRegistry, parse_model_str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One scored search hit: the chunk text plus its provenance payload."""

    text: str
    uri: str
    score: float
    doc_id: str = ""
    kind: str = ""
    chunk_index: int = 0


def _to_chunk(hit: SearchHit) -> RetrievedChunk:
    """Map a raw Qdrant hit onto the retrieval-layer value object."""
    payload: dict[str, Any] = hit.payload
    return RetrievedChunk(
        text=str(payload.get("text", "")),
        uri=str(payload.get("uri", "")),
        score=float(hit.score),
        doc_id=str(payload.get("doc_id", "")),
        kind=str(payload.get("kind", "")),
        chunk_index=int(payload.get("chunk_index", 0) or 0),
    )


def retrieve_chunks(
    kb_id: uuid.UUID | str,
    question: str,
    config: EffectiveConfig,
    registry: ProviderRegistry,
    store: QdrantStore,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Embed ``question`` with the KB's effective model and return the top hits.

    ``top_k`` overrides the KB's configured ``top_k`` when given; the store
    returns the nearest points best-first. A KB whose collection does not exist
    yet (nothing indexed) yields an empty result, mirroring the delete path.
    """
    limit = top_k if top_k is not None else config.top_k
    if not store.collection_exists(kb_id):
        return []
    _, embed_model = parse_model_str(config.embed_model)
    embedder = registry.get_embedder(config.embed_model)
    vectors = embedder.embed([question], embed_model)
    if not vectors:
        return []
    return [_to_chunk(hit) for hit in store.search(kb_id, vectors[0], limit)]
