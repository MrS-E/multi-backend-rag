"""Tests for the retrieval stage (fake store, mocked provider ports).

Covers effective-model routing (the KB's embed model is the one called), top-k
passthrough and score/payload mapping. No Qdrant or Ollama is contacted.
"""

from __future__ import annotations

import uuid

from app.domain.config_resolution import EffectiveConfig
from app.persistence.qdrant_store import SearchHit
from app.providers.registry import ProviderBundle, ProviderRegistry
from app.rag.retrieve import retrieve_chunks
from tests.conftest import FakeQdrantStore, StubChat, StubEmbedder, StubVision

_KB_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _config(embed_model: str = "ollama/bge-m3", top_k: int = 6) -> EffectiveConfig:
    return EffectiveConfig(
        embed_model=embed_model,
        chat_model="ollama/llama3.2",
        vision_model="ollama/llama3.2-vision",
        chunk_size=700,
        chunk_overlap=80,
        top_k=top_k,
        max_ctx_chars=3000,
    )


def _registry(embedder: StubEmbedder) -> ProviderRegistry:
    return ProviderRegistry(
        {
            "ollama": ProviderBundle(embedder=embedder, chat=StubChat(), vision=StubVision()),
            "openai": ProviderBundle(embedder=embedder, chat=StubChat(), vision=StubVision()),
        }
    )


def _hit(text: str, uri: str = "notes.md", score: float = 0.87) -> SearchHit:
    return SearchHit(
        payload={
            "text": text,
            "uri": uri,
            "doc_id": "doc-1",
            "kind": "text",
            "chunk_index": 2,
        },
        score=score,
    )


def test_given_effective_embed_model_when_retrieving_then_provider_uses_bare_model() -> None:
    embedder = StubEmbedder()
    store = FakeQdrantStore()
    store.create_collection(_KB_ID, 8)
    store.search_hits = [_hit("alpha")]

    retrieve_chunks(_KB_ID, "question", _config(), _registry(embedder), store)

    assert embedder.calls == [(["question"], "bge-m3")]


def test_given_override_embed_model_when_retrieving_then_override_routes() -> None:
    embedder = StubEmbedder()
    store = FakeQdrantStore()
    store.create_collection(_KB_ID, 8)
    store.search_hits = [_hit("alpha")]

    retrieve_chunks(
        _KB_ID,
        "question",
        _config(embed_model="openai/text-embedding-3-large"),
        _registry(embedder),
        store,
    )

    assert embedder.calls[0][1] == "text-embedding-3-large"


def test_given_config_top_k_when_retrieving_then_limit_passed_through() -> None:
    store = FakeQdrantStore()
    store.create_collection(_KB_ID, 8)
    store.search_hits = [_hit("alpha")]

    retrieve_chunks(_KB_ID, "question", _config(top_k=4), _registry(StubEmbedder()), store)

    assert store.search_calls[0][2] == 4


def test_given_top_k_override_when_retrieving_then_override_wins() -> None:
    store = FakeQdrantStore()
    store.create_collection(_KB_ID, 8)
    store.search_hits = [_hit("alpha")]

    retrieve_chunks(
        _KB_ID, "question", _config(top_k=4), _registry(StubEmbedder()), store, top_k=2
    )

    assert store.search_calls[0][2] == 2


def test_given_hits_when_retrieving_then_payload_and_score_mapped() -> None:
    store = FakeQdrantStore()
    store.create_collection(_KB_ID, 8)
    store.search_hits = [_hit("alpha", uri="docs/a.md", score=0.5)]

    chunks = retrieve_chunks(_KB_ID, "question", _config(), _registry(StubEmbedder()), store)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "alpha"
    assert chunk.uri == "docs/a.md"
    assert chunk.score == 0.5
    assert chunk.doc_id == "doc-1"
    assert chunk.kind == "text"
    assert chunk.chunk_index == 2


def test_given_no_hits_when_retrieving_then_empty_list() -> None:
    store = FakeQdrantStore()
    store.create_collection(_KB_ID, 8)

    chunks = retrieve_chunks(_KB_ID, "question", _config(), _registry(StubEmbedder()), store)

    assert chunks == []


def test_given_kb_without_collection_when_querying_then_empty_result() -> None:
    store = FakeQdrantStore()

    chunks = retrieve_chunks(_KB_ID, "question", _config(), _registry(StubEmbedder()), store)

    assert chunks == []
    assert store.search_calls == []
