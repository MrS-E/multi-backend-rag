"""Embedding stage: turn chunk texts into vectors via the embedding port.

Texts are embedded in bounded batches so a large document does not build one
oversized provider request. The provider port already accepts a list of texts,
so the batching here is about request size, not per-text calls.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.ports import EmbeddingProvider

#: Maximum number of texts sent to the provider in a single call.
EMBED_BATCH_SIZE = 64


def embed_texts(
    texts: Sequence[str],
    provider: EmbeddingProvider,
    model: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed ``texts`` in order, batching calls to ``provider``.

    Returns one vector per input text, preserving input order. Empty input
    yields an empty list without calling the provider.
    """
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        vectors.extend(provider.embed(batch, model))
    return vectors
