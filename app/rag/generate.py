"""Generation stage: call the KB's effective chat model on the assembled prompt.

The chat options are ported verbatim from the PoC (``temperature`` 0.2,
``num_ctx`` 4096, ``top_p`` 0.9). The returned :class:`Source` values carry the
*faithful* stored excerpt (not the sanitized text the prompt carried), so the
API can show the client what the document really says without re-reading Qdrant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.config_resolution import EffectiveConfig
from app.providers.registry import ProviderRegistry, parse_model_str
from app.rag.prompts import build_prompt
from app.rag.retrieve import RetrievedChunk

#: Chat generation options, ported from the PoC's ``generate_answer``.
CHAT_OPTIONS: dict[str, object] = {
    "temperature": 0.2,
    "num_ctx": 4096,
    "top_p": 0.9,
}


@dataclass(frozen=True, slots=True)
class Source:
    """A cited source: the document URI, its score and a faithful excerpt."""

    uri: str
    score: float
    snippet: str


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """The model's answer plus the sources it was given."""

    answer: str
    sources: list[Source]


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    config: EffectiveConfig,
    registry: ProviderRegistry,
) -> GeneratedAnswer:
    """Assemble the injection-safe prompt and answer it with the KB's chat model."""
    assembled = build_prompt(question, chunks, config.max_ctx_chars)
    _, chat_model = parse_model_str(config.chat_model)
    provider = registry.get_chat(config.chat_model)
    answer = provider.chat(assembled.messages, chat_model, **CHAT_OPTIONS)
    sources = [
        Source(uri=citation.uri, score=citation.score, snippet=citation.faithful_snippet)
        for citation in assembled.citations
    ]
    return GeneratedAnswer(answer=answer, sources=sources)
