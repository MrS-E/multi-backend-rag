"""Tests for provider-registry routing (providers mocked at the port boundary)."""

from __future__ import annotations

from typing import Any

import pytest

from app.domain.models import Message
from app.providers.registry import (
    ProviderBundle,
    ProviderRegistry,
    UnknownProviderError,
    parse_model_str,
)


class _StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append((texts, model))
        return [[0.0, 1.0] for _ in texts]


class _StubChat:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], str, dict[str, Any]]] = []

    def chat(self, messages: list[Message], model: str, **opts: object) -> str:
        self.calls.append((messages, model, dict(opts)))
        return "answer"


class _StubVision:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str, str]] = []

    def caption(self, image: bytes, prompt: str, model: str) -> str:
        self.calls.append((image, prompt, model))
        return "caption"


def _bundle() -> tuple[ProviderBundle, _StubEmbedder, _StubChat, _StubVision]:
    embedder, chat, vision = _StubEmbedder(), _StubChat(), _StubVision()
    return ProviderBundle(embedder=embedder, chat=chat, vision=vision), embedder, chat, vision


def test_given_prefixed_model_when_parsing_then_split_into_provider_and_model() -> None:
    assert parse_model_str("ollama/bge-m3") == ("ollama", "bge-m3")
    assert parse_model_str("openai/text-embedding-3-large") == (
        "openai",
        "text-embedding-3-large",
    )


def test_given_unprefixed_model_when_parsing_then_raises() -> None:
    with pytest.raises(UnknownProviderError):
        parse_model_str("bge-m3")


def test_given_empty_model_after_prefix_when_parsing_then_raises() -> None:
    with pytest.raises(UnknownProviderError):
        parse_model_str("ollama/")


def test_given_registered_provider_when_getting_embedder_then_returns_it() -> None:
    bundle, embedder, _, _ = _bundle()
    registry = ProviderRegistry({"ollama": bundle})

    resolved = registry.get_embedder("ollama/bge-m3")

    assert resolved is embedder


def test_given_registered_provider_when_getting_chat_and_vision_then_returns_them() -> None:
    bundle, _, chat, vision = _bundle()
    registry = ProviderRegistry({"openai": bundle})

    assert registry.get_chat("openai/gpt-4o") is chat
    assert registry.get_vision("openai/gpt-4o") is vision


def test_given_unknown_prefix_when_getting_provider_then_raises() -> None:
    registry = ProviderRegistry({"ollama": _bundle()[0]})

    with pytest.raises(UnknownProviderError):
        registry.get_embedder("anthropic/claude")


def test_given_unprefixed_model_when_getting_provider_then_raises() -> None:
    registry = ProviderRegistry({"ollama": _bundle()[0]})

    with pytest.raises(UnknownProviderError):
        registry.get_chat("llama3.2")


def test_given_registered_bundle_when_registering_again_then_replaced() -> None:
    registry = ProviderRegistry()
    first, _, _, _ = _bundle()
    second, second_embedder, _, _ = _bundle()
    registry.register("custom", first)

    registry.register("custom", second)

    assert registry.get_embedder("custom/model") is second_embedder
