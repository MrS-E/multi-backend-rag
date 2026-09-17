"""Provider registry: route provider-prefixed model strings to implementations.

Model strings are ``<provider>/<model>`` (e.g. ``ollama/bge-m3``,
``openai/text-embedding-3-large``). The registry resolves the prefix to a
provider instance and strips it so callers pass the bare model name to the port.
Providers are built lazily on first use, so importing the registry never opens a
connection or requires an API key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.domain.errors import DomainError
from app.domain.ports import ChatProvider, EmbeddingProvider, VisionProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatProvider


class UnknownProviderError(DomainError):
    """Raised when a model string names a provider prefix we do not know."""


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    """The three provider capabilities served by one backend.

    Both bundled implementations expose all three ports from a single class, but
    keeping them separate lets a future backend supply only some capabilities.
    """

    embedder: EmbeddingProvider
    chat: ChatProvider
    vision: VisionProvider


def parse_model_str(model_str: str) -> tuple[str, str]:
    """Split ``<provider>/<model>`` into its parts.

    Raises :class:`UnknownProviderError` if the string is not provider-prefixed.
    """
    provider, sep, model = model_str.partition("/")
    if not sep or not provider or not model:
        raise UnknownProviderError(
            f"model {model_str!r} must be provider-prefixed, e.g. 'ollama/bge-m3'"
        )
    return provider, model


class ProviderRegistry:
    """Resolve model strings to provider instances, caching them by prefix."""

    def __init__(self, providers: Mapping[str, ProviderBundle] | None = None) -> None:
        self._providers: dict[str, ProviderBundle] = dict(providers or {})

    def register(self, name: str, bundle: ProviderBundle) -> None:
        """Register a ready-made provider bundle under ``name``."""
        self._providers[name] = bundle

    def known_providers(self) -> set[str]:
        """Return the provider prefixes this registry can resolve."""
        return {"ollama", "openai"} | set(self._providers)

    def validate_model_str(self, model_str: str) -> None:
        """Raise :class:`UnknownProviderError` unless ``model_str`` names a known provider."""
        provider, _ = parse_model_str(model_str)
        if provider not in self.known_providers():
            raise UnknownProviderError(
                f"unknown provider prefix {provider!r}; known providers: "
                f"{sorted(self.known_providers())}"
            )

    def _bundle(self, provider: str) -> ProviderBundle:
        bundle = self._providers.get(provider)
        if bundle is not None:
            return bundle
        if provider == "ollama":
            instance: OllamaProvider | OpenAICompatProvider = OllamaProvider()
        elif provider == "openai":
            instance = OpenAICompatProvider()
        else:
            raise UnknownProviderError(
                f"unknown provider prefix {provider!r}; known providers: "
                f"{sorted(self.known_providers())}"
            )
        bundle = ProviderBundle(embedder=instance, chat=instance, vision=instance)
        self._providers[provider] = bundle
        return bundle

    def get_embedder(self, model_str: str) -> EmbeddingProvider:
        """Return the embedding provider for a ``<provider>/<model>`` string."""
        provider, _ = parse_model_str(model_str)
        return self._bundle(provider).embedder

    def get_chat(self, model_str: str) -> ChatProvider:
        """Return the chat provider for a ``<provider>/<model>`` string."""
        provider, _ = parse_model_str(model_str)
        return self._bundle(provider).chat

    def get_vision(self, model_str: str) -> VisionProvider:
        """Return the vision provider for a ``<provider>/<model>`` string."""
        provider, _ = parse_model_str(model_str)
        return self._bundle(provider).vision


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the process-wide registry singleton."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
