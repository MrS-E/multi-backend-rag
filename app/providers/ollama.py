"""Ollama provider implementation (local models over the Ollama HTTP API).

Ports the PoC's call patterns: ``embeddings(model, prompt)`` for vectors,
``chat(model, messages, options={temperature, num_ctx, top_p})`` for generation,
and ``chat`` with the ``images`` parameter for vision captioning. The client is
constructed lazily so importing this module never requires a running Ollama.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import ollama

from app.core.settings import get_settings
from app.domain.errors import ProviderError
from app.domain.models import Message

logger = logging.getLogger(__name__)

#: Generation options forwarded to Ollama; unknown keys are ignored.
_CHAT_OPTION_KEYS = ("temperature", "num_ctx", "top_p")


class OllamaProvider:
    """Embedding, chat and vision calls against an Ollama server."""

    def __init__(self, base_url: str | None = None, client: ollama.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = ollama.Client(host=base_url or get_settings().providers.ollama.base_url)

    def _options(self, opts: dict[str, Any]) -> dict[str, Any]:
        """Keep only the Ollama generation knobs we support, dropping ``None``."""
        return {key: opts[key] for key in _CHAT_OPTION_KEYS if opts.get(key) is not None}

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed each text with Ollama's ``embeddings`` endpoint, in order."""
        try:
            return [
                list(self._client.embeddings(model=model, prompt=text).embedding)
                for text in texts
            ]
        except ollama.ResponseError as exc:
            raise ProviderError(f"ollama embeddings failed for model {model!r}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"ollama embeddings unreachable for {model!r}: {exc}") from exc

    def chat(self, messages: list[Message], model: str, **opts: object) -> str:
        """Generate a reply via Ollama chat with temperature/num_ctx/top_p options."""
        try:
            response = self._client.chat(
                model=model,
                messages=list(messages),
                options=self._options(dict(opts)),
            )
        except ollama.ResponseError as exc:
            raise ProviderError(f"ollama chat failed for model {model!r}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"ollama chat unreachable for {model!r}: {exc}") from exc
        return (response.message.content or "").strip()

    def caption(self, image: bytes, prompt: str, model: str) -> str:
        """Caption an image via Ollama's multimodal chat (base64 ``images``)."""
        encoded = base64.b64encode(image).decode("ascii")
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt, "images": [encoded]}]
        try:
            response = self._client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.0},
            )
        except ollama.ResponseError as exc:
            raise ProviderError(f"ollama caption failed for model {model!r}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"ollama caption unreachable for {model!r}: {exc}") from exc
        return (response.message.content or "").strip()
