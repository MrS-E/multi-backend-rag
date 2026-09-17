"""OpenAI-compatible provider (OpenAI, LM Studio, vLLM, ...).

Uses the official ``openai`` client pointed at ``providers.openai.base_url`` with
the API key read from the environment variable named by
``providers.openai.api_key_env``. Works with any OpenAI-compatible endpoint;
local servers that ignore auth still expect a non-empty key, so set the env var
to a placeholder for those.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import openai

from app.core.settings import get_settings
from app.domain.errors import ConfigurationError, ProviderError
from app.domain.models import Message

logger = logging.getLogger(__name__)

#: Vision captioning is deterministic, matching the PoC's temperature 0.
_CAPTION_TEMPERATURE = 0.0


def _image_mime(data: bytes) -> str:
    """Best-effort MIME sniffing for the data-URL prefix (no external deps)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return "image/png"


class OpenAICompatProvider:
    """Embedding, chat and vision calls against an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        settings = get_settings().providers.openai
        env_var = api_key_env or settings.api_key_env
        resolved_key = api_key if api_key is not None else os.environ.get(env_var)
        if not resolved_key:
            raise ConfigurationError(
                f"openai provider requires an API key in environment variable {env_var!r}"
            )
        self._client = openai.OpenAI(api_key=resolved_key, base_url=base_url or settings.base_url)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed texts in one request, preserving input order."""
        try:
            response = self._client.embeddings.create(input=texts, model=model)
        except openai.OpenAIError as exc:
            raise ProviderError(f"openai embeddings failed for model {model!r}: {exc}") from exc
        return [list(item.embedding) for item in response.data]

    def chat(self, messages: list[Message], model: str, **opts: object) -> str:
        """Generate a reply, mapping ``temperature``/``top_p`` to OpenAI params.

        ``num_ctx`` has no OpenAI-compatible equivalent and is ignored.
        """
        params: dict[str, Any] = {}
        if opts.get("temperature") is not None:
            params["temperature"] = opts["temperature"]
        if opts.get("top_p") is not None:
            params["top_p"] = opts["top_p"]
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=list(messages),  # type: ignore[arg-type]
                **params,
            )
        except openai.OpenAIError as exc:
            raise ProviderError(f"openai chat failed for model {model!r}: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        return (content or "").strip()

    def caption(self, image: bytes, prompt: str, model: str) -> str:
        """Caption an image using a base64 data-URL in the multimodal message."""
        mime = _image_mime(image)
        encoded = base64.b64encode(image).decode("ascii")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ]
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=_CAPTION_TEMPERATURE,
            )
        except openai.OpenAIError as exc:
            raise ProviderError(f"openai caption failed for model {model!r}: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        return (content or "").strip()
