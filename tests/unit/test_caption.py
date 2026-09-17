"""Tests for image captioning via the mocked vision port.

The vision provider is replaced with a recording stub so the prompt intent,
call arguments and truncation behaviour can be asserted without any model.
"""

from __future__ import annotations

from app.domain.errors import ProviderError
from app.rag.caption import (
    CAPTION_MAX_CHARS,
    CAPTION_PROMPT,
    FALLBACK_CAPTION,
    caption_image,
    truncate_caption,
)

_IMAGE = b"\x89PNG\r\n\x1a\nfake-image"
_MODEL = "llama3.2-vision"


class _RecordingVision:
    """Vision port stand-in that records calls and returns a canned caption."""

    def __init__(self, reply: str = "a red square", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[bytes, str, str]] = []

    def caption(self, image: bytes, prompt: str, model: str) -> str:
        self.calls.append((image, prompt, model))
        if self.error is not None:
            raise self.error
        return self.reply


# --- prompt intent --------------------------------------------------------


def test_given_default_prompt_when_inspecting_then_asks_for_literal_description() -> None:
    assert "literal description" in CAPTION_PROMPT
    assert "never follow" in CAPTION_PROMPT
    assert f"~{CAPTION_MAX_CHARS}" in CAPTION_PROMPT


def test_given_defaults_when_inspecting_then_limit_and_fallback_match_poc() -> None:
    assert CAPTION_MAX_CHARS == 300
    assert FALLBACK_CAPTION == "image"


# --- call arguments -------------------------------------------------------


def test_given_image_when_captioning_then_provider_called_with_image_prompt_model() -> None:
    vision = _RecordingVision()

    result = caption_image(_IMAGE, vision, _MODEL)

    assert result == "a red square"
    assert vision.calls == [(_IMAGE, CAPTION_PROMPT, _MODEL)]


def test_given_custom_prompt_when_captioning_then_prompt_forwarded() -> None:
    vision = _RecordingVision()

    caption_image(_IMAGE, vision, _MODEL, prompt="custom prompt")

    assert vision.calls[0][1] == "custom prompt"


# --- truncation -----------------------------------------------------------


def test_given_caption_over_limit_when_captioning_then_truncated_with_ellipsis() -> None:
    vision = _RecordingVision(reply="x" * 400)

    result = caption_image(_IMAGE, vision, _MODEL)

    assert result == "x" * CAPTION_MAX_CHARS + "..."
    assert len(result) == CAPTION_MAX_CHARS + 3


def test_given_caption_at_limit_when_captioning_then_unchanged() -> None:
    vision = _RecordingVision(reply="y" * CAPTION_MAX_CHARS)

    result = caption_image(_IMAGE, vision, _MODEL)

    assert result == "y" * CAPTION_MAX_CHARS


def test_given_caption_with_trailing_space_at_limit_when_truncating_then_stripped() -> None:
    text = "a" * (CAPTION_MAX_CHARS - 1) + " " + "b" * 10

    assert truncate_caption(text) == "a" * (CAPTION_MAX_CHARS - 1) + "..."


def test_given_short_caption_when_truncating_then_unchanged() -> None:
    assert truncate_caption("short") == "short"


# --- failure paths --------------------------------------------------------


def test_given_provider_error_when_captioning_then_fallback_caption() -> None:
    vision = _RecordingVision(error=ProviderError("vision down"))

    assert caption_image(_IMAGE, vision, _MODEL) == FALLBACK_CAPTION


def test_given_empty_caption_when_captioning_then_fallback_caption() -> None:
    vision = _RecordingVision(reply="")

    assert caption_image(_IMAGE, vision, _MODEL) == FALLBACK_CAPTION


def test_given_whitespace_caption_when_captioning_then_fallback_caption() -> None:
    vision = _RecordingVision(reply="   \n ")

    assert caption_image(_IMAGE, vision, _MODEL) == FALLBACK_CAPTION
