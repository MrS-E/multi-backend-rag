"""Standalone raster-image converter.

A standalone image has no text of its own: the converter returns an empty text
body and a single :class:`ExtractedImage` for the captioning stage to describe.
"""

from __future__ import annotations

from app.rag.convert.base import KIND_TEXT, ConvertedDocument, ExtractedImage


def convert(data: bytes, uri: str) -> ConvertedDocument:
    """Return the image as the sole extracted image, with no text body."""
    return ConvertedDocument(
        text="",
        images=[ExtractedImage(data=data, alt="", position=0, uri=uri)],
        kind=KIND_TEXT,
    )
