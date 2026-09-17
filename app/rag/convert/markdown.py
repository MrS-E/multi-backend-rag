"""Markdown/text passthrough converter.

The text is kept as-is (the PoC indexed markdown verbatim); embedded
``![alt](path)`` references are extracted and resolved to bytes via the optional
``resolve_image`` callback so they can be captioned downstream. A reference whose
image cannot be resolved is simply skipped — the text chunk still carries the
alt text and path.
"""

from __future__ import annotations

from app.rag.convert.base import (
    KIND_TEXT,
    ConvertedDocument,
    ExtractedImage,
    ImageResolver,
    extract_image_refs,
)


def convert(
    data: bytes,
    uri: str,
    resolve_image: ImageResolver | None = None,
) -> ConvertedDocument:
    """Decode ``data`` as text and collect its resolvable image references.

    ``uri`` is used only for diagnostics; markdown has no language hint.
    """
    text = data.decode("utf-8", errors="replace")
    images: list[ExtractedImage] = []
    if resolve_image is not None:
        for ref in extract_image_refs(text):
            image_bytes = resolve_image(ref.path)
            if image_bytes:
                images.append(
                    ExtractedImage(
                        data=image_bytes,
                        alt=ref.alt,
                        position=ref.position,
                        uri=ref.path,
                    )
                )
    return ConvertedDocument(text=text, images=images, kind=KIND_TEXT)
