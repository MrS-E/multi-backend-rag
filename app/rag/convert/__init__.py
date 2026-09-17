"""Converter dispatch: pick a converter from the file extension.

The pipeline calls :func:`convert_document`; each format module stays a small,
independently testable unit. ``resolve_image`` is threaded through to the
markdown converter so embedded ``![alt](path)`` references can be resolved to
bytes by the caller (the pipeline resolves them against the KB's stored files).
"""

from __future__ import annotations

from app.domain.filetypes import (
    CODE_LANGUAGE_BY_EXTENSION,
    IMAGE_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    PDF_EXTENSIONS,
    extension_of,
)
from app.rag.convert import code, image, markdown, pdf_docling
from app.rag.convert.base import ConvertedDocument, ImageResolver


class UnsupportedFileTypeError(ValueError):
    """Raised when no converter handles the given file extension."""


def convert_document(
    data: bytes,
    uri: str,
    resolve_image: ImageResolver | None = None,
) -> ConvertedDocument:
    """Convert ``data`` to text + images, dispatching on ``uri``'s extension."""
    extension = extension_of(uri)
    if extension in MARKDOWN_EXTENSIONS:
        return markdown.convert(data, uri, resolve_image=resolve_image)
    if extension in CODE_LANGUAGE_BY_EXTENSION:
        return code.convert(data, uri)
    if extension in PDF_EXTENSIONS:
        return pdf_docling.convert(data, uri)
    if extension in IMAGE_EXTENSIONS:
        return image.convert(data, uri)
    raise UnsupportedFileTypeError(f"no converter for extension {extension!r} ({uri!r})")
