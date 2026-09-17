"""Source-code converter: wrap the file in a fenced block with a language hint.

The language is derived from the file extension (:data:`CODE_LANGUAGE_BY_EXTENSION`);
unknown extensions fall back to an unlabelled fence. The fence keeps the code
recognisable to the embedding model and marks the chunk ``kind=code``.
"""

from __future__ import annotations

from app.domain.filetypes import CODE_LANGUAGE_BY_EXTENSION, extension_of
from app.rag.convert.base import KIND_CODE, ConvertedDocument


def language_for(uri: str) -> str:
    """Return the language hint for ``uri``, or ``""`` when unknown."""
    return CODE_LANGUAGE_BY_EXTENSION.get(extension_of(uri), "")


def convert(data: bytes, uri: str) -> ConvertedDocument:
    """Decode ``data`` and wrap it in a fenced code block labelled by extension."""
    source = data.decode("utf-8", errors="replace")
    language = language_for(uri)
    text = f"```{language}\n{source}\n```"
    return ConvertedDocument(text=text, images=[], kind=KIND_CODE)
