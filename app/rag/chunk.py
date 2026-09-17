"""Chunking: ``RecursiveCharacterTextSplitter`` with per-KB size/overlap.

The PoC used 700/80; the values now come from a KB's effective config so each
knowledge base can tune them. ``chunk_text`` is pure and returns an empty list
for blank input, matching the splitter's own behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.convert.base import KIND_TEXT


@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexable unit: its text, kind and position within the document."""

    text: str
    kind: str = KIND_TEXT
    chunk_index: int = 0


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` with ``chunk_overlap``.

    Blank/whitespace-only input yields no chunks. ``chunk_overlap`` is clamped
    below ``chunk_size`` because the splitter rejects an overlap that is not
    smaller than the chunk size.
    """
    if not text.strip():
        return []
    overlap = min(chunk_overlap, max(chunk_size - 1, 0))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def split_into_chunks(
    text: str, chunk_size: int, chunk_overlap: int, kind: str = KIND_TEXT
) -> list[Chunk]:
    """Split ``text`` and label every resulting chunk with ``kind``."""
    return [
        Chunk(text=piece, kind=kind, chunk_index=index)
        for index, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap))
    ]
