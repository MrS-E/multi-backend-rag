"""Tests for chunking: per-KB size/overlap and kind labelling."""

from __future__ import annotations

from app.rag.chunk import Chunk, chunk_text, split_into_chunks
from app.rag.convert.base import KIND_CODE, KIND_TEXT


def test_given_short_text_when_chunking_then_single_chunk() -> None:
    chunks = chunk_text("short text", chunk_size=700, chunk_overlap=80)

    assert chunks == ["short text"]


def test_given_blank_text_when_chunking_then_no_chunks() -> None:
    assert chunk_text("   \n\t ", chunk_size=700, chunk_overlap=80) == []


def test_given_empty_text_when_chunking_then_no_chunks() -> None:
    assert chunk_text("", chunk_size=700, chunk_overlap=80) == []


def test_given_long_text_when_chunking_then_multiple_chunks_within_size() -> None:
    text = "abcdefghij" * 50

    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_given_long_text_when_chunking_then_consecutive_overlap_matches() -> None:
    text = "abcdefghij" * 50

    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert chunks[1].startswith(chunks[0][-20:])


def test_given_larger_overlap_when_chunking_then_more_chunks() -> None:
    text = "abcdefghij" * 50

    small = chunk_text(text, chunk_size=100, chunk_overlap=10)
    large = chunk_text(text, chunk_size=100, chunk_overlap=50)

    assert len(large) > len(small)


def test_given_overlap_not_smaller_than_size_when_chunking_then_clamped_not_raised() -> None:
    chunks = chunk_text("abcdefghij" * 5, chunk_size=20, chunk_overlap=50)

    assert chunks


def test_given_text_when_splitting_then_kind_and_indices_assigned() -> None:
    chunks = split_into_chunks("abcdefghij" * 50, chunk_size=100, chunk_overlap=20, kind=KIND_CODE)

    assert all(chunk.kind == KIND_CODE for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_given_no_kind_when_splitting_then_defaults_to_text() -> None:
    chunks = split_into_chunks("short text", chunk_size=700, chunk_overlap=80)

    assert chunks == [Chunk(text="short text", kind=KIND_TEXT, chunk_index=0)]
