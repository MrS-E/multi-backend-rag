"""Tests for the Qdrant collection-name helper (no server required)."""

from __future__ import annotations

import uuid

from app.persistence.qdrant_store import collection_name


def test_given_kb_id_when_building_collection_name_then_prefixed() -> None:
    kb_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    assert collection_name(kb_id) == "kb_12345678-1234-5678-1234-567812345678"


def test_given_string_kb_id_when_building_collection_name_then_used_as_is() -> None:
    assert collection_name("abc") == "kb_abc"
