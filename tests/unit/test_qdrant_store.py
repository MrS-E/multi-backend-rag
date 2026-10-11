"""Tests for the Qdrant collection-name helper (no server required)."""

from __future__ import annotations

import uuid

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from app.persistence.qdrant_store import QdrantStore, collection_name


def test_given_kb_id_when_building_collection_name_then_prefixed() -> None:
    kb_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    assert collection_name(kb_id) == "kb_12345678-1234-5678-1234-567812345678"


def test_given_string_kb_id_when_building_collection_name_then_used_as_is() -> None:
    assert collection_name("abc") == "kb_abc"


class _DeletingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def delete_collection(self, _: str) -> None:
        raise self.error


def test_given_missing_collection_when_deleting_then_operation_is_idempotent() -> None:
    missing = UnexpectedResponse(404, "not found", b"", httpx.Headers())
    store = QdrantStore(_DeletingClient(missing))

    store.delete_collection(uuid.uuid4())


def test_given_qdrant_delete_error_when_deleting_then_error_is_retained() -> None:
    failure = UnexpectedResponse(500, "server error", b"", httpx.Headers())
    store = QdrantStore(_DeletingClient(failure))

    with pytest.raises(UnexpectedResponse):
        store.delete_collection(uuid.uuid4())
