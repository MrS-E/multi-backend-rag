"""Tests for content-addressed file storage."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest

from app.persistence.file_storage import FileStorage


def test_given_bytes_when_saving_then_path_is_sha256_named(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    kb_id = uuid.uuid4()

    rel_path, digest = storage.save(kb_id, b"hello world")

    assert digest == hashlib.sha256(b"hello world").hexdigest()
    assert rel_path == f"{kb_id}/{digest}"
    assert storage.read(rel_path) == b"hello world"


def test_given_identical_content_when_saving_twice_then_same_path(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    kb_id = uuid.uuid4()

    first, _ = storage.save(kb_id, b"same")
    second, _ = storage.save(kb_id, b"same")

    assert first == second


def test_given_stored_file_when_deleting_then_it_is_gone(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    rel_path, _ = storage.save(uuid.uuid4(), b"data")

    storage.delete(rel_path)

    assert not (tmp_path / rel_path).exists()


def test_given_traversal_path_when_reading_then_raises(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.read("../escape")
