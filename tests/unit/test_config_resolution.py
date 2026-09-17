"""Tests for effective-config resolution (``override ?? global``)."""

from __future__ import annotations

from app.core.settings import ModelDefaults
from app.domain.config_resolution import OVERRIDABLE_FIELDS, resolve_effective_config


def _defaults() -> ModelDefaults:
    return ModelDefaults(
        embed_model="ollama/bge-m3",
        chat_model="ollama/llama3.2",
        vision_model="ollama/llama3.2-vision",
        chunk_size=700,
        chunk_overlap=80,
        top_k=6,
        max_ctx_chars=3000,
    )


def test_given_no_overrides_when_resolving_then_all_globals_are_used() -> None:
    config = resolve_effective_config(_defaults(), None)

    assert config.embed_model == "ollama/bge-m3"
    assert config.chat_model == "ollama/llama3.2"
    assert config.vision_model == "ollama/llama3.2-vision"
    assert config.chunk_size == 700
    assert config.chunk_overlap == 80
    assert config.top_k == 6
    assert config.max_ctx_chars == 3000


def test_given_empty_overrides_when_resolving_then_all_globals_are_used() -> None:
    config = resolve_effective_config(_defaults(), {})

    assert config.embed_model == "ollama/bge-m3"
    assert config.top_k == 6


def test_given_partial_overrides_when_resolving_then_only_those_fields_change() -> None:
    config = resolve_effective_config(_defaults(), {"chat_model": "openai/gpt-4o"})

    assert config.chat_model == "openai/gpt-4o"
    assert config.embed_model == "ollama/bge-m3"
    assert config.vision_model == "ollama/llama3.2-vision"
    assert config.chunk_size == 700


def test_given_every_field_overridden_when_resolving_then_no_global_survives() -> None:
    overrides = {
        "embed_model": "openai/text-embedding-3-large",
        "chat_model": "openai/gpt-4o",
        "vision_model": "openai/gpt-4o",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k": 10,
    }

    config = resolve_effective_config(_defaults(), overrides)

    assert config.embed_model == "openai/text-embedding-3-large"
    assert config.chat_model == "openai/gpt-4o"
    assert config.vision_model == "openai/gpt-4o"
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.top_k == 10
    assert config.max_ctx_chars == 3000


def test_given_explicit_none_override_when_resolving_then_global_is_used() -> None:
    overrides = {"embed_model": None, "chunk_size": None}

    config = resolve_effective_config(_defaults(), overrides)

    assert config.embed_model == "ollama/bge-m3"
    assert config.chunk_size == 700


def test_given_unknown_override_key_when_resolving_then_it_is_ignored() -> None:
    config = resolve_effective_config(_defaults(), {"unknown_field": "whatever"})

    assert config.embed_model == "ollama/bge-m3"
    assert not hasattr(config, "unknown_field")


def test_given_zero_valued_override_when_resolving_then_zero_is_kept() -> None:
    config = resolve_effective_config(_defaults(), {"chunk_overlap": 0})

    assert config.chunk_overlap == 0


def test_given_each_field_overridden_individually_when_resolving_then_others_inherit() -> None:
    defaults = _defaults()
    for field in OVERRIDABLE_FIELDS:
        override = "override" if field.endswith("_model") else 1
        config = resolve_effective_config(defaults, {field: override})
        assert getattr(config, field) == override
        for other in OVERRIDABLE_FIELDS:
            if other != field:
                assert getattr(config, other) == getattr(defaults, other)
