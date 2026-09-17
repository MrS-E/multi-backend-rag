"""Tests for the settings loader (YAML defaults + env overrides)."""

from __future__ import annotations

from app.core.settings import Settings


def test_given_config_yaml_when_loading_then_defaults_are_present() -> None:
    settings = Settings()

    assert settings.defaults.embed_model == "ollama/bge-m3"
    assert settings.defaults.chunk_size == 700
    assert settings.qdrant.host == "localhost"
    assert settings.jobs.keep_result_s == 3600
    assert settings.auth.provider == "none"
    assert settings.auth.oidc.roles_claim == "groups"


def test_given_nested_env_vars_when_loading_then_they_override_yaml(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFAULTS__CHAT_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("QDRANT__HOST", "qdrant")

    settings = Settings()

    assert settings.defaults.chat_model == "openai/gpt-4o"
    assert settings.qdrant.host == "qdrant"
