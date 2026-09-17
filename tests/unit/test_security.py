"""Tests for the auth seam: identity providers and the KB access choke point."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import (
    ApiKeyIdentityProvider,
    NoneIdentityProvider,
    build_identity_provider,
    require_kb_access,
)
from app.core.settings import AuthConfig, Settings
from app.domain.errors import ConfigurationError
from app.domain.models import SYSTEM_SUBJECT, Principal, RequestCredentials
from app.main import create_app
from app.persistence.file_storage import FileStorage
from app.providers.registry import ProviderBundle, ProviderRegistry
from tests.conftest import StubChat, StubEmbedder, StubQdrantStore, StubVision


def _api_key_app(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: object,
) -> TestClient:
    """Build a TestClient for an app using api_key auth and mocked backends."""
    application = create_app(settings)
    application.state.session_factory = session_factory
    application.state.provider_registry = ProviderRegistry(
        {"ollama": ProviderBundle(embedder=StubEmbedder(), chat=StubChat(), vision=StubVision())}
    )
    application.state.qdrant_store = StubQdrantStore()
    application.state.file_storage = FileStorage(tmp_path)  # type: ignore[arg-type]
    return TestClient(application)


def test_given_none_provider_when_authenticating_then_system_principal() -> None:
    provider = NoneIdentityProvider()

    principal = provider.authenticate(RequestCredentials(authorization=None))

    assert principal == Principal(subject=SYSTEM_SUBJECT, groups=())


def test_given_api_key_provider_when_token_matches_then_principal() -> None:
    provider = ApiKeyIdentityProvider("secret")

    principal = provider.authenticate(RequestCredentials(authorization="Bearer secret"))

    assert principal is not None
    assert principal.subject == "api-key"


def test_given_api_key_provider_when_token_missing_then_none() -> None:
    provider = ApiKeyIdentityProvider("secret")

    assert provider.authenticate(RequestCredentials(authorization=None)) is None
    assert provider.authenticate(RequestCredentials(authorization="Bearer wrong")) is None


def test_given_api_key_provider_when_bearer_casing_differs_then_still_matches() -> None:
    provider = ApiKeyIdentityProvider("secret")

    principal = provider.authenticate(RequestCredentials(authorization="bearer secret"))

    assert principal is not None


def test_given_api_key_provider_and_unset_env_when_building_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    settings = Settings(auth=AuthConfig(provider="api_key", api_key_env="RAG_API_KEY"))

    with pytest.raises(ConfigurationError):
        build_identity_provider(settings)


def test_given_api_key_provider_and_set_env_when_building_then_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "secret")
    settings = Settings(auth=AuthConfig(provider="api_key", api_key_env="RAG_API_KEY"))

    provider = build_identity_provider(settings)

    assert provider.authenticate(RequestCredentials(authorization="Bearer secret")) is not None


def test_given_unknown_auth_provider_when_building_then_raises() -> None:
    settings = Settings(auth=AuthConfig(provider="magic"))

    with pytest.raises(ConfigurationError):
        build_identity_provider(settings)


def test_given_require_kb_access_when_called_then_allow_all() -> None:
    require_kb_access(uuid.uuid4(), "viewer", Principal(subject="someone"))


def test_given_require_kb_access_with_unknown_role_when_called_then_raises() -> None:
    with pytest.raises(ValueError):
        require_kb_access(uuid.uuid4(), "admin", Principal(subject="someone"))


def test_given_none_auth_when_requesting_kbs_then_no_401(client: TestClient) -> None:
    response = client.get("/kbs")

    assert response.status_code == 200


def test_given_kb_scoped_config_request_when_called_then_passes_through_choke_point(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[uuid.UUID | str, str]] = []

    def spy(kb_id: uuid.UUID | str, role: str, principal: Principal | None = None) -> None:
        calls.append((kb_id, role))

    monkeypatch.setattr("app.api.deps.require_kb_access", spy)
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.get("/config/effective", params={"kb_id": kb_id})

    assert response.status_code == 200
    assert calls == [(uuid.UUID(kb_id), "viewer")]


def test_given_global_config_request_when_called_then_no_choke_point(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[uuid.UUID | str, str]] = []

    def spy(kb_id: uuid.UUID | str, role: str, principal: Principal | None = None) -> None:
        calls.append((kb_id, role))

    monkeypatch.setattr("app.api.deps.require_kb_access", spy)

    response = client.get("/config/effective")

    assert response.status_code == 200
    assert calls == []


def test_given_api_key_auth_when_no_token_then_401(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
    tmp_path: object,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "secret")
    settings = Settings(auth=AuthConfig(provider="api_key", api_key_env="RAG_API_KEY"))

    with _api_key_app(settings, session_factory, tmp_path) as client:
        response = client.get("/kbs")

    assert response.status_code == 401


def test_given_api_key_auth_when_valid_token_then_200(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
    tmp_path: object,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "secret")
    settings = Settings(auth=AuthConfig(provider="api_key", api_key_env="RAG_API_KEY"))

    with _api_key_app(settings, session_factory, tmp_path) as client:
        response = client.get("/kbs", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200


def test_given_api_key_auth_when_bearer_token_is_non_ascii_then_401(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
    tmp_path: object,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "secret")
    settings = Settings(auth=AuthConfig(provider="api_key", api_key_env="RAG_API_KEY"))

    with _api_key_app(settings, session_factory, tmp_path) as client:
        response = client.get(
            "/kbs", headers=[(b"authorization", b"Bearer s\xc3\xa9cret")]
        )

    assert response.status_code == 401
