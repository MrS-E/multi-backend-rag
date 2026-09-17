"""Auth seam: principal resolution and the single KB-authorization choke point.

v1 is single-tenant: ``auth.provider: none`` makes every caller the system
principal, and ``require_kb_access`` allows everything. The seam exists so that
turning on multi-user auth later is *config plus one class*:

* add an ``oidc`` :class:`IdentityProvider` that validates the bearer token
  against the issuer's JWKS — ``get_principal`` and every router stay unchanged;
* back :func:`require_kb_access` with a ``kb_memberships`` lookup — the role
  argument is already threaded through every KB-scoped endpoint.

Workers and the RAG core never see principals; authorization happens only here,
at the API edge, before jobs are enqueued.
"""

from __future__ import annotations

import hmac
import logging
import os
import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.core.settings import Settings
from app.domain.errors import ConfigurationError
from app.domain.models import SYSTEM_SUBJECT, Principal, RequestCredentials
from app.domain.ports import IdentityProvider

logger = logging.getLogger(__name__)

#: KB roles, ordered from least to most privileged (used by future membership checks).
KB_ROLES: tuple[str, ...] = ("viewer", "editor", "owner")

_BEARER_PREFIX = "bearer "


class NoneIdentityProvider:
    """Single-tenant provider: every caller is the system principal."""

    def __init__(self, subject: str = SYSTEM_SUBJECT) -> None:
        self._subject = subject

    def authenticate(self, creds: RequestCredentials) -> Principal | None:
        """Always authenticate as the system principal, ignoring credentials."""
        return Principal(subject=self._subject, groups=())


class ApiKeyIdentityProvider:
    """Static bearer-token provider: the token must equal the configured key."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ConfigurationError("api_key auth provider requires a non-empty key")
        self._api_key = api_key

    def authenticate(self, creds: RequestCredentials) -> Principal | None:
        """Return a principal when the bearer token matches, else ``None``."""
        token = _extract_bearer_token(creds.authorization)
        if token is None:
            return None
        # Compare bytes: ``hmac.compare_digest`` rejects ``str`` arguments that
        # contain non-ASCII characters, which would surface as a 500 instead of
        # a clean 401 for a malformed token.
        if not hmac.compare_digest(token.encode("utf-8"), self._api_key.encode("utf-8")):
            return None
        return Principal(subject="api-key", groups=())


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Parse ``Authorization: Bearer <token>``, tolerating case and whitespace."""
    if not authorization:
        return None
    if not authorization.lower().startswith(_BEARER_PREFIX):
        return None
    token = authorization[len(_BEARER_PREFIX) :].strip()
    return token or None


def build_identity_provider(settings: Settings) -> IdentityProvider:
    """Construct the configured identity provider.

    Raises :class:`ConfigurationError` at startup when ``api_key`` is selected
    but the named environment variable is unset or empty.
    """
    provider = settings.auth.provider.lower()
    if provider == "none":
        return NoneIdentityProvider()
    if provider == "api_key":
        env_var = settings.auth.api_key_env
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise ConfigurationError(
                f"auth.provider is 'api_key' but environment variable {env_var!r} is unset"
            )
        return ApiKeyIdentityProvider(api_key)
    if provider == "oidc":
        raise ConfigurationError("auth.provider 'oidc' is reserved for the multi-user roadmap")
    raise ConfigurationError(f"unknown auth.provider {settings.auth.provider!r}")


def _identity_provider(request: Request) -> IdentityProvider:
    """Return the app-scoped identity provider, building it lazily if absent."""
    provider: IdentityProvider | None = getattr(request.app.state, "identity_provider", None)
    if provider is None:
        provider = build_identity_provider(request.app.state.settings)
        request.app.state.identity_provider = provider
    return provider


async def get_principal(request: Request) -> Principal:
    """Resolve the caller's :class:`Principal`, or raise ``401``.

    This is the only place identity is established; every KB-scoped router
    depends on it.
    """
    creds = RequestCredentials(authorization=request.headers.get("Authorization"))
    principal = _identity_provider(request).authenticate(creds)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_kb_access(
    kb_id: uuid.UUID | str, role: str, principal: Principal | None = None
) -> None:
    """Authorization choke point for KB-scoped endpoints.

    v1 is allow-all: the role argument is validated and the principal is accepted
    without a membership lookup. When multi-user lands, this function becomes the
    single place that checks ``kb_memberships`` and raises ``403``.
    """
    if role not in KB_ROLES:
        raise ValueError(f"unknown KB role {role!r}; expected one of {KB_ROLES}")
    return None


#: Convenience alias so routers can write ``PrincipalDep`` instead of ``Depends(get_principal)``.
PrincipalDep = Annotated[Principal, Depends(get_principal)]
