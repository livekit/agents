"""LiveKit credentials and thread grants, backed by the native authorization core."""

from __future__ import annotations

from collections.abc import Mapping

from .._core import _core_Credentials
from .._generated.a2a_auth import AgentThreadGrant, Principal

GRANT_HEADER = "X-LiveKit-Thread-Grant"


class AuthError(Exception):
    """An HTTP authorization refusal, outside JSON-RPC."""

    status = 401

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Unauthorized(AuthError):
    status = 401


class Forbidden(AuthError):
    status = 403


class Credentials:
    """The worker's project key pair; signing and verification use the LiveKit SDK."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._native = _core_Credentials(api_key, api_secret)

    def verify(self, token: str, *, agent_name: str = "", deployment: str = "") -> Principal:
        return self._native.verify(token, agent_name=agent_name, deployment=deployment)

    def _principal(
        self,
        headers: Mapping[str, str] | None,
        *,
        agent_name: str,
        deployment: str = "",
        node_authorized: bool = False,
    ) -> Principal:
        headers = headers or {}
        return self._native.principal(
            headers.get("authorization", headers.get("Authorization")),
            headers.get(GRANT_HEADER.lower(), headers.get(GRANT_HEADER)),
            node_authorized,
            agent_name,
            deployment,
        )

    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str:
        return self._native.project_token(identity=identity, ttl_seconds=ttl_seconds)

    def thread_grant_token(
        self, grant: AgentThreadGrant, *, identity: str = "", ttl_seconds: int = 3600
    ) -> str:
        return self._native.thread_grant_token(grant, identity=identity, ttl_seconds=ttl_seconds)


PROJECT = Principal(kind="project", grant=None, identity="", claims={}, endpoint_call=True)

__all__ = [
    "AgentThreadGrant",
    "AuthError",
    "Credentials",
    "Forbidden",
    "Principal",
    "Unauthorized",
]
