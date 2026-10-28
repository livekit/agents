"""LiveKit credentials and thread grants, backed by the native authorization core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from .._core import _core_auth_grant_allows, _core_Credentials
from .._generated.a2a_auth import AuthGrant, AuthPrincipal
from .._generated.storage_class import ThreadInfo

GRANT_CLAIM = "agentThread"
GRANT_HEADER = "X-LiveKit-Thread-Grant"
Flag = Literal["list", "create", "write", "read", "delete"]


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


class AgentThreadGrant(AuthGrant):
    """A grant's authoring surface; the native core owns every permission decision."""

    def __init__(
        self,
        *,
        all: bool = False,
        scope: str = "",
        thread_ids: Sequence[str] = (),
        list: bool = False,
        create: bool = False,
        write: bool = False,
        read: bool = False,
        delete: bool = False,
    ) -> None:
        super().__init__(
            all=all,
            scope=scope,
            thread_ids=[*thread_ids],
            list=list,
            create=create,
            write=write,
            read=read,
            delete=delete,
        )

    def can_create(self, scope: str | None = None) -> bool:
        return _core_auth_grant_allows(self, "create", "", scope)

    def can_list(self, scope: str) -> bool:
        return _core_auth_grant_allows(self, "list", "", scope)

    def can_read(self, thread: ThreadInfo) -> bool:
        return _core_auth_grant_allows(self, "read", thread.thread_id, thread.scope)

    def can_write(self, thread: ThreadInfo) -> bool:
        return _core_auth_grant_allows(self, "write", thread.thread_id, thread.scope)

    def can_delete(self, thread: ThreadInfo) -> bool:
        return _core_auth_grant_allows(self, "delete", thread.thread_id, thread.scope)


class Principal(AuthPrincipal):
    """The verified endpoint caller, with endpoint and thread authority kept independent."""

    grant: AgentThreadGrant | None

    @property
    def unscoped(self) -> bool:
        return self.kind == "project"

    @property
    def anonymous(self) -> bool:
        return False

    @classmethod
    def _from_native(cls, principal: AuthPrincipal) -> Principal:
        grant = principal.grant
        return cls(
            kind=principal.kind,
            grant=AgentThreadGrant(
                all=grant.all,
                scope=grant.scope,
                thread_ids=grant.thread_ids,
                list=grant.list,
                create=grant.create,
                write=grant.write,
                read=grant.read,
                delete=grant.delete,
            )
            if grant is not None
            else None,
            identity=principal.identity,
            claims=principal.claims,
            endpoint_call=principal.endpoint_call,
        )


class Credentials:
    """The worker's project key pair; signing and verification use the LiveKit SDK."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._native = _core_Credentials(api_key, api_secret)

    def verify(self, token: str, *, agent_name: str = "", deployment: str = "") -> Principal:
        return Principal._from_native(
            self._native.verify(token, agent_name=agent_name, deployment=deployment)
        )

    def principal(
        self,
        headers: Mapping[str, str] | None,
        *,
        agent_name: str,
        deployment: str = "",
        node_authorized: bool = False,
        query_params: Mapping[str, str] | None = None,
    ) -> Principal:
        headers = headers or {}
        return Principal._from_native(
            self._native.principal(
                headers.get("authorization", headers.get("Authorization")),
                headers.get(GRANT_HEADER.lower(), headers.get(GRANT_HEADER)),
                node_authorized,
                agent_name,
                deployment,
            )
        )

    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str:
        return self._native.project_token(identity=identity, ttl_seconds=ttl_seconds)

    def thread_grant_token(
        self, grant: AgentThreadGrant, *, identity: str = "", ttl_seconds: int = 3600
    ) -> str:
        return self._native.thread_grant_token(grant, identity=identity, ttl_seconds=ttl_seconds)


# Internal calls must explicitly choose a trusted principal. HTTP requests never receive this
# missing-principal sentinel; the endpoint authenticates them before dispatch.
ANONYMOUS = None
PROJECT = Principal(kind="project", grant=None, identity="", claims={}, endpoint_call=True)

__all__ = [
    "AgentThreadGrant",
    "AuthError",
    "Credentials",
    "Forbidden",
    "Principal",
    "Unauthorized",
]
