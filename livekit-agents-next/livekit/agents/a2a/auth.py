"""Who is calling a session endpoint: the project credential, a thread grant, or nobody.

A frontend talks to threads with a scoped LiveKit token whose ``agentThread`` grant names
threads by id, a scope to list and create in, or every thread of the project (``all``), and
the verbs. The project credential is a token whose ``agentEndpoint`` grant may call this
agent, as the node decides it. Through a node the project credential never reaches the
worker: the node verifies it and asserts ``authorized`` in the preamble. A thread grant
reaches a public endpoint as the bearer it is, and the worker verifies it with the project's
API key and secret, as a worker reached directly, with no node in front, verifies every
token. Any other token the project signed reaches public endpoints only.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import jwt

from .._generated.storage_threads import ThreadInfo

GRANT_CLAIM = "agentThread"
GRANT_HEADER = "X-LiveKit-Thread-Grant"
"""A thread grant token for a worker reached directly, with no node in front, beside the
bearer. Through a node the grant travels in the preamble and the node strips this header from
every forwarded request, so a client cannot plant one."""

Flag = Literal["list", "create", "write", "read", "delete"]
FLAGS: tuple[Flag, ...] = ("list", "create", "write", "read", "delete")

Claims = dict[str, object]
"""A JWT payload as this module reads and writes it: the wire, checked claim by claim."""

GrantClaims = dict[str, str | bool | list[str]]
"""The ``agentThread`` claim as a grant token carries it."""


class AuthError(Exception):
    """Refused at the door; ``status`` is the HTTP status the session API answers with."""

    status = 401

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Unauthorized(AuthError):
    """No token, or a token the project did not sign."""

    status = 401


class Forbidden(AuthError):
    """A grant for another scope, or without the flag the call needs."""

    status = 403


@dataclass(frozen=True)
class AgentThreadGrant:
    """``ClaimGrants.agentThread`` (livekit/protocol ``auth/grants.go``), checked the same way:
    read, write (send) and delete the threads in ``thread_ids``; list and create in
    ``scope``; ``all`` extends both to every thread and scope of the project, the verbs
    still applying. It covers thread data only; which endpoints the holder may call is the
    ``agentEndpoint`` grant's, and a token with only this one reaches public endpoints."""

    all: bool = False
    scope: str = ""
    thread_ids: tuple[str, ...] = ()
    list: bool = False
    create: bool = False
    write: bool = False
    read: bool = False
    delete: bool = False

    def to_claims(self) -> GrantClaims:
        out: GrantClaims = {}
        if self.all:
            out["all"] = True
        if self.scope:
            out["scope"] = self.scope
        if self.thread_ids:
            out["threadIds"] = list(self.thread_ids)

        for flag in FLAGS:
            if getattr(self, flag):
                out[flag] = True
        return out

    def can_create(self, scope: str) -> bool:
        """A scoped grant creates only in its own scope."""
        return self.create and self._in_scope(scope)

    def can_list(self, scope: str) -> bool:
        return self.list and self._in_scope(scope)

    def can_update(self) -> bool:
        """A thread's attributes and idle TTL change only under a project-wide grant."""
        return self.all and self.write

    def can_read(self, thread: ThreadInfo) -> bool:
        return self.read and self._includes(thread)

    def can_write(self, thread: ThreadInfo) -> bool:
        return self.write and self._includes(thread)

    def can_delete(self, thread: ThreadInfo) -> bool:
        return self.delete and self._includes(thread)

    def _in_scope(self, scope: str) -> bool:
        return self.all or (bool(self.scope) and self.scope == scope)

    def _includes(self, thread: ThreadInfo) -> bool:
        """A thread is in the grant by id, or as any thread under ``all``; scope covers list
        and create only."""
        return bool(thread.thread_id) and (self.all or thread.thread_id in self.thread_ids)


def _checked(grant: AgentThreadGrant) -> AgentThreadGrant:
    """A grant is refused when it names no thread, no scope and not ``all``."""
    if not (grant.all or grant.scope or grant.thread_ids):
        raise Unauthorized("an agentThread grant needs all, scope or threadIds")
    return grant


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str):
        return ()
    return tuple(str(item) for item in value)


def _grant_from_claims(raw: Mapping[str, object]) -> AgentThreadGrant:
    """The ``agentThread`` claim of a verified token, camelCase or snake_case."""
    grant = AgentThreadGrant(
        all=bool(raw.get("all")),
        scope=str(raw.get("scope") or ""),
        thread_ids=_strings(raw.get("threadIds") or raw.get("thread_ids")),
        list=bool(raw.get("list")),
        create=bool(raw.get("create")),
        write=bool(raw.get("write")),
        read=bool(raw.get("read")),
        delete=bool(raw.get("delete")),
    )
    return _checked(grant)


@dataclass
class Principal:
    """The verified caller of one request."""

    kind: Literal["anonymous", "project", "grant"]
    grant: AgentThreadGrant | None = None
    identity: str = ""
    claims: Claims = field(default_factory=dict)

    @property
    def unscoped(self) -> bool:
        """The project credential: every call is open."""
        return self.kind == "project"

    @property
    def anonymous(self) -> bool:
        return self.kind == "anonymous"


ANONYMOUS = Principal(kind="anonymous")
PROJECT = Principal(kind="project")


def _bearer(headers: Mapping[str, str]) -> str | None:
    """The bearer token of the ``Authorization`` header, when it carries one."""
    authorization = headers.get("authorization") or headers.get("Authorization") or ""
    scheme, _, token = str(authorization).partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _token_of(
    headers: Mapping[str, str] | None, query_params: Mapping[str, str] | None
) -> str | None:
    """The bearer, else the grant header, else ``access_token`` on the query string."""
    token = None
    if headers:
        token = _bearer(headers)
        if token is None:
            token = headers.get(GRANT_HEADER.lower()) or headers.get(GRANT_HEADER) or None

    if token is None and query_params is not None:
        token = query_params.get("access_token") or None
    return token


@dataclass
class Credentials:
    """The worker's API key and secret: what verifies a caller's token and mints its own."""

    api_key: str
    api_secret: str

    def verify(self, token: str, *, agent_name: str) -> Principal:
        """A thread grant, the project credential when the ``agentEndpoint`` grant may call
        ``agent_name`` (as the node decides it), else anonymous: a valid token with neither,
        a room token for one, reaches public endpoints only. ``Unauthorized`` when the token
        does not verify."""
        if not self.api_key or not self.api_secret:
            raise Unauthorized("the worker holds no API key and secret to verify tokens with")

        claims = self._decode(token)
        identity = str(claims.get("sub") or "")
        raw = claims.get(GRANT_CLAIM)
        if isinstance(raw, Mapping):
            grant = _grant_from_claims(raw)
            return Principal(kind="grant", grant=grant, identity=identity, claims=claims)

        endpoint = claims.get("agentEndpoint")
        if (
            isinstance(endpoint, Mapping)
            and endpoint.get("call") is True
            and str(endpoint.get("agentName") or "") in ("", agent_name)
        ):
            return Principal(kind="project", identity=identity, claims=claims)
        return Principal(kind="anonymous", identity=identity, claims=claims)

    def _decode(self, token: str) -> Claims:
        try:
            decoded = jwt.decode(
                token,
                key=self.api_secret,
                issuer=self.api_key,
                algorithms=["HS256"],
                leeway=60,
            )
        except jwt.PyJWTError as e:
            raise Unauthorized(f"invalid token: {e}") from e
        return dict(decoded)

    def principal(
        self,
        headers: Mapping[str, str] | None,
        *,
        agent_name: str,
        node_authorized: bool = False,
        query_params: Mapping[str, str] | None = None,
    ) -> Principal:
        """The caller of a request: the bearer token, the ``X-LiveKit-Thread-Grant`` header or
        ``access_token`` on the query string, verified here; with none of these,
        ``node_authorized`` is the project credential the node verified and stripped."""
        token = _token_of(headers, query_params)
        if token is not None:
            return self.verify(token, agent_name=agent_name)
        return PROJECT if node_authorized else ANONYMOUS

    def project_token(self, *, identity: str = "worker", ttl_seconds: int = 3600) -> str:
        """A project credential this worker mints for its own calls: the ``agentEndpoint``
        call grant lets it through the node to a ``public=False`` endpoint of any agent in the
        project; the worker behind it reads it as the project credential."""
        return self._sign(identity, ttl_seconds, {"agentEndpoint": {"call": True}})

    def thread_grant_token(
        self, grant: AgentThreadGrant, *, identity: str = "", ttl_seconds: int = 3600
    ) -> str:
        """A thread grant, as a backend mints one for a browser."""
        return self._sign(identity, ttl_seconds, {GRANT_CLAIM: grant.to_claims()})

    def _sign(self, identity: str, ttl_seconds: int, grants: Mapping[str, object]) -> str:
        now = int(time.time())
        claims: Claims = {
            "iss": self.api_key,
            "sub": identity,
            "nbf": now - 5,
            "exp": now + ttl_seconds,
            **grants,
        }
        return jwt.encode(claims, self.api_secret, algorithm="HS256")


__all__ = [
    "ANONYMOUS",
    "GRANT_CLAIM",
    "GRANT_HEADER",
    "PROJECT",
    "AgentThreadGrant",
    "AuthError",
    "Claims",
    "Credentials",
    "Forbidden",
    "GrantClaims",
    "Principal",
    "Unauthorized",
]
