"""The text door over HTTP: ``@server.a2a_session`` as an ASGI application (plan 7.3).

One endpoint is one :class:`A2AEndpoint`: its handler, card, visibility, policies and the
storage its lane lives in. Bound to a server it serves its :class:`A2AService` (``methods.py``)
as the card at ``/.well-known/agent-card.json`` and JSON-RPC at ``POST /v1``, with server-sent
events for the streams; the caller is checked by ``auth.py`` and the developer's ``auth``.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import timedelta
from typing import Generic

from a2a.server.jsonrpc_models import (
    InternalError as RpcInternalError,
    InvalidRequestError as RpcInvalidRequestError,
    JSONParseError as RpcParseError,
)
from a2a.server.request_handlers.response_helpers import build_error_response
from a2a.utils.errors import InternalError
from pydantic import JsonValue
from starlette.applications import Starlette
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .._generated.storage_turn import Policy, TurnOptions
from ..storage.agentdb import AgentDBStorage
from ..storage.context import Handler, S
from ..storage.protocol import dump_state
from ..storage.resolve import AgentRegistry
from .auth import ANONYMOUS, AuthError, Credentials, Principal, Unauthorized
from .extension import EXTENSIONS_HEADER, Extensions, agent_card, to_json
from .methods import METHODS, STREAMING, A2AService, CallContext, JsonRpc

AGENT_CARD_PATH = "/.well-known/agent-card.json"
RPC_PATH = "/v1"
"""Where the binding's methods live under the endpoint, and what the card's URL points at."""

AuthCallable = Callable[[HTTPConnection], "bool | Awaitable[bool]"]


def a2a_session(
    *,
    endpoint: str,
    description: str,
    state: type[S] | None = None,
    public: bool = True,
    auth: AuthCallable | None = None,
    inbox: Policy | Mapping[str, Policy] | None = None,
    debounce: timedelta | Mapping[str, timedelta] | None = None,
    storage: AgentDBStorage | None = None,
    version: str | None = None,
    lane: str | None = None,
) -> Callable[[Handler[S]], A2AEndpoint[S]]:
    """The text door as a decorator with no server: the door is an object a module defines and
    ``AgentServer.add()`` serves (``server.a2a_session(...)`` does both). ``lane`` names the lane
    it writes when that is not the worker's own (docs/lanes.md)."""

    def decorator(handler: Handler[S]) -> A2AEndpoint[S]:
        return A2AEndpoint(
            handler,
            endpoint=endpoint,
            description=description,
            state=state,
            public=public,
            auth=auth,
            inbox=inbox,
            debounce=debounce,
            storage=storage,
            version=version or os.environ.get("LIVEKIT_AGENT_VERSION") or "1.0.0",
            lane=lane,
        )

    return decorator


class A2AEndpoint(Generic[S]):
    """What ``@server.a2a_session`` returns: the handler, plus its name, card, visibility and a
    ``url`` resolved at runtime. Pass it wherever a delegate or a client wants an endpoint."""

    def __init__(
        self,
        handler: Handler[S],
        *,
        endpoint: str,
        description: str,
        state: type[S] | None = None,
        public: bool = True,
        auth: AuthCallable | None = None,
        inbox: Policy | Mapping[str, Policy] | None = None,
        debounce: timedelta | Mapping[str, timedelta] | None = None,
        storage: AgentDBStorage | None = None,
        version: str = "1.0.0",
        lane: str | None = None,
    ) -> None:
        if not endpoint or "/" in endpoint:
            raise ValueError("endpoint is one path segment, the last of the endpoint's URL")
        policies = inbox.values() if isinstance(inbox, Mapping) else [inbox]
        if "concurrent" in policies and state is not None:
            raise ValueError("the concurrent policy requires a door with no state")

        self.handler = handler
        self.name = endpoint
        self.description = description
        self.state_type = state
        self.public = public
        self.auth = auth
        self.turn_options = TurnOptions(
            policy=dict(inbox) if isinstance(inbox, Mapping) else inbox,
            debounce=_seconds(debounce),
            state_default=dump_state(state(), state) if state is not None else None,
        )
        self.version = version
        self.lane = lane
        self._storage = storage
        self._service: A2AService[S] | None = None
        self._url: str | None = None
        self._url_base: Callable[[], str | None] | None = None
        self._app: Starlette | None = None

    # -- binding to a server --

    def bind(
        self,
        *,
        storage: AgentDBStorage | None,
        agent_name: Callable[[], str] | str,
        credentials: Credentials | None,
        url_base: Callable[[], str | None] | None = None,
        agents: AgentRegistry | None = None,
        lane: Callable[[], str] | None = None,
    ) -> A2AService[S]:
        """Attach the endpoint to a server's storage, agent name, credentials and registry;
        ``lane`` when the door writes a lane other than the worker's."""
        chosen = self._storage or storage
        if chosen is None:
            raise ValueError(
                f"a2a_session(endpoint={self.name!r}) needs a storage: AgentServer(storage=...) "
                "or a2a_session(storage=...)"
            )

        name_fn = (lambda: agent_name) if isinstance(agent_name, str) else agent_name
        own = self.lane
        self._service = A2AService(
            self,
            storage=chosen,
            agent_name=name_fn,
            credentials=credentials,
            agents=agents,
            lane=lane or ((lambda: own) if own else None),
        )
        self._url_base = url_base
        return self._service

    @property
    def service(self) -> A2AService[S]:
        if self._service is None:
            raise RuntimeError(f"endpoint {self.name!r} is not attached to a server yet")
        return self._service

    @property
    def url(self) -> str | None:
        """``https://<project>.livekit.cloud/agents/<agent>/<deployment>/<endpoint>`` when the
        server knows its project and deployment; ``None`` otherwise."""
        if self._url:
            return self._url
        if self._url_base is not None:
            base = self._url_base()
            if base:
                return f"{base.rstrip('/')}/{self.name}"
        return None

    @url.setter
    def url(self, value: str | None) -> None:
        self._url = value

    def card(self, base_url: str | None = None) -> dict[str, JsonValue]:
        """The agent card as JSON (plan 7.3.1). ``base_url`` is where the endpoint is reached
        when its own ``url`` is unknown, the request's base URL plus the mount path."""
        root = (self.url or base_url or f"/{self.name}").rstrip("/")
        return to_json(
            agent_card(
                url=f"{root}{RPC_PATH}",
                name=self.name,
                description=self.description,
                version=self.version,
                public=self.public,
            )
        )

    # -- the ASGI app --

    @property
    def app(self) -> Starlette:
        """The endpoint as an ASGI application: the card and the JSON-RPC route at its root;
        the server mounts it under ``/<endpoint>``."""
        if self._app is None:
            self._app = Starlette(
                routes=[
                    Route(AGENT_CARD_PATH, self._serve_card, methods=["GET"]),
                    Route(RPC_PATH, self._serve_rpc, methods=["POST"]),
                ]
            )
        return self._app

    async def _serve_card(self, request: Request) -> Response:
        # the mount's path is the scope's root_path; base_url carries the host alone
        base = str(request.base_url).rstrip("/") + str(request.scope.get("root_path") or "")
        return JSONResponse(self.card(base.rstrip("/")))

    async def authenticate(self, connection: HTTPConnection) -> Principal:
        """The caller of a request: the preamble's grant and verdict, a bearer token when no
        node is in front, the developer's ``auth``.

        Raises ``Unauthorized`` when ``auth`` refuses or the token does not verify, and when
        a non-public endpoint sees nobody.
        """
        lk_request = getattr(connection.state, "lk_request", None)
        node_authorized = bool(getattr(lk_request, "authorized", False))
        credentials = self.service.credentials
        if credentials is not None:
            principal = credentials.principal(
                connection.headers,
                agent_name=self.service.agent_name,
                node_authorized=node_authorized,
                query_params=connection.query_params,
            )
        else:
            principal = Principal(kind="project") if node_authorized else ANONYMOUS

        if self.auth is not None:
            allowed = self.auth(connection)
            if inspect.isawaitable(allowed):
                allowed = await allowed
            if not allowed:
                raise Unauthorized("refused by the endpoint's auth")
        if not self.public and principal.anonymous:
            raise Unauthorized("this endpoint is not public: a project token is required")
        return principal

    def call_context(self, connection: HTTPConnection, principal: Principal) -> CallContext:
        return CallContext(
            principal=principal,
            extensions=Extensions.from_headers(connection.headers),
            resume_offset=connection.headers.get("last-event-id"),
        )

    async def _serve_rpc(self, request: Request) -> Response:
        try:
            principal = await self.authenticate(request)
        except AuthError as e:
            return _auth_response(e)

        call = self.call_context(request, principal)
        echo = call.extensions.header()
        headers = {EXTENSIONS_HEADER: echo} if echo else {}
        try:
            body = await request.json()
        except ValueError as e:
            return JSONResponse(
                build_error_response(None, RpcParseError(message=str(e))), headers=headers
            )
        if isinstance(body, list):
            return JSONResponse(
                build_error_response(
                    None, RpcInvalidRequestError(message="batch requests are not supported")
                ),
                headers=headers,
            )

        method = body.get("method") if isinstance(body, Mapping) else None
        streaming = METHODS.get(str(method or "")) in STREAMING
        responses = self.service.dispatch(body, call)
        # the first response before the head: an upfront error is a JSON error body
        try:
            first = await anext(responses)
        except AuthError as e:
            return _auth_response(e)
        except StopAsyncIteration:
            first = build_error_response(None, RpcInternalError(message="no response"))
        if not streaming or "error" in first:
            return JSONResponse(first, headers=headers)

        async def sse() -> AsyncIterator[bytes]:
            yield _sse_frame(first)
            try:
                async for item in responses:
                    yield _sse_frame(item)
            except AuthError as e:
                yield _sse_frame(
                    build_error_response(body.get("id"), InternalError(message=e.message))
                )

        headers["Cache-Control"] = "no-cache"
        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


def _seconds(
    debounce: timedelta | Mapping[str, timedelta] | None,
) -> float | dict[str, float] | None:
    """The decorator's ``debounce`` as the turn's options carry it."""
    if debounce is None:
        return None
    if isinstance(debounce, timedelta):
        return debounce.total_seconds()
    return {channel: wait.total_seconds() for channel, wait in debounce.items()}


def _sse_frame(item: JsonRpc) -> bytes:
    data = json.dumps(item, separators=(",", ":"))
    prefix = b"event: error\n" if "error" in item else b""
    return prefix + b"data: " + data.encode() + b"\n\n"


def _auth_response(error: AuthError) -> Response:
    return JSONResponse({"error": error.message}, status_code=error.status)


__all__ = ["AGENT_CARD_PATH", "RPC_PATH", "A2AEndpoint", "AuthCallable"]
