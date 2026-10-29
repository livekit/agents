"""The text door over HTTP: ``@server.a2a_session`` as an ASGI application (plan 7.3).

One endpoint is one :class:`A2AEndpoint`: its handler, card, visibility, policies and the
storage its lane lives in. Bound to a server it serves its :class:`A2AService` (``methods.py``)
as the card at ``/.well-known/agent-card.json`` and JSON-RPC at ``POST /v1``, with server-sent
events for the streams; the caller is checked by ``auth.py`` and the developer's ``auth``.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import timedelta
from typing import TypedDict

from pydantic import JsonValue
from starlette.applications import Starlette
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from typing_extensions import Unpack

from .._core import ServerConfig, endpoint_url_base
from .._generated.storage_options import Policy, TurnOptions
from ..storage.backends import Storage
from ..storage.context import _Handler
from .auth import PROJECT, AuthError, Credentials, Principal, Unauthorized
from .extension import EXTENSIONS_HEADER, Extensions, agent_card, to_json
from .methods import A2AService

AGENT_CARD_PATH = "/.well-known/agent-card.json"
RPC_PATH = "/v1"
"""Where the binding's methods live under the endpoint, and what the card's URL points at."""

AuthCallable = Callable[[HTTPConnection], "bool | Awaitable[bool]"]


class A2ASessionOptions(TypedDict, total=False):
    """``a2a_session``'s optional parameters (docs/server.md)."""

    auth: AuthCallable | None
    """The developer's own check, a callable taking the request."""
    inbox: Policy | Mapping[str, Policy] | None
    """The turn policy for a message that arrives while one runs, or a map by channel."""
    debounce: timedelta | Mapping[str, timedelta] | None
    """How long to wait for more of a burst, or a map by channel."""
    storage: Storage | None
    """This endpoint's storage when it is not the server's."""
    version: str | None
    """The card's version; by default the deployment the worker registered under, else
    ``LIVEKIT_AGENT_VERSION`` for a local worker."""


def a2a_session(
    *, endpoint: str, description: str, **options: Unpack[A2ASessionOptions]
) -> Callable[[_Handler], A2AEndpoint]:
    """The text door as a decorator with no handler: the door is an object a module defines and
    ``AgentServer.add()`` serves (``server.a2a_session(...)`` does both). ``endpoint`` is the
    last path segment of its URL and ``description`` what another agent reads to decide whether
    to call. It writes the worker's lane; an ``AgentHandler(lane=)`` door writes its own."""
    return lambda handler: A2AEndpoint(
        handler, endpoint=endpoint, description=description, **options
    )


class A2AEndpoint:
    """What ``@server.a2a_session`` returns: the handler, plus its name, card, visibility and a
    ``url`` resolved at runtime. Pass it wherever a delegate or a client wants an endpoint."""

    def __init__(
        self,
        handler: _Handler,
        *,
        endpoint: str,
        description: str,
        auth: AuthCallable | None = None,
        inbox: Policy | Mapping[str, Policy] | None = None,
        debounce: timedelta | Mapping[str, timedelta] | None = None,
        storage: Storage | None = None,
        version: str | None = None,
        lane: str | None = None,
    ) -> None:
        if not endpoint or "/" in endpoint:
            raise ValueError("endpoint is one path segment, the last of the endpoint's URL")
        self.handler = handler
        self.name = endpoint
        self.description = description
        self.auth = auth
        self.turn_options = TurnOptions(
            policy=dict(inbox) if isinstance(inbox, Mapping) else inbox,
            debounce=_seconds(debounce),
        )
        self.version = version or ""
        self.lane = lane
        self._storage = storage
        self._service: A2AService | None = None
        self._url: str | None = None
        self._url_base: Callable[[], str | None] | None = None
        self._app: Starlette | None = None
        self._deployment = ""

    # -- binding to a server --

    def bind(self, *, storage: Storage | None, config: ServerConfig) -> A2AService:
        """Attach the endpoint to a server's storage and options: its agent name (the lane,
        unless the door writes its own), credentials and URL."""
        chosen = self._storage or storage
        if chosen is None:
            raise ValueError(
                f"a2a_session(endpoint={self.name!r}) needs a storage: AgentServer(storage=...) "
                "or a2a_session(storage=...)"
            )

        self._deployment = config.deployment
        # the card's version is the deployment's; a local worker, registered under none, falls
        # back to LIVEKIT_AGENT_VERSION
        self.version = (
            self.version or config.deployment or os.environ.get("LIVEKIT_AGENT_VERSION") or "1.0.0"
        )
        own = self.lane
        self._service = A2AService(
            self,
            storage=chosen,
            agent_name=lambda: config.agent_name,
            credentials=(
                Credentials(config.api_key, config.api_secret)
                if config.api_key and config.api_secret
                else None
            ),
            lane=(lambda: own) if own else None,
        )
        self._url_base = lambda: endpoint_url_base(
            config.ws_url, config.agent_name, config.deployment
        )
        return self._service

    @property
    def service(self) -> A2AService:
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
        try:
            await self.authenticate(request)
        except AuthError as error:
            return _auth_response(error)
        # the mount's path is the scope's root_path; base_url carries the host alone
        base = str(request.base_url).rstrip("/") + str(request.scope.get("root_path") or "")
        return JSONResponse(self.card(base.rstrip("/")))

    async def authenticate(self, connection: HTTPConnection) -> Principal:
        """The caller of a request: the preamble's grant and verdict, a bearer token when no
        node is in front, the developer's ``auth``.

        Raises ``Unauthorized`` when ``auth`` refuses or the token does not verify.
        """
        if self.auth is not None:
            allowed = self.auth(connection)
            if inspect.isawaitable(allowed):
                allowed = await allowed
            if not allowed:
                raise Unauthorized("refused by the endpoint's auth")
        lk_request = getattr(connection.state, "lk_request", None)
        node_authorized = bool(getattr(lk_request, "authorized", False))
        credentials = self.service.credentials
        if credentials is not None:
            principal = credentials._principal(
                connection.headers,
                agent_name=self.service.agent_name,
                node_authorized=node_authorized,
                deployment=self._deployment,
            )
        else:
            if not node_authorized:
                raise Unauthorized("a LiveKit bearer token is required")
            principal = PROJECT

        return principal

    async def _serve_rpc(self, request: Request) -> Response:
        try:
            principal = await self.authenticate(request)
        except AuthError as error:
            return _auth_response(error)

        headers: dict[str, str] = {}
        if negotiated := Extensions.from_headers(request.headers).header():
            headers[EXTENSIONS_HEADER] = negotiated
        native = await self.service.native()
        body = bytearray()
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > native.max_payload_bytes:
                break
            body.extend(chunk)
        responses = await self.service.dispatch(
            bytes(body),
            principal,
            version=request.headers.get("a2a-version"),
            offset=request.headers.get("last-event-id"),
            body_size=received,
        )
        try:
            first = await responses.next()
        except AuthError as error:
            await responses.aclose()
            return _auth_response(error)
        if first is None:
            await responses.aclose()
            raise RuntimeError("the native A2A door produced no response")
        if not responses.streaming or first.is_error:
            await responses.aclose()
            return Response(first.json, media_type="application/json", headers=headers)

        async def sse() -> AsyncIterator[bytes]:
            try:
                yield first.sse.encode()
                while (frame := await responses.next()) is not None:
                    yield frame.sse.encode()
            finally:
                await responses.aclose()

        return StreamingResponse(
            sse(), media_type="text/event-stream", headers={**headers, "Cache-Control": "no-cache"}
        )


def _seconds(
    debounce: timedelta | Mapping[str, timedelta] | None,
) -> float | dict[str, float] | None:
    """The decorator's ``debounce`` as the turn's options carry it."""
    if debounce is None:
        return None
    if isinstance(debounce, timedelta):
        return debounce.total_seconds()
    return {channel: wait.total_seconds() for channel, wait in debounce.items()}


def _auth_response(error: AuthError) -> Response:
    return JSONResponse({"error": error.message}, status_code=error.status)


__all__ = ["AGENT_CARD_PATH", "RPC_PATH", "A2AEndpoint", "AuthCallable"]
