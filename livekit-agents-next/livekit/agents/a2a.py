"""The text door: A2A endpoints, ``AgentClient``, ``RemoteAgent`` and endpoint tokens."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import os
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, TypedDict, cast

from a2a.types import a2a_pb2 as pb
from google.protobuf import json_format, struct_pb2
from google.protobuf.message import Message as ProtoMessage
from pydantic import JsonValue
from starlette.applications import Starlette
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from typing_extensions import Unpack

from . import _core
from ._core import (
    AgentClient,
    Reply,
    ServerConfig,
    _core_a2a_agent_card,
    _core_a2a_extensions_header,
    _core_Credentials,
)
from ._generated.a2a_auth import AgentThreadGrant, Principal
from ._generated.a2a_card import EndpointDescriptor
from ._generated.a2a_client import (
    Artifact,
    DataPart,
    FilePart,
    Message,
    MessageMetadata,
    TaskUpdate,
    TextPart,
)
from ._generated.a2a_door import DoorConfig
from ._generated.a2a_metadata import EXTENSION_URI, EXTENSIONS_HEADER
from ._generated.a2a_remote import (
    RemoteEvent,
    RemoteEventFollow,
    RemoteEventInputRequired,
    RemoteEventOutcome,
    RemoteEventProgress,
)
from ._generated.storage_options import Policy, TurnOptions
from ._generated.tool_report import ToolOutcomeInputRequired, ToolReport
from .job import _server_credentials
from .llm.tool_context import InputRequired, RawFunctionTool, Tool, Toolset, function_tool
from .storage import Storage, Thread
from .storage.session import A2ASessionContext, _Handler, _TurnHost
from .types import APIConnectOptions
from .voice.events import RunContext

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


Metadata = dict[str, JsonValue]


@dataclass(frozen=True)
class Extensions:
    """Whether a request activated the extension: read from ``A2A-Extensions``, echoed back.

    A stock client that sends nothing gets ``Extensions()``: plain A2A, text only. The
    server-owned keys (``interrupted``, ``mergedInto``) are written either way, since
    ``metadata`` is core A2A; the typed chat context part follows the extension.
    """

    livekit: bool = False

    @classmethod
    def from_header(cls, value: str | None) -> Extensions:
        return cls(livekit=_core_a2a_extensions_header(value) is not None)

    @classmethod
    def from_headers(cls, headers: Mapping[str, str] | None) -> Extensions:
        """From a Starlette ``Headers`` or any mapping with a case-insensitive ``get``."""
        if headers is None:
            return cls()

        value = headers.get(EXTENSIONS_HEADER)
        if value is None:
            value = headers.get(EXTENSIONS_HEADER.lower())
        return cls.from_header(value)

    def header(self) -> str | None:
        """The echo: the URI when activated, ``None`` when not."""
        return EXTENSION_URI if self.livekit else None


# -- protobuf helpers --


def struct(value: Mapping[str, JsonValue]) -> struct_pb2.Struct:
    """A metadata field: A2A types them ``Struct``."""
    return json_format.ParseDict(dict(value), struct_pb2.Struct())


def value(data: JsonValue) -> struct_pb2.Value:
    """A data part's payload: A2A types it ``Value``."""
    holder = json_format.ParseDict({"data": data}, struct_pb2.Struct())
    return holder.fields["data"]


def as_dict(field: ProtoMessage) -> Metadata:
    """A ``Struct`` or ``Value`` back as JSON; an empty dict for anything that is not a map."""
    out = json_format.MessageToDict(field)
    return out if isinstance(out, dict) else {}


def to_json(message: ProtoMessage) -> dict[str, JsonValue]:
    """A proto message as the JSON the HTTP binding carries (camelCase, enums by name)."""
    return json_format.MessageToDict(message, preserving_proto_field_name=False)


# -- the agent card --


def agent_card(
    *,
    url: str,
    name: str,
    description: str,
    version: str = "1.0.0",
) -> pb.AgentCard:
    """Adapt the native card to the Python A2A SDK's protobuf representation."""
    descriptor = EndpointDescriptor(name=name, description=description, version=version, url=url)
    return json_format.Parse(_core_a2a_agent_card(descriptor), pb.AgentCard())


if TYPE_CHECKING:
    pass


class A2AService:
    def __init__(
        self,
        endpoint: A2AEndpoint,
        *,
        storage: Storage,
        agent_name: Callable[[], str],
        credentials: Credentials | None,
        lane: Callable[[], str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._storage = storage
        self._agent_name = agent_name
        self._lane = lane
        self.credentials = credentials
        self._door: _core._core_A2ADoor | None = None

    @property
    def agent_name(self) -> str:
        return self._agent_name()

    @property
    def lane(self) -> str:
        return (self._lane() if self._lane is not None else None) or self.agent_name

    def _open(self) -> _core._core_A2ADoor:
        endpoint = self._endpoint
        host = _TurnHost(
            endpoint.handler,
            thread=Thread(self._storage, "", agent_name=self.lane),
        )
        return _core._core_A2ADoor(
            self._storage._core,
            DoorConfig(
                lane=self.lane,
                holder=self._storage.holder,
                endpoint=EndpointDescriptor(
                    name=endpoint.name,
                    description=endpoint.description,
                    version=endpoint.version,
                    url=endpoint.url or f"/{endpoint.name}/v1",
                ),
                options=endpoint.turn_options,
            ),
            host,
        )

    async def native(self) -> _core._core_A2ADoor:
        if self._door is None:
            self._door = self._open()
        return self._door

    async def dispatch(
        self,
        body: bytes,
        principal: Principal,
        *,
        version: str | None = None,
        offset: str | None = None,
        body_size: int | None = None,
        traceparent: str | None = None,
    ) -> _core._core_A2AResponse:
        return await (await self.native()).dispatch(
            body,
            principal,
            version=version,
            offset=offset,
            body_size=body_size,
            host_context=(asyncio.get_running_loop(), contextvars.copy_context()),
            traceparent=traceparent,
        )

    async def aclose(self) -> None:
        if self._door is not None:
            await self._door.aclose()


if TYPE_CHECKING:
    pass


class RemoteAgent:
    """An A2A endpoint as a delegation tool: an endpoint in this codebase, or a URL with the
    tool's name and description."""

    def __init__(
        self,
        target: A2AEndpoint | str,
        *,
        name: str | None = None,
        description: str | None = None,
        token: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        if isinstance(target, str):
            url = target
            self._url: Callable[[], str] = lambda: url
        else:
            name = name or target.name
            description = description or target.description
            self._url = target.delegation_url
            _referenced.add(target)
        self._card = _core._core_delegation_tool(name or "", description or "")
        self._token = token
        self._conn_options = conn_options

    def as_tool(self) -> Toolset:
        return _RemoteTool(self)


# what the servers in this process warn about when they do not serve it
_referenced: weakref.WeakSet[A2AEndpoint] = weakref.WeakSet()


class _ExpertTool(RawFunctionTool):
    """The remote tool: its endpoint and name are its configured identity as an expert."""

    def __init__(self, tool: RawFunctionTool, url: Callable[[], str]) -> None:
        super().__init__(tool._func, tool.info)
        self._url = url

    @property
    def endpoint(self) -> str:
        return self._url()


class _RemoteTool(Toolset):
    def __init__(self, agent: RemoteAgent) -> None:
        self._agent = agent
        card = agent._card
        tool = function_tool(
            self._invoke,
            raw_schema={
                "name": card.name,
                "description": card.description,
                "parameters": card.input_schema,
            },
        )
        self._tool = _ExpertTool(tool, agent._url)
        super().__init__(id=card.name)

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return [self._tool]

    async def _invoke(self, ctx: RunContext, raw_arguments: dict[str, object]) -> ToolReport:
        agent = self._agent
        url = agent._url()
        if ctx._session is not None:
            history = ctx._session.history
        else:
            assert ctx._run is not None
            history = ctx._run.chat_ctx
        project = _server_credentials()
        interaction = _core._core_RemoteInteraction(
            url,
            agent._card.name,
            ctx.function_call,
            history.items,
            *project,
        )
        # with no token of its own, the call shares its thread with the endpoint under a grant
        # minted for that thread (docs/internals/storage.md "Two brains, one thread")
        thread = ctx._run.thread if ctx._run is not None else ctx._call().thread
        thread_id = thread.id if agent._token is None and thread is not None else None
        token = _core._core_remote_token(url, *project, agent._token, thread_id)
        shared = thread_id if agent._token is None and token is not None else None
        client = AgentClient(url, token=token, conn_options=agent._conn_options)
        reply: Reply | None = None
        try:
            await client.connect()
            message = interaction.message(
                raw_arguments,
                agent._card.structured_input,
                client.offers_extension,
                history.items,
                shared,
                token if shared else None,
            )
            reply = await client.send(message)
            if ctx._session is not None:
                # The live session owns the interaction from here: the call waits on the expert
                # while the conversation goes on.
                await ctx.update(_DELEGATED)
            while True:
                event: RemoteEvent | None = None
                async for update in reply:
                    event = interaction.update(update)
                    ctx.function_call.extra.update(_json_object(interaction.extra))
                    if isinstance(event, RemoteEventProgress):
                        await ctx.update(event.text)
                    elif isinstance(
                        event, RemoteEventInputRequired | RemoteEventOutcome | RemoteEventFollow
                    ):
                        break
                await reply.aclose()
                if isinstance(event, RemoteEventFollow):
                    reply = await client._core_subscribe(event.task_id, event.thread_id)
                elif isinstance(event, RemoteEventInputRequired) and ctx._session is not None:
                    answer = await _ask_the_person(ctx, event)
                    reply = await client.send(interaction.answer(event.request_id, answer))
                elif isinstance(event, RemoteEventInputRequired):
                    transition = InputRequired(
                        event.question, tool=self._tool, verbatim=event.verbatim_full
                    )
                    return ToolReport(outcome=event.outcome, transition=transition)
                elif isinstance(event, RemoteEventOutcome):
                    return ToolReport(outcome=event.outcome)
                else:
                    raise RuntimeError("the remote interaction ended without an outcome")
        except asyncio.CancelledError:
            interaction.cancelled()
            ctx.function_call.extra.update(_json_object(interaction.extra))
            raise
        finally:
            if reply is not None:
                await reply.aclose()
            await client.aclose()


_DELEGATED = "The request is with the expert: its progress, questions and result reach the person."


async def _ask_the_person(ctx: RunContext, event: RemoteEventInputRequired) -> str:
    """The expert's question goes to the live session, which presents it when the floor permits
    and forwards the person's reply here."""
    assert ctx._session is not None
    ctx._expert_answer = asyncio.get_running_loop().create_future()
    question = ToolOutcomeInputRequired(question=event.question, verbatim=event.verbatim_full)
    try:
        ctx._tool_update(ToolReport(outcome=question))
        return await ctx._expert_answer
    finally:
        ctx._expert_answer = None


def _json_object(value: dict[str, object]) -> dict[str, JsonValue]:
    """The core's JSON object; its stub types every JSON output as ``object``."""
    return cast("dict[str, JsonValue]", value)


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
        self._config: ServerConfig | None = None
        self._app: Starlette | None = None
        self._deployment = ""

    # -- binding to a server --

    def bind(self, *, storage: Storage, config: ServerConfig) -> A2AService:
        """Attach the endpoint to a server's storage and options: its agent name (the lane,
        unless the door writes its own), credentials and URL."""
        chosen = self._storage or storage
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
        self._config = config
        return self._service

    @property
    def service(self) -> A2AService:
        if self._service is None:
            raise RuntimeError(f"endpoint {self.name!r} is not attached to a server yet")
        return self._service

    @property
    def url(self) -> str | None:
        """``https://<project>.livekit.cloud/agents/<agent>/<deployment>/<endpoint>`` when
        LiveKit Cloud routes the endpoint; its path when the worker serves it itself."""
        if self._url:
            return self._url
        return self._config.endpoint_url(self.name) if self._config is not None else None

    @url.setter
    def url(self, value: str | None) -> None:
        self._url = value

    def delegation_url(self) -> str:
        """Where the worker serving this endpoint delegates to it: Cloud, in the worker's own
        deployment, or the worker's public listener."""
        if self._config is None:
            raise RuntimeError(
                f"endpoint {self.name!r} is not served by any AgentServer in this process: "
                "add it with server.add(...), or use RemoteAgent(url, name=, description=)"
            )
        return self._config.delegation_url(self.name)

    def card(self, base_url: str | None = None) -> dict[str, JsonValue]:
        """The agent card as JSON (plan 7.3.1). ``base_url`` is where the endpoint is reached
        when its own ``url`` is unknown, the request's base URL plus the mount path."""
        url = self.url
        cloud = url if url and not url.startswith("/") else None
        root = (cloud or base_url or f"/{self.name}").rstrip("/")
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
        # served by the worker: its path on the address the request reached; otherwise the
        # mount's path is the scope's root_path and base_url carries the host alone
        url = self.url
        if url and url.startswith("/"):
            base = f"{request.url.scheme}://{request.url.netloc}{url}"
        else:
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
        if getattr(lk_request, "unauthenticated", False):
            return _core._core_unauthenticated_principal()
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
        # the caller's W3C trace context: the turn this request starts continues its trace
        responses = await self.service.dispatch(
            bytes(body),
            principal,
            version=request.headers.get("a2a-version"),
            offset=request.headers.get("last-event-id"),
            body_size=received,
            traceparent=request.headers.get("traceparent"),
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


__all__ = [
    "a2a_session",
    "A2ASessionContext",
    "AgentClient",
    "Artifact",
    "DataPart",
    "FilePart",
    "Message",
    "MessageMetadata",
    "RemoteAgent",
    "TextPart",
    "AgentThreadGrant",
    "Credentials",
    "Forbidden",
    "Principal",
    "Reply",
    "TaskUpdate",
    "Unauthorized",
    "AuthError",
    "A2AEndpoint",
]
