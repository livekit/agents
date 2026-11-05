"""The text door: A2A endpoints, ``AgentClient``, ``RemoteAgent`` and endpoint tokens."""

from __future__ import annotations

import asyncio
import json
import os
import weakref
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import TypedDict, cast

from google.protobuf import json_format, struct_pb2
from google.protobuf.message import Message as ProtoMessage
from pydantic import JsonValue
from typing_extensions import Unpack

from . import _core
from ._core import (
    AgentClient,
    Reply,
    ServerConfig,
    _core_a2a_agent_card,
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
from ._generated.a2a_metadata import (
    AGENT_CARD_PATH as AGENT_CARD_PATH,
    EXTENSION_URI as EXTENSION_URI,
    GRANT_HEADER,
    RPC_PATH,
)
from ._generated.a2a_remote import RemoteEventInputRequired, RemoteEventOutcome
from ._generated.storage_options import Policy, TurnOptions
from ._generated.tool_report import ToolOutcomeInputRequired, ToolReport
from .job import _server_credentials
from .llm.tool_context import InputRequired, RawFunctionTool, Tool, Toolset, function_tool
from .storage import Storage, Thread
from .storage.session import A2ASessionContext, _Handler, _TurnHost
from .types import APIConnectOptions
from .voice.events import RunContext


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


Metadata = dict[str, JsonValue]


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


class A2AService:
    def __init__(
        self,
        endpoint: A2AEndpoint,
        *,
        storage: Storage,
        agent_name: Callable[[], str],
        lane: Callable[[], str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._storage = storage
        self._agent_name = agent_name
        self._lane = lane
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

    async def aclose(self) -> None:
        if self._door is not None:
            await self._door.aclose()


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
        try:
            event = await interaction.run(
                _Delegation(ctx),
                raw_arguments,
                agent._card.structured_input,
                history.items,
                token,
                shared,
                agent._conn_options,
                ctx._session is not None,
            )
        except asyncio.CancelledError:
            # stopped in this turn, then its reply (cancelling the remote task) and client closed
            interaction.cancelled()
            ctx.function_call.extra.update(_json_object(interaction.extra))
            await interaction.settled()
            raise
        if isinstance(event, RemoteEventInputRequired):
            transition = InputRequired(
                event.question, tool=self._tool, verbatim=event.verbatim_full
            )
            return ToolReport(outcome=event.outcome, transition=transition)
        assert isinstance(event, RemoteEventOutcome)
        return ToolReport(outcome=event.outcome)


_DELEGATED = "The request is with the expert: its progress, questions and result reach the person."


class _Delegation:
    """What ``_core_RemoteInteraction.run`` reaches on the host for one tool call: its progress
    and retained state, and on a call the person."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def _core_progress(self, text: str | None, extra: dict[str, object]) -> None:
        self._ctx.function_call.extra.update(_json_object(extra))
        if text is not None:
            await self._ctx.update(text)

    async def _core_delegated(self) -> None:
        # The live session owns the interaction from here: the call waits on the expert while
        # the conversation goes on.
        await self._ctx.update(_DELEGATED)

    async def _core_ask(self, question: str, verbatim: bool) -> str:
        """The expert's question goes to the live session, which presents it when the floor
        permits and forwards the person's reply here."""
        ctx = self._ctx
        assert ctx._session is not None
        ctx._expert_answer = asyncio.get_running_loop().create_future()
        outcome = ToolOutcomeInputRequired(question=question, verbatim=verbatim)
        try:
            ctx._tool_update(ToolReport(outcome=outcome))
            return await ctx._expert_answer
        finally:
            ctx._expert_answer = None


def _json_object(value: dict[str, object]) -> dict[str, JsonValue]:
    """The core's JSON object; its stub types every JSON output as ``object``."""
    return cast("dict[str, JsonValue]", value)


class A2ASessionOptions(TypedDict, total=False):
    """``a2a_session``'s optional parameters (docs/server.md)."""

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

    # -- binding to a server --

    def bind(self, *, storage: Storage, config: ServerConfig) -> A2AService:
        """Attach the endpoint to a server's storage and options: its agent name (the lane,
        unless the door writes its own) and URL."""
        chosen = self._storage or storage
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
        """The agent card as JSON, as the endpoint serves it (plan 7.3.1). ``base_url`` is where
        the endpoint is reached when its own ``url`` is unknown."""
        url = self.url
        cloud = url if url and not url.startswith("/") else None
        root = (cloud or base_url or f"/{self.name}").rstrip("/")
        descriptor = EndpointDescriptor(
            name=self.name, description=self.description, version=self.version, url=root + RPC_PATH
        )
        return cast("dict[str, JsonValue]", json.loads(_core_a2a_agent_card(descriptor)))


def _seconds(
    debounce: timedelta | Mapping[str, timedelta] | None,
) -> float | dict[str, float] | None:
    """The decorator's ``debounce`` as the turn's options carry it."""
    if debounce is None:
        return None
    if isinstance(debounce, timedelta):
        return debounce.total_seconds()
    return {channel: wait.total_seconds() for channel, wait in debounce.items()}


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
