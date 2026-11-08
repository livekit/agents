"""The text door: A2A endpoints, ``AgentClient``, ``RemoteAgent`` and endpoint tokens."""

from __future__ import annotations

import asyncio
import json
import weakref
from collections.abc import Callable, Sequence
from typing import cast

from pydantic import JsonValue
from typing_extensions import Unpack

from . import _core
from ._core import ServerConfig, _core_a2a_agent_card
from ._generated import public_a2a as _public
from ._generated.a2a_card import EndpointDescriptor
from ._generated.a2a_door import DoorConfig
from ._generated.a2a_metadata import RPC_PATH
from ._generated.a2a_remote import RemoteEventInputRequired, RemoteEventOutcome
from ._generated.public_a2a import *  # noqa: F403
from ._generated.storage_options import TurnOptions, TurnOptionsKwargs
from ._generated.tool_report import ToolOutcomeInputRequired, ToolReport
from .job import _server_credentials
from .llm.tool_context import InputRequired, RawFunctionTool, Tool, Toolset, function_tool
from .storage import Storage, Thread
from .storage.session import A2ASessionContext, _Handler, _TurnHost
from .types import APIConnectOptions
from .voice.events import RunContext


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


class A2ASessionOptions(TurnOptionsKwargs, total=False):
    """``a2a_session``'s optional parameters (docs/server.md)."""

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
        storage: Storage | None = None,
        version: str | None = None,
        lane: str | None = None,
        **turn: Unpack[TurnOptionsKwargs],
    ) -> None:
        if not endpoint or "/" in endpoint:
            raise ValueError("endpoint is one path segment, the last of the endpoint's URL")
        self.handler = handler
        self.name = endpoint
        self.description = description
        self.turn_options = TurnOptions(**turn)
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
        self.version = self.version or config.card_version
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


__all__ = [
    "a2a_session",
    "A2ASessionContext",
    "RemoteAgent",
    "A2AEndpoint",
]
__all__ += _public.__all__
