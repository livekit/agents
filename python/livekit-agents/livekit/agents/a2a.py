from __future__ import annotations

import asyncio
import dataclasses
import json
import weakref
from collections.abc import Callable, Sequence
from typing import Self, cast

from pydantic import JsonValue
from typing_extensions import Unpack

from . import _core
from ._core import AgentClient, ServerConfig, _core_a2a_agent_card
from ._generated import public_a2a as _public
from ._generated.a2a_card import EndpointDescriptor
from ._generated.a2a_client import AgentCard
from ._generated.a2a_door import DoorConfig
from ._generated.a2a_metadata import RPC_PATH
from ._generated.a2a_remote import DelegationPartition, RemoteEventInputRequired, RemoteEventOutcome
from ._generated.public_a2a import *  # noqa: F403
from ._generated.storage_options import TurnOptions, TurnOptionsKwargs
from ._generated.tool_report import ToolOutcomeInputRequired, ToolReport
from .job import _server_credentials
from .llm.tool_context import InputRequired, RawFunctionTool, Tool, Toolset, function_tool
from .storage import DEFAULT_PARTITION_KEY, Storage, Thread
from .storage.session import A2ASessionContext, _Handler, _TurnHost
from .types import APIConnectOptions
from .voice.events import RunContext


class RemoteAgent:
    """An A2A endpoint as a delegation tool: an endpoint in this codebase, or a URL.

    ``name`` and ``description`` override the endpoint's own. A URL given without both reads
    them, and the tool's input schema, from the remote card when a run first sets the tool up.
    """

    def __init__(
        self,
        target: A2AEndpoint | str,
        *,
        name: str | None = None,
        description: str | None = None,
        token: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        self._token = token
        self._conn_options = conn_options
        self._name = name
        self._description = description
        self._card: AgentCard | None = None
        if isinstance(target, str):
            url = target
            self._url: Callable[[], str] = lambda: url
            if name is not None and description is not None:
                self._card = _core._core_delegation_tool(name, description)
        else:
            self._url = target.delegation_url
            self._card = _core._core_delegation_tool(
                name or target.name, description or target.description
            )
            _referenced.add(target)

    async def _resolved_card(self) -> AgentCard:
        """The tool's card: declared in code, else the remote card's, read once."""
        if self._card is None:
            client = AgentClient(self._url(), token=self._token, conn_options=self._conn_options)
            try:
                card = await client.connect()
            finally:
                await client.aclose()
            self._card = dataclasses.replace(
                card,
                name=self._name or card.name,
                description=self._description or card.description,
            )
        return self._card

    def as_tool(self) -> Toolset:
        return _RemoteTool(self)


# endpoints a RemoteAgent targets; servers in this process warn about any they do not serve
_referenced: weakref.WeakSet[A2AEndpoint] = weakref.WeakSet()


class _ExpertTool(RawFunctionTool):
    def __init__(self, tool: RawFunctionTool, url: Callable[[], str]) -> None:
        super().__init__(tool._func, tool.info)
        self._url = url

    @property
    def endpoint(self) -> str:
        return self._url()


class _RemoteTool(Toolset):
    def __init__(self, agent: RemoteAgent) -> None:
        self._agent = agent
        self._tool: _ExpertTool | None = None
        if agent._card is not None:
            self._define(agent._card)
        super().__init__(id=agent._card.name if agent._card is not None else agent._url())

    def _define(self, card: AgentCard) -> None:
        tool = function_tool(
            self._invoke,
            raw_schema={
                "name": card.name,
                "description": card.description,
                "parameters": card.input_schema,
            },
        )
        self._tool = _ExpertTool(tool, self._agent._url)

    async def setup(self) -> Self:
        if self._tool is None:
            self._define(await self._agent._resolved_card())
        return await super().setup()

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        # a URL's card is read when a run first sets the tool up
        return [self._tool] if self._tool is not None else []

    async def _invoke(self, ctx: RunContext, raw_arguments: dict[str, object]) -> ToolReport:
        agent = self._agent
        assert agent._card is not None and self._tool is not None
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
        # with no token of its own, the call shares its thread under a grant minted for it
        thread = ctx._run.thread if ctx._run is not None else ctx._call().thread
        if thread is not None:
            from .voice.agent_run import door_hooks

            hooks = ctx._run.hooks if ctx._run is not None else door_hooks()
            chain = list(hooks.delegation_chain) if hooks is not None else []
            chain.append(
                DelegationPartition(
                    storage_location=thread._storage._core.location,
                    storage_project=thread._storage._core.project_id,
                    thread_id=thread.id,
                    partition_key=thread._partition_key,
                )
            )
            interaction.set_partition_chain(chain)
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
            # stop in this turn, then wait for the remote task's cancellation and client close
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
    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def _core_progress(self, text: str | None, extra: dict[str, object]) -> None:
        self._ctx.function_call.extra.update(_json_object(extra))
        if text is not None:
            await self._ctx.update(text)

    async def _core_delegated(self) -> None:
        await self._ctx.update(_DELEGATED)

    async def _core_ask(self, question: str, verbatim: bool) -> str:
        ctx = self._ctx
        assert ctx._session is not None
        ctx._expert_answer = asyncio.get_running_loop().create_future()
        ctx._session._settled_by_events(ctx._expert_answer)
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
    """Optional parameters of ``a2a_session``."""

    storage: Storage | None
    """This endpoint's storage when it is not the server's."""
    partition_key: str
    """Saved history and state within a thread; defaults to ``default``."""
    version: str | None
    """The card's version; by default the deployment the worker registered under, else
    ``LIVEKIT_AGENT_VERSION`` for a local worker."""


def a2a_session(
    *, endpoint: str, description: str, **options: Unpack[A2ASessionOptions]
) -> Callable[[_Handler], A2AEndpoint]:
    """Declare an A2A text endpoint for ``AgentServer.add()`` to serve.

    Args:
        endpoint: The last path segment of the endpoint's URL.
        description: What another agent reads to decide whether to call it.
    """
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
        partition_key: str = DEFAULT_PARTITION_KEY,
        **turn: Unpack[TurnOptionsKwargs],
    ) -> None:
        if not endpoint or "/" in endpoint:
            raise ValueError("endpoint is one path segment, the last of the endpoint's URL")
        if not isinstance(partition_key, str) or not partition_key:
            raise ValueError("partition_key must be a non-empty string")
        self.handler = handler
        self.name = endpoint
        self.description = description
        self.turn_options = TurnOptions(**turn)
        self.version = version or ""
        self.partition_key = partition_key
        self._storage = storage
        self._native: _core._core_A2ADoor | None = None
        self._url: str | None = None
        self._config: ServerConfig | None = None

    def bind(self, *, storage: Storage, config: ServerConfig) -> None:
        """Attach the endpoint to a server's storage and configuration."""
        self._storage = self._storage or storage
        self._config = config

    def _card_version(self) -> str:
        return self.version or (self._config.card_version if self._config is not None else "")

    def _door(self) -> _core._core_A2ADoor:
        if self._config is None or self._storage is None:
            raise RuntimeError(f"endpoint {self.name!r} is not attached to a server yet")
        if self._native is None:
            partition_key = self.partition_key
            self._native = _core._core_A2ADoor(
                self._storage._core,
                DoorConfig(
                    partition_key=partition_key,
                    holder=self._storage.holder,
                    endpoint=EndpointDescriptor(
                        name=self.name,
                        description=self.description,
                        version=self._card_version(),
                        url=self.url or f"/{self.name}/v1",
                    ),
                    options=self.turn_options,
                ),
                _TurnHost(
                    self.handler, thread=Thread(self._storage, "", partition_key=partition_key)
                ),
                server=self._config,
            )
        return self._native

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
        """The agent card as JSON, as the endpoint serves it.

        Args:
            base_url: Where the endpoint is reached when its own ``url`` is unknown.
        """
        url = self.url
        cloud = url if url and not url.startswith("/") else None
        root = (cloud or base_url or f"/{self.name}").rstrip("/")
        descriptor = EndpointDescriptor(
            name=self.name,
            description=self.description,
            version=self._card_version(),
            url=root + RPC_PATH,
        )
        return cast("dict[str, JsonValue]", json.loads(_core_a2a_agent_card(descriptor)))


__all__ = [
    "a2a_session",
    "A2ASessionContext",
    "RemoteAgent",
    "A2AEndpoint",
]
__all__ += _public.__all__
