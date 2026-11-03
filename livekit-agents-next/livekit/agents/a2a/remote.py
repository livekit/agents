"""Remote tools: host callbacks around the native A2A client and interaction decisions."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from .. import _core
from .._generated.a2a_remote import (
    RemoteEvent,
    RemoteEventFollow,
    RemoteEventInputRequired,
    RemoteEventOutcome,
    RemoteEventProgress,
)
from .._generated.tool_report import ToolOutcomeInputRequired, ToolReport
from ..job import _server_credentials
from ..llm.tool_context import InputRequired, RawFunctionTool, Tool, Toolset, function_tool
from ..telemetry._transport import capture_traceparent
from ..types import APIConnectOptions
from ..voice.events import RunContext
from .client import AgentClient, Reply

if TYPE_CHECKING:
    from .endpoint import A2AEndpoint


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
            reply = await client.send(message, traceparent=capture_traceparent())
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
                    reply = await client._core_subscribe(
                        event.task_id, event.thread_id, capture_traceparent()
                    )
                elif isinstance(event, RemoteEventInputRequired) and ctx._session is not None:
                    answer = await _ask_the_person(ctx, event)
                    reply = await client.send(
                        interaction.answer(event.request_id, answer),
                        traceparent=capture_traceparent(),
                    )
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


__all__ = ["RemoteAgent"]


def _json_object(value: dict[str, object]) -> dict[str, JsonValue]:
    """The core's JSON object; its stub types every JSON output as ``object``."""
    return cast("dict[str, JsonValue]", value)
