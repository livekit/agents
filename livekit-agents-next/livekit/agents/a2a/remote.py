"""Remote tools: host callbacks around the native A2A client and interaction decisions."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .. import _core
from .._core_view import to_core_item, to_core_items
from .._generated.a2a_client import AgentCard
from .._generated.a2a_remote import (
    RemoteEvent,
    RemoteEventFollow,
    RemoteEventInputRequired,
    RemoteEventOutcome,
    RemoteEventProgress,
)
from .._generated.tool_report import ToolOutcomeInputRequired, ToolReport
from ..llm.tool_context import InputRequired, RawFunctionTool, Tool, Toolset, function_tool
from ..types import APIConnectOptions
from ..voice.events import RunContext
from .client import AgentClient, Reply


class RemoteAgent:
    """An A2A endpoint exposed through its advertised tool definition."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        self._url = url
        self._token = token
        self._conn_options = conn_options

    def as_tool(self) -> Toolset:
        return _RemoteTool(self._url, self._token, self._conn_options)


class _ExpertTool(RawFunctionTool):
    """The remote tool: its endpoint and name are its configured identity as an expert."""

    def __init__(self, tool: RawFunctionTool, endpoint: str) -> None:
        super().__init__(tool._func, tool.info)
        self.endpoint = endpoint


class _RemoteTool(Toolset):
    def __init__(self, url: str, token: str | None, conn_options: APIConnectOptions | None) -> None:
        self._url = url
        self._token = token
        self._conn_options = conn_options
        self._card: AgentCard | None = None
        self._tool: _ExpertTool | None = None
        super().__init__(id=url)

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return [self._tool] if self._tool is not None else []

    def _client(self, token: str | None) -> AgentClient:
        return AgentClient(self._url, token=token, conn_options=self._conn_options)

    async def setup(self) -> _RemoteTool:
        if self._tool is None:
            client = self._client(_core._core_remote_token(self._url, self._token))
            try:
                self._card = await client.connect()
            finally:
                await client.aclose()
            self._id = self._card.name
            tool = function_tool(
                self._invoke,
                raw_schema={
                    "name": self._card.name,
                    "description": self._card.description,
                    "parameters": self._card.input_schema,
                },
            )
            self._tool = _ExpertTool(tool, self._url)
        return self

    async def _invoke(self, ctx: RunContext, raw_arguments: dict[str, object]) -> ToolReport:
        assert self._card is not None and self._tool is not None
        if ctx._session is not None:
            history = ctx._session.history
        else:
            assert ctx._run is not None
            history = ctx._run.chat_ctx
        interaction = _core._core_RemoteInteraction(
            self._url,
            self._card.name,
            to_core_item(ctx.function_call),
            to_core_items(history.items),
        )
        # with no token of its own, the call shares its thread with the endpoint under a grant
        # minted for that thread (docs/internals/storage.md "Two brains, one thread")
        thread_id = ctx.thread.id if self._token is None else None
        token = _core._core_remote_token(self._url, self._token, thread_id)
        shared = thread_id if self._token is None and token is not None else None
        client = self._client(token)
        reply: Reply | None = None
        try:
            await client.connect()
            message = interaction.message(
                raw_arguments,
                self._card.structured_input,
                client.offers_extension,
                to_core_items(history.items),
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
                    ctx.function_call.extra.update(interaction.extra)
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
            ctx.function_call.extra.update(interaction.extra)
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
        ctx._session._core.tool_update(ctx.function_call.call_id, ToolReport(outcome=question))
        return await ctx._expert_answer
    finally:
        ctx._expert_answer = None


__all__ = ["RemoteAgent"]
