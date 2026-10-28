"""Remote tools: host callbacks around the native A2A client and interaction decisions."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .. import _core
from .._core_view import to_core_item, to_core_items
from .._generated.a2a_client import AgentCard
from .._generated.a2a_remote import (
    RemoteEventFollow,
    RemoteEventInputRequired,
    RemoteEventOutcome,
    RemoteEventProgress,
)
from .._generated.tool_report import ToolReport
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
        self._token = _core._core_remote_token(url, token)
        self._conn_options = conn_options

    def as_tool(self) -> Toolset:
        return _RemoteTool(self._url, self._token, self._conn_options)


class _RemoteTool(Toolset):
    def __init__(self, url: str, token: str | None, conn_options: APIConnectOptions | None) -> None:
        self._url = url
        self._token = token
        self._conn_options = conn_options
        self._card: AgentCard | None = None
        self._tool: RawFunctionTool | None = None
        super().__init__(id=url)

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return [self._tool] if self._tool is not None else []

    def _client(self) -> AgentClient:
        return AgentClient(self._url, token=self._token, conn_options=self._conn_options)

    async def setup(self) -> _RemoteTool:
        if self._tool is None:
            client = self._client()
            try:
                self._card = await client.connect()
            finally:
                await client.aclose()
            self._id = self._card.name
            self._tool = function_tool(
                self._invoke,
                raw_schema={
                    "name": self._card.name,
                    "description": self._card.description,
                    "parameters": self._card.input_schema,
                },
            )
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
        client = self._client()
        reply: Reply | None = None
        try:
            await client.connect()
            message = interaction.message(
                raw_arguments,
                self._card.structured_input,
                client.offers_extension,
                to_core_items(history.items),
            )
            reply = await client.send(message)
            while True:
                follow: RemoteEventFollow | None = None
                async for update in reply:
                    event = interaction.update(update)
                    ctx.function_call.extra.update(interaction.extra)
                    if isinstance(event, RemoteEventProgress):
                        await ctx.update(event.text)
                    elif isinstance(event, RemoteEventInputRequired):
                        transition = InputRequired(
                            event.question,
                            tool=self._tool,
                            verbatim=event.verbatim_full,
                        )
                        return ToolReport(outcome=event.outcome, transition=transition)
                    elif isinstance(event, RemoteEventOutcome):
                        return ToolReport(outcome=event.outcome)
                    elif isinstance(event, RemoteEventFollow):
                        follow = event
                        break
                if follow is None:
                    raise RuntimeError("the remote interaction ended without an outcome")
                await reply.aclose()
                reply = await client._core_subscribe(follow.task_id, follow.thread_id)
        except asyncio.CancelledError:
            interaction.cancelled()
            ctx.function_call.extra.update(interaction.extra)
            raise
        finally:
            if reply is not None:
                await reply.aclose()
            await client.aclose()


__all__ = ["RemoteAgent"]
