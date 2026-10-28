"""Closing a host provider settles work and releases blocked readers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.llm import LLM, ChatContext, LLMStream
from livekit.agents.llm.tool_context import Tool

pytestmark = pytest.mark.unit


class _WaitingLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        return _WaitingStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _WaitingStream(LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, _WaitingLLM)
        self._llm.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self._llm.stopped.set()


async def test_closing_host_stream_settles_provider_and_unblocks_reader() -> None:
    model = _WaitingLLM()
    stream = model.chat(chat_ctx=ChatContext.empty(), conn_options=APIConnectOptions(max_retry=0))
    reader = asyncio.create_task(anext(stream))
    try:
        async with asyncio.timeout(5):
            await model.started.wait()
            await stream.aclose()
            assert model.stopped.is_set()
            with pytest.raises(StopAsyncIteration):
                await reader
    finally:
        await stream.aclose()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
