from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.langchain import LLMAdapter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("use_generator", [False, True])
async def test_closing_chat_releases_the_graph_iterator(use_generator: bool) -> None:
    closed = asyncio.Event()

    class GraphIterator:
        def __aiter__(self):
            return self

        def __init__(self):
            self._sent = False

        async def __anext__(self):
            if self._sent:
                await asyncio.Event().wait()
            self._sent = True
            return "hello"

        async def aclose(self):
            closed.set()

    async def stream_graph():
        try:
            yield "hello"
            await asyncio.Event().wait()
        finally:
            closed.set()

    class Graph:
        def astream(self, *args, **kwargs):
            return stream_graph() if use_generator else GraphIterator()

    model = LLMAdapter(Graph(), stream_mode="custom")
    async with model.chat(
        chat_ctx=llm.ChatContext(), conn_options=APIConnectOptions(max_retry=0)
    ) as stream:
        assert (await anext(stream)).delta.content == "hello"
    assert closed.is_set(), "the graph iterator outlived its chat stream"
