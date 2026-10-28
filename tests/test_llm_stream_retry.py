"""A plugin LLM stream's retries: output already handed to the reader is never replayed."""

from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.agents.llm import LLM, ChatChunk, ChatContext, ChoiceDelta, LLMStream
from livekit.agents.llm.tool_context import Tool

pytestmark = pytest.mark.unit


class _DroppingLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = APIConnectOptions(),
        **kwargs: Any,
    ) -> LLMStream:
        return _DroppingStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _DroppingStream(LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, _DroppingLLM)
        self._llm.attempts += 1
        self.send(ChatChunk(id="a", delta=ChoiceDelta(content="partial")))
        raise APIConnectionError("dropped mid-stream", retryable=True)


# Catches: a retryable failure after a chunk reached the reader running the plugin again, which
# would repeat the output the caller already has.
async def test_a_stream_that_sent_output_is_not_run_again() -> None:
    llm = _DroppingLLM()
    chunks: list[ChatChunk] = []
    with pytest.raises(APIConnectionError):
        async with llm.chat(
            chat_ctx=ChatContext.empty(),
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0),
        ) as stream:
            async for chunk in stream:
                chunks.append(chunk)
    assert llm.attempts == 1
    assert [c.delta.content for c in chunks if c.delta] == ["partial"]
