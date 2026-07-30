from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    CompletionUsage,
    FallbackAdapter,
    LLMStream,
    Tool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .fake_llm import FakeLLM, FakeLLMResponse


def _chat_ctx(text: str) -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=text)
    return chat_ctx


class _SilentLLMStream(LLMStream):
    """Stream that opens successfully but never yields any useful chunk.

    Optionally emits chunks that carry no spoken content (e.g. a usage-only
    chunk) to mimic a provider that streams "empty" data while staying silent.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        emit_empty_chunk: bool,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._emit_empty_chunk = emit_empty_chunk

    async def _run(self) -> None:
        if self._emit_empty_chunk:
            # a chunk with neither content nor tool calls is not "useful"
            self._event_ch.send_nowait(
                ChatChunk(
                    id="silent",
                    delta=ChoiceDelta(role="assistant", content=None),
                    usage=CompletionUsage(
                        completion_tokens=0,
                        prompt_tokens=1,
                        total_tokens=1,
                    ),
                )
            )

        # connection stays open but no useful chunk is ever produced
        await asyncio.Future()


class _SilentLLM(LLM):
    def __init__(self, *, emit_empty_chunk: bool = False) -> None:
        super().__init__()
        self._emit_empty_chunk = emit_empty_chunk

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _SilentLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            emit_empty_chunk=self._emit_empty_chunk,
        )


async def _collect(stream: LLMStream) -> str:
    text = ""
    async with stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
    return text


async def _drain_recovery(fallback: FallbackAdapter) -> None:
    """Wait for any background recovery tasks to settle.

    A silent LLM marked unavailable spawns a recovery task that also times out
    after attempt_timeout; await it so the test doesn't observe leaked tasks
    (recovery-task cancellation in aclose() is tracked separately)."""
    for status in fallback._status:
        task = status.recovering_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


async def test_llm_fallback_on_silent_attempt() -> None:
    """A provider that opens the connection but never sends a useful chunk
    must trigger a fallback to the next LLM within attempt_timeout."""
    silent = _SilentLLM()
    working = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="hi there", ttft=0.0, duration=0.0)]
    )

    fallback = FallbackAdapter([silent, working], attempt_timeout=0.5)

    text = await asyncio.wait_for(_collect(fallback.chat(chat_ctx=_chat_ctx("hello"))), 5.0)
    assert text == "hi there"

    await _drain_recovery(fallback)
    await fallback.aclose()


async def test_llm_fallback_on_non_useful_chunks() -> None:
    """A provider that streams only non-useful chunks (no content / tool calls)
    must still fall back to the next LLM within attempt_timeout."""
    silent = _SilentLLM(emit_empty_chunk=True)
    working = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="hi there", ttft=0.0, duration=0.0)]
    )

    fallback = FallbackAdapter([silent, working], attempt_timeout=0.5)

    text = await asyncio.wait_for(_collect(fallback.chat(chat_ctx=_chat_ctx("hello"))), 5.0)
    assert text == "hi there"

    await _drain_recovery(fallback)
    await fallback.aclose()


async def test_llm_fallback_all_silent_raises() -> None:
    """If every LLM stays silent, the adapter should give up with an
    APIConnectionError rather than hanging forever."""
    fallback = FallbackAdapter(
        [_SilentLLM(), _SilentLLM()],
        attempt_timeout=0.3,
    )

    with pytest.raises(APIConnectionError):
        await asyncio.wait_for(_collect(fallback.chat(chat_ctx=_chat_ctx("hello"))), 5.0)

    await _drain_recovery(fallback)
    await fallback.aclose()
