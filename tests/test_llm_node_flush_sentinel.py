from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable

import pytest

from livekit.agents import llm
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FunctionToolCall,
)
from livekit.agents.llm.tool_context import ToolContext
from livekit.agents.types import FlushSentinel
from livekit.agents.utils import aio
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.generation import (
    _llm_inference_task,
    _LLMGenerationData,
)

pytestmark = pytest.mark.unit


def _text_chunk(text: str) -> ChatChunk:
    return ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content=text))


def _tool_chunk(name: str, arguments: str = "{}") -> ChatChunk:
    return ChatChunk(
        id="c",
        delta=ChoiceDelta(
            role="assistant",
            content=None,
            tool_calls=[
                FunctionToolCall(
                    name=name,
                    arguments=arguments,
                    call_id="call_001",
                )
            ],
        ),
    )


def _text_and_tool_chunk(text: str, name: str, arguments: str = "{}") -> ChatChunk:
    return ChatChunk(
        id="c",
        delta=ChoiceDelta(
            role="assistant",
            content=text,
            tool_calls=[
                FunctionToolCall(
                    name=name,
                    arguments=arguments,
                    call_id="call_001",
                )
            ],
        ),
    )


def _fake_node(chunks: list[ChatChunk]):
    async def node(
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[ChatChunk]:
        del chat_ctx, tools, model_settings
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    return node


async def _drain_text_ch(text_ch: aio.Chan[str | FlushSentinel]) -> list[str | FlushSentinel]:
    items: list[str | FlushSentinel] = []
    async for item in text_ch:
        items.append(item)
    return items


async def _run(chunks: list[ChatChunk]) -> tuple[_LLMGenerationData, list[str | FlushSentinel]]:
    text_ch: aio.Chan[str | FlushSentinel] = aio.Chan()
    function_ch: aio.Chan = aio.Chan()
    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)

    # perform_llm_inference closes these channels via done callbacks; this focused
    # test calls _llm_inference_task directly, so it mirrors that cleanup here.
    task = asyncio.create_task(
        _llm_inference_task(
            _fake_node(chunks),
            ChatContext.empty(),
            ToolContext.empty(),
            ModelSettings(),
            data,
        )
    )
    task.add_done_callback(lambda _: text_ch.close())
    task.add_done_callback(lambda _: function_ch.close())

    items = await _drain_text_ch(text_ch)
    await task
    return data, items


class TestFlushSentinelOnToolCall:
    async def test_flush_sentinel_emitted_after_text_then_tool(self) -> None:
        chunks = [
            _text_chunk("Sure, "),
            _text_chunk("let me check."),
            _tool_chunk("get_weather"),
        ]
        _, items = await _run(chunks)

        text_items = [item for item in items if isinstance(item, str)]
        sentinels = [item for item in items if isinstance(item, FlushSentinel)]

        assert "".join(text_items) == "Sure, let me check."
        assert len(sentinels) == 1

    async def test_no_flush_sentinel_when_tool_only(self) -> None:
        _, items = await _run([_tool_chunk("search")])

        assert not any(isinstance(item, FlushSentinel) for item in items)

    async def test_function_call_still_dispatched(self) -> None:
        chunks = [
            _text_chunk("On it."),
            _tool_chunk("send_email", arguments='{"to":"a@b.com"}'),
        ]
        data, _ = await _run(chunks)

        assert len(data.generated_functions) == 1
        assert data.generated_functions[0].name == "send_email"
        assert data.generated_functions[0].arguments == '{"to":"a@b.com"}'

    async def test_only_one_sentinel_for_multiple_parallel_tool_calls(self) -> None:
        chunk_with_two_tools = ChatChunk(
            id="c",
            delta=ChoiceDelta(
                role="assistant",
                content=None,
                tool_calls=[
                    FunctionToolCall(name="foo", arguments="{}", call_id="call_1"),
                    FunctionToolCall(name="bar", arguments="{}", call_id="call_2"),
                ],
            ),
        )
        _, items = await _run([_text_chunk("Doing both."), chunk_with_two_tools])

        sentinels = [item for item in items if isinstance(item, FlushSentinel)]
        assert len(sentinels) == 1

    async def test_only_one_sentinel_across_multiple_tool_chunks(self) -> None:
        chunks = [
            _text_chunk("Sure, "),
            _tool_chunk("first_tool"),
            _tool_chunk("second_tool"),
        ]
        _, items = await _run(chunks)

        sentinels = [item for item in items if isinstance(item, FlushSentinel)]
        assert len(sentinels) == 1

    async def test_flush_sentinel_after_text_in_same_tool_chunk(self) -> None:
        _, items = await _run([_text_and_tool_chunk("Let me check.", "get_weather")])

        assert isinstance(items[0], str)
        assert items[0] == "Let me check."
        assert isinstance(items[1], FlushSentinel)

    async def test_text_between_tool_chunks_gets_flushed(self) -> None:
        chunks = [
            _text_chunk("First preface."),
            _tool_chunk("first_tool"),
            _text_chunk("Second preface."),
            _tool_chunk("second_tool"),
        ]
        _, items = await _run(chunks)

        assert [type(item) for item in items] == [
            str,
            FlushSentinel,
            str,
            FlushSentinel,
        ]
        assert [item for item in items if isinstance(item, str)] == [
            "First preface.",
            "Second preface.",
        ]
