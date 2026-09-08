from __future__ import annotations

"""Tests that _llm_inference_task emits a FlushSentinel into text_ch as soon
as a tool call arrives, so the in-progress TTS segment can be spoken without
waiting for the entire tool round-trip.

Root cause of the bug:
- _parse_choice() returns None on the first tool delta (correct).
- Nothing in the default streaming path ever closed the TTS text channel,
  so audio was delayed until the full llm_task finished.
Fix:
- generation.py now sends FlushSentinel() into text_ch the moment the first
  tool call chunk arrives, *if* any text has already been generated.
"""

import asyncio

import pytest

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
    _LLMGenerationData,
    _llm_inference_task,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _fake_node(chunks: list[ChatChunk]):
    """Minimal LLM node fixture matching the io.LLMNode signature."""

    async def node(chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    return node


async def _drain_text_ch(text_ch: aio.Chan) -> list[str | FlushSentinel]:
    """Collect everything the text channel produced."""
    items: list[str | FlushSentinel] = []
    async for item in text_ch:
        items.append(item)
    return items


async def _run(chunks: list[ChatChunk]) -> tuple[_LLMGenerationData, list[str | FlushSentinel]]:
    text_ch: aio.Chan[str | FlushSentinel] = aio.Chan()
    function_ch: aio.Chan = aio.Chan()
    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)

    # _llm_inference_task closes text_ch via a done-callback added by
    # perform_llm_inference. Here we replicate that manually.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlushSentinelOnToolCall:
    async def test_flush_sentinel_emitted_after_text_then_tool(self) -> None:
        """The canonical case: model speaks then calls a tool.

        text_ch must contain the text tokens followed by exactly one
        FlushSentinel before the channel closes.
        """
        chunks = [
            _text_chunk("Sure, "),
            _text_chunk("let me check."),
            _tool_chunk("get_weather"),
        ]
        _, items = await _run(chunks)

        text_items = [i for i in items if isinstance(i, str)]
        sentinels = [i for i in items if isinstance(i, FlushSentinel)]

        assert "".join(text_items) == "Sure, let me check."
        assert len(sentinels) == 1, "exactly one FlushSentinel must be emitted"

    async def test_no_flush_sentinel_when_tool_only(self) -> None:
        """Tool-only response with no preceding text must not emit a sentinel.

        Emitting an empty-segment sentinel would be harmless but is wasteful
        and could confuse TTS providers that treat a flush as 'speak now'.
        """
        chunks = [_tool_chunk("search")]
        _, items = await _run(chunks)

        assert not any(isinstance(i, FlushSentinel) for i in items)

    async def test_function_call_still_dispatched(self) -> None:
        """Flushing TTS must not drop the tool call from generated_functions."""
        chunks = [
            _text_chunk("On it."),
            _tool_chunk("send_email", arguments='{"to":"a@b.com"}'),
        ]
        data, _ = await _run(chunks)

        assert len(data.generated_functions) == 1
        assert data.generated_functions[0].name == "send_email"
        assert data.generated_functions[0].arguments == '{"to":"a@b.com"}'

    async def test_only_one_sentinel_for_multiple_parallel_tool_calls(self) -> None:
        """Parallel tool calls arrive in a single ChatChunk.

        There should still be at most one FlushSentinel regardless of how many
        tools the model calls simultaneously.
        """
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
        chunks = [_text_chunk("Doing both."), chunk_with_two_tools]
        _, items = await _run(chunks)

        sentinels = [i for i in items if isinstance(i, FlushSentinel)]
        assert len(sentinels) == 1

    async def test_only_one_sentinel_across_multiple_tool_chunks(self) -> None:
        """Multiple tool chunks after text should still flush only once.

        Some models stream tool calls across more than one delta. The first
        tool delta should close the TTS segment, and later tool deltas must not
        enqueue extra flush sentinels.
        """
        chunks = [
            _text_chunk("Sure, "),
            _tool_chunk("first_tool"),
            _tool_chunk("second_tool"),
        ]
        _, items = await _run(chunks)

        sentinels = [i for i in items if isinstance(i, FlushSentinel)]
        assert len(sentinels) == 1
