"""Regression tests for Nova Sonic tool-set changes during realtime replies."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import function_tool, utils
from livekit.agents.llm import ToolContext

pytestmark = pytest.mark.unit

# The realtime plugin imports the optional AWS Smithy/Bedrock SDK. Keep these tests
# hermetic so they can exercise the session state machine without AWS credentials.
_AWS_STUBS = [
    "boto3",
    "aws_sdk_bedrock_runtime",
    "aws_sdk_bedrock_runtime.client",
    "aws_sdk_bedrock_runtime.models",
    "aws_sdk_bedrock_runtime.config",
    "smithy_aws_core",
    "smithy_aws_core.identity",
    "smithy_aws_event_stream",
    "smithy_aws_event_stream.exceptions",
    "smithy_core",
    "smithy_core.aio",
    "smithy_core.aio.interfaces",
    "smithy_core.aio.interfaces.identity",
]
for _mod in _AWS_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


@function_tool
async def first_tool() -> None:
    """First test tool."""


@function_tool
async def second_tool() -> None:
    """Second test tool."""


@function_tool
async def third_tool() -> None:
    """Third test tool."""


def _session():
    from livekit.plugins.aws.experimental.realtime.realtime_model import RealtimeSession

    session = object.__new__(RealtimeSession)
    session._tools = ToolContext([first_tool])
    session._is_sess_active = asyncio.Event()
    session._is_sess_active.set()
    session._pending_tools = set()
    session._tool_results_ch = utils.aio.Chan()
    session._tools_ready = asyncio.get_running_loop().create_future()
    session._tools_ready.set_result(True)
    session._tool_recycle_task = None
    session._active_tool_names = {"first_tool"}
    session._current_generation = None
    return session


async def test_update_tools_coalesces_pending_session_recycles() -> None:
    session = _session()
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    await session.update_tools([third_tool])

    assert session._tool_recycle_task is recycle_task
    await recycle_task

    assert recycle_calls == 1
    assert set(session.tools.function_tools) == {"third_tool"}


async def test_restored_tool_set_skips_pending_recycle() -> None:
    session = _session()
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    await session.update_tools([first_tool])

    assert session._tool_recycle_task is recycle_task
    await recycle_task

    assert recycle_calls == 0
    assert set(session.tools.function_tools) == {"first_tool"}


async def test_tool_recycle_waits_for_active_generation() -> None:
    session = _session()
    generation_done = asyncio.get_running_loop().create_future()
    session._current_generation = SimpleNamespace(_done_fut=generation_done)
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    await asyncio.sleep(0.2)

    assert recycle_calls == 0
    assert not recycle_task.done()

    generation_done.set_result(None)
    await recycle_task
    assert recycle_calls == 1


async def test_generate_reply_waits_for_pending_tool_recycle() -> None:
    from livekit.plugins.aws.experimental.realtime.realtime_model import RealtimeSession

    session = object.__new__(RealtimeSession)
    session._realtime_model = SimpleNamespace(modalities="mixed", _generate_reply_timeout=1.0)
    session._pending_generation_fut = None
    session._stream_ready = asyncio.Event()
    session._stream_ready.set()
    send_called = asyncio.Event()

    async def _send_text_message(*args, **kwargs) -> None:
        send_called.set()

    session._send_text_message = AsyncMock(side_effect=_send_text_message)

    recycle_finished = asyncio.Event()

    async def _wait_for_recycle() -> None:
        await recycle_finished.wait()

    session._tool_recycle_task = asyncio.create_task(_wait_for_recycle())

    generation_fut = session.generate_reply(instructions="say hello")
    await asyncio.sleep(0)
    session._send_text_message.assert_not_awaited()

    recycle_finished.set()
    await asyncio.wait_for(send_called.wait(), timeout=1.0)
    session._send_text_message.assert_awaited_once_with("say hello", interactive=True)
    generation_fut.cancel()


async def test_consecutive_on_enter_replies_coalesce_tool_set_changes() -> None:
    """A restore before the next reply should reuse the pending session recycle."""
    session = _session()
    session._realtime_model = SimpleNamespace(modalities="mixed", _generate_reply_timeout=1.0)
    session._stream_ready = asyncio.Event()
    session._stream_ready.set()

    sent_instructions: list[str] = []

    async def _send_text_message(text: str, *, interactive: bool) -> None:
        assert interactive
        sent_instructions.append(text)

    session._send_text_message = AsyncMock(side_effect=_send_text_message)
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1
        session._active_tool_names = set(session._tools.function_tools)

    session._graceful_session_recycle = _fake_recycle

    for index, instructions in enumerate(("say hello", "introduce yourself", "say goodbye")):
        await session.update_tools([second_tool])
        generation_fut = session.generate_reply(instructions=instructions)

        while len(sent_instructions) <= index:
            await asyncio.sleep(0)

        assert sent_instructions[index] == instructions
        generation_fut.cancel()

        if index < 2:
            await session.update_tools([first_tool])

    await session.update_tools([first_tool])
    recycle_task = session._tool_recycle_task
    assert recycle_task is not None
    await recycle_task

    assert sent_instructions == ["say hello", "introduce yourself", "say goodbye"]
    assert recycle_calls == 2
