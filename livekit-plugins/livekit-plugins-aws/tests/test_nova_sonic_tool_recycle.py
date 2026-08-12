"""Regression tests for Nova Sonic tool-set changes during realtime replies."""

from __future__ import annotations

import asyncio
import base64
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import function_tool, llm, utils
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
        session._active_tool_names = set(session._tools.function_tools)

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
        session._active_tool_names = set(session._tools.function_tools)

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    await asyncio.sleep(0.2)

    assert recycle_calls == 0
    assert not recycle_task.done()

    generation_done.set_result(None)
    await recycle_task
    assert recycle_calls == 1


async def test_tool_recycle_waits_for_generation_that_replaces_completed_one() -> None:
    session = _session()
    first_done = asyncio.get_running_loop().create_future()
    second_done = asyncio.get_running_loop().create_future()
    first_generation = SimpleNamespace(_done_fut=first_done)
    second_generation = SimpleNamespace(_done_fut=second_done)
    session._current_generation = first_generation
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1
        session._active_tool_names = set(session._tools.function_tools)

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    assert recycle_task is not None
    await asyncio.sleep(0.2)

    session._current_generation = second_generation
    first_done.set_result(None)
    await asyncio.sleep(0)

    assert recycle_calls == 0
    assert not recycle_task.done()

    second_done.set_result(None)
    await recycle_task
    assert recycle_calls == 1


async def test_tool_recycle_keeps_audio_available_until_generation_finishes() -> None:
    session = _session()
    generation_done = asyncio.get_running_loop().create_future()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    session._current_generation = SimpleNamespace(
        _done_fut=generation_done,
        content_id_map={"audio-content": "ASSISTANT_AUDIO"},
        message_gen=SimpleNamespace(audio_ch=audio_ch),
    )
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1
        session._active_tool_names = set(session._tools.function_tools)

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    assert recycle_task is not None
    await asyncio.sleep(0.2)

    audio = b"\x01\x02" * 320
    await session._handle_audio_output_content_event(
        {
            "event": {
                "audioOutput": {
                    "contentId": "audio-content",
                    "content": base64.b64encode(audio).decode(),
                }
            }
        }
    )

    frame = audio_ch.recv_nowait()
    assert bytes(frame.data) == audio
    assert not audio_ch.closed
    assert recycle_calls == 0

    generation_done.set_result(None)
    await recycle_task
    assert recycle_calls == 1


async def test_initialize_stream_captures_prompt_tool_snapshot() -> None:
    from livekit.plugins.aws.experimental.realtime.realtime_model import RealtimeSession

    session = object.__new__(RealtimeSession)
    session._realtime_model = SimpleNamespace(
        model="amazon.nova-2-sonic-v1:0",
        _model="amazon.nova-2-sonic-v1:0",
        _opts=SimpleNamespace(
            voice="tiffany",
            max_tokens=100,
            top_p=0.9,
            temperature=0.5,
            tool_choice=None,
            turn_detection="MEDIUM",
        ),
    )
    session._bedrock_client = SimpleNamespace(
        invoke_model_with_bidirectional_stream=AsyncMock(return_value=object())
    )
    session._stream_response = object()
    session._tools = ToolContext([first_tool])
    session._chat_ctx = llm.ChatContext.empty()
    session._instructions = "test instructions"
    session._event_builder = MagicMock()
    session._event_builder.create_prompt_start_block.return_value = (["session-start"], [])
    tool_configuration = object()
    session._serialize_tool_config = MagicMock(return_value=tool_configuration)
    session._report_connection_acquired = MagicMock()
    session._start_session_recycle_timer = MagicMock()
    session._response_task = None
    session._audio_input_task = None
    session._tools_ready = asyncio.get_running_loop().create_future()
    session._tools_ready.set_result(True)
    session._is_sess_active = asyncio.Event()
    session._stream_ready = asyncio.Event()
    session._tool_recycle_task = None
    session._session_recycle_task = None
    session._active_tool_names = set()
    session._pending_tools = set()
    session._current_generation = None
    session._tool_results_ch = utils.aio.Chan()
    session._audio_input_chan = utils.aio.Chan()

    async def _send_raw_event(_: str) -> None:
        await session.update_tools([second_tool])

    async def _noop() -> None:
        return

    session._send_raw_event = _send_raw_event
    session._process_responses = _noop
    session._process_audio_input = _noop

    await session.initialize_streams(is_restart=True)

    assert session._active_tool_names == {"first_tool"}
    assert set(session._tools.function_tools) == {"second_tool"}
    assert session._tool_recycle_task is not None
    assert (
        session._event_builder.create_prompt_start_block.call_args.kwargs["tool_configuration"]
        is tool_configuration
    )

    await asyncio.gather(session._response_task, session._audio_input_task)

    session._tool_recycle_task.cancel()
    try:
        await session._tool_recycle_task
    except asyncio.CancelledError:
        pass


async def test_tool_recycle_times_out_on_stuck_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.aws.experimental.realtime import realtime_model

    session = _session()
    monkeypatch.setattr(realtime_model, "TOOL_RECYCLE_GENERATION_TIMEOUT", 0.01)
    session._current_generation = SimpleNamespace(_done_fut=asyncio.Future())
    session._close_current_generation = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: setattr(session, "_current_generation", None)
    )
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1
        session._active_tool_names = set(session._tools.function_tools)

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    assert recycle_task is not None
    await recycle_task

    session._close_current_generation.assert_called_once()
    assert recycle_calls == 1


async def test_tool_changes_during_recycle_are_applied() -> None:
    session = _session()
    first_recycle_started = asyncio.Event()
    allow_first_recycle = asyncio.Event()
    recycle_calls = 0

    async def _fake_recycle() -> None:
        nonlocal recycle_calls
        recycle_calls += 1
        target_tool_names = set(session._tools.function_tools)
        if recycle_calls == 1:
            first_recycle_started.set()
            await allow_first_recycle.wait()
        session._active_tool_names = target_tool_names

    session._graceful_session_recycle = _fake_recycle

    await session.update_tools([second_tool])
    recycle_task = session._tool_recycle_task
    assert recycle_task is not None
    await first_recycle_started.wait()

    await session.update_tools([third_tool])
    allow_first_recycle.set()
    await recycle_task

    assert recycle_calls == 2
    assert session._active_tool_names == {"third_tool"}


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
