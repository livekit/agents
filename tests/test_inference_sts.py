import asyncio

import pytest

from livekit.agents import llm, utils
from livekit.agents.inference.sts import STS, STSSession, _ResponseGeneration

pytestmark = pytest.mark.unit


def _make_session() -> STSSession:
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )
    return model.session()


def _new_generation() -> _ResponseGeneration:
    return _ResponseGeneration(
        message_ch=utils.aio.Chan(),
        function_ch=utils.aio.Chan(),
        messages={},
        response_id="resp_1",
    )


@pytest.mark.asyncio
async def test_function_call_emitted_only_when_arguments_complete():
    """Function-call arguments only arrive by output_item.done, so the FunctionCall
    must be emitted there (not at output_item.added, where arguments are empty)."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_response_output_item_added(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
            }
        }
    )
    # nothing should be emitted while arguments are still empty
    assert session._current_generation.function_ch.qsize() == 0

    session._handle_response_output_item_done(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }
        }
    )

    fc = session._current_generation.function_ch.recv_nowait()
    assert isinstance(fc, llm.FunctionCall)
    assert fc.id == "fc_1"
    assert fc.call_id == "call_1"
    assert fc.name == "get_weather"
    assert fc.arguments == '{"city": "SF"}'
    # exactly one function call emitted
    assert session._current_generation.function_ch.qsize() == 0


@pytest.mark.asyncio
async def test_incomplete_function_call_on_done_is_skipped():
    """A function_call item missing call_id/name should not emit a partial FunctionCall."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_response_output_item_done(
        {"item": {"id": "fc_1", "type": "function_call", "call_id": "", "name": "", "arguments": ""}}
    )
    assert session._current_generation.function_ch.qsize() == 0


@pytest.mark.asyncio
async def test_update_chat_ctx_forwards_function_call_output():
    """Tool results reach the server as conversation.item.create, and only once.

    STS manages conversation history server-side, but function_call_output is a
    client->server event the model needs before it can produce a tool reply, so
    update_chat_ctx must forward it (otherwise tool calls hang)."""
    session = _make_session()
    # Mark started+connected so _send queues onto _msg_ch instead of opening a
    # websocket (_send gates lifecycle startup on _started).
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.items.append(llm.FunctionCallOutput(call_id="call_1", output="sunny", is_error=False))

    await session.update_chat_ctx(chat_ctx)

    ev = session._msg_ch.recv_nowait()
    assert ev == {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    }

    # Idempotent: the same output is not forwarded twice.
    await session.update_chat_ctx(chat_ctx)
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_input_audio_transcription_completed_emitted():
    """When transcription is enabled the pipeline skips its own STT, so STS must
    surface the user transcript from OpenAI's transcription.completed event."""
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._handle_input_audio_transcription_completed(
        {"item_id": "item_1", "transcript": "what's the weather"}
    )

    assert len(events) == 1
    assert events[0].item_id == "item_1"
    assert events[0].transcript == "what's the weather"
    assert events[0].is_final is True


@pytest.mark.asyncio
async def test_input_audio_transcription_delta_accumulates():
    """Interim deltas emit non-final events carrying the full transcript so far,
    and the final .completed clears the per-item accumulator."""
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._handle_input_audio_transcription_delta({"item_id": "item_1", "delta": "what's "})
    session._handle_input_audio_transcription_delta({"item_id": "item_1", "delta": "the weather"})

    assert [e.transcript for e in events] == ["what's ", "what's the weather"]
    assert all(e.is_final is False for e in events)
    assert session._input_transcripts["item_1"] == "what's the weather"

    session._handle_input_audio_transcription_completed(
        {"item_id": "item_1", "transcript": "what's the weather?"}
    )

    assert events[-1].is_final is True
    assert events[-1].transcript == "what's the weather?"
    # accumulator is cleared once the turn finalizes
    assert "item_1" not in session._input_transcripts


@pytest.mark.asyncio
async def test_input_audio_transcription_failed_clears_partial():
    """A failed transcription drops the partial and does not emit a transcript."""
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._input_transcripts["item_1"] = "what's th"
    session._handle_input_audio_transcription_failed(
        {"item_id": "item_1", "error": {"message": "boom"}}
    )

    assert events == []
    assert "item_1" not in session._input_transcripts


@pytest.mark.asyncio
async def test_update_chat_ctx_ignores_non_output_items():
    """Messages are server-owned; only function outputs are forwarded."""
    session = _make_session()
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    await session.update_chat_ctx(chat_ctx)
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_replay_session_state_resets_and_requeues():
    """On reconnect the session is fresh: in-flight response futures must fail
    (not hang), per-turn state resets, and non-default tool_choice is re-applied
    since it rides on session.update rather than session.create."""
    session = _make_session()
    session._connected = True
    session._ws = object()

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    session._response_created_futures["evt_1"] = fut
    session._input_transcripts["item_1"] = "partial"
    session._sent_fnc_outputs.add("call_1")
    session._tool_choice = "required"

    session._replay_session_state()

    # pending response future is failed rather than left hanging on a dead turn
    assert fut.done()
    assert isinstance(fut.exception(), llm.RealtimeError)
    assert session._response_created_futures == {}
    assert session._input_transcripts == {}
    assert session._sent_fnc_outputs == set()

    # non-default tool_choice is replayed on the fresh session
    ev = session._msg_ch.recv_nowait()
    assert ev == {
        "type": "session.update",
        "session": {"type": "realtime", "tool_choice": "required"},
    }
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_replay_session_state_default_tool_choice_no_requeue():
    """A default 'auto' tool_choice and no tools produce no replay events."""
    session = _make_session()
    session._connected = True
    session._ws = object()

    session._replay_session_state()

    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_generation_done_gates_session_recycle():
    """The recycle timer waits on _generation_done, which clears while a turn is
    in flight and sets once the generation closes, so a proactive reconnect lands
    between turns rather than mid-response."""
    session = _make_session()
    assert session._generation_done.is_set()  # idle at start

    session._handle_response_created({"response": {"id": "resp_1"}})
    assert not session._generation_done.is_set()  # turn in flight, recycle held off

    session._close_current_generation()
    assert session._generation_done.is_set()  # turn done, recycle may proceed
