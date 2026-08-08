import asyncio
import base64
import json

import aiohttp
import pytest
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemDeletedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from livekit.agents import llm, utils
from livekit.agents.types import APIConnectOptions
from livekit.plugins.boson import realtime

pytestmark = pytest.mark.unit


async def _idle_run(_session):
    await asyncio.Future()


class _FakeWebSocket:
    def __init__(self, close_message=None) -> None:
        self.closed = False
        self.sent: list[str] = []
        self.close_event = asyncio.Event()
        self.close_message = close_message
        self.close_code = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.close_event.wait()
        raise StopAsyncIteration

    async def receive(self):
        await self.close_event.wait()
        if self.close_message is not None:
            self.close_code = self.close_message.data
            return self.close_message
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def exception(self):
        return None


class _FakeHTTPClient:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws

    async def ws_connect(self, *_args, **_kwargs):
        return self.ws


class _ScriptedFakeWebSocket(_FakeWebSocket):
    """A _FakeWebSocket that first delivers scripted messages, then behaves normally."""

    def __init__(self, messages=None, close_message=None) -> None:
        super().__init__(close_message)
        self.messages = list(messages or [])

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return await super().receive()


class _SequencedHTTPClient:
    """Returns one scripted outcome (websocket or exception) per ws_connect call."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.connect_urls: list[str] = []

    async def ws_connect(self, url, **_kwargs):
        self.connect_urls.append(url)
        if not self.outcomes:
            raise aiohttp.ClientConnectionError("no more scripted connections")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# type -> (RealtimeSession handler, event model), mirroring the dispatch in the
# OpenAI base recv loop that the plugin inherits.
_SERVER_EVENT_HANDLERS = {
    "input_audio_buffer.speech_started": (
        "_handle_input_audio_buffer_speech_started",
        InputAudioBufferSpeechStartedEvent,
    ),
    "conversation.item.added": ("_handle_conversion_item_added", ConversationItemAdded),
    "conversation.item.deleted": ("_handle_conversion_item_deleted", ConversationItemDeletedEvent),
    "conversation.item.input_audio_transcription.completed": (
        "_handle_conversion_item_input_audio_transcription_completed",
        ConversationItemInputAudioTranscriptionCompletedEvent,
    ),
    "response.created": ("_handle_response_created", ResponseCreatedEvent),
    "response.output_item.added": (
        "_handle_response_output_item_added",
        ResponseOutputItemAddedEvent,
    ),
    "response.content_part.added": (
        "_handle_response_content_part_added",
        ResponseContentPartAddedEvent,
    ),
    "response.output_text.delta": ("_handle_response_text_delta", ResponseTextDeltaEvent),
    "response.output_text.done": ("_handle_response_text_done", ResponseTextDoneEvent),
    "response.output_audio.delta": ("_handle_response_audio_delta", ResponseAudioDeltaEvent),
    "response.output_item.done": ("_handle_response_output_item_done", ResponseOutputItemDoneEvent),
    "response.done": ("_handle_response_done", ResponseDoneEvent),
    "error": ("_handle_error", RealtimeErrorEvent),
}


def _server_event(session, event: dict) -> None:
    """Inject a server event as if it had been received on the socket."""
    session.emit("openai_server_event_received", event)
    entry = _SERVER_EVENT_HANDLERS.get(event["type"])
    if entry is not None:
        method, event_cls = entry
        getattr(session, method)(event_cls.construct(**event))


@pytest.mark.asyncio
async def test_boson_realtime_session_sends_full_session_update(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        model="test-model",
        voice="coral",
        instructions="Be concise.",
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["type"] == "session.update"
        assert event["session"]["model"] == "test-model"
        assert event["session"]["instructions"] == "Be concise."
        assert event["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert event["session"]["audio"]["output"]["voice"] == "coral"
        assert "transcription" not in event["session"]["audio"]["input"]
        assert event["session"]["audio"]["input"]["turn_detection"]["type"] == "server_vad"
        assert event["session"]["output_modalities"] == ["audio"]
        assert model.capabilities.audio_output is True
        # The server preserves client-supplied item ids, so the base
        # diff/create/delete chat-context synchronization works.
        assert model.capabilities.mutable_chat_context is True
        # item.create is a pure insert server-side; the framework must send
        # response.create after tool outputs.
        assert model.capabilities.auto_tool_reply_generation is False
        # No transcription model set -> the server emits no user-transcript events.
        assert model.capabilities.user_transcription is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_session_omits_explicit_none_transcription(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        input_audio_transcription=None,
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert "transcription" not in event["session"]["audio"]["input"]
        assert model.capabilities.user_transcription is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_session_transcription_without_model_disables_user_transcription(
    monkeypatch,
):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    # A transcription block without a `model` still runs ASR internally on the
    # server but emits no client-facing transcription events.
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        input_audio_transcription={"language": "english"},
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["audio"]["input"]["transcription"] == {"language": "english"}
        assert model.capabilities.user_transcription is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_session_sends_boson_options(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    turn_detection = {
        "type": "server_vad",
        "silence_duration_ms": 800,
        "threshold": 0.6,
        "prefix_padding_ms": 200,
        "interrupt_response": False,
    }

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        model="llm-model",
        voice="en_woman",
        input_audio_transcription={
            "model": "asr-model",
            "language": "english",
            # `prompt` is not honored by the server; the wrapper drops it even
            # from a raw transcription dict (see test below).
            "prompt": "Transcribe names carefully.",
            "temperature": 0.1,
        },
        turn_detection=turn_detection,
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        session_update = event["session"]
        assert session_update["audio"]["input"]["transcription"] == {
            "model": "asr-model",
            "language": "english",
            "temperature": 0.1,
        }
        assert session_update["audio"]["input"]["turn_detection"] == turn_detection
        assert session_update["audio"]["output"]["voice"] == "en_woman"
        assert model.capabilities.user_transcription is True
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_session_can_disable_turn_detection(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        turn_detection=None,
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["audio"]["input"]["turn_detection"] is None
        assert model.capabilities.turn_detection is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_session_turn_detection_false_disables_like_none(monkeypatch):
    # README documents turn_detection=None/False as equivalent; False must
    # not crash (dict(False) raises TypeError - bool isn't iterable).
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        turn_detection=False,
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["audio"]["input"]["turn_detection"] is None
        assert model.capabilities.turn_detection is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_options_turn_detection_false_disables(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        model.update_options(turn_detection=False)

        event = await session._msg_ch.recv()
        assert event["session"]["audio"]["input"]["turn_detection"] is None
        assert model.capabilities.turn_detection is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_text_only_modality_session_update(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        output_modalities=["text"],
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["output_modalities"] == ["text"]
        assert model.capabilities.audio_output is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize("output_modalities", [["text", "audio"], [], ["video"]])
def test_boson_realtime_rejects_invalid_output_modalities(output_modalities):
    with pytest.raises(ValueError):
        realtime.RealtimeModel(
            url="ws://localhost:8000/v1/realtime/",
            api_key="test-key",
            output_modalities=output_modalities,
        )


@pytest.mark.asyncio
async def test_boson_realtime_text_only_generation_streams_text(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        output_modalities=["text"],
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        generation = await generation_fut

        _server_event(
            session,
            {
                "type": "response.output_item.added",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            },
        )
        message = await generation.message_stream.recv()

        _server_event(
            session,
            {
                "type": "response.content_part.added",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "text", "text": ""},
            },
        )
        assert await message.modalities == ["text"]

        _server_event(
            session,
            {
                "type": "response.output_text.delta",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi there.",
            },
        )
        assert await message.text_stream.recv() == "Hi there."

        _server_event(
            session,
            {
                "type": "response.output_text.done",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Hi there.",
            },
        )
        _server_event(
            session,
            {
                "type": "response.output_item.done",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            },
        )
        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_1", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_noise_reduction_string_normalized(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        input_audio_noise_reduction="near_field",
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_noise_reduction_dict_passthrough(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        input_audio_noise_reduction={"type": "far_field"},
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_noise_reduction_omitted_by_default(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert "noise_reduction" not in event["session"]["audio"]["input"]
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_options_sends_boson_options(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
    )
    session = model.session()
    try:
        await session._msg_ch.recv()
        model.update_options(
            # `prompt` is not honored by the server; the wrapper drops it even
            # from a raw transcription dict.
            input_audio_transcription={"model": "asr-v2", "prompt": "ASR hints."},
            input_audio_noise_reduction="near_field",
            turn_detection={"type": "server_vad", "silence_duration_ms": 900},
        )

        event = await session._msg_ch.recv()
        assert event["type"] == "session.update"
        assert event["session"]["audio"]["input"]["transcription"] == {"model": "asr-v2"}
        assert event["session"]["audio"]["input"]["turn_detection"] == {
            "type": "server_vad",
            "silence_duration_ms": 900,
        }
        assert event["session"]["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
        assert model.capabilities.user_transcription is True
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_options_omits_explicit_none_transcription(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        input_audio_transcription={"model": "asr-v1"},
    )
    session = model.session()
    try:
        initial_event = await session._msg_ch.recv()
        assert initial_event["session"]["audio"]["input"]["transcription"] == {"model": "asr-v1"}

        model.update_options(input_audio_transcription=None)

        event = await session._msg_ch.recv()
        assert event["type"] == "session.update"
        assert "transcription" not in event["session"]["audio"]["input"]
        assert model.capabilities.user_transcription is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_tool_choice_uses_boson_shape(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        tool_choice={"type": "function", "function": {"name": "lookup_policy"}},
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["tool_choice"] == {"type": "function", "name": "lookup_policy"}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_raw_function_tool_schema_uses_boson_shape(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    async def lookup_policy():
        return "ok"

    tool = llm.function_tool(
        lookup_policy,
        raw_schema={
            "name": "lookup_policy",
            "description": "Lookup a policy.",
            "parameters": {"type": "object", "properties": {}},
            "meta": {"provider": "test"},
        },
    )

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        await session.update_tools([tool])
        event = await session._msg_ch.recv()

        assert event["session"]["tools"] == [
            {
                "type": "function",
                "name": "lookup_policy",
                "description": "Lookup a policy.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generation_audio_mapping(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()
        assert response_create["type"] == "response.create"

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        generation = await generation_fut
        assert generation.response_id == "resp_1"
        assert generation.user_initiated is True

        _server_event(
            session,
            {
                "type": "response.output_item.added",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            },
        )
        message = await generation.message_stream.recv()
        assert message.message_id == "item_1"

        _server_event(
            session,
            {
                "type": "response.content_part.added",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio", "transcript": ""},
            },
        )
        assert await message.modalities == ["audio", "text"]

        pcm = b"\x00\x00" * 240
        _server_event(
            session,
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(pcm).decode("utf-8"),
            },
        )
        frame = await message.audio_stream.recv()
        assert frame.sample_rate == 24000
        assert frame.num_channels == 1
        assert frame.samples_per_channel == 240

        _server_event(
            session,
            {
                "type": "response.output_item.done",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            },
        )
        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_1", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_uses_response_metadata(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()
        event_id = response_create["event_id"]

        assert response_create["response"]["metadata"]["client_event_id"] == event_id

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {"id": "resp_auto", "status": "in_progress"},
            },
        )
        assert not generation_fut.done()
        assert event_id in session._response_created_futures

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_manual",
                    "status": "in_progress",
                    "metadata": {"client_event_id": event_id},
                },
            },
        )
        generation = await generation_fut
        assert generation.response_id == "resp_manual"
        assert generation.user_initiated is True
        assert session._response_created_futures == {}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_capabilities_disable_per_response_tool_choice(monkeypatch):
    # Boson has no per-response override: the framework must scope
    # tool_choice/tools at the session level around generate_reply() instead
    # of embedding them in response.create.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        assert model.capabilities.per_response_tool_choice is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_instructions_scoped_via_session_update(monkeypatch):
    # Boson has no per-response instructions override: sending it inside
    # response.create makes the server discard the real conversation history
    # for that turn. generate_reply(instructions=...) must scope it at the
    # session level instead — a session.update before response.create that
    # carries no override fields — then restore the original afterwards.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        instructions="Base instructions.",
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        generation_fut = session.generate_reply(instructions="Greet the user.")

        scoped_update = await session._msg_ch.recv()
        assert scoped_update["type"] == "session.update"
        assert scoped_update["session"]["instructions"] == "Base instructions.\nGreet the user."

        response_create = await session._msg_ch.recv()
        assert response_create["type"] == "response.create"
        assert response_create["response"].get("instructions") is None

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        await generation_fut

        restore_update = await session._msg_ch.recv()
        assert restore_update["type"] == "session.update"
        assert restore_update["session"]["instructions"] == "Base instructions."
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_tool_choice_scoped_via_session_update(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        generation_fut = session.generate_reply(tool_choice="none")

        scoped_update = await session._msg_ch.recv()
        assert scoped_update["type"] == "session.update"
        assert scoped_update["session"]["tool_choice"] == "none"

        response_create = await session._msg_ch.recv()
        assert response_create["type"] == "response.create"
        assert "tool_choice" not in response_create["response"]

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        await generation_fut

        restore_update = await session._msg_ch.recv()
        assert restore_update["type"] == "session.update"
        assert restore_update["session"]["tool_choice"] == "auto"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_tools_emits_followup_session_update(monkeypatch):
    # Scope of this test is the client side of the exchange only: which events
    # generate_reply(tools=...) puts on the wire, in what order, carrying what.
    # It asserts nothing about the session state the peer ends up in — that is
    # not observable from here and is not what this test is for.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    async def lookup_policy():
        return "ok"

    tool = llm.function_tool(
        lookup_policy,
        raw_schema={
            "name": "lookup_policy",
            "description": "Lookup a policy.",
            "parameters": {"type": "object", "properties": {}},
            "meta": {"provider": "test"},
        },
    )

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        generation_fut = session.generate_reply(tools=[tool])

        scoped_update = await session._msg_ch.recv()
        assert scoped_update["type"] == "session.update"
        assert scoped_update["session"]["tools"] == [
            {
                "type": "function",
                "name": "lookup_policy",
                "description": "Lookup a policy.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        response_create = await session._msg_ch.recv()
        assert response_create["type"] == "response.create"
        assert "tools" not in response_create["response"]

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        await generation_fut

        # A follow-up session.update goes out once the response is created,
        # carrying the tools that were configured before the override.
        followup_update = await session._msg_ch.recv()
        assert followup_update["type"] == "session.update"
        assert followup_update["session"]["tools"] == []
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_scoped_overrides_are_serialized(monkeypatch):
    # Two overlapping scoped generate_reply() calls must not race: the second
    # must not swap in its override until the first has fully restored,
    # otherwise the first's restore could stomp the second's override, or the
    # second could capture the first's temporary value as "original".
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        instructions="Base instructions.",
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        first_fut = session.generate_reply(instructions="First.")
        second_fut = session.generate_reply(instructions="Second.")

        # The first call's scoped update and response.create go out; the
        # second is blocked on the lock until the first fully restores (the
        # recv() ordering below is the actual proof of serialization).
        first_scoped = await session._msg_ch.recv()
        assert first_scoped["session"]["instructions"] == "Base instructions.\nFirst."

        first_response_create = await session._msg_ch.recv()
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": first_response_create["event_id"]},
                },
            },
        )
        await first_fut

        first_restore = await session._msg_ch.recv()
        assert first_restore["session"]["instructions"] == "Base instructions."

        # The second call's scoped update carries the true original, not
        # whatever the first call happened to leave in place mid-flight.
        second_scoped = await session._msg_ch.recv()
        assert second_scoped["session"]["instructions"] == "Base instructions.\nSecond."

        second_response_create = await session._msg_ch.recv()
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_2",
                    "status": "in_progress",
                    "metadata": {"client_event_id": second_response_create["event_id"]},
                },
            },
        )
        await second_fut

        second_restore = await session._msg_ch.recv()
        assert second_restore["session"]["instructions"] == "Base instructions."
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_restores_on_cancel_during_swap_in(monkeypatch):
    # update_tools() awaits the base class's own lock — a genuine suspension
    # point. If generate_reply() is cancelled while still stuck there mid
    # swap-in (e.g. aclose() cancelling it), the restore must still run, not
    # be skipped because the swap-in lived outside the try/finally.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    async def lookup_policy():
        return "ok"

    tool = llm.function_tool(
        lookup_policy,
        raw_schema={
            "name": "lookup_policy",
            "description": "Lookup a policy.",
            "parameters": {"type": "object", "properties": {}},
            "meta": {"provider": "test"},
        },
    )

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        instructions="Base instructions.",
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        # Hold the base class's tools lock so update_tools() inside the
        # scoped call blocks mid swap-in, after instructions already went out.
        await session._update_fnc_ctx_lock.acquire()
        try:
            generation_fut = session.generate_reply(instructions="Scoped!", tools=[tool])

            scoped_update = await session._msg_ch.recv()
            assert scoped_update["session"]["instructions"] == "Base instructions.\nScoped!"
            assert session._instructions == "Base instructions.\nScoped!"
            assert session._msg_ch.empty()

            generation_fut.cancel()
            with pytest.raises(asyncio.CancelledError):
                await generation_fut
        finally:
            session._update_fnc_ctx_lock.release()

        # The restore must still happen even though cancellation landed
        # mid-swap-in, before generate_reply() itself was ever called.
        restore_update = await asyncio.wait_for(session._msg_ch.recv(), timeout=1.0)
        assert restore_update["type"] == "session.update"
        assert restore_update["session"]["instructions"] == "Base instructions."
        assert session._instructions == "Base instructions."
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_new_response_closes_previous_generation(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {"id": "resp_1", "status": "in_progress"},
            },
        )
        first_generation = session._current_generation
        assert first_generation is not None

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {"id": "resp_2", "status": "in_progress"},
            },
        )
        assert first_generation.message_ch.closed
        assert first_generation.function_ch.closed
        assert session._current_generation is not first_generation
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_discarded_response_keeps_active_generation_streams(monkeypatch):
    # A response that timed out before the server created it must be discarded
    # without cutting off the streams of a newer in-progress generation.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        session._discarded_event_ids.add("evt_stale")

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {"id": "resp_active", "status": "in_progress"},
            },
        )
        active_generation = session._current_generation
        assert active_generation is not None

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_stale",
                    "status": "in_progress",
                    "metadata": {"client_event_id": "evt_stale"},
                },
            },
        )

        # the active generation's streams stay open and interrupt() still
        # targets the active response, not the discarded one
        assert not active_generation.message_ch.closed
        assert not active_generation.function_ch.closed
        assert session._current_generation is active_generation
        assert session._current_response_id == "resp_active"

        # the base cancelled the stale response by id
        cancel_event = await session._msg_ch.recv()
        assert cancel_event["type"] == "response.cancel"
        assert cancel_event["response_id"] == "resp_stale"

        # the stale response's terminal event must not close the active
        # generation either
        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_stale", "status": "cancelled", "usage": None},
            },
        )
        assert not active_generation.message_ch.closed
        assert session._current_generation is active_generation
        assert session._current_response_id == "resp_active"

        # the active response still completes normally afterwards
        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_active", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is None
        assert session._current_response_id is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_discarded_response_without_active_generation_is_skipped(monkeypatch):
    # With no generation in progress, the base discard marker must stay in the
    # slot so the stale response's trailing events are eaten.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        session._discarded_event_ids.add("evt_stale")

        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_stale",
                    "status": "in_progress",
                    "metadata": {"client_event_id": "evt_stale"},
                },
            },
        )
        assert session._current_generation is not None  # the discard marker
        assert session._current_response_id is None

        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_stale", "status": "cancelled", "usage": None},
            },
        )
        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_stale_response_done_does_not_corrupt_new_generation(monkeypatch):
    # A's response.done can legitimately arrive after B's response.created:
    # A's terminal cleanup is queued behind its own buffered audio output on
    # the server, while B's response.created is emitted immediately on a
    # separate path. Every response-scoped handler must ignore events whose
    # response_id isn't the current one, not just response.done.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        _server_event(
            session,
            {"type": "response.created", "response": {"id": "resp_a", "status": "in_progress"}},
        )
        _server_event(
            session,
            {
                "type": "response.output_item.added",
                "response_id": "resp_a",
                "output_index": 0,
                "item": {"id": "item_a", "type": "message"},
            },
        )

        # B supersedes A before A's response.done arrives.
        _server_event(
            session,
            {"type": "response.created", "response": {"id": "resp_b", "status": "in_progress"}},
        )
        generation_b = session._current_generation
        assert generation_b is not None
        assert session._current_response_id == "resp_b"

        _server_event(
            session,
            {
                "type": "response.output_item.added",
                "response_id": "resp_b",
                "output_index": 0,
                "item": {"id": "item_b", "type": "message"},
            },
        )
        message_b = await generation_b.message_ch.recv()
        assert message_b.message_id == "item_b"

        _server_event(
            session,
            {
                "type": "response.content_part.added",
                "response_id": "resp_b",
                "item_id": "item_b",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "text", "text": ""},
            },
        )
        assert await message_b.modalities == ["text"]

        # A's late trailing delta targets an item that only exists on A's
        # (already-discarded) generation; it must be dropped before it can
        # even be looked up against B's, not silently misapplied to B.
        _server_event(
            session,
            {
                "type": "response.output_text.delta",
                "response_id": "resp_a",
                "item_id": "item_a",
                "output_index": 0,
                "content_index": 0,
                "delta": "stale from A",
            },
        )

        # A's late response.done must not close B.
        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_a", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is generation_b
        assert session._current_response_id == "resp_b"
        assert not generation_b.message_ch.closed
        assert not generation_b.function_ch.closed

        # B keeps streaming and completes normally afterwards.
        _server_event(
            session,
            {
                "type": "response.output_text.delta",
                "response_id": "resp_b",
                "item_id": "item_b",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello from B",
            },
        )
        assert await message_b.text_stream.recv() == "hello from B"

        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_b", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is None
        assert session._current_response_id is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_late_event_after_generation_finished_is_dropped(monkeypatch):
    # Trailing events for a superseded response can still arrive after the
    # response that superseded it has itself finished, leaving no generation
    # in the slot. Filtering must not depend on one being there: every
    # response-scoped handler in the base except response.done asserts on a
    # missing generation, so letting these through raises instead of dropping.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        _server_event(
            session,
            {"type": "response.created", "response": {"id": "resp_a", "status": "in_progress"}},
        )
        _server_event(
            session,
            {
                "type": "response.output_item.added",
                "response_id": "resp_a",
                "output_index": 0,
                "item": {"id": "item_a", "type": "message"},
            },
        )

        # B supersedes A, then runs to completion — clearing the slot.
        _server_event(
            session,
            {"type": "response.created", "response": {"id": "resp_b", "status": "in_progress"}},
        )
        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_b", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is None
        assert session._current_response_id is None

        # Every response-scoped event for A must now be dropped, not forwarded.
        assert session._is_stale_response_scoped_event("resp_a") is True
        for late_event in (
            {
                "type": "response.output_text.delta",
                "response_id": "resp_a",
                "item_id": "item_a",
                "output_index": 0,
                "content_index": 0,
                "delta": "stale from A",
            },
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_a",
                "item_id": "item_a",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(b"\x00\x00").decode(),
            },
            {
                "type": "response.output_item.added",
                "response_id": "resp_a",
                "output_index": 0,
                "item": {"id": "item_a", "type": "message"},
            },
            {
                "type": "response.output_item.done",
                "response_id": "resp_a",
                "output_index": 0,
                "item": {"id": "item_a", "type": "message"},
            },
        ):
            _server_event(session, late_event)

        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_discarded_response_trailing_event_after_active_done(monkeypatch):
    # A response discarded on arrival (its generate_reply timed out or was
    # interrupted first) does not get the base's discard placeholder parked in
    # the slot when a real generation is already streaming there — it is
    # filtered by response id instead. Once that real response finishes and
    # empties the slot, the discarded response's own trailing events must
    # still be dropped rather than forwarded to a missing generation.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        _server_event(
            session,
            {"type": "response.created", "response": {"id": "resp_a", "status": "in_progress"}},
        )
        generation_a = session._current_generation

        session._discarded_event_ids.add("evt_discarded")
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_d",
                    "status": "in_progress",
                    "metadata": {"client_event_id": "evt_discarded"},
                },
            },
        )
        # A keeps the slot; the discarded response never becomes current.
        assert session._current_generation is generation_a
        assert session._current_response_id == "resp_a"
        assert (await session._msg_ch.recv())["type"] == "response.cancel"

        _server_event(
            session,
            {
                "type": "response.done",
                "response": {"id": "resp_a", "status": "completed", "usage": None},
            },
        )
        assert session._current_generation is None

        _server_event(
            session,
            {
                "type": "response.output_item.added",
                "response_id": "resp_d",
                "output_index": 0,
                "item": {"id": "item_d", "type": "message"},
            },
        )
        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_remote_items_are_not_recreated(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(
            session,
            {
                "type": "conversation.item.added",
                "previous_item_id": None,
                "item": {
                    "id": "user_1",
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        )

        await session.update_chat_ctx(
            llm.ChatContext([llm.ChatMessage(id="user_1", role="user", content=["hello"])])
        )

        assert session._msg_ch.empty()
    finally:
        await session.aclose()
        await model.aclose()


def _user_item_added(item_id: str, content: list[dict]) -> dict:
    return {
        "type": "conversation.item.added",
        "previous_item_id": None,
        "item": {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": content,
        },
    }


def _item_added_echo(item: dict, previous_item_id: str | None = None) -> dict:
    """The server's conversation.item.added echo for a client-created item."""
    return {
        "type": "conversation.item.added",
        "previous_item_id": previous_item_id,
        "item": item,
    }


@pytest.mark.asyncio
async def test_boson_realtime_merged_item_readd_updates_text(monkeypatch):
    # The server merges consecutive same-role turns into one item and re-emits
    # conversation.item.added with the same id and cumulative content.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(
            session, _user_item_added("user_1", [{"type": "input_text", "text": "what's"}])
        )
        _server_event(
            session,
            _user_item_added("user_1", [{"type": "input_text", "text": "what's the weather"}]),
        )

        assert len(session.chat_ctx.items) == 1
        remote_item = session._remote_chat_ctx.get("user_1")
        assert remote_item is not None
        assert remote_item.item.content == ["what's the weather"]
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_merged_item_readd_without_text_keeps_transcript(monkeypatch):
    # Audio-input configs re-add the merged item with an empty input_audio part;
    # a transcript already applied by input_audio_transcription.completed must
    # not be wiped.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(session, _user_item_added("user_1", [{"type": "input_audio", "audio": ""}]))
        _server_event(
            session,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "user_1",
                "content_index": 0,
                "transcript": "hello there",
                "logprobs": None,
            },
        )
        _server_event(session, _user_item_added("user_1", [{"type": "input_audio", "audio": ""}]))

        remote_item = session._remote_chat_ctx.get("user_1")
        assert remote_item is not None
        assert remote_item.item.content == ["hello there"]
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_input_transcription_replaces_remote_item_text(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    completed = []
    session.on("input_audio_transcription_completed", completed.append)
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(
            session,
            {
                "type": "conversation.item.added",
                "previous_item_id": None,
                "item": {
                    "id": "user_1",
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        )
        _server_event(
            session,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "user_1",
                "content_index": 0,
                "transcript": "hello world",
                "logprobs": None,
            },
        )

        remote_item = session._remote_chat_ctx.get("user_1")
        assert remote_item is not None
        assert remote_item.item.content == ["hello world"]
        assert completed
        assert completed[-1].transcript == "hello world"
        assert completed[-1].is_final is True
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_chat_ctx_message_schema_uses_boson_shape(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext(
                    [llm.ChatMessage(id="asst_1", role="assistant", content=["Assistant note."])]
                )
            )
        )
        assistant_event = await session._msg_ch.recv()

        assert assistant_event["item"]["role"] == "assistant"
        assert assistant_event["item"]["content"] == [{"type": "text", "text": "Assistant note."}]
        assert assistant_event["previous_item_id"] is None

        # update_chat_ctx resolves once the server echoes the created item
        # (the echo carries the client-supplied id).
        assert not update_task.done()
        _server_event(session, _item_added_echo(assistant_event["item"]))
        await asyncio.wait_for(update_task, timeout=1.0)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_chat_ctx_drops_system_and_developer_items(monkeypatch):
    # The server's conversation store only accepts role "assistant"/"user"
    # (add_resource_item raises ValueError otherwise); system/developer items
    # would never see their conversation.item.added echo. The plugin must
    # drop them client-side instead of sending a create the server rejects,
    # and remap surrounding previous_item_id links around the drop.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext(
                    [
                        llm.ChatMessage(id="user_1", role="user", content=["hi"]),
                        llm.ChatMessage(id="sys_1", role="system", content=["System note."]),
                        llm.ChatMessage(id="dev_1", role="developer", content=["Developer note."]),
                        llm.ChatMessage(id="user_2", role="user", content=["again"]),
                    ]
                )
            )
        )
        first_event = await session._msg_ch.recv()
        second_event = await session._msg_ch.recv()
        assert session._msg_ch.empty()

        assert first_event["item"]["id"] == "user_1"
        assert second_event["item"]["id"] == "user_2"
        assert second_event["previous_item_id"] == "user_1"

        _server_event(session, _item_added_echo(first_event["item"]))
        _server_event(session, _item_added_echo(second_event["item"], "user_1"))
        await asyncio.wait_for(update_task, timeout=1.0)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_server_error_fails_pending_generate_reply(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()

        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_request",
                    "message": "response.create failed",
                    "event_id": response_create["event_id"],
                },
            },
        )

        with pytest.raises(llm.RealtimeError, match="response.create failed"):
            await generation_fut
        assert session._response_created_futures == {}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_unrelated_error_does_not_fail_pending_generate_reply(monkeypatch):
    # An error whose event_id can't be correlated to a pending generate_reply()
    # (e.g. it belongs to an unrelated conversation.item.create) must not
    # blanket-fail every other in-flight generate_reply() call.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        await session._msg_ch.recv()  # response.create

        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_request",
                    "message": "conversation.item.create failed",
                    "event_id": "chat_ctx_create_unrelated",
                },
            },
        )

        # The session-level error still surfaces (so it isn't silently lost)...
        assert errors
        # ...but the unrelated pending generate_reply() is left alone.
        assert not generation_fut.done()
        assert generation_fut in session._response_created_futures.values()
    finally:
        generation_fut.cancel()
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_truncation_defaults_to_auto(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["truncation"] == "auto"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_truncation_can_be_disabled(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        truncation="disabled",
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["truncation"] == "disabled"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_options_sends_truncation(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        model.update_options(truncation="disabled")

        event = await session._msg_ch.recv()
        assert event["session"]["truncation"] == "disabled"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_options_ignores_none_truncation(monkeypatch):
    # Unlike turn_detection/noise_reduction, the server's `truncation` field is
    # non-Optional ("auto"/"disabled" only); sending null would fail server
    # validation and tear down the whole session. `None` must be a no-op here,
    # not forwarded to the wire.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        truncation="disabled",
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        model.update_options(truncation=None, voice="coral")

        event = await session._msg_ch.recv()
        assert event["session"]["truncation"] == "disabled"
        assert event["session"]["audio"]["output"]["voice"] == "coral"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_non_retryable_billing_refusal_close_code_is_terminal():
    # 4429 (WS_CLOSE_CODE_ENTITLEMENT_REFUSED) is a permanent billing refusal;
    # reconnecting cannot fix it.
    ws = _FakeWebSocket(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 4429, "insufficient_quota"))
    ws.close_event.set()
    client = _SequencedHTTPClient([ws, _FakeWebSocket()])
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        for _ in range(200):
            if session._closed:
                break
            await asyncio.sleep(0.01)
        assert session._closed is True
        assert len(client.connect_urls) == 1  # no reconnect attempt
        assert errors
        assert errors[0].recoverable is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_nonfatal_response_not_active_error_is_swallowed(monkeypatch):
    # A redundant response.cancel (nothing active) must not surface as a
    # user-facing recoverable error, matching Boson's own client (pipecat).
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "response_not_active",
                    "message": "No active response to cancel.",
                    "event_id": "evt_cancel_1",
                },
            },
        )
        assert errors == []
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_nonfatal_invalid_previous_item_id_is_swallowed(monkeypatch):
    # Raised without an event_id/code the client can correlate; must not
    # blanket-fail unrelated pending generate_reply() futures.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        await session._msg_ch.recv()

        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_previous_item_id",
                    "message": "previous_item_id 'user_9' not found in conversation",
                },
            },
        )
        assert errors == []
        assert not generation_fut.done()
    finally:
        generation_fut.cancel()
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_invalid_previous_item_id_fails_update_chat_ctx_fast(monkeypatch):
    # invalid_previous_item_id carries no event_id, and names the missing
    # previous_item_id rather than the rejected item's own id. Without
    # matching it to the specific pending create that used it, update_chat_ctx()
    # would sit until the base class's own 5s timeout and raise a generic
    # "timed out" instead of this specific, immediate error.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(session, _user_item_added("user_1", [{"type": "input_text", "text": "hi"}]))

        # user_2 chains onto user_1 as its previous_item_id.
        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext(
                    [
                        llm.ChatMessage(id="user_1", role="user", content=["hi"]),
                        llm.ChatMessage(id="user_2", role="user", content=["again"]),
                    ]
                )
            )
        )
        create_event = await session._msg_ch.recv()
        assert create_event["item"]["id"] == "user_2"
        assert create_event["previous_item_id"] == "user_1"

        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_previous_item_id",
                    "message": "previous_item_id 'user_1' not found in conversation",
                },
            },
        )

        with pytest.raises(llm.RealtimeError, match="previous_item_id 'user_1' not found"):
            await asyncio.wait_for(update_task, timeout=0.5)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_unparseable_invalid_previous_item_id_blames_nobody(monkeypatch):
    # Which item an invalid_previous_item_id error refers to is only
    # recoverable by parsing its message, so a wording change makes it
    # unrecoverable. It must not then be pinned on an unrelated item: `None`
    # is a real previous_item_id (append-at-tail), so a failed parse yielding
    # None must not match the pending append below.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        # An append onto an empty remote context: previous_item_id is None.
        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext([llm.ChatMessage(id="user_1", role="user", content=["hi"])])
            )
        )
        create_event = await session._msg_ch.recv()
        assert create_event["item"]["id"] == "user_1"
        assert create_event["previous_item_id"] is None
        assert session._pending_item_create_previous_ids == {"user_1": None}

        # Same error type, wording the regex cannot parse.
        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_previous_item_id",
                    "message": "the item you referenced does not exist",
                },
            },
        )

        # The append is untouched: no exception pinned on it, still pending.
        fut = session._item_create_future["user_1"]
        assert not fut.done()
        assert session._chat_ctx_sync_error is None
        assert not update_task.done()

        # It still resolves normally when its own echo arrives.
        _server_event(session, _user_item_added("user_1", [{"type": "input_text", "text": "hi"}]))
        await asyncio.wait_for(update_task, timeout=0.5)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_concurrent_update_chat_ctx_keeps_sync_error(monkeypatch):
    # The sync error travels through a single session-level slot, and the base
    # gathers its item futures with return_exceptions=True — so that slot is
    # the only channel a per-item failure has. A second update_chat_ctx()
    # entering while the first is parked inside the base must not clear the
    # slot out from under it, or the first call reports success for a sync
    # that actually failed.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    second_task: asyncio.Task | None = None
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(session, _user_item_added("user_1", [{"type": "input_text", "text": "hi"}]))

        first_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext(
                    [
                        llm.ChatMessage(id="user_1", role="user", content=["hi"]),
                        llm.ChatMessage(id="user_2", role="user", content=["again"]),
                    ]
                )
            )
        )
        create_event = await session._msg_ch.recv()
        assert create_event["item"]["id"] == "user_2"

        # Queued before the error lands, so its body runs before the first
        # call gets to look at the slot.
        second_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext([llm.ChatMessage(id="user_1", role="user", content=["hi"])])
            )
        )

        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_previous_item_id",
                    "message": "previous_item_id 'user_1' not found in conversation",
                },
            },
        )

        with pytest.raises(llm.RealtimeError, match="previous_item_id 'user_1' not found"):
            await asyncio.wait_for(first_task, timeout=0.5)
    finally:
        if second_task is not None:
            await utils.aio.cancel_and_wait(second_task)
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_unsupported_role_at_head_does_not_rebuild(monkeypatch):
    # A system/developer item is never sent, so it never lands in the remote
    # context — which means the diff would see it as missing remotely on every
    # single sync. At the head of the context that reads as an insert before
    # everything else and triggers the full delete/recreate rebuild, forever,
    # for an item that is dropped either way. Unsupported roles must be
    # filtered before the diff, not while walking its output.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(session, _user_item_added("user_1", [{"type": "input_text", "text": "hi"}]))

        target = llm.ChatContext(
            [
                llm.ChatMessage(id="sys_1", role="system", content=["You are helpful."]),
                llm.ChatMessage(id="user_1", role="user", content=["hi"]),
            ]
        )
        # Already in sync apart from the undeliverable system item: nothing to do.
        assert session._create_update_chat_ctx_events(target) == []
        # And still nothing to do on the next sync of the same context.
        assert session._create_update_chat_ctx_events(target) == []

        # A genuine head insert of a *supported* item still rebuilds.
        with_summary = llm.ChatContext(
            [
                llm.ChatMessage(id="sys_1", role="system", content=["You are helpful."]),
                llm.ChatMessage(id="summary_1", role="assistant", content=["Summary."]),
                llm.ChatMessage(id="user_1", role="user", content=["hi"]),
            ]
        )
        events = session._create_update_chat_ctx_events(with_summary)
        assert [ev.type for ev in events] == [
            "conversation.item.delete",
            "conversation.item.create",
            "conversation.item.create",
        ]
        assert events[0].item_id == "user_1"
        assert events[1].item.id == "summary_1"
        assert events[1].previous_item_id is None
        assert events[2].item.id == "user_1"
        assert events[2].previous_item_id == "summary_1"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_invalid_previous_item_id_does_not_affect_unrelated_call(monkeypatch):
    # A late invalid_previous_item_id error for a create that's no longer
    # pending (its update_chat_ctx() call already gave up and returned) must
    # not be misattributed to a different, currently in-flight update_chat_ctx()
    # call just because something happens to be pending right now.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        # Simulate call 1 having already timed out and cleaned up: an item's
        # previous_item_id mapping is recorded, but its future is gone.
        stale_fut: asyncio.Future[None] = asyncio.Future()
        session._pending_item_create_previous_ids["stale_item"] = "root_1"
        session._item_create_future["stale_item"] = stale_fut
        stale_fut.set_result(None)  # already resolved; simulates prior cleanup
        del session._item_create_future["stale_item"]  # base class pops on timeout

        # Call 2 is a genuinely different, currently in-flight sync.
        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext([llm.ChatMessage(id="user_9", role="user", content=["hi"])])
            )
        )
        create_event = await session._msg_ch.recv()
        assert create_event["item"]["id"] == "user_9"

        # Call 1's stale error arrives late, referencing its own (unrelated,
        # already-gone) previous_item_id.
        _server_event(
            session,
            {
                "type": "error",
                "error": {
                    "type": "invalid_previous_item_id",
                    "message": "previous_item_id 'root_1' not found in conversation",
                },
            },
        )
        assert not update_task.done()

        # Call 2 completes normally, unaffected by call 1's stale error.
        _server_event(session, _item_added_echo(create_event["item"]))
        await asyncio.wait_for(update_task, timeout=1.0)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_server_vad_interrupt_skips_duplicate_response_cancel(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        await generation_fut

        session.on("input_speech_started", lambda _event: session.interrupt())
        _server_event(session, {"type": "input_audio_buffer.speech_started"})
        assert session._msg_ch.empty()

        session.interrupt()
        cancel_event = await session._msg_ch.recv()
        assert cancel_event["type"] == "response.cancel"
        assert cancel_event["response_id"] == "resp_1"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_interrupt_response_false_still_skips_duplicate_cancel(monkeypatch):
    # `interrupt_response` in the turn_detection payload does not currently
    # change Higgs Realtime's actual auto-interrupt behavior: server VAD always
    # auto-cancels the active response on speech start regardless of this
    # field. The client must suppress its own duplicate response.cancel
    # whenever server VAD is enabled at all, even if the caller set
    # `interrupt_response: False` expecting it to disable auto-interrupt.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        turn_detection={"type": "server_vad", "interrupt_response": False},
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        await generation_fut

        session.on("input_speech_started", lambda _event: session.interrupt())
        _server_event(session, {"type": "input_audio_buffer.speech_started"})
        assert session._msg_ch.empty()
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_client_side_interrupt_sends_response_cancel(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        turn_detection=None,
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()
        _server_event(
            session,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            },
        )
        await generation_fut

        session.interrupt()
        cancel_event = await session._msg_ch.recv()
        assert cancel_event["type"] == "response.cancel"
        assert cancel_event["response_id"] == "resp_1"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_chat_ctx_audio_content_uses_transcript(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        chat_ctx = llm.ChatContext(
            [
                llm.ChatMessage(
                    id="user_audio_1",
                    role="user",
                    content=[llm.AudioContent(frame=[], transcript="hello from audio")],
                )
            ]
        )

        update_task = asyncio.create_task(session.update_chat_ctx(chat_ctx))
        create_event = await session._msg_ch.recv()

        assert create_event["type"] == "conversation.item.create"
        assert create_event["item"]["id"] == "user_audio_1"
        assert create_event["item"]["content"] == [
            {"type": "input_text", "text": "hello from audio"}
        ]

        _server_event(session, _item_added_echo(create_event["item"]))
        await asyncio.wait_for(update_task, timeout=1.0)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_chat_ctx_deletes_removed_items(monkeypatch):
    # Client-supplied item ids are preserved by the server, so removed items
    # are addressable: the base diff issues conversation.item.delete for them.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(session, _user_item_added("user_1", [{"type": "input_text", "text": "hi"}]))
        _server_event(
            session, _user_item_added("user_2", [{"type": "input_text", "text": "again"}])
        )

        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext([llm.ChatMessage(id="user_1", role="user", content=["hi"])])
            )
        )
        delete_event = await session._msg_ch.recv()
        assert delete_event["type"] == "conversation.item.delete"
        assert delete_event["item_id"] == "user_2"
        assert session._msg_ch.empty()

        _server_event(session, {"type": "conversation.item.deleted", "item_id": "user_2"})
        await asyncio.wait_for(update_task, timeout=1.0)
        assert session._remote_chat_ctx.get("user_2") is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_chat_ctx_skips_textless_items_and_remaps_prev(monkeypatch):
    # Items the server cannot store (no text content) are not sent, and any
    # previous_item_id pointing at one is remapped to its predecessor so the
    # server never sees an unknown previous_item_id.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext(
                    [
                        llm.ChatMessage(id="user_1", role="user", content=["first"]),
                        llm.ChatMessage(
                            id="img_1",
                            role="user",
                            content=[llm.ImageContent(image="https://example.com/a.png")],
                        ),
                        llm.ChatMessage(id="user_2", role="user", content=["second"]),
                    ]
                )
            )
        )
        first_event = await session._msg_ch.recv()
        second_event = await session._msg_ch.recv()
        assert session._msg_ch.empty()

        assert first_event["item"]["id"] == "user_1"
        assert second_event["item"]["id"] == "user_2"
        assert second_event["previous_item_id"] == "user_1"

        _server_event(session, _item_added_echo(first_event["item"]))
        _server_event(session, _item_added_echo(second_event["item"], "user_1"))
        await asyncio.wait_for(update_task, timeout=1.0)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_update_chat_ctx_rebuilds_on_head_insert(monkeypatch):
    # The server has no insert-at-head primitive (previous_item_id=None only
    # means append-at-tail). Prepending a new item ahead of turns the server
    # already has (e.g. a context summary) must not be silently mapped to
    # append-at-tail, which would misorder the conversation — it must rebuild
    # the remote conversation in the target order instead.
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        _server_event(session, _user_item_added("user_1", [{"type": "input_text", "text": "hi"}]))
        _server_event(
            session,
            _item_added_echo(
                {
                    "id": "asst_1",
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "text", "text": "hello"}],
                },
                "user_1",
            ),
        )
        assert len(session.chat_ctx.items) == 2

        update_task = asyncio.create_task(
            session.update_chat_ctx(
                llm.ChatContext(
                    [
                        llm.ChatMessage(id="summary_1", role="assistant", content=["Summary."]),
                        llm.ChatMessage(id="user_1", role="user", content=["hi"]),
                        llm.ChatMessage(id="asst_1", role="assistant", content=["hello"]),
                    ]
                )
            )
        )

        delete_1 = await session._msg_ch.recv()
        delete_2 = await session._msg_ch.recv()
        assert delete_1["type"] == "conversation.item.delete"
        assert delete_2["type"] == "conversation.item.delete"
        assert {delete_1["item_id"], delete_2["item_id"]} == {"user_1", "asst_1"}

        create_summary = await session._msg_ch.recv()
        create_user = await session._msg_ch.recv()
        create_asst = await session._msg_ch.recv()
        assert session._msg_ch.empty()

        assert create_summary["item"]["id"] == "summary_1"
        assert create_summary["previous_item_id"] is None
        assert create_user["item"]["id"] == "user_1"
        assert create_user["previous_item_id"] == "summary_1"
        assert create_asst["item"]["id"] == "asst_1"
        assert create_asst["previous_item_id"] == "user_1"

        _server_event(
            session, {"type": "conversation.item.deleted", "item_id": delete_1["item_id"]}
        )
        _server_event(
            session, {"type": "conversation.item.deleted", "item_id": delete_2["item_id"]}
        )
        _server_event(session, _item_added_echo(create_summary["item"]))
        _server_event(session, _item_added_echo(create_user["item"], "summary_1"))
        _server_event(session, _item_added_echo(create_asst["item"], "user_1"))
        await asyncio.wait_for(update_task, timeout=1.0)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_commit_audio_skips_short_buffers(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update

        session._pushed_duration_s = 0.05
        session.commit_audio()
        assert session._msg_ch.empty()
        assert session._pushed_duration_s == 0.05

        session._pushed_duration_s = 0.11
        session.commit_audio()
        commit_event = await session._msg_ch.recv()
        assert commit_event["type"] == "input_audio_buffer.commit"
        assert session._pushed_duration_s == 0.0
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_run_ws_cancellation_preserves_cancelled_error(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    ws = _FakeWebSocket()
    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    run_task = asyncio.create_task(session._run_ws(ws))
    try:
        await asyncio.sleep(0)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_websocket_close_fails_pending_generate_reply():
    ws = _FakeWebSocket()
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=_FakeHTTPClient(ws),
        conn_options=APIConnectOptions(max_retry=0, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        generation_fut = session.generate_reply()
        for _ in range(100):
            if any('"type": "response.create"' in sent for sent in ws.sent):
                break
            await asyncio.sleep(0.01)
        assert any('"type": "response.create"' in sent for sent in ws.sent)

        ws.close_event.set()

        with pytest.raises(llm.RealtimeError, match="closed unexpectedly"):
            await asyncio.wait_for(generation_fut, timeout=1.0)
        assert session._closed is True
        queued_before = session._msg_ch.qsize()
        session.send_event({"type": "input_audio_buffer.clear"})
        assert session._msg_ch.qsize() == queued_before
        assert errors
        assert errors[0].recoverable is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_websocket_close_includes_close_code():
    ws = _FakeWebSocket(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 3000, "Invalid API key"))
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="bad-key",
        http_session=_FakeHTTPClient(ws),
        conn_options=APIConnectOptions(max_retry=0, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        generation_fut = session.generate_reply()
        for _ in range(100):
            if any('"type": "response.create"' in sent for sent in ws.sent):
                break
            await asyncio.sleep(0.01)
        assert any('"type": "response.create"' in sent for sent in ws.sent)

        ws.close_event.set()

        with pytest.raises(llm.RealtimeError) as exc_info:
            await asyncio.wait_for(generation_fut, timeout=1.0)
        assert "close_code=3000" in str(exc_info.value)
        assert errors
        assert "close_code=3000" in str(errors[0].error)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_reconnects_and_replays_chat_ctx_after_ws_drop():
    # First connection delivers one conversation item, then drops.
    item_added = {
        "type": "conversation.item.added",
        "previous_item_id": None,
        "item": {
            "id": "user_1",
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    }
    ws1 = _ScriptedFakeWebSocket(
        messages=[aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(item_added), None)]
    )
    ws1.close_event.set()
    ws2 = _FakeWebSocket()
    client = _SequencedHTTPClient([ws1, ws2])
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    # A response active on the first connection when it drops never sees its
    # response.done on the new connection; reconnection must not carry the id
    # over as "current" (it can no longer match anything real).
    session._current_response_id = "resp_stale"
    errors = []
    reconnected = []
    session.on("error", errors.append)
    session.on("session_reconnected", reconnected.append)
    try:
        for _ in range(200):
            if reconnected:
                break
            await asyncio.sleep(0.01)
        assert reconnected
        assert len(client.connect_urls) == 2
        assert session._current_response_id is None

        # The session config is re-sent first on the new connection, then the
        # local chat-context mirror is replayed via conversation.item.create
        # (the server keeps no state across connections).
        assert ws2.sent
        reconnect_update = json.loads(ws2.sent[0])
        assert reconnect_update["type"] == "session.update"
        replayed = [
            json.loads(sent)
            for sent in ws2.sent
            if json.loads(sent)["type"] == "conversation.item.create"
        ]
        assert len(replayed) == 1
        assert replayed[0]["item"]["id"] == "user_1"
        assert replayed[0]["item"]["content"] == [{"type": "input_text", "text": "hello"}]

        # The drop surfaced as a recoverable error and the session stayed usable.
        assert errors
        assert errors[0].recoverable is True
        assert session._closed is False
        assert not session._msg_ch.closed
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_tracks_server_assigned_session_id(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        assert session.session_id is None
        _server_event(
            session,
            {
                "type": "session.created",
                "event_id": "evt_1",
                "session": {"id": "sess_server123", "object": "realtime.session"},
            },
        )
        assert session.session_id == "sess_server123"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_non_retryable_close_code_is_terminal():
    ws = _FakeWebSocket(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 3000, "Invalid API key"))
    ws.close_event.set()
    client = _SequencedHTTPClient([ws, _FakeWebSocket()])
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="bad-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        for _ in range(200):
            if session._closed:
                break
            await asyncio.sleep(0.01)
        assert session._closed is True
        assert len(client.connect_urls) == 1  # no reconnect attempt
        assert errors
        assert errors[0].recoverable is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_retries_exhausted_is_terminal():
    client = _SequencedHTTPClient(
        [
            aiohttp.ClientConnectionError("refused"),
            aiohttp.ClientConnectionError("refused"),
            aiohttp.ClientConnectionError("refused"),
        ]
    )
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        for _ in range(200):
            if session._closed:
                break
            await asyncio.sleep(0.01)
        assert session._closed is True
        assert len(client.connect_urls) == 3  # initial attempt + max_retry
        assert [e.recoverable for e in errors] == [True, True, False]
        assert session._msg_ch.closed
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_idle_timeout_close_is_terminal():
    idle_event = {"type": "session.idle_timeout", "seconds_idle": 120}
    ws = _ScriptedFakeWebSocket(
        messages=[aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(idle_event), None)]
    )
    ws.close_event.set()  # server closes right after announcing the idle timeout
    client = _SequencedHTTPClient([ws, _FakeWebSocket()])
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        for _ in range(200):
            if session._closed:
                break
            await asyncio.sleep(0.01)
        assert session._closed is True
        assert len(client.connect_urls) == 1  # the server ended the session; no reconnect
        assert errors
        assert errors[0].recoverable is False
        assert "idle timeout" in str(errors[0].error)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_max_duration_close_is_terminal():
    max_duration_event = {"type": "session.max_duration_reached", "max_duration_sec": 7200}
    ws = _ScriptedFakeWebSocket(
        messages=[aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(max_duration_event), None)]
    )
    ws.close_event.set()  # server closes right after announcing the max duration
    client = _SequencedHTTPClient([ws, _FakeWebSocket()])
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    errors = []
    session.on("error", errors.append)
    try:
        for _ in range(200):
            if session._closed:
                break
            await asyncio.sleep(0.01)
        assert session._closed is True
        assert len(client.connect_urls) == 1  # the server ended the session; no reconnect
        assert errors
        assert errors[0].recoverable is False
        assert "max session duration" in str(errors[0].error)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_pending_reply_fails_recoverably_on_reconnect():
    ws1 = _FakeWebSocket()
    ws2 = _FakeWebSocket()
    client = _SequencedHTTPClient([ws1, ws2])
    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        http_session=client,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    reconnected = []
    session.on("session_reconnected", reconnected.append)
    try:
        generation_fut = session.generate_reply()
        for _ in range(100):
            if any('"type": "response.create"' in sent for sent in ws1.sent):
                break
            await asyncio.sleep(0.01)
        assert any('"type": "response.create"' in sent for sent in ws1.sent)

        ws1.close_event.set()  # drop the connection with the reply in flight

        with pytest.raises(llm.RealtimeError, match="session reconnection"):
            await asyncio.wait_for(generation_fut, timeout=2.0)
        for _ in range(200):
            if reconnected:
                break
            await asyncio.sleep(0.01)
        assert reconnected
        assert session._closed is False
    finally:
        await session.aclose()
        await model.aclose()
