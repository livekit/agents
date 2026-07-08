import asyncio
import base64
import json

import aiohttp
import pytest
from openai.types.realtime import (
    ConversationItemAdded,
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

from livekit.agents import llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.boson import realtime

pytestmark = pytest.mark.plugin("boson")


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
        assert model.capabilities.mutable_chat_context is False
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
            "prompt": "Transcribe names carefully.",
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
async def test_boson_realtime_text_only_modality_session_update(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        modalities=["text"],
    )
    session = model.session()
    try:
        event = await session._msg_ch.recv()
        assert event["session"]["output_modalities"] == ["text"]
        assert model.capabilities.audio_output is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize("modalities", [["text", "audio"], [], ["video"]])
def test_boson_realtime_rejects_invalid_modalities(modalities):
    with pytest.raises(ValueError):
        realtime.RealtimeModel(
            url="ws://localhost:8000/v1/realtime/",
            api_key="test-key",
            modalities=modalities,
        )


@pytest.mark.asyncio
async def test_boson_realtime_text_only_generation_streams_text(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(
        url="ws://localhost:8000/v1/realtime/",
        api_key="test-key",
        modalities=["text"],
    )
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply(instructions="Say hi.")
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
            input_audio_transcription={"model": "asr-v2", "prompt": "ASR hints."},
            input_audio_noise_reduction="near_field",
            turn_detection={"type": "server_vad", "silence_duration_ms": 900},
        )

        event = await session._msg_ch.recv()
        assert event["type"] == "session.update"
        assert event["session"]["audio"]["input"]["transcription"] == {
            "model": "asr-v2",
            "prompt": "ASR hints.",
        }
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
async def test_boson_realtime_model_aclose_closes_sessions(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    await session._msg_ch.recv()  # initial session.update

    await model.aclose()

    assert session._closed is True
    assert session._msg_ch.closed


@pytest.mark.asyncio
async def test_boson_realtime_generation_audio_mapping(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply(instructions="Say hi.")
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
        assert event_id in session._pending_response_futures

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
        assert session._pending_response_futures == {}
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
        await session.update_chat_ctx(
            llm.ChatContext(
                [
                    llm.ChatMessage(id="dev_1", role="developer", content=["System note."]),
                    llm.ChatMessage(id="asst_1", role="assistant", content=["Assistant note."]),
                ]
            )
        )
        developer_event = await session._msg_ch.recv()
        assistant_event = await session._msg_ch.recv()

        assert developer_event["item"]["role"] == "system"
        assert developer_event["item"]["content"] == [
            {"type": "input_text", "text": "System note."}
        ]
        assert assistant_event["item"]["role"] == "assistant"
        assert assistant_event["item"]["content"] == [{"type": "text", "text": "Assistant note."}]
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
        generation_fut = session.generate_reply(instructions="Say hi.")
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
        assert session._pending_response_futures == {}
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
        generation_fut = session.generate_reply(instructions="Say hi.")
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
        generation_fut = session.generate_reply(instructions="Say hi.")
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

        await session.update_chat_ctx(chat_ctx)
        create_event = await session._msg_ch.recv()

        assert create_event["type"] == "conversation.item.create"
        assert create_event["item"]["id"] == "user_audio_1"
        assert create_event["item"]["content"] == [
            {"type": "input_text", "text": "hello from audio"}
        ]
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
        generation_fut = session.generate_reply(instructions="Say hi.")
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
        generation_fut = session.generate_reply(instructions="Say hi.")
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
        generation_fut = session.generate_reply(instructions="Say hi.")
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
