import asyncio
import base64

import aiohttp
import pytest

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


class _ScriptedWebSocket:
    def __init__(self, messages: list[aiohttp.WSMessage]) -> None:
        self.messages = messages
        self.close_event = asyncio.Event()

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        await self.close_event.wait()
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)


@pytest.mark.asyncio
async def test_boson_realtime_session_sends_full_session_update(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
        # No transcription model set -> the server emits no user-transcript events.
        assert model.capabilities.user_transcription is False
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_session_omits_explicit_none_transcription(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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

        session._handle_server_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            }
        )
        generation = await generation_fut

        session._handle_server_event(
            {
                "type": "response.output_item.added",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            }
        )
        message = await generation.message_stream.recv()

        session._handle_server_event(
            {
                "type": "response.content_part.added",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "text", "text": ""},
            }
        )
        assert await message.modalities == ["text"]

        session._handle_server_event(
            {
                "type": "response.output_text.delta",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi there.",
            }
        )
        assert await message.text_stream.recv() == "Hi there."

        session._handle_server_event(
            {
                "type": "response.output_text.done",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Hi there.",
            }
        )
        session._handle_server_event(
            {
                "type": "response.output_item.done",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            }
        )
        session._handle_server_event(
            {
                "type": "response.done",
                "response": {"id": "resp_1", "status": "completed", "usage": None},
            }
        )
        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_noise_reduction_string_normalized(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    await session._msg_ch.recv()  # initial session.update

    await model.aclose()

    assert session._closed is True
    assert session._msg_ch.closed


@pytest.mark.asyncio
async def test_boson_realtime_generation_audio_mapping(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply(instructions="Say hi.")
        response_create = await session._msg_ch.recv()
        assert response_create["type"] == "response.create"

        session._handle_server_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            }
        )
        generation = await generation_fut
        assert generation.response_id == "resp_1"
        assert generation.user_initiated is True

        session._handle_server_event(
            {
                "type": "response.output_item.added",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            }
        )
        message = await generation.message_stream.recv()
        assert message.message_id == "item_1"

        session._handle_server_event(
            {
                "type": "response.content_part.added",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio", "transcript": ""},
            }
        )
        assert await message.modalities == ["audio", "text"]

        pcm = b"\x00\x00" * 240
        session._handle_server_event(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(pcm).decode("utf-8"),
            }
        )
        frame = await message.audio_stream.recv()
        assert frame.sample_rate == 24000
        assert frame.num_channels == 1
        assert frame.samples_per_channel == 240

        session._handle_server_event(
            {
                "type": "response.output_item.done",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message"},
            }
        )
        session._handle_server_event(
            {
                "type": "response.done",
                "response": {"id": "resp_1", "status": "completed", "usage": None},
            }
        )
        assert session._current_generation is None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_generate_reply_uses_response_metadata(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply()
        response_create = await session._msg_ch.recv()
        event_id = response_create["event_id"]

        assert response_create["response"]["metadata"]["client_event_id"] == event_id

        session._handle_server_event(
            {
                "type": "response.created",
                "response": {"id": "resp_auto", "status": "in_progress"},
            }
        )
        assert not generation_fut.done()
        assert event_id in session._pending_response_futures

        session._handle_server_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_manual",
                    "status": "in_progress",
                    "metadata": {"client_event_id": event_id},
                },
            }
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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        session._handle_server_event(
            {
                "type": "response.created",
                "response": {"id": "resp_1", "status": "in_progress"},
            }
        )
        first_generation = session._current_generation
        assert first_generation is not None

        session._handle_server_event(
            {
                "type": "response.created",
                "response": {"id": "resp_2", "status": "in_progress"},
            }
        )
        assert first_generation.message_ch.closed
        assert first_generation.function_ch.closed
        assert session._current_generation is not first_generation
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_remote_items_are_not_recreated(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        session._handle_server_event(
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
            }
        )

        await session.update_chat_ctx(
            llm.ChatContext([llm.ChatMessage(id="user_1", role="user", content=["hello"])])
        )

        assert session._msg_ch.empty()
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_input_transcription_replaces_remote_item_text(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    completed = []
    session.on("input_audio_transcription_completed", completed.append)
    try:
        await session._msg_ch.recv()  # initial session.update
        session._handle_server_event(
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
            }
        )
        session._handle_server_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "user_1",
                "content_index": 0,
                "transcript": "hello world",
                "logprobs": None,
            }
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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply(instructions="Say hi.")
        response_create = await session._msg_ch.recv()

        session._handle_server_event(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_request",
                    "message": "response.create failed",
                    "event_id": response_create["event_id"],
                },
            }
        )

        with pytest.raises(llm.RealtimeError, match="response.create failed"):
            await generation_fut
        assert session._pending_response_futures == {}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_server_vad_interrupt_skips_duplicate_response_cancel(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    try:
        await session._msg_ch.recv()  # initial session.update
        generation_fut = session.generate_reply(instructions="Say hi.")
        response_create = await session._msg_ch.recv()
        session._handle_server_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            }
        )
        await generation_fut

        session.on("input_speech_started", lambda _event: session.interrupt())
        session._handle_server_event({"type": "input_audio_buffer.speech_started"})
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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
        session._handle_server_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "status": "in_progress",
                    "metadata": {"client_event_id": response_create["event_id"]},
                },
            }
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
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

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
async def test_boson_realtime_recv_loop_ignores_invalid_json(monkeypatch):
    monkeypatch.setattr(realtime.RealtimeSession, "_run", _idle_run)

    ws = _ScriptedWebSocket([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, "not-json", None)])
    model = realtime.RealtimeModel(url="ws://localhost:8000/v1/realtime/", api_key="test-key")
    session = model.session()
    errors = []
    session.on("error", errors.append)
    recv_task = asyncio.create_task(session._recv_loop(ws))
    try:
        for _ in range(100):
            if errors:
                break
            await asyncio.sleep(0.01)

        assert errors
        assert errors[0].recoverable is True
        assert "Invalid Boson realtime JSON message" in str(errors[0].error)

        session._closing = True
        ws.close_event.set()
        await asyncio.wait_for(recv_task, timeout=1.0)
    finally:
        recv_task.cancel()
        await asyncio.gather(recv_task, return_exceptions=True)
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_boson_realtime_websocket_close_includes_code_and_reason():
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
        error_message = str(exc_info.value)
        assert "close_code=3000" in error_message
        assert "Invalid API key" in error_message
        assert errors
        assert "close_code=3000" in str(errors[0].error)
        assert "Invalid API key" in str(errors[0].error)
    finally:
        await session.aclose()
        await model.aclose()
