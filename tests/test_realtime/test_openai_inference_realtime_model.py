from __future__ import annotations

import asyncio
import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit.agents import llm, utils
from livekit.agents._exceptions import APIError
from livekit.agents.inference import RealtimeModel
from livekit.agents.inference.realtime import openai as inference_realtime
from livekit.agents.inference.realtime.openai import RealtimeSession
from livekit.agents.llm._realtime.openai import _DiscardedGeneration, _MessageGeneration

pytestmark = pytest.mark.unit


def test_plugin_import_is_canonical_class() -> None:
    from livekit.plugins.openai.realtime import InferenceRealtimeModel
    from livekit.plugins.openai.realtime.inference_realtime_model import (
        InferenceRealtimeSession,
    )

    assert InferenceRealtimeModel is RealtimeModel
    assert InferenceRealtimeSession is RealtimeSession


def test_hosted_model_keeps_canonical_label() -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )

    assert model.label == "livekit.agents.inference.realtime.openai.RealtimeModel"


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        pass


class _FakeHTTPSession:
    def __init__(self) -> None:
        self.connections: list[tuple[str, dict[str, str], _FakeWebSocket]] = []

    async def ws_connect(self, *, url: str, headers: dict[str, str]) -> _FakeWebSocket:
        ws = _FakeWebSocket()
        self.connections.append((url, headers, ws))
        return ws


@pytest.fixture
def paused_realtime_main(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _paused_main(self: RealtimeSession) -> None:
        await self._msg_ch._close_ev.wait()

    monkeypatch.setattr(RealtimeSession, "_main_task", _paused_main)


def test_requires_provider_prefixed_model() -> None:
    with pytest.raises(ValueError, match="provider-prefixed"):
        RealtimeModel("gpt-realtime", api_key="key", api_secret="secret")


def test_points_duplex_models_at_duplex_model() -> None:
    with pytest.raises(ValueError, match="inference.DuplexModel"):
        RealtimeModel("openai/gpt-live-1", api_key="key", api_secret="secret")


@pytest.mark.parametrize(
    ("model", "replacement"),
    [
        ("openai/gpt-realtime-2", "openai/gpt-realtime-2.1"),
        ("openai/gpt-realtime-1.5", "openai/gpt-realtime-2.1"),
        ("xai/grok-voice-think-fast-2.0", "xai/grok-voice-3.0"),
    ],
)
def test_deprecated_models_still_connect_and_name_their_replacement(
    model: str, replacement: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        resolved = RealtimeModel(model, api_key="key", api_secret="secret")

    assert resolved.model == model
    assert f"{model} is deprecated, use {replacement} instead" in caplog.text


def test_inference_class_warns_that_it_has_no_effect(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        RealtimeModel(
            "openai/gpt-realtime", api_key="key", api_secret="secret", inference_class="priority"
        )

    assert "inference_class is deprecated" in caplog.text


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("key", "api_key is required"),
        ("secret", "api_secret is required"),
    ],
)
def test_requires_livekit_credentials(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    message: str,
) -> None:
    for name in (
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
        "LIVEKIT_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    kwargs = {"api_key": "key", "api_secret": "secret"}
    kwargs[f"api_{missing}"] = None

    with pytest.raises(ValueError, match=message):
        RealtimeModel("openai/gpt-realtime", **kwargs)


def test_credentials_and_url_follow_inference_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "inference-key")
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_SECRET", "inference-secret")
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://inference.example/v1")

    model = RealtimeModel("openai/gpt-realtime")

    assert model._inference_opts.api_key == "inference-key"
    assert model._inference_opts.api_secret == "inference-secret"
    assert model._opts.base_url == "https://inference.example/v1"


async def test_openai_environment_does_not_enable_azure_mode(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-04-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    model = RealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret-with-at-least-32-bytes-for-jwt",
    )
    session = model.session()

    url, _ = session._create_ws_url_and_headers()
    parsed = urlparse(url)
    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert model._opts.is_azure is False
    assert model._opts.api_version is None
    assert parsed.scheme == "wss"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query) == {"model": ["openai/gpt-realtime"]}
    assert dumped["session"]["type"] == "realtime"
    await session.aclose()


async def test_connection_refreshes_livekit_auth_and_custom_headers(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    tokens = iter(("token-one", "token-two"))
    monkeypatch.setattr(
        inference_realtime,
        "create_access_token",
        lambda key, secret: f"{next(tokens)}:{key}:{secret}",
    )
    monkeypatch.setattr(inference_realtime, "get_inference_headers", lambda: {"X-Test": "custom"})
    http_session = _FakeHTTPSession()
    model = RealtimeModel(
        "openai/gpt-realtime",
        provider="openai",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
    )
    session = model.session()

    await session._create_ws_conn()
    await session._create_ws_conn()

    parsed = urlparse(http_session.connections[0][0])
    assert parsed.scheme == "wss"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query) == {"model": ["openai/gpt-realtime"]}
    assert [headers["Authorization"] for _, headers, _ in http_session.connections] == [
        "Bearer token-one:key:secret",
        "Bearer token-two:key:secret",
    ]
    assert http_session.connections[0][1]["X-Test"] == "custom"
    assert http_session.connections[0][1]["X-LiveKit-Inference-Provider"] == "openai"
    await session.aclose()


async def test_initial_session_update_omits_gateway_model_field(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert dumped["type"] == "session.update"
    assert dumped["session"]["type"] == "realtime"
    assert "model" not in dumped["session"]
    await session.aclose()


@pytest.mark.parametrize(
    "code",
    [
        "unsupported_transcription_model",
        "unsupported_audio_transport",
        "unsupported_audio_format",
        "insufficient_quota",
    ],
)
async def test_gateway_configuration_errors_are_fatal(
    code: str,
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    assert session._is_fatal_error(SimpleNamespace(code=code))
    await session.aclose()


@pytest.mark.parametrize(
    ("code", "type_", "fatal"),
    [
        ("some_new_code", "unsupported_audio_format", True),
        ("some_new_code", "insufficient_quota", True),
        (None, "unsupported_audio_transport", True),
        ("", "insufficient_quota", True),
        ("invalid_audio_payload", "invalid_request_error", False),
        ("", "", False),
        (None, None, False),
    ],
)
async def test_gateway_fatal_errors_match_either_code_or_type(
    code: str | None,
    type_: str | None,
    fatal: bool,
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    try:
        assert session._is_fatal_error(SimpleNamespace(code=code, type=type_)) is fatal
    finally:
        await session.aclose()


async def test_gateway_invalid_audio_payload_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    emitted: list[tuple[APIError, bool]] = []
    monkeypatch.setattr(
        session,
        "_emit_error",
        lambda error, recoverable: emitted.append((error, recoverable)),
    )

    session._handle_error(
        SimpleNamespace(
            error=SimpleNamespace(
                event_id=None,
                message="malformed audio frame",
                code="invalid_audio_payload",
            )
        )
    )

    assert len(emitted) == 1
    assert emitted[0][0].retryable is True
    assert emitted[0][1] is True
    await session.aclose()


async def test_gateway_configuration_error_stops_reconnect(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    with pytest.raises(APIError) as exc_info:
        session._handle_error(
            SimpleNamespace(
                error=SimpleNamespace(
                    event_id=None,
                    message="unsupported transport",
                    code="unsupported_audio_transport",
                )
            )
        )

    assert exc_info.value.retryable is False
    await session.aclose()


async def test_gateway_malformed_error_code_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    emitted: list[tuple[APIError, bool]] = []
    monkeypatch.setattr(
        session,
        "_emit_error",
        lambda error, recoverable: emitted.append((error, recoverable)),
    )

    session._handle_error(
        SimpleNamespace(
            error=SimpleNamespace(
                event_id=None,
                message="malformed error event",
                code=["unsupported_audio_transport"],
            )
        )
    )

    assert len(emitted) == 1
    assert emitted[0][0].retryable is True
    assert emitted[0][1] is True
    await session.aclose()


def test_new_api_does_not_expose_deprecated_temperature() -> None:
    parameters = inspect.signature(RealtimeModel).parameters

    assert "temperature" not in parameters


async def test_xai_models_use_gateway_compatible_defaults(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-latest",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert model._opts.voice == "eve"
    assert model.capabilities.turn_detection is True
    assert model.capabilities.can_disable_turn_detection is True
    assert dumped["session"]["audio"]["input"]["transcription"]["model"] == "grok-transcribe"
    assert dumped["session"]["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    await session.aclose()


async def test_xai_gateway_defaults_can_be_overridden(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-latest",
        api_key="key",
        api_secret="secret",
        voice="Ara",
        input_audio_transcription=None,
        turn_detection=None,
    )
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert model._opts.voice == "Ara"
    assert model.capabilities.turn_detection is False
    assert model.capabilities.can_disable_turn_detection is False
    assert dumped["session"]["audio"]["input"].get("transcription") is None
    assert dumped["session"]["audio"]["input"].get("turn_detection") is None
    await session.aclose()


def test_xai_explicit_turn_detection_is_preserved() -> None:
    turn_detection = ServerVad(
        type="server_vad",
        threshold=0.8,
        create_response=False,
        interrupt_response=False,
    )

    model = RealtimeModel(
        "xai/grok-voice-latest",
        api_key="key",
        api_secret="secret",
        turn_detection=turn_detection,
    )

    assert model._opts.turn_detection == turn_detection
    assert model.capabilities.turn_detection is False
    assert model.capabilities.can_disable_turn_detection is False


def _transcript(session: RealtimeSession, item_id: str, text: str, *, status: str | None) -> None:
    payload: dict[str, Any] = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "evt",
        "item_id": item_id,
        "content_index": 0,
        "transcript": text,
    }
    if status is not None:
        payload["status"] = status
    session._handle_conversion_item_input_audio_transcription_completed(
        ConversationItemInputAudioTranscriptionCompletedEvent.construct(**payload)
    )


def _agent_speaks(session: RealtimeSession) -> None:
    session._handle_response_audio_delta(
        ResponseAudioDeltaEvent.construct(
            type="response.output_audio.delta",
            event_id="evt",
            item_id="reply",
            response_id="resp",
            output_index=0,
            content_index=0,
            delta="",
        )
    )


async def test_xai_session_emits_one_final_transcript_per_turn(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-3.0",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    session._current_generation = _DiscardedGeneration()
    session._remote_chat_ctx.insert(None, llm.ChatMessage(id="item_1", role="user", content=[""]))
    emitted: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", emitted.append)

    _transcript(session, "item_1", "Hello, how", status="in_progress")
    _transcript(session, "item_1", "Hello, how are you? Last ones.", status="completed")
    _transcript(
        session, "item_1", "Hello, how are you? Last once. I paid twice.", status="completed"
    )
    assert [ev for ev in emitted if ev.is_final] == []

    _agent_speaks(session)

    assert [(ev.item_id, ev.transcript) for ev in emitted if ev.is_final] == [
        ("item_1", "Hello, how are you? Last once. I paid twice.")
    ]
    remote = session._remote_chat_ctx.get("item_1")
    assert remote is not None
    assert remote.item.content == ["Hello, how are you? Last once. I paid twice."]
    await session.aclose()


async def test_openai_session_finalizes_transcript_immediately(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    emitted: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", emitted.append)

    _transcript(session, "item_1", "what is the weather", status=None)

    assert [(ev.item_id, ev.transcript, ev.is_final) for ev in emitted] == [
        ("item_1", "what is the weather", True)
    ]
    await session.aclose()


async def test_xai_session_flushes_held_transcript_on_next_turn_and_close(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-3.0",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    emitted: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", emitted.append)

    _transcript(session, "item_1", "first turn", status="completed")
    session._handle_input_audio_buffer_speech_started(
        InputAudioBufferSpeechStartedEvent.construct(
            type="input_audio_buffer.speech_started",
            event_id="evt",
            item_id="item_1",
            audio_start_ms=0,
        )
    )
    assert [ev for ev in emitted if ev.is_final] == []

    session._handle_input_audio_buffer_speech_started(
        InputAudioBufferSpeechStartedEvent.construct(
            type="input_audio_buffer.speech_started",
            event_id="evt",
            item_id="item_2",
            audio_start_ms=0,
        )
    )
    assert [(ev.item_id, ev.transcript) for ev in emitted if ev.is_final] == [
        ("item_1", "first turn")
    ]

    _transcript(session, "item_2", "second turn", status="completed")
    await session.aclose()
    assert [(ev.item_id, ev.transcript) for ev in emitted if ev.is_final] == [
        ("item_1", "first turn"),
        ("item_2", "second turn"),
    ]


async def test_xai_session_preserves_held_turn_in_chat_ctx_and_appends_unanchored_items(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-3.0",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    session._remote_chat_ctx.insert(
        None, llm.ChatMessage(id="earlier", role="assistant", content=["hi"])
    )
    session._remote_chat_ctx.insert(
        "earlier", llm.ChatMessage(id="item_1", role="user", content=[""])
    )
    session._remote_chat_ctx.insert(
        "item_1", llm.ChatMessage(id="reply", role="assistant", content=["the full reply"])
    )
    _transcript(session, "item_1", "are you there", status="completed")

    agent_ctx = llm.ChatContext.empty()
    agent_ctx.items.append(llm.ChatMessage(id="earlier", role="assistant", content=["hi"]))
    agent_ctx.items.append(llm.ChatMessage(id="reply", role="assistant", content=["the trunc"]))
    events = session._create_update_chat_ctx_events(agent_ctx)

    deleted = {ev.item_id for ev in events if isinstance(ev, ConversationItemDeleteEvent)}
    created = {
        ev.item.id: ev.previous_item_id
        for ev in events
        if isinstance(ev, ConversationItemCreateEvent)
    }
    assert "item_1" not in deleted
    assert created == {"reply": "item_1"}

    session._handle_conversion_item_added(
        ConversationItemAdded.construct(
            type="conversation.item.added",
            event_id="evt",
            previous_item_id=None,
            item={
                "id": "tail_item",
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
    )
    assert [item.id for item in session._remote_chat_ctx.to_chat_ctx().items] == [
        "earlier",
        "item_1",
        "reply",
        "tail_item",
    ]
    await session.aclose()


async def test_xai_session_discards_abandoned_response(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-3.0",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    session._handle_response_created(
        ResponseCreatedEvent.construct(
            type="response.created",
            event_id="evt",
            response={"id": "resp", "object": "realtime.response", "output": []},
        )
    )
    session._remote_chat_ctx.insert(
        None, llm.ChatMessage(id="phantom", role="assistant", content=[])
    )
    session._current_generation.messages["phantom"] = _MessageGeneration(  # type: ignore[union-attr]
        message_id="phantom",
        text_ch=utils.aio.Chan(),
        audio_ch=utils.aio.Chan(),
        modalities=asyncio.Future(),
    )
    generation = session._current_generation

    session.interrupt()

    assert generation is not None and generation._done_fut.done()
    assert isinstance(session._current_generation, _DiscardedGeneration)
    await session.aclose()


async def test_update_options_voice_omits_empty_audio_input_subobject(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(
        "xai/grok-voice-3.0",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    session._msg_ch.recv_nowait()

    session.update_options(voice="rex")
    update = session._msg_ch.recv_nowait()

    assert update["session"]["audio"] == {"output": {"voice": "rex"}}
    await session.aclose()
