from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

import livekit.agents.inference.stt as inference_stt
import livekit.agents.inference.vad as inference_vad
from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    EndpointingOptions,
    InterruptionOptions,
    TurnHandlingOptions,
)
from livekit.agents.inference._utils import HEADER_CLIENT_FEATURE, HEADER_SESSION_ID
from livekit.agents.inference.stt import (
    STT,
    FallbackModel,
    OpenAIOptions,
    SpeechStream,
    _keyterms_extra_for_model,
    _normalize_fallback,
    _parse_model_string,
)
from livekit.agents.llm import AgentHandoff, ChatMessage
from livekit.agents.stt import FallbackAdapter, RecognizeStream, SpeechEvent, SpeechEventType
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.aio.channel import Chan, ChanEmpty
from livekit.agents.voice import audio_recognition
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_llm import FakeLLM
from .fake_stt import FakeUserSpeech
from .fake_tts import FakeTTS
from .fake_turn_stt import ScriptedTurn, TurnScriptedSTT
from .fake_vad import FakeVAD
from .test_transcription_delay_anchor import _make_recognition

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

# a cartesia/ink-2 final as the gateway sends it: no words, and a duration that is the audio
# billed since the previous turn rather than the utterance span
GATEWAY_INK2_FINAL = {
    "transcript": "are you open on sunday",
    "confidence": 1.0,
    "start": 0.0,
    "duration": 12.5,
    "words": [],
    "language": "en",
}


class _DefaultVAD(FakeVAD):
    """Stands in for the Silero VAD that inference.STT builds when none is passed."""


@pytest.fixture(autouse=True)
def _default_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_vad, "VAD", _DefaultVAD)


def _make_stt(model: str = "deepgram/nova-3", **kwargs: Any) -> STT:
    return STT(
        model=model,
        **{
            "api_key": "test-key",
            "api_secret": "test-secret-long-enough-for-hs256-signing",
            "base_url": "https://example.livekit.cloud",
            **kwargs,
        },
    )


# -- model strings and fallback --


@pytest.mark.parametrize(
    ("model_str", "expected_model", "expected_lang"),
    [
        ("deepgram", "deepgram", NOT_GIVEN),
        ("deepgram/nova-3", "deepgram/nova-3", NOT_GIVEN),
        ("auto", "auto", NOT_GIVEN),
        ("deepgram:en", "deepgram", "en"),
        ("deepgram/nova-3:en", "deepgram/nova-3", "en"),
        ("cartesia/ink-whisper:de", "cartesia/ink-whisper", "de"),
        ("deepgram/nova-2-medical:ja", "deepgram/nova-2-medical", "ja"),
        ("deepgram/nova-3:multi", "deepgram/nova-3", "multi"),
        ("auto:pt", "auto", "pt"),
    ],
)
def test_parse_model_string(model_str: str, expected_model: str, expected_lang: Any) -> None:
    model, language = _parse_model_string(model_str)
    assert model == expected_model
    assert language == expected_lang


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        ("deepgram/nova-3", [{"model": "deepgram/nova-3"}]),
        (FallbackModel(model="deepgram/nova-3"), [{"model": "deepgram/nova-3"}]),
        ([], []),
        # a language suffix is dropped, split on the last colon
        ("deepgram/nova-3:en", [{"model": "deepgram/nova-3"}]),
        ("some:model:part:fr", [{"model": "some:model:part"}]),
        (
            [
                "deepgram/nova-3",
                FallbackModel(model="cartesia/ink-whisper"),
                FallbackModel(model="assemblyai", extra_kwargs={"format_turns": True}),
            ],
            [
                {"model": "deepgram/nova-3"},
                {"model": "cartesia/ink-whisper"},
                {"model": "assemblyai", "extra_kwargs": {"format_turns": True}},
            ],
        ),
    ],
)
def test_normalize_fallback(fallback: Any, expected: list[dict[str, Any]]) -> None:
    assert _normalize_fallback(fallback) == expected


def test_constructor_normalizes_fallback() -> None:
    assert _make_stt()._opts.fallback is NOT_GIVEN
    assert _make_stt(
        fallback=["deepgram/nova-3:en", FallbackModel(model="assemblyai")]
    )._opts.fallback == [
        {"model": "deepgram/nova-3"},
        {"model": "assemblyai"},
    ]


# -- capabilities --


@pytest.mark.parametrize(
    ("model", "extra", "expected"),
    [
        ("deepgram", {}, False),
        ("deepgram", {"diarize": True}, True),
        ("deepgram", {"diarize": False}, False),
        ("xai/stt-1", {"diarize": True}, True),
        ("assemblyai/universal-streaming", {"speaker_labels": True}, True),
    ],
)
def test_diarization_capability_follows_extra(
    model: str, extra: dict[str, Any], expected: bool
) -> None:
    assert _make_stt(model, extra_kwargs=extra).capabilities.diarization is expected


def test_update_options_merges_extra_and_toggles_diarization() -> None:
    stt = _make_stt(extra_kwargs={"diarize": True})
    stt.update_options(extra={"endpointing": 500})
    assert stt._opts.extra_kwargs == {"diarize": True, "endpointing": 500}
    assert stt.capabilities.diarization is True
    stt.update_options(extra={"diarize": False})
    assert stt.capabilities.diarization is False


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("cartesia/ink-whisper", "word"),
        ("deepgram/nova-3", "word"),
        ("assemblyai/universal-streaming", "word"),
        # ink-2 forwards no word timings
        ("cartesia/ink-2", False),
        # the provider is resolved server-side per language, so alignment can't be promised
        ("auto", False),
        ("inworld/inworld-stt-1", False),
        ("new-provider/new-turn-model", False),
    ],
)
def test_aligned_transcript_claim_follows_the_model(model: str, expected: object) -> None:
    assert _make_stt(model).capabilities.aligned_transcript == expected


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        ("cartesia/ink-whisper", "word"),
        ("cartesia/ink-2", False),
        ("new-provider/new-turn-model", False),
    ],
)
def test_fallback_models_constrain_alignment_claim(fallback: str, expected: object) -> None:
    assert _make_stt(fallback=fallback).capabilities.aligned_transcript == expected


def test_alignment_is_recomputed_on_model_update_with_fallback() -> None:
    stt = _make_stt()
    stt.update_options(model="cartesia/ink-2")
    assert stt.capabilities.aligned_transcript is False

    stt = _make_stt("cartesia/ink-2", fallback="new-provider/new-turn-model")
    stt.update_options(model="deepgram/nova-3")
    assert stt.capabilities.aligned_transcript is False


def test_inference_and_plugin_agree_about_ink2(monkeypatch: pytest.MonkeyPatch) -> None:
    cartesia = pytest.importorskip("livekit.plugins.cartesia")
    monkeypatch.setenv("CARTESIA_API_KEY", "fake")

    assert cartesia.STT(model="ink-2").capabilities.aligned_transcript is False
    assert _make_stt("cartesia/ink-2").capabilities.aligned_transcript is False


def test_gateway_ink2_payload_carries_no_word_alignment() -> None:
    stream = SpeechStream.__new__(SpeechStream)
    stream._input_ch = SimpleNamespace(start_time_offset=0.0)
    stream._opts = SimpleNamespace(language="en")

    data = stream._build_speech_data(GATEWAY_INK2_FINAL)

    assert data.words == []
    assert (data.start_time, data.end_time) == (0.0, 12.5)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("assemblyai/u3-rt-pro", True),
        ("assemblyai/universal-3-5-pro", True),
        ("assemblyai/universal-3-6-pro", True),
        ("assemblyai/universal-streaming", False),
        ("deepgram/nova-3", False),
    ],
)
def test_chat_context_capability_follows_native_support(
    model: str, expected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(model, extra_kwargs={"previous_context_n_turns": 0})
    assert stt.capabilities.chat_context is expected
    assert "agent_context" not in caplog.text


def test_chat_context_capability_reevaluated_on_model_update() -> None:
    stt = _make_stt("assemblyai/universal-streaming")
    stt.update_options(model="assemblyai/universal-3-5-pro")
    assert stt.capabilities.chat_context is True


# -- agent_context carryover --


def _assistant_item(text: str) -> ConversationItemAddedEvent:
    return ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=[text]))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Your room is booked for Tuesday.", "Your room is booked for Tuesday."),
        # oversize replies keep their last 1750 characters
        ("a" * 2000 + "b" * 1750, "b" * 1750),
    ],
)
def test_carryover_forwards_assistant_text(text: str, expected: str) -> None:
    stt = _make_stt("assemblyai/universal-3-5-pro", extra_kwargs={"agent_context": "earlier"})
    stt._push_conversation_item(_assistant_item(text))
    assert stt._opts.extra_kwargs["agent_context"] == expected


@pytest.mark.parametrize(
    "item",
    [
        ChatMessage(role="user", content=["hi there"]),
        ChatMessage(role="assistant", content=[]),
        AgentHandoff(new_agent_id="agent-2"),
    ],
    ids=["user", "textless", "handoff"],
)
def test_carryover_ignores_other_items(item: Any) -> None:
    stt = _make_stt("assemblyai/universal-3-5-pro")
    stt._push_conversation_item(ConversationItemAddedEvent(item=item))
    assert "agent_context" not in stt._opts.extra_kwargs


def test_carryover_skipped_after_switch_to_model_without_native_support() -> None:
    stt = _make_stt("assemblyai/universal-3-6-pro")
    stt.update_options(model="openai/gpt-live-transcribe")
    stt._push_conversation_item(_assistant_item("What is your date of birth?"))
    assert "agent_context" not in stt._opts.extra_kwargs


def test_carryover_reaches_only_fallback_instances_with_native_support() -> None:
    u3 = _make_stt("assemblyai/universal-3-6-pro")
    openai = _make_stt("openai/gpt-live-transcribe")
    adapter = FallbackAdapter(stt=[openai, u3])
    assert adapter.capabilities.chat_context is True

    adapter._push_conversation_item(_assistant_item("What is your date of birth?"))

    assert u3._opts.extra_kwargs["agent_context"] == "What is your date of birth?"
    assert "agent_context" not in openai._opts.extra_kwargs


# -- OpenAI gpt-live-transcribe: client endpointing and 24 kHz --


@pytest.mark.parametrize(
    ("model", "gets_vad", "rate"),
    [
        ("openai/gpt-live-transcribe", True, 24000),
        ("openai/gpt-live-transcribe:en", True, 24000),
        ("openai", True, 24000),
        ("speechmatics/enhanced", True, 16000),
        ("speechmatics/linden-1", False, 16000),
        ("deepgram/nova-3", False, 16000),
    ],
)
def test_default_vad_and_rate_follow_the_model(
    model: str, gets_vad: bool, rate: int, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(model)

    assert isinstance(stt._vad, _DefaultVAD) is gets_vad
    assert stt._opts.sample_rate == rate
    assert "takes audio only" not in caplog.text


def test_model_string_language_is_parsed_before_the_rate_is_chosen() -> None:
    stt = _make_stt("openai/gpt-live-transcribe:en")
    assert (stt._opts.model, stt._opts.language) == ("openai/gpt-live-transcribe", "en")


def test_openai_keeps_a_passed_vad() -> None:
    vad = FakeVAD()
    assert _make_stt("openai/gpt-live-transcribe", vad=vad)._vad is vad


def test_vad_is_dropped_for_models_that_end_turns_server_side(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(vad=FakeVAD())

    assert stt._vad is None
    assert "`vad` will be ignored" in caplog.text


def test_switching_model_re_resolves_vad_and_default_rate() -> None:
    stt = _make_stt()
    stt.update_options(model="openai/gpt-live-transcribe")
    assert isinstance(stt._vad, _DefaultVAD)
    assert stt._opts.sample_rate == 24000

    stt.update_options(model="deepgram/nova-3")
    assert stt._vad is None
    assert stt._opts.sample_rate == 16000


def test_an_explicit_refused_rate_is_kept_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt("openai/gpt-live-transcribe", sample_rate=16000)
    assert stt._opts.sample_rate == 16000
    assert "takes audio only at 24000 Hz" in caplog.text

    caplog.clear()
    stt = _make_stt(sample_rate=16000)
    with caplog.at_level(logging.WARNING):
        stt.update_options(model="openai/gpt-live-transcribe")
    assert stt._opts.sample_rate == 16000
    assert "takes audio only at 24000 Hz" in caplog.text


def test_openai_keyterms_map_to_keywords() -> None:
    opts: OpenAIOptions = {"keywords": ["LiveKit"], "delay": "low"}
    stt = _make_stt("openai/gpt-live-transcribe", extra_kwargs=opts)
    assert stt._opts.extra_kwargs == opts
    assert stt.capabilities.keyterms is True
    assert _keyterms_extra_for_model(
        "openai/gpt-live-transcribe",
        extra_kwargs={"keywords": "LiveKit"},
        session_keyterms=["LiveKit", "Aura"],
    ) == {"keywords": ["LiveKit", "Aura"]}


# -- transcript events --


def _bare_stream() -> SpeechStream:
    stream = SpeechStream.__new__(SpeechStream)
    stream._event_ch = Chan()  # type: ignore[assignment]
    stream._speaking = False
    stream._request_id = "req-1"
    stream._speech_duration = 0.0
    stream._input_ch = SimpleNamespace(start_time_offset=0.0)
    stream._pending_extra = None
    stream._opts = SimpleNamespace(language="en", model="auto")
    return stream


def _drain(stream: SpeechStream) -> list[SpeechEventType]:
    out = []
    while True:
        try:
            out.append(stream._event_ch.recv_nowait().type)
        except ChanEmpty:
            return out


def test_start_of_speech_is_reported_once_per_turn() -> None:
    stream = _bare_stream()

    stream._process_start_of_speech()
    stream._process_start_of_speech()
    stream._process_transcript({"transcript": "are you"}, is_final=False)
    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH, SpeechEventType.INTERIM_TRANSCRIPT]

    stream._process_transcript({"transcript": "are you open", "duration": 1.0}, is_final=True)
    assert stream._speaking is False
    _drain(stream)
    stream._process_start_of_speech()
    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]


def test_first_transcript_reports_onset_without_a_start_event() -> None:
    stream = _bare_stream()
    stream._process_transcript({"transcript": "are you"}, is_final=False)
    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH, SpeechEventType.INTERIM_TRANSCRIPT]


def test_empty_interim_never_reports_onset() -> None:
    stream = _bare_stream()
    stream._process_transcript({"transcript": ""}, is_final=False)
    assert _drain(stream) == []
    assert stream._speaking is False


@pytest.mark.parametrize("previous_turn", [False, True])
def test_empty_final_reports_usage_without_starting_a_silent_turn(previous_turn: bool) -> None:
    stream = _bare_stream()
    if previous_turn:
        stream._process_transcript(
            {"transcript": "hello", "words": [{"end": 44.26}]}, is_final=True
        )
        _drain(stream)
    stream._speech_duration = 4.05

    stream._process_transcript(
        {"transcript": "", "start": 44.58, "duration": 4.05, "words": []}, is_final=True
    )

    usage = stream._event_ch.recv_nowait()
    assert usage.type == SpeechEventType.RECOGNITION_USAGE
    assert usage.recognition_usage.audio_duration == pytest.approx(4.05)
    assert _drain(stream) == []
    assert stream._speech_duration == 0
    assert stream._speaking is False


def test_empty_final_still_closes_an_active_turn() -> None:
    stream = _bare_stream()
    stream._process_start_of_speech()
    _drain(stream)

    stream._process_transcript({"transcript": ""}, is_final=True)

    assert _drain(stream) == [SpeechEventType.FINAL_TRANSCRIPT, SpeechEventType.END_OF_SPEECH]
    assert stream._speaking is False


@pytest.mark.parametrize(
    ("timing", "end_time"),
    [
        ({"start": 1.0, "duration": 2.0, "words": [{"word": "hello", "end": 2.4}]}, 10.9),
        ({"start": 1.0, "duration": 1.4}, None),
        ({"words": [{"word": "hello", "end": 2.4}]}, 10.9),
        ({}, None),
        ({"words": [{"word": "hello"}]}, None),
    ],
)
def test_eos_reports_provider_speech_end_time(
    timing: dict[str, Any], end_time: float | None
) -> None:
    stream = _bare_stream()
    stream.start_time_offset = 8.5
    stream._process_transcript({"transcript": "hello", **timing}, is_final=True)

    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    final = stream._event_ch.recv_nowait()
    endpoint = stream._event_ch.recv_nowait()
    assert final.type == SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].end_time == pytest.approx(
        8.5 + timing.get("start", 0) + timing.get("duration", 0)
    )
    assert endpoint.type == SpeechEventType.END_OF_SPEECH
    if end_time is None:
        assert endpoint.alternatives == []
    else:
        assert endpoint.alternatives[0].end_time == pytest.approx(end_time)


@pytest.mark.parametrize("timed", [True, False], ids=["utterance-timing", "arrival-time"])
async def test_eos_sets_the_last_speaking_time(
    timed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider timing anchors the end of speech; without it the arrival time does."""
    now = 1_700_000_010.0
    monkeypatch.setattr(audio_recognition.time, "time", lambda: now)
    stream = _bare_stream()
    if timed:
        stream._opts.model = "deepgram/nova-3"
        stream.start_time_offset = 8.0
        stream._process_transcript(
            {"transcript": "hello", "start": 0.0, "duration": 1.4, "words": [{"end": 1.4}]},
            is_final=True,
        )
    else:
        stream._opts.model = "cartesia/ink-2"
        stream.start_time_offset = 8.5
        stream._process_transcript(GATEWAY_INK2_FINAL, is_final=True)
    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    recognition = _make_recognition(vad=None, input_started_at=now - 10.0, mode="stt")
    recognition._stt_aligned_transcript = timed

    while True:
        try:
            event = stream._event_ch.recv_nowait()
        except ChanEmpty:
            break
        await recognition._on_stt_event(event)

    assert recognition._user_turn_committed
    if not timed:
        assert recognition._last_speaking_time == now
        return
    assert recognition._last_speaking_time == pytest.approx(now - 0.6, abs=1e-6, rel=0)
    metrics = audio_recognition._compute_end_of_turn_metrics(
        speech_start_time=now - 2.0,
        last_speaking_time=recognition._last_speaking_time,
        last_final_transcript_time=recognition._last_final_transcript_time,
        now=now,
    )
    assert metrics.transcription_delay == pytest.approx(0.6)


# -- adaptive interruption does not depend on the alignment claim --


@pytest.fixture
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "fake")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fakesecret")
    monkeypatch.delenv("LIVEKIT_REMOTE_EOT_URL", raising=False)


def _activity(*, aligned_transcript: object, mode: str | None, streaming: bool) -> AgentActivity:
    interruption = InterruptionOptions(resume_false_interruption=False)
    if mode is not None:
        interruption["mode"] = mode  # type: ignore[typeddict-item]
    stt_impl = TurnScriptedSTT(
        turns=[ScriptedTurn(speech_start=0.55, final_at=2.6, final_text="hello")],
        aligned_transcript=aligned_transcript,  # type: ignore[arg-type]
    )
    stt_impl._capabilities.streaming = streaming
    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=[
                FakeUserSpeech(start_time=0.5, end_time=2.0, transcript="hello", stt_delay=0.0)
            ]
        ),
        stt=stt_impl,
        llm=FakeLLM(fake_responses=[]),
        tts=FakeTTS(fake_responses=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            endpointing=EndpointingOptions(min_delay=0.2),
            interruption=interruption,
        ),
        aec_warmup_duration=None,
    )
    # AgentActivity resolves interruption detection in __init__, before any I/O starts
    return AgentActivity(Agent(instructions="You are a helpful assistant."), session)


@pytest.mark.usefixtures("_fake_credentials")
@pytest.mark.parametrize(
    ("aligned_transcript", "mode", "streaming", "dev_mode", "expected"),
    [
        ("word", None, True, True, True),
        (False, None, True, True, True),
        (False, None, False, True, True),
        (False, "adaptive", True, False, True),
        ("word", None, True, False, False),
    ],
    ids=["aligned-dev", "unaligned-dev", "non-streaming-dev", "explicit-adaptive", "production"],
)
def test_adaptive_interruption_ignores_the_alignment_claim(
    aligned_transcript: object,
    mode: str | None,
    streaming: bool,
    dev_mode: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if dev_mode:
        monkeypatch.setenv("LIVEKIT_DEV_MODE", "1")
    else:
        monkeypatch.delenv("LIVEKIT_DEV_MODE", raising=False)

    activity = _activity(aligned_transcript=aligned_transcript, mode=mode, streaming=streaming)

    assert activity._interruption_detection_enabled is expected


@pytest.mark.usefixtures("_fake_credentials")
@pytest.mark.parametrize("aligned_transcript", ["word", False])
def test_mode_vad_keeps_fast_interruption_on_either_stt(
    aligned_transcript: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "1")

    activity = _activity(aligned_transcript=aligned_transcript, mode="vad", streaming=True)

    assert activity._session.interruption_detection == "vad"
    assert activity._interruption_detection_enabled is False
    assert activity._interruption_by_audio_activity_enabled is True
    assert activity.vad is activity._session.vad is not None


# -- gateway websocket session --

Handler = Callable[["_GatewaySocket", dict[str, Any]], Awaitable[None]]


class _GatewaySocket:
    """One in-memory gateway connection; `reply` and `hang_up` act as the server."""

    def __init__(self, gateway: _Gateway, index: int) -> None:
        self._gateway = gateway
        self.index = index
        self.closed = False
        self.close_code: int | None = None
        self.received: list[dict[str, Any]] = []
        self._inbox: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        if self.closed:
            raise ConnectionResetError("socket is closed")
        event = json.loads(data)
        self.received.append(event)
        self._gateway.seen[event["type"]].set()
        if self._gateway.handler is not None:
            await self._gateway.handler(self, event)

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        msg = await self._inbox.get()
        if msg.type == aiohttp.WSMsgType.CLOSE:
            self.closed = True
            self.close_code = msg.data
        return msg

    async def ping(self, message: bytes = b"") -> None:
        if self._gateway.answers_pings:
            self._inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", None))

    async def close(self, **kwargs: Any) -> bool:
        self.closed = True
        return True

    def exception(self) -> BaseException | None:
        return None

    def reply(self, msg_type: str, **fields: Any) -> None:
        payload = json.dumps({"type": msg_type, **fields})
        self._inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, payload, None))

    def hang_up(self, code: int = 1000) -> None:
        self._inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, code, None))


class _Gateway:
    """In-memory stand-in for the gateway's `/stt` endpoint, passed as the `http_session`."""

    closed = False

    def __init__(self, handler: Handler | None = None, *, answers_pings: bool = True) -> None:
        self.handler = handler
        self.answers_pings = answers_pings
        self.sockets: list[_GatewaySocket] = []
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.seen: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)

    async def ws_connect(self, url: str, **kwargs: Any) -> _GatewaySocket:
        self.urls.append(url)
        self.headers.append(dict(kwargs.get("headers") or {}))
        socket = _GatewaySocket(self, len(self.sockets))
        self.sockets.append(socket)
        return socket

    def messages(self, msg_type: str) -> list[dict[str, Any]]:
        return [e for s in self.sockets for e in s.received if e["type"] == msg_type]

    def types(self, index: int = 0) -> list[str]:
        return [e["type"] for e in self.sockets[index].received]


async def _final_then_hang_up(ws: _GatewaySocket, event: dict[str, Any]) -> None:
    if event["type"] == "session.finalize":
        ws.reply("final_transcript", transcript="hello", language="en")
        ws.hang_up()


async def _final_then_finalized(ws: _GatewaySocket, event: dict[str, Any]) -> None:
    if event["type"] == "session.finalize":
        ws.reply("final_transcript", transcript="hello", language="en")
        ws.reply("session.finalized")


def _gateway_stt(gateway: _Gateway, model: str = "deepgram/nova-3", **kwargs: Any) -> STT:
    return _make_stt(model, base_url="http://gateway", http_session=gateway, **kwargs)


def _no_retry(**kwargs: Any) -> APIConnectOptions:
    return APIConnectOptions(**{"max_retry": 0, "timeout": 1.0, **kwargs})


def _one_retry() -> APIConnectOptions:
    return APIConnectOptions(max_retry=1, retry_interval=0.001, timeout=1.0)


async def _finals(stream: RecognizeStream) -> list[str]:
    return [
        event.alternatives[0].text
        async for event in stream
        if event.type == SpeechEventType.FINAL_TRANSCRIPT
    ]


async def _first_final(stream: RecognizeStream) -> SpeechEvent:
    async for event in stream:
        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            return event
    raise AssertionError("stream ended without a final transcript")


async def _live(stream: SpeechStream) -> None:
    while stream._ws is None or stream._ws.closed:
        await asyncio.sleep(0.01)


async def _run_to_end(stt: STT, **stream_kwargs: Any) -> list[str]:
    stream = stt.stream(conn_options=_no_retry(), **stream_kwargs)
    try:
        stream.end_input()
        return await asyncio.wait_for(_finals(stream), timeout=5.0)
    finally:
        await stream.aclose()


def _silence(sample_rate: int, seconds: float = 0.1) -> rtc.AudioFrame:
    samples = int(sample_rate * seconds)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def test_session_create_carries_model_and_connection_options() -> None:
    gateway = _Gateway(_final_then_hang_up)
    stt = _gateway_stt(gateway, conn_options=APIConnectOptions(timeout=30.0, max_retry=5))

    assert await _run_to_end(stt) == ["hello"]

    assert "model=deepgram%2Fnova-3" in gateway.urls[0]
    created = gateway.messages("session.create")[0]
    assert created["model"] == "deepgram/nova-3"
    assert created["connection"] == {"timeout": 30.0, "retries": 5}
    assert _make_stt()._opts.conn_options == DEFAULT_API_CONNECT_OPTIONS


async def test_connection_session_id_becomes_the_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _Gateway()
    monkeypatch.setattr(
        inference_stt, "get_inference_headers", lambda: {HEADER_SESSION_ID: "inference_connection"}
    )
    stream = _gateway_stt(gateway).stream(conn_options=_no_retry())
    try:
        await _live(stream)
        assert stream._request_id == "inference_connection"
    finally:
        await stream.aclose()


async def test_extra_headers_go_out_below_the_sdk_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _Gateway()
    monkeypatch.setattr(
        inference_stt, "get_inference_headers", lambda: {HEADER_SESSION_ID: "inference_sdk"}
    )
    monkeypatch.setattr(inference_stt, "create_access_token", lambda *_args: "token")
    stream = _gateway_stt(gateway).stream(
        conn_options=_no_retry(),
        extra_headers={HEADER_CLIENT_FEATURE: "AMD", HEADER_SESSION_ID: "caller_value"},
    )
    try:
        await _live(stream)
    finally:
        await stream.aclose()

    headers = gateway.headers[0]
    assert headers[HEADER_CLIENT_FEATURE] == "AMD"
    assert headers[HEADER_SESSION_ID] == "inference_sdk"
    assert headers["Authorization"] == "Bearer token"


@pytest.mark.parametrize(
    ("model", "language", "stream_language", "sent"),
    [
        ("deepgram/flux-general-multi", NOT_GIVEN, NOT_GIVEN, True),
        # flux gates a hint on the model, not on the session language
        ("deepgram/flux-general-multi", "en", NOT_GIVEN, True),
        ("deepgram/nova-3", "multi", NOT_GIVEN, True),
        ("deepgram/nova-3", "en", NOT_GIVEN, False),
        # the guard reads the stream's effective language
        ("deepgram/nova-3", "en", "multi", True),
    ],
)
async def test_language_hint_reaches_session_create_only_where_usable(
    model: str,
    language: Any,
    stream_language: Any,
    sent: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = _Gateway(_final_then_hang_up)
    stt = _gateway_stt(
        gateway, model, language=language, extra_kwargs={"language_hint": ["en", "es"]}
    )

    with caplog.at_level(logging.WARNING):
        await _run_to_end(stt, language=stream_language)

    extra = gateway.messages("session.create")[0]["settings"]["extra"]
    if sent:
        assert extra["language_hint"] == ["en", "es"]
        assert "removed `language_hint`" not in caplog.text
    else:
        assert "language_hint" not in extra
        assert "removed `language_hint`" in caplog.text
        assert "model 'deepgram/nova-3'" in caplog.text
        assert "language is 'en'" in caplog.text
        # the suppressed send lives on the payload, never on the stored options
        assert stt._opts.extra_kwargs["language_hint"] == ["en", "es"]


async def test_a_later_multi_mode_sends_the_hint_again(caplog: pytest.LogCaptureFixture) -> None:
    gateway = _Gateway()
    stt = _gateway_stt(gateway, language="en", extra_kwargs={"language_hint": ["en", "es"]})
    stream = stt.stream(conn_options=_one_retry())

    with caplog.at_level(logging.WARNING):
        try:
            await _live(stream)
            stt.update_options(language="multi")
            await gateway.seen["session.update"].wait()
            # a dropped socket rebuilds the connection on the new language
            gateway.sockets[0].hang_up()
            while len(gateway.messages("session.create")) < 2:
                await asyncio.sleep(0.01)
        finally:
            await stream.aclose()

    first, second = gateway.messages("session.create")
    assert "language_hint" not in first["settings"]["extra"]
    assert second["settings"]["language"] == "multi"
    assert second["settings"]["extra"]["language_hint"] == ["en", "es"]
    assert caplog.text.count("removed `language_hint`") == 1


@pytest.mark.parametrize("into_multi", [False, True])
async def test_a_live_update_guards_the_language_hint(
    into_multi: bool, caplog: pytest.LogCaptureFixture
) -> None:
    gateway = _Gateway()
    stt = _gateway_stt(gateway, language="en")
    stream = stt.stream(conn_options=_no_retry())

    with caplog.at_level(logging.WARNING):
        try:
            await _live(stream)
            if into_multi:
                stt.update_options(language="multi", extra={"language_hint": ["en", "es"]})
            else:
                # a repeated hint is dropped silently: the stream already reported it
                stt.update_options(extra={"language_hint": ["en", "es"]})
                stt.update_options(extra={"language_hint": ["en", "es"]})
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    updates = gateway.messages("session.update")
    if into_multi:
        assert [u["settings"]["extra"]["language_hint"] for u in updates] == [["en", "es"]]
        assert "removed `language_hint`" not in caplog.text
    else:
        assert len(updates) == 2
        assert all("language_hint" not in u["settings"]["extra"] for u in updates)
        assert caplog.text.count("removed `language_hint`") == 1


async def test_linden_session_keyterms_reach_session_create() -> None:
    gateway = _Gateway(_final_then_hang_up)
    user_entry = {"content": "gnocchi", "sounds_like": ["nyohki"]}
    stt = _gateway_stt(
        gateway, "speechmatics/linden-1", extra_kwargs={"additional_vocab": [user_entry]}
    )
    assert stt.capabilities.keyterms is True
    stt._update_session_keyterms(["gnocchi", "LiveKit"])

    await _run_to_end(stt)

    # the user's entry keeps its sounds_like; a session term it already covers is not duplicated
    extra = gateway.messages("session.create")[0]["settings"]["extra"]
    assert extra["additional_vocab"] == [user_entry, {"content": "LiveKit"}]


@pytest.mark.parametrize("close_code", [1000, 1011])
async def test_socket_close_after_input_end_is_accepted(close_code: int) -> None:
    async def handler(ws: _GatewaySocket, event: dict[str, Any]) -> None:
        if event["type"] == "session.finalize":
            ws.reply("final_transcript", transcript="hello", language="en")
            ws.hang_up(close_code)

    gateway = _Gateway(handler)
    stt = _gateway_stt(gateway)
    errors: list[Exception] = []
    stt.on("error", lambda event: errors.append(event.error))
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=3, retry_interval=0.001))
    try:
        stream.end_input()
        assert await _finals(stream) == ["hello"]
    finally:
        await stream.aclose()

    assert len(gateway.sockets) == 1
    assert not errors


async def test_session_closed_after_input_end_finishes_without_another_close() -> None:
    async def handler(ws: _GatewaySocket, event: dict[str, Any]) -> None:
        if event["type"] == "session.finalize":
            ws.reply("final_transcript", transcript="hello", language="en")
            ws.reply("session.closed")

    gateway = _Gateway(handler)
    assert await _run_to_end(_gateway_stt(gateway)) == ["hello"]
    assert gateway.types() == ["session.create", "session.finalize"]


@pytest.mark.parametrize("close_session", [False, True], ids=["socket", "session"])
async def test_close_before_input_end_reconnects(close_session: bool) -> None:
    async def handler(ws: _GatewaySocket, event: dict[str, Any]) -> None:
        if ws.index == 0 and event["type"] == "session.create":
            if close_session:
                ws.reply("session.closed")
            else:
                ws.hang_up()
        if ws.index == 1 and event["type"] == "session.finalize":
            ws.reply("final_transcript", transcript="hello", language="en")
            ws.reply("session.closed")

    gateway = _Gateway(handler)
    stt = _gateway_stt(gateway)
    errors: list[Exception] = []
    stt.on("error", lambda event: errors.append(event.error))
    stream = stt.stream(conn_options=_one_retry())
    try:
        while len(gateway.sockets) < 2:
            await asyncio.sleep(0.01)
        stream.end_input()
        assert await _finals(stream) == ["hello"]
    finally:
        await stream.aclose()

    assert len(gateway.sockets) == 2
    assert len(errors) == 1
    assert isinstance(errors[0], APIError) and errors[0].retryable


@pytest.mark.parametrize("end_input", [False, True], ids=["active", "draining"])
async def test_stream_close_sends_session_close_and_closes_socket(end_input: bool) -> None:
    gateway = _Gateway()
    stream = _gateway_stt(gateway).stream(conn_options=_no_retry())
    await _live(stream)
    if end_input:
        stream.end_input()
        await gateway.seen["session.finalize"].wait()
    await stream.aclose()

    # aclose closes the input channel before it cancels the stream task
    assert gateway.types() in (
        ["session.create", "session.close"],
        ["session.create", "session.finalize", "session.close"],
    )
    if end_input:
        assert gateway.types()[1] == "session.finalize"
    assert gateway.sockets[0].closed


@pytest.mark.parametrize(
    "message",
    [
        "start_of_speech",
        "interim_transcript",
        "preflight_transcript",
        "final_transcript",
        "session.finalized",
        "session.created",
        "unknown",
        "pong",
    ],
)
async def test_provider_events_reset_retries_but_setup_and_pongs_do_not(message: str) -> None:
    progress = message not in ("session.created", "unknown", "pong")
    gateway = _Gateway()
    stream = _gateway_stt(gateway).stream()
    stream._num_retries = 2
    try:
        await _live(stream)
        ws = gateway.sockets[0]
        if message == "pong":
            ws._inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", None))
        else:
            ws.reply(message, transcript="words", language="en")
        await asyncio.sleep(0.5)
        assert stream._num_retries == (0 if progress else 2)
    finally:
        await stream.aclose()


@pytest.mark.parametrize("message", ["error", "session.closed"])
async def test_provider_failure_events_do_not_reset_retries(message: str) -> None:
    gateway = _Gateway()
    stream = _gateway_stt(gateway).stream()
    stream._num_retries = 3
    try:
        await _live(stream)
        gateway.sockets[0].reply(message)
        with pytest.raises(APIError):
            await stream._task
        assert stream._num_retries == 3
    finally:
        await stream.aclose()


@pytest.mark.parametrize(
    ("events", "expected_duration"),
    [
        ([], 30.0),
        ([(5.0, "interim_transcript")], 35.0),
        ([(5.0, "preflight_transcript")], 35.0),
        ([(5.0, "session.finalized")], 30.0),
        ([(5.0, "final_transcript")], 8.0),
        # an acknowledgment may precede a trailing final
        ([(0.0, "session.finalized"), (0.025, "final_transcript")], 3.025),
        ([(5.0, "final_transcript"), (2.0, "interim_transcript")], 10.0),
        ([(5.0, "final_transcript"), (2.0, "preflight_transcript")], 10.0),
        ([(5.0, "final_transcript"), (2.0, "final_transcript")], 10.0),
    ],
    ids=[
        "silent",
        "interim",
        "preflight",
        "ack",
        "final",
        "ack-final",
        "final-interim",
        "final-preflight",
        "final-final",
    ],
)
async def test_input_end_waits_for_transcript_inactivity(
    events: list[tuple[float, str]], expected_duration: float
) -> None:
    gateway = _Gateway()
    stream = _gateway_stt(gateway).stream(conn_options=_no_retry())

    async def send_events() -> None:
        await gateway.seen["session.finalize"].wait()
        for delay, msg_type in events:
            await asyncio.sleep(delay)
            gateway.sockets[0].reply(msg_type, transcript="words", language="en")

    sender = asyncio.create_task(send_events())
    start = asyncio.get_running_loop().time()
    try:
        stream.end_input()
        finals = await _finals(stream)
        elapsed = asyncio.get_running_loop().time() - start
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender
        await stream.aclose()

    assert elapsed == pytest.approx(expected_duration, abs=0.01)
    assert len(finals) == sum(msg_type == "final_transcript" for _, msg_type in events)
    assert gateway.types() == ["session.create", "session.finalize", "session.close"]
    assert gateway.sockets[0].closed


async def test_final_before_input_end_uses_shorter_inactivity_window() -> None:
    gateway = _Gateway()
    stream = _gateway_stt(gateway).stream(conn_options=_no_retry())
    try:
        await _live(stream)
        gateway.sockets[0].reply("final_transcript", transcript="words", language="en")
        await _first_final(stream)
        await asyncio.sleep(40.0)
        assert not gateway.sockets[0].closed
        start = asyncio.get_running_loop().time()
        stream.end_input()
        await _finals(stream)
        elapsed = asyncio.get_running_loop().time() - start
    finally:
        await stream.aclose()

    assert elapsed == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("error_code", [2006, 2007], ids=["connection", "inactivity"])
async def test_error_before_input_end_reconnects(error_code: int) -> None:
    async def handler(ws: _GatewaySocket, event: dict[str, Any]) -> None:
        if ws.index == 0 and event["type"] == "session.create":
            ws.reply("error", code=error_code, message="customer content must not leak")
        elif event["type"] == "session.finalize":
            ws.reply("final_transcript", transcript="hello", language="en")
            ws.reply("session.closed")

    gateway = _Gateway(handler)
    stt = _gateway_stt(gateway, sample_rate=16000)
    errors: list[Exception] = []
    stt.on("error", lambda event: errors.append(event.error))
    stream = stt.stream(conn_options=_one_retry())
    try:
        while len(gateway.sockets) < 2:
            await asyncio.sleep(0.01)
        stream.push_frame(_silence(16000, 0.05))
        stream.end_input()
        assert await _finals(stream) == ["hello"]
    finally:
        await stream.aclose()

    assert [s.index for s in gateway.sockets if "input_audio" in gateway.types(s.index)] == [1]
    [error] = errors
    assert isinstance(error, APIError)
    assert error.message == "LiveKit Inference STT returned an error"
    assert error.body == {"code": error_code}
    assert error.retryable is True


@pytest.mark.parametrize("error_code", [2006, 2007], ids=["connection", "inactivity"])
async def test_error_after_input_end_is_not_retried(error_code: int) -> None:
    async def handler(ws: _GatewaySocket, event: dict[str, Any]) -> None:
        if event["type"] == "session.finalize":
            ws.reply("error", code=error_code, message="customer content must not leak")

    gateway = _Gateway(handler)
    stt = _gateway_stt(gateway)
    errors: list[Exception] = []
    stt.on("error", lambda event: errors.append(event.error))
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=3, retry_interval=0.001))
    try:
        stream.end_input()
        with pytest.raises(APIError) as exc_info:
            await _finals(stream)
    finally:
        await stream.aclose()

    assert len(gateway.sockets) == 1
    assert errors == [exc_info.value]
    assert exc_info.value.body == {"code": error_code}
    assert exc_info.value.retryable is False


async def test_half_open_socket_raises_retryable_error() -> None:
    gateway = _Gateway(answers_pings=False)
    stream = _gateway_stt(gateway).stream(conn_options=_no_retry(liveness_timeout=0.15))
    try:
        with pytest.raises(APIConnectionError) as exc_info:
            await _finals(stream)
    finally:
        await stream.aclose()

    assert exc_info.value.retryable is True
    assert isinstance(exc_info.value.__cause__, aiohttp.ServerTimeoutError)


async def test_end_of_speech_finalizes_the_turn_while_input_is_open() -> None:
    """Gpt-live-transcribe detects no turns, so only a VAD-driven finalize yields a final."""
    gateway = _Gateway(_final_then_finalized)
    vad = FakeVAD(
        fake_user_speeches=[
            FakeUserSpeech(start_time=0.0, end_time=0.05, transcript="hello", stt_delay=0.0)
        ],
        min_speech_duration=0.01,
        min_silence_duration=0.05,
    )
    stream = _gateway_stt(gateway, "openai/gpt-live-transcribe", vad=vad).stream(
        conn_options=_no_retry()
    )
    try:
        # room audio arrives at 48 kHz; the stream must resample it to 24 kHz
        stream.push_frame(_silence(48000))
        final = await _first_final(stream)
    finally:
        await stream.aclose()

    assert final.alternatives[0].text == "hello"
    assert gateway.messages("session.create")[0]["settings"]["sample_rate"] == "24000"
    audio_bytes = sum(len(base64.b64decode(m["audio"])) for m in gateway.messages("input_audio"))
    assert 0 < audio_bytes <= 24000 * 2 // 10, "more than 100 ms of 24 kHz PCM was sent"


async def test_a_closed_vad_still_lets_input_end_finalize() -> None:
    gateway = _Gateway(_final_then_finalized)
    # a VAD with nothing to report closes its stream at once
    stream = _gateway_stt(gateway, "openai/gpt-live-transcribe", vad=FakeVAD()).stream(
        conn_options=_no_retry()
    )
    try:
        await _live(stream)
        await asyncio.sleep(0.05)
        stream.end_input()
        final = await _first_final(stream)
    finally:
        await stream.aclose()

    assert final.alternatives[0].text == "hello"


async def test_live_switch_off_client_endpointing_keeps_finalizing_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The running Speechmatics session keeps its client VAD after the parent moves on."""

    class _SpeechyVAD(FakeVAD):
        def __init__(self) -> None:
            super().__init__(
                fake_user_speeches=[
                    FakeUserSpeech(start_time=0.0, end_time=0.05, transcript="hello", stt_delay=0.0)
                ],
                min_speech_duration=0.01,
                min_silence_duration=0.05,
            )

    monkeypatch.setattr(inference_vad, "VAD", _SpeechyVAD)
    gateway = _Gateway(_final_then_finalized)
    # 24 kHz so the Deepgram switch is not refused for its rate
    stt = _gateway_stt(gateway, "speechmatics/enhanced", sample_rate=24000)
    stream = stt.stream(conn_options=_no_retry())

    async def feed() -> None:
        while True:
            stream.push_frame(_silence(48000))
            await asyncio.sleep(0.02)

    feeder = asyncio.create_task(feed())
    try:
        await _live(stream)
        stt.update_options(model="deepgram/nova-3")
        final = await _first_final(stream)
    finally:
        feeder.cancel()
        await stream.aclose()

    assert final.alternatives[0].text == "hello"
    assert stream._vad is not None


async def _updates_after(
    action: Callable[[STT, SpeechStream], None], *, sample_rate: int | None = None
) -> tuple[STT, SpeechStream, list[dict[str, Any]]]:
    """Run `action` on a live Deepgram stream and return the session.update settings it sent."""
    gateway = _Gateway()
    kwargs = {"sample_rate": sample_rate} if sample_rate else {}
    stt = _gateway_stt(gateway, **kwargs)
    stream = stt.stream(conn_options=_no_retry())
    try:
        await _live(stream)
        stream.push_frame(_silence(stt._opts.sample_rate))
        action(stt, stream)
        await asyncio.sleep(0.1)
    finally:
        await stream.aclose()
    return stt, stream, [u["settings"] for u in gateway.messages("session.update")]


async def test_live_switch_to_openai_keeps_the_running_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The gateway does not switch a live session's provider, so the stream keeps its model."""
    with caplog.at_level(logging.WARNING):
        stt, stream, updates = await _updates_after(
            lambda stt, _: stt.update_options(model="openai/gpt-live-transcribe:fr")
        )

    # the parent re-resolves its default, so streams created from now on work
    assert (stt._opts.model, stt._opts.sample_rate, stt._opts.language) == (
        "openai/gpt-live-transcribe",
        24000,
        "fr",
    )
    assert stream._opts.model == "deepgram/nova-3"
    assert stream._vad is None
    assert stream._opts.language in (NOT_GIVEN, None)
    assert all("language" not in u for u in updates)
    assert "takes audio only at 24000 Hz" in caplog.text


@pytest.mark.parametrize("sample_rate", [16000, 24000], ids=["refused-rate", "other-provider"])
async def test_a_direct_stream_suffix_language_is_not_applied_to_another_model(
    sample_rate: int,
) -> None:
    _, stream, updates = await _updates_after(
        lambda _, stream: stream.update_options(model="openai/gpt-live-transcribe:fr"),
        sample_rate=sample_rate,
    )

    assert stream._opts.model == "deepgram/nova-3"
    assert stream._opts.language in (NOT_GIVEN, None)
    assert all("language" not in u for u in updates)


async def test_an_explicit_language_still_reaches_a_live_stream() -> None:
    _, _, updates = await _updates_after(
        lambda stt, _: stt.update_options(model="deepgram/nova-3", language="fr")
    )
    assert any(u.get("language") == "fr" for u in updates)


async def test_a_model_switch_does_not_send_the_new_providers_extra() -> None:
    stt, _, updates = await _updates_after(
        lambda stt, _: stt.update_options(
            model="openai/gpt-live-transcribe", extra={"keywords": ["Acme"]}
        )
    )
    assert not any("keywords" in u.get("extra", {}) for u in updates)
    assert stt._opts.extra_kwargs == {"keywords": ["Acme"]}


async def test_a_live_user_keyword_update_is_not_dropped_for_a_keyterm_model() -> None:
    """Deepgram's weighted `keywords` is a user option distinct from its framework `keyterm`."""
    _, _, updates = await _updates_after(
        lambda stt, _: stt.update_options(extra={"keywords": [("LiveKit", 2.0)]})
    )
    assert any(u.get("extra", {}).get("keywords") == [["LiveKit", 2.0]] for u in updates)


async def test_live_switch_keeps_keyterms_on_the_running_model() -> None:
    def action(stt: STT, _: SpeechStream) -> None:
        stt.update_options(model="openai/gpt-live-transcribe")
        stt._update_session_keyterms(["Acme"])

    _, _, updates = await _updates_after(action)

    extras = [u["extra"] for u in updates if "extra" in u]
    assert not any("keywords" in e for e in extras)
    assert {"keyterm": ["Acme"]} in extras


async def test_a_cleared_keyterm_list_reaches_the_live_session() -> None:
    gateway = _Gateway()
    stt = _gateway_stt(gateway)
    stream = stt.stream(conn_options=_no_retry())
    try:
        await _live(stream)
        stt._update_session_keyterms(["Acme"])
        await asyncio.sleep(0.05)
        stt._update_session_keyterms([])
        await asyncio.sleep(0.05)
    finally:
        await stream.aclose()

    extras = [u["settings"].get("extra", {}) for u in gateway.messages("session.update")]
    assert [e["keyterm"] for e in extras if "keyterm" in e] == [["Acme"], []]


async def test_an_update_during_speech_does_not_flush_deferred_keyterms() -> None:
    """Keyterms that change mid-utterance wait for the end of speech, even past other updates."""
    gateway = _Gateway()
    stt = _gateway_stt(gateway)
    stream = stt.stream(conn_options=_no_retry())

    def keyterm_updates() -> list[Any]:
        extras = [u["settings"].get("extra", {}) for u in gateway.messages("session.update")]
        return [e["keyterm"] for e in extras if "keyterm" in e]

    try:
        await _live(stream)
        stream._speaking = True
        stt._update_session_keyterms(["Acme"])
        stt.update_options(language="fr")
        await asyncio.sleep(0.05)

        assert keyterm_updates() == []
        assert any(
            u["settings"].get("language") == "fr" for u in gateway.messages("session.update")
        )

        stream._speaking = False
        stream._on_end_of_speech()
        await asyncio.sleep(0.05)
    finally:
        await stream.aclose()

    assert keyterm_updates() == [["Acme"]]
