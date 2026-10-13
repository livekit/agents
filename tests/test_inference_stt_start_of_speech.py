"""`inference.STT` must report speech onset when the provider detects it server-side.

Cartesia Ink-2 signals onset with `turn.start`, which the gateway sends on as
`start_of_speech`. That used to arrive as an `interim_transcript` with an empty string,
which `_process_transcript` drops, so onset fell back to the first transcript carrying
words — around 800ms late against the same audio through the Cartesia plugin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import livekit.agents.inference.stt as inference_stt
from livekit.agents import NOT_GIVEN, APIConnectOptions
from livekit.agents.inference._utils import HEADER_SESSION_ID
from livekit.agents.inference.stt import SpeechStream as InferenceSpeechStream
from livekit.agents.stt import SpeechEventType
from livekit.agents.utils.aio.channel import Chan, ChanEmpty
from livekit.agents.voice import audio_recognition

from .test_inference_stt_aligned_transcript_claim import GATEWAY_INK2_FINAL
from .test_transcription_delay_anchor import _make_recognition

pytestmark = [pytest.mark.unit]


def _stream() -> InferenceSpeechStream:
    stream = InferenceSpeechStream.__new__(InferenceSpeechStream)
    stream._event_ch = Chan()  # type: ignore[assignment]
    stream._speaking = False
    stream._request_id = "req-1"
    stream._speech_duration = 0.0
    stream._start_time_offset = 0.0
    stream._pending_extra = None
    stream._opts = SimpleNamespace(language="en", model="auto", fallback=NOT_GIVEN)
    return stream


def _drain(stream: InferenceSpeechStream) -> list[SpeechEventType]:
    out = []
    while True:
        try:
            out.append(stream._event_ch.recv_nowait().type)
        except ChanEmpty:
            return out


def test_start_of_speech_message_reports_onset_immediately() -> None:
    """The onset event must not wait for a transcript."""
    stream = _stream()

    stream._process_start_of_speech()

    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]
    assert stream._speaking is True


def test_onset_is_not_reported_twice() -> None:
    """A later transcript must not re-announce onset the provider already gave us."""
    stream = _stream()

    stream._process_start_of_speech()
    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]

    stream._process_transcript({"transcript": "are you"}, is_final=False)

    # only the interim, no second onset
    assert _drain(stream) == [SpeechEventType.INTERIM_TRANSCRIPT]


def test_duplicate_start_of_speech_is_ignored() -> None:
    stream = _stream()

    stream._process_start_of_speech()
    stream._process_start_of_speech()

    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]


def test_providers_without_onset_still_fall_back_to_the_first_transcript() -> None:
    """Models that send no onset event keep today's behaviour."""
    stream = _stream()

    stream._process_transcript({"transcript": "are you"}, is_final=False)

    assert _drain(stream) == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
    ]


def test_empty_interim_alone_never_reports_onset() -> None:
    """The shape the gateway sends today: an empty interim carries no onset.

    This is the bug. The gateway must send `start_of_speech` instead; the SDK cannot
    infer onset from an empty transcript without firing for providers that emit empty
    interims for unrelated reasons.
    """
    stream = _stream()

    stream._process_transcript({"transcript": ""}, is_final=False)

    assert _drain(stream) == []
    assert stream._speaking is False


def test_onset_resets_after_the_turn_ends() -> None:
    """A second turn on the same stream must report its own onset."""
    stream = _stream()

    stream._process_start_of_speech()
    stream._process_transcript(
        {"transcript": "are you open on sunday", "start": 0.0, "duration": 1.0},
        is_final=True,
    )
    assert stream._speaking is False

    stream._process_start_of_speech()
    assert SpeechEventType.START_OF_SPEECH in _drain(stream)


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
def test_eos_reports_provider_speech_end_time(timing, end_time):
    stream = _stream()
    stream._start_time_offset = 8.5
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


@pytest.mark.parametrize(
    ("model", "words"),
    [
        ("deepgram/nova-3", [{"end": 1.4}]),
        ("soniox/stt-rt-v5", []),
        ("speechmatics/linden-1", []),
    ],
)
async def test_eos_preserves_transcription_delay_in_stt_turn_detection(monkeypatch, model, words):
    now = 1_700_000_010.0
    monkeypatch.setattr(audio_recognition.time, "time", lambda: now)
    stream = _stream()
    stream._opts.model = model
    stream._start_time_offset = 8.0
    stream._process_transcript(
        {"transcript": "hello", "start": 0.0, "duration": 1.4, "words": words},
        is_final=True,
    )
    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    recognition = _make_recognition(vad=None, input_started_at=now - 10.0, mode="stt")
    recognition._stt_aligned_transcript = bool(words)

    while True:
        try:
            event = stream._event_ch.recv_nowait()
        except ChanEmpty:
            break
        await recognition._on_stt_event(event)

    assert recognition._user_turn_committed
    assert recognition._last_speaking_time == pytest.approx(now - 0.6, abs=1e-6, rel=0)
    metrics = audio_recognition._compute_end_of_turn_metrics(
        speech_start_time=now - 2.0,
        last_speaking_time=recognition._last_speaking_time,
        last_final_transcript_time=recognition._last_final_transcript_time,
        now=now,
    )
    assert metrics.transcription_delay == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("model", "fallback", "timed"),
    [
        ("soniox/stt-rt", NOT_GIVEN, True),
        ("soniox/stt-rt-v4", NOT_GIVEN, True),
        ("soniox/stt-rt-v5", NOT_GIVEN, True),
        ("speechmatics/linden-1", NOT_GIVEN, True),
        ("soniox/stt-rt-v5", [{"model": "speechmatics/linden-1"}], True),
        ("soniox/stt-rt-v5", [{"model": "cartesia/ink-2"}], False),
        ("cartesia/ink-2", [{"model": "soniox/stt-rt-v5"}], False),
        ("soniox/stt-rt-v5", [{"model": "auto"}], False),
        ("soniox/stt-rt-v5", [{"model": "unknown/model"}], False),
        ("auto", NOT_GIVEN, False),
        ("unknown/model", NOT_GIVEN, False),
    ],
)
def test_eos_uses_duration_only_when_every_configured_model_reports_utterance_timing(
    model, fallback, timed
):
    stream = _stream()
    stream._opts.model = model
    stream._opts.fallback = fallback
    stream._start_time_offset = 8.5
    stream._process_transcript(
        {"transcript": "hello", "start": 1.0, "duration": 1.4, "words": []}, is_final=True
    )
    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    assert stream._event_ch.recv_nowait().type == SpeechEventType.FINAL_TRANSCRIPT
    endpoint = stream._event_ch.recv_nowait()
    assert endpoint.type == SpeechEventType.END_OF_SPEECH
    if timed:
        assert endpoint.alternatives[0].end_time == pytest.approx(10.9)
    else:
        assert endpoint.alternatives == []


async def test_eos_without_utterance_timing_uses_arrival_time(monkeypatch):
    now = 1_700_000_030.0
    monkeypatch.setattr(audio_recognition.time, "time", lambda: now)
    stream = _stream()
    stream._opts.model = "cartesia/ink-2"
    stream._start_time_offset = 8.5
    stream._process_transcript(GATEWAY_INK2_FINAL, is_final=True)
    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    recognition = _make_recognition(vad=None, input_started_at=now - 30.0, mode="stt")

    while True:
        try:
            event = stream._event_ch.recv_nowait()
        except ChanEmpty:
            break
        await recognition._on_stt_event(event)

    assert recognition._user_turn_committed
    assert recognition._last_speaking_time == now


async def test_connection_session_id_becomes_fallback_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWebSocket:
        async def send_str(self, data: str) -> None:
            pass

    class _FakeHTTPSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        async def ws_connect(self, url: str, *, headers: dict[str, str]) -> _FakeWebSocket:
            self.headers = headers
            return _FakeWebSocket()

    stream = InferenceSpeechStream.__new__(InferenceSpeechStream)
    stream._stt = SimpleNamespace(_session_keyterms=[])
    stream._opts = SimpleNamespace(
        sample_rate=16000,
        encoding="pcm_s16le",
        extra_kwargs={},
        model="auto",
        language=None,
        fallback=None,
        conn_options=None,
        base_url="https://example.livekit.cloud/v1",
        api_key="key",
        api_secret="secret",
    )
    stream._conn_options = APIConnectOptions(timeout=1.0, max_retry=0)
    stream._request_id = "stt_request_fallback"
    http_session = _FakeHTTPSession()

    monkeypatch.setattr(
        inference_stt,
        "get_inference_headers",
        lambda: {HEADER_SESSION_ID: "inference_connection"},
    )
    monkeypatch.setattr(inference_stt, "create_access_token", lambda *_args: "token")

    await stream._connect_ws(http_session)  # type: ignore[arg-type]

    assert http_session.headers[HEADER_SESSION_ID] == "inference_connection"
    assert stream._request_id == "inference_connection"
