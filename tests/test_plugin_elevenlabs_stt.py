"""Unit tests for ElevenLabs STT plugin configuration."""

from __future__ import annotations

import pytest
from multidict import CIMultiDict

from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import stt as elevenlabs_stt
from livekit.plugins.elevenlabs._utils import trace_id_from_headers

pytestmark = pytest.mark.plugin("elevenlabs")


class _EventSink:
    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


def _new_stream(*, server_vad=NOT_GIVEN) -> elevenlabs_stt.SpeechStream:
    stream = object.__new__(elevenlabs_stt.SpeechStream)
    stream._opts = elevenlabs_stt.STTOptions(
        model_id="scribe_v2_realtime",
        api_key="test-key",
        base_url=elevenlabs_stt.API_BASE_URL_V1,
        language_code=None,
        tag_audio_events=True,
        include_timestamps=False,
        sample_rate=16000,
        server_vad=server_vad,
        keyterms=NOT_GIVEN,
        no_verbatim=False,
        enable_logging=True,
    )
    stream._language = None
    stream._event_ch = _EventSink()
    stream._speaking = False
    stream._start_time_offset = 0.0
    return stream


def _committed_transcript(text: str) -> dict:
    return {
        "message_type": "committed_transcript",
        "text": text,
        "words": [
            {"text": text, "start": 0.1, "end": 0.4},
        ]
        if text
        else [],
    }


def test_server_vad_commit_emits_end_of_speech() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._event_ch.events[1].alternatives[0].text == "hello"
    assert stream._speaking is False


def test_manual_commit_still_waits_for_empty_commit() -> None:
    stream = _new_stream(server_vad=None)

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert stream._speaking is True

    stream._process_stream_event(_committed_transcript(""))

    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert stream._speaking is False


def _stt(**kwargs: object) -> elevenlabs_stt.STT:
    return elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime", **kwargs)


def test_no_verbatim_defaults_to_false() -> None:
    assert _stt()._opts.no_verbatim is False


def test_no_verbatim_can_be_enabled() -> None:
    assert _stt(no_verbatim=True)._opts.no_verbatim is True


def test_update_options_sets_no_verbatim() -> None:
    instance = _stt()
    assert instance._opts.no_verbatim is False
    instance.update_options(no_verbatim=True)
    assert instance._opts.no_verbatim is True


def test_update_options_leaves_no_verbatim_untouched_when_not_given() -> None:
    instance = _stt(no_verbatim=True)
    instance.update_options(tag_audio_events=False)
    assert instance._opts.no_verbatim is True


def test_update_options_forwards_no_verbatim_to_active_streams() -> None:
    # no_verbatim is a WebSocket query param applied at connect time, so a live
    # realtime stream must be told to reconnect. Verify STT.update_options
    # forwards it to active streams (which trigger a reconnect).
    instance = _stt()
    captured: dict[str, object] = {}

    class _FakeStream:
        def update_options(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake = _FakeStream()
    instance._streams.add(fake)
    instance.update_options(no_verbatim=True)
    assert captured.get("no_verbatim") is True


def test_enable_logging_defaults_to_true() -> None:
    assert _stt()._opts.enable_logging is True


def test_enable_logging_can_be_disabled() -> None:
    assert _stt(enable_logging=False)._opts.enable_logging is False


@pytest.mark.parametrize(("enable_logging", "expected"), [(True, "true"), (False, "false")])
async def test_connect_ws_includes_enable_logging(enable_logging: bool, expected: str) -> None:
    # enable_logging is a WebSocket query param. Verify it is forwarded to the
    # realtime connect URL with the expected lowercase boolean string.
    stream = _new_stream()
    stream._opts.enable_logging = enable_logging

    class _ConnOptions:
        timeout = 5.0

    stream._conn_options = _ConnOptions()

    captured: dict[str, object] = {}

    class _FakeSession:
        async def ws_connect(self, url: str, **kwargs: object) -> object:
            captured["url"] = url
            return object()

    stream._session = _FakeSession()

    await stream._connect_ws()

    assert f"enable_logging={expected}" in captured["url"]


def test_trace_id_from_headers() -> None:
    # header lookup is case-insensitive, and an absent header returns None
    assert trace_id_from_headers(CIMultiDict({"X-Trace-Id": "trace-1"})) == "trace-1"
    assert trace_id_from_headers(CIMultiDict()) is None
    assert trace_id_from_headers(None) is None


def test_speech_confidence_from_word_logprobs() -> None:
    # confident transcription: word logprobs near 0 -> confidence near 1 (spacing tokens ignored)
    words = [
        {"type": "word", "logprob": -0.01},
        {"type": "spacing", "logprob": -2.0},
        {"type": "word", "logprob": -0.05},
    ]
    assert 0.9 < elevenlabs_stt._speech_confidence(words) <= 1.0


def test_speech_confidence_flags_low_quality_transcription() -> None:
    # uncertain transcription (very negative logprobs) -> low confidence
    words = [{"type": "word", "logprob": -2.5}, {"type": "word", "logprob": -3.0}]
    assert elevenlabs_stt._speech_confidence(words) < 0.2


def test_speech_confidence_defaults_to_zero_without_logprobs() -> None:
    # no words, or words without logprobs (e.g. non-timestamped commit) -> default 0.0
    assert elevenlabs_stt._speech_confidence(None) == 0.0
    assert elevenlabs_stt._speech_confidence([{"text": "hi", "start": 0.1, "end": 0.4}]) == 0.0


def test_committed_transcript_sets_confidence() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(
        {
            "message_type": "committed_transcript",
            "text": "hello",
            "words": [
                {"text": "hello", "start": 0.1, "end": 0.4, "type": "word", "logprob": -0.02}
            ],
        }
    )

    final = stream._event_ch.events[1]
    assert final.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].confidence > 0.9
