import asyncio
import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.agents.stt.clock import _ConnectionClock
from livekit.plugins.azure import stt as azure_stt

from .fake_stt import attach_provider_audio_channels

pytestmark = pytest.mark.plugin("azure")


@pytest.mark.parametrize("offset", [4_000_000, 0, -1])
def test_azure_endpoint_uses_event_offset_before_final(offset: int) -> None:
    stream = object.__new__(azure_stt.SpeechStream)
    stream._speaking = True
    stream._final_alternatives = []
    stream._on_provider_progress = None
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    events = attach_provider_audio_channels(stream, start_time_offset=7.0)

    stream._on_speech_end(SimpleNamespace(offset=offset))
    stream._on_speech_end(SimpleNamespace(offset=offset))

    assert len(events) == 1
    assert events[0].type == stt.SpeechEventType.END_OF_SPEECH
    if offset > 0:
        assert events[0].alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert events[0].alternatives == []


def test_azure_recognized_emits_final_transcript(monkeypatch):
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "AutoDetectSourceLanguageResult",
        lambda result: SimpleNamespace(language="en-US"),
    )

    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._opts = SimpleNamespace(language=["en-US"], pronunciation_assessment=None)
    stream._on_provider_progress = None
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    events = attach_provider_audio_channels(stream)

    stream._on_recognized(
        SimpleNamespace(
            result=SimpleNamespace(
                text="hello",
                offset=10_000_000,
                duration=20_000_000,
                result_id="azure-result-id",
            )
        )
    )

    assert [event.type for event in events] == [stt.SpeechEventType.FINAL_TRANSCRIPT]


@pytest.fixture
def timed_stream(monkeypatch):
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "AutoDetectSourceLanguageResult",
        lambda result: SimpleNamespace(language="en-US"),
    )
    stream = object.__new__(azure_stt.SpeechStream)
    stream._opts = SimpleNamespace(language=["en-US"], pronunciation_assessment=None)
    stream._speaking = True
    stream._final_alternatives = []
    stream._on_provider_progress = None
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    events = attach_provider_audio_channels(stream, start_time_offset=7.0)
    return stream, events


def test_azure_endpoint_prefers_final_speech_timing_over_terminal_silence(timed_stream):
    stream, events = timed_stream
    stream._on_recognized(
        SimpleNamespace(
            result=SimpleNamespace(text="hello", offset=301_900_000, duration=144_400_000)
        )
    )
    stream._on_speech_end(SimpleNamespace(offset=486_400_000))

    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert events[-1].alternatives[0].end_time == pytest.approx(7.0 + 44.63)


@pytest.mark.parametrize("new_activity", ["start", "interim", "untimed_final"])
def test_azure_endpoint_does_not_reuse_final_timing_after_new_speech(timed_stream, new_activity):
    stream, events = timed_stream
    stream._on_recognized(
        SimpleNamespace(
            result=SimpleNamespace(text="first", offset=10_000_000, duration=20_000_000)
        )
    )
    if new_activity == "start":
        stream._on_speech_start(SimpleNamespace(offset=40_000_000))
    else:
        callback = stream._on_recognizing if new_activity == "interim" else stream._on_recognized
        callback(
            SimpleNamespace(result=SimpleNamespace(text="next", offset=40_000_000, duration=0))
        )
    stream._on_speech_end(SimpleNamespace(offset=50_000_000))

    assert events[-1].alternatives[0].end_time == pytest.approx(7.0 + 5.0)


def test_azure_stream_emits_usage_for_processed_audio():
    events = []
    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    stream._audio_duration = 3.5
    stream._last_audio_duration_report_time = 0.0

    stream._emit_recognition_usage()

    assert [event.type for event in events] == [stt.SpeechEventType.RECOGNITION_USAGE]
    assert events[0].recognition_usage is not None
    assert events[0].recognition_usage.audio_duration == 3.5
    assert stream._audio_duration == 0.0


def _canceled_event(reason, code=None, error_details=""):
    return SimpleNamespace(
        cancellation_details=SimpleNamespace(reason=reason, code=code, error_details=error_details)
    )


def test_azure_canceled_error_unblocks_run():
    # An Error cancellation (e.g. a service timeout) must wake _run via the
    # stopped event and stash the details, so the base retry/fallback path runs
    # instead of the stream hanging on a dead recognizer.
    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    stream._session_stopped_event = asyncio.Event()
    stream._cancellation_error = None

    details = SimpleNamespace(
        reason=azure_stt.speechsdk.CancellationReason.Error,
        code=azure_stt.speechsdk.CancellationErrorCode.ServiceTimeout,
        error_details="timeout",
    )
    stream._on_canceled(SimpleNamespace(cancellation_details=details))

    assert stream._session_stopped_event.is_set()
    assert stream._cancellation_error is details


def test_azure_canceled_without_error_is_ignored():
    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    stream._session_stopped_event = asyncio.Event()
    stream._cancellation_error = None

    stream._on_canceled(_canceled_event(azure_stt.speechsdk.CancellationReason.EndOfStream))

    assert not stream._session_stopped_event.is_set()
    assert stream._cancellation_error is None


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from unittest.mock import MagicMock

    from livekit.agents import APIConnectOptions
    from livekit.plugins.azure import stt as azure_stt

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)

    def create_recognizer(*, config, stream):
        recognizer = MagicMock()
        recognizer.start_continuous_recognition.side_effect = lambda: (
            recognizer.session_started.connect.call_args.args[0](None)
        )
        stream.close.side_effect = lambda: recognizer.session_stopped.connect.call_args.args[0](
            None
        )
        recognizer.stop_continuous_recognition.side_effect = stream.close
        clock.sockets.append(recognizer)
        return recognizer

    monkeypatch.setattr(
        azure_stt.speechsdk.audio, "PushAudioInputStream", lambda **kwargs: MagicMock()
    )
    monkeypatch.setattr(azure_stt, "_create_speech_recognizer", create_recognizer)
    provider = azure_stt.STT(speech_key="test-key", speech_region="test")
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
    finally:
        await stream.aclose()
        await provider.aclose()


def _recognized_stream(monkeypatch, *, assessment: bool) -> azure_stt.SpeechStream:
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "AutoDetectSourceLanguageResult",
        lambda result: SimpleNamespace(language="ja-JP"),
    )
    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._opts = SimpleNamespace(
        language=["ja-JP"], pronunciation_assessment='{"x": 1}' if assessment else None
    )
    stream._on_provider_progress = None
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    attach_provider_audio_channels(stream)
    return stream


def _recognized_event(json_result: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        result=SimpleNamespace(text="こんにちは", offset=0, duration=10_000_000, json=json_result)
    )


ASSESSED = {
    "Display": "こんにちは",
    "PronunciationAssessment": {"AccuracyScore": 92.0, "FluencyScore": 88.0},
    "Words": [{"Word": "こんにちは", "PronunciationAssessment": {"AccuracyScore": 92.0}}],
}


def test_azure_final_transcript_carries_pronunciation_assessment(monkeypatch):
    stream = _recognized_stream(monkeypatch, assessment=True)
    events = stream._event_ch.events

    stream._on_recognized(_recognized_event(json.dumps({"NBest": [ASSESSED]})))

    assert [event.type for event in events] == [stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert events[0].alternatives[0].metadata == {"pronunciation_assessment": ASSESSED}


def test_azure_final_transcript_has_no_metadata_without_assessment(monkeypatch):
    stream = _recognized_stream(monkeypatch, assessment=False)
    events = stream._event_ch.events

    stream._on_recognized(_recognized_event(json.dumps({"NBest": [ASSESSED]})))

    assert events[0].alternatives[0].metadata is None


@pytest.mark.parametrize(
    "json_result",
    [None, "", "not json", "{}", '{"NBest": []}', '{"NBest": [{"Display": "x"}]}'],
)
def test_azure_unreadable_assessment_still_emits_transcript(monkeypatch, caplog, json_result):
    stream = _recognized_stream(monkeypatch, assessment=True)
    events = stream._event_ch.events

    stream._on_recognized(_recognized_event(json_result))

    assert [event.type for event in events] == [stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert events[0].alternatives[0].metadata is None
    assert "こんにちは" not in caplog.text


def test_azure_pronunciation_assessment_survives_stream_copy_and_applies():
    config = azure_stt.speechsdk.PronunciationAssessmentConfig(
        grading_system=azure_stt.speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=azure_stt.speechsdk.PronunciationAssessmentGranularity.Phoneme,
    )
    engine = azure_stt.STT(
        speech_key="test-key", speech_region="japaneast", pronunciation_assessment=config
    )
    opts = copy.deepcopy(engine._config)
    assert opts.pronunciation_assessment == config.to_json()

    recognizer = azure_stt._create_speech_recognizer(
        config=opts, stream=azure_stt.speechsdk.audio.PushAudioInputStream()
    )

    applied = recognizer.properties.get_property_by_name("PronunciationAssessment_Params")
    assert json.loads(applied)["granularity"] == "Phoneme"


@pytest.mark.parametrize(
    "callback", ["_on_recognized", "_on_recognizing", "_on_speech_start", "_on_speech_end"]
)
def test_provider_progress_resets_retries_without_transcript(monkeypatch, callback):
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "AutoDetectSourceLanguageResult",
        lambda result: SimpleNamespace(language="en-US"),
    )
    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    stream._event_ch = SimpleNamespace(send_nowait=lambda event: None)
    stream._on_provider_progress = None
    stream._num_retries = 2
    stream._speaking = False
    getattr(stream, callback)(
        SimpleNamespace(result=SimpleNamespace(text="", offset=0, duration=0), offset=0)
    )
    assert stream._num_retries == 0


async def test_azure_places_sdk_results_on_the_event_loop_across_an_input_gap(monkeypatch):
    """Enqueue SDK-thread events on the loop and preserve post-gap timing."""
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    # the input clock measures elapsed time on the monotonic clock
    monkeypatch.setattr(time, "perf_counter", lambda: now)
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "AutoDetectSourceLanguageResult",
        lambda result: SimpleNamespace(language="en-US"),
    )
    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._opts = SimpleNamespace(language=["en-US"], pronunciation_assessment=None)
    stream._speaking = False
    stream._final_alternatives = []
    stream._on_provider_progress = None
    stream._num_retries = 0
    stream._loop = asyncio.get_running_loop()
    stream._init_channels(translate_timestamps=True)
    stream.start_time_offset = 7.0

    frame = rtc.AudioFrame.create(16000, 1, 1600)
    for captured_from in (1000.0, 1006.0):  # the input paused for 5s
        for i in range(10):
            now = captured_from + (i + 1) * frame.duration
            stream._input_ch.send_nowait(frame)
            stream._input_ch.recv_nowait()

    # the event channel places the times of an event when it is sent
    placed_on: set[int] = set()
    connection_to_session_timestamps = _ConnectionClock.connection_to_session_timestamps

    def spy(clock: _ConnectionClock, speech_data: stt.SpeechData) -> stt.SpeechData:
        placed_on.add(threading.get_ident())
        return connection_to_session_timestamps(clock, speech_data)

    monkeypatch.setattr(_ConnectionClock, "connection_to_session_timestamps", spy)

    def result(text: str, duration: int) -> SimpleNamespace:
        return SimpleNamespace(
            result=SimpleNamespace(text=text, offset=10_000_000, duration=duration)
        )

    def sdk_thread() -> None:
        stream._on_speech_start(SimpleNamespace(offset=10_000_000))
        stream._on_recognizing(result("hel", 2_000_000))
        stream._on_recognized(result("hello", 3_000_000))
        stream._on_speech_end(SimpleNamespace(offset=20_000_000))

    await asyncio.to_thread(sdk_thread)
    events = [await asyncio.wait_for(stream._event_ch.recv(), 1) for _ in range(4)]

    assert [event.type for event in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    interim, final, end = (event.alternatives[0] for event in events[1:])
    assert (interim.start_time, interim.end_time) == pytest.approx((7.0 + 6.0, 7.0 + 6.2))
    assert (final.start_time, final.end_time) == pytest.approx((7.0 + 6.0, 7.0 + 6.3))
    assert end.end_time == pytest.approx(7.0 + 6.3)
    assert placed_on == {threading.get_ident()}
