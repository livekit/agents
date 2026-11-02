import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.azure import stt as azure_stt

pytestmark = pytest.mark.plugin("azure")


def test_azure_recognized_emits_final_transcript(monkeypatch):
    events = []
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "AutoDetectSourceLanguageResult",
        lambda result: SimpleNamespace(language="en-US"),
    )

    stream = azure_stt.SpeechStream.__new__(azure_stt.SpeechStream)
    stream._opts = SimpleNamespace(language=["en-US"], pronunciation_assessment=None)
    stream._on_provider_progress = None
    stream._loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args))
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    stream._start_time_offset = 0.0

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


def _recognized_stream(monkeypatch, *, assessment: bool, events: list) -> azure_stt.SpeechStream:
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
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    stream._start_time_offset = 0.0
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
    events: list = []
    stream = _recognized_stream(monkeypatch, assessment=True, events=events)

    stream._on_recognized(_recognized_event(json.dumps({"NBest": [ASSESSED]})))

    assert [event.type for event in events] == [stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert events[0].alternatives[0].metadata == {"pronunciation_assessment": ASSESSED}


def test_azure_final_transcript_has_no_metadata_without_assessment(monkeypatch):
    events: list = []
    stream = _recognized_stream(monkeypatch, assessment=False, events=events)

    stream._on_recognized(_recognized_event(json.dumps({"NBest": [ASSESSED]})))

    assert events[0].alternatives[0].metadata is None


@pytest.mark.parametrize(
    "json_result",
    [None, "", "not json", "{}", '{"NBest": []}', '{"NBest": [{"Display": "x"}]}'],
)
def test_azure_unreadable_assessment_still_emits_transcript(monkeypatch, caplog, json_result):
    events: list = []
    stream = _recognized_stream(monkeypatch, assessment=True, events=events)

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
    getattr(stream, callback)(SimpleNamespace(result=SimpleNamespace(text="")))
    assert stream._num_retries == 0
