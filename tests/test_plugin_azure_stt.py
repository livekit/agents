import asyncio
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
    stream._opts = SimpleNamespace(language=["en-US"])
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
