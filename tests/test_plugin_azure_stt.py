import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from livekit.agents import LanguageCode, stt
from livekit.agents.types import NOT_GIVEN
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


def _stt_options(**overrides):
    defaults = {
        "speech_key": "key",
        "speech_region": "region",
        "speech_host": NOT_GIVEN,
        "speech_auth_token": NOT_GIVEN,
        "sample_rate": 16000,
        "num_channels": 1,
        "segmentation_silence_timeout_ms": NOT_GIVEN,
        "segmentation_max_time_ms": NOT_GIVEN,
        "segmentation_strategy": NOT_GIVEN,
        "language": [LanguageCode("en-US")],
    }
    defaults.update(overrides)
    return azure_stt.STTOptions(**defaults)


def _patch_recognizer_deps(monkeypatch, grammar):
    monkeypatch.setattr(
        azure_stt.speechsdk,
        "PhraseListGrammar",
        SimpleNamespace(from_recognizer=lambda recognizer: grammar),
    )
    monkeypatch.setattr(azure_stt.speechsdk, "SpeechConfig", lambda **kwargs: MagicMock())
    monkeypatch.setattr(azure_stt.speechsdk.audio, "AudioConfig", lambda **kwargs: MagicMock())
    monkeypatch.setattr(azure_stt.speechsdk, "SpeechRecognizer", lambda **kwargs: MagicMock())


def test_azure_phrase_list_weight_applied_to_grammar(monkeypatch):
    grammar = MagicMock()
    _patch_recognizer_deps(monkeypatch, grammar)
    config = _stt_options(phrase_list=["LiveKit", "WebRTC"], phrase_list_weight=1.8)

    azure_stt._create_speech_recognizer(config=config, stream=MagicMock())

    assert [call.args[0] for call in grammar.addPhrase.call_args_list] == [
        "LiveKit",
        "WebRTC",
    ]
    grammar.setWeight.assert_called_once_with(1.8)


def test_azure_phrase_list_without_weight_leaves_grammar_default(monkeypatch):
    grammar = MagicMock()
    _patch_recognizer_deps(monkeypatch, grammar)
    config = _stt_options(phrase_list=["LiveKit"])

    azure_stt._create_speech_recognizer(config=config, stream=MagicMock())

    grammar.setWeight.assert_not_called()
