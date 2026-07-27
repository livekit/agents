from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from livekit.agents import APIStatusError
from livekit.plugins.sarvam import stt_streaming

pytestmark = pytest.mark.unit


class _FakeEventChannel:
    def __init__(self) -> None:
        self.events = []

    def send_nowait(self, event: object) -> None:
        self.events.append(event)


def _make_stream(
    *,
    endpointing: str = "vad",
) -> stt_streaming.StreamingSpeechStream:
    stream = object.__new__(stt_streaming.StreamingSpeechStream)
    stream._event_ch = _FakeEventChannel()
    stream._request_id = ""
    stream._session_id = ""
    stream._session_ended = False
    stream._utterance_idx = None
    stream._utterance_in_progress = False
    stream._active_endpointing = endpointing
    stream._pending_endpointing = None
    stream._endpointing_update_acknowledged = False
    stream._pending_config_update = None
    stream._total_reported_audio_duration = 0.0
    stream._local_audio_duration = 0.0
    stream._server_audio_duration_reported = False
    stream._conn_options = stt_streaming.DEFAULT_API_CONNECT_OPTIONS
    stream._opts = stt_streaming.StreamingSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        endpointing=endpointing,
    )
    stream._logger = stt_streaming.logger.getChild("StreamingSpeechStream")
    stream._build_log_context = stt_streaming.StreamingSpeechStream._build_log_context.__get__(
        stream, stt_streaming.StreamingSpeechStream
    )
    stream._pending_eos = False
    stream._pending_eos_time = None
    stream._pending_final_data = None
    stream._utterance_start_audio_pos = 0.0
    stream._utterance_speech_end_audio_pos = None
    stream._utterance_speech_end_wall = None
    stream._final_received_for_utterance = False
    stream._eos_emitted_for_utterance = False
    stream._eos_fallback_task = None
    stream._manual_speech_started = False
    stream._stream_started_at = stt_streaming.time.time()
    stream._audio_position = 0.0
    stream._audio_duration_collector = stt_streaming.PeriodicCollector(
        callback=stt_streaming.StreamingSpeechStream._on_audio_duration_report.__get__(
            stream, stt_streaming.StreamingSpeechStream
        ),
        duration=5.0,
    )
    return stream


def _parse_ws_url(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return {key: value[0] for key, value in qs.items()}


def test_realtime_ws_url_includes_core_and_vad_params() -> None:
    opts = stt_streaming.StreamingSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        stream_type="fast",
        endpointing="vad",
        sample_rate=16000,
        vad_sot_threshold=0.4,
        vad_min_speech_ms=300,
        vad_min_silence_ms=800,
    )

    url = stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)

    assert url.startswith(stt_streaming.SARVAM_STT_REALTIME_URL)
    assert _parse_ws_url(url) == {
        "language_code": "hi-IN",
        "stream_type": "fast",
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": "16000",
        "model": "saaras:v3-realtime",
        "mode": "transcribe",
        "return_timestamps": "false",
        "threshold": "0.4",
        "min_speech_duration_ms": "300",
        "silence_duration_ms": "800",
    }


def test_realtime_ws_url_omits_vad_params_for_manual_endpointing() -> None:
    opts = stt_streaming.StreamingSTTOptions(
        language="en-IN",
        api_key="sk_test",
        endpointing="manual",
        vad_sot_threshold=0.4,
        vad_min_speech_ms=300,
    )

    url = stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)

    params = _parse_ws_url(url)
    assert params["endpointing"] == "manual"
    assert "threshold" not in params
    assert "min_speech_duration_ms" not in params


def test_realtime_ws_url_includes_prompt_and_timestamp_controls() -> None:
    opts = stt_streaming.StreamingSTTOptions(
        language="en-IN",
        api_key="sk_test",
        prompt="LiveKit terminology",
        return_timestamps=True,
    )

    params = _parse_ws_url(
        stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    )

    assert params["prompt"] == "LiveKit terminology"
    assert params["return_timestamps"] == "true"


def test_streaming_options_validate_realtime_contract() -> None:
    with pytest.raises(ValueError, match="sample_rate must be one of"):
        stt_streaming.StreamingSTTOptions(language="hi-IN", api_key="sk_test", sample_rate=44100)

    with pytest.raises(ValueError, match="language od-IN is not supported"):
        stt_streaming.StreamingSTTOptions(language="od-IN", api_key="sk_test")


def test_streaming_options_reject_server_tuned_vad_smoothing() -> None:
    with pytest.raises(TypeError, match="vad_smoothing_alpha"):
        stt_streaming.StreamingSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            vad_smoothing_alpha=0.5,
        )


@pytest.mark.parametrize("endpointing", ["vad", "manual"])
def test_auto_language_is_valid_for_all_contract_endpointing_modes(endpointing: str) -> None:
    opts = stt_streaming.StreamingSTTOptions(
        language="auto",
        api_key="sk_test",
        stream_type="fast",
        endpointing=endpointing,
    )

    params = _parse_ws_url(
        stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    )

    assert params["language_code"] == "auto"
    assert params["endpointing"] == endpointing


def test_simulated_streaming_allows_auto_language_and_mode() -> None:
    opts = stt_streaming.StreamingSTTOptions(
        language="auto",
        api_key="sk_test",
        stream_type="simulated",
        mode="translate",
    )

    params = _parse_ws_url(
        stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    )

    assert params["language_code"] == "auto"
    assert params["stream_type"] == "simulated"
    assert params["mode"] == "translate"


def test_streaming_option_update_uses_in_band_contract_config_message() -> None:
    class _ReconnectEvent:
        def __init__(self) -> None:
            self.set_called = False

        def set(self) -> None:
            self.set_called = True

    stream = _make_stream()
    stream._ws = None
    stream._reconnect_event = _ReconnectEvent()
    updated_options = stt_streaming.StreamingSTTOptions(
        language="en-IN",
        api_key="sk_test",
        stream_type="fast",
        mode="translate",
        endpointing="vad",
        prompt="LiveKit",
        vad_sot_threshold=0.6,
    )

    stream.update_options(updated_options)

    assert stream._pending_config_update == {
        "event": "config.update",
        "language_code": "en-IN",
        "stream_type": "fast",
        "mode": "translate",
        "prompt": "LiveKit",
        "threshold": 0.6,
    }
    assert stream._reconnect_event.set_called is False


def test_active_stream_keeps_connection_only_options_on_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    stream._pending_config_update = None
    stream._reconnect_event = SimpleNamespace(set_called=False)
    stream._reconnect_event.set = lambda: setattr(stream._reconnect_event, "set_called", True)
    updated_options = stt_streaming.StreamingSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        sample_rate=8000,
        return_timestamps=True,
        prompt="LiveKit",
    )

    with caplog.at_level(logging.WARNING, logger=stt_streaming.logger.name):
        stream.update_options(updated_options)

    assert stream._opts.sample_rate == 16000
    assert stream._opts.return_timestamps is False
    assert stream._pending_config_update == {
        "event": "config.update",
        "prompt": "LiveKit",
    }
    assert stream._reconnect_event.set_called is False
    assert "only apply to new streams" in caplog.text


def test_streaming_option_updates_merge_before_the_next_audio_frame() -> None:
    stream = _make_stream()
    stream._pending_config_update = None
    stream.update_options(
        stt_streaming.StreamingSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            prompt="LiveKit",
        )
    )
    stream.update_options(
        stt_streaming.StreamingSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            prompt="LiveKit",
            mode="translate",
        )
    )

    assert stream._pending_config_update == {
        "event": "config.update",
        "prompt": "LiveKit",
        "mode": "translate",
    }


def test_streaming_option_update_clears_prompt_with_empty_string() -> None:
    previous = stt_streaming.StreamingSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        prompt="LiveKit",
    )
    current = stt_streaming.StreamingSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        prompt=None,
    )

    assert stt_streaming.StreamingSpeechStream._config_update_payload(previous, current) == {
        "event": "config.update",
        "prompt": "",
    }


def test_active_stream_defers_endpointing_until_config_acknowledgement() -> None:
    stream = _make_stream()
    stream._active_endpointing = "vad"
    stream._utterance_in_progress = True
    stream._pending_endpointing = None
    stream._endpointing_update_acknowledged = False
    stream._pending_config_update = None
    stream.update_options(
        stt_streaming.StreamingSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            endpointing="manual",
        )
    )

    assert stream._active_endpointing == "vad"
    assert stream._pending_endpointing == "manual"

    stream._handle_config_updated()
    assert stream._active_endpointing == "vad"

    stream._utterance_in_progress = False
    stream._apply_pending_endpointing()
    assert stream._active_endpointing == "manual"


@pytest.mark.asyncio
async def test_streaming_event_mapping_emits_speech_and_transcript_events() -> None:
    stream = _make_stream()
    stream._audio_position = 1.25

    await stream._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_123",
            "request_id": "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68",
        }
    )
    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message(
        {
            "event": "transcript.partial",
            "utterance_idx": 0,
            "text": "नमस्ते",
            "confidence": 0.91,
        }
    )
    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.INTERIM_TRANSCRIPT,
    ]

    stream._audio_position = 1.75
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    event_types = [event.type for event in stream._event_ch.events]
    assert event_types == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    final_event = stream._event_ch.events[3]
    assert stream._session_id == "sess_123"
    assert stream._request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
    assert final_event.request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
    assert final_event.alternatives[0].text == "नमस्ते आप कैसे हैं"
    assert final_event.alternatives[0].language == "hi-IN"
    assert final_event.alternatives[0].confidence == 0.99
    assert final_event.alternatives[0].end_time == 1.75

    eos_event = stream._event_ch.events[2]
    assert eos_event.alternatives[0].end_time == 1.75
    assert eos_event.alternatives[0].metadata is not None
    assert eos_event.alternatives[0].metadata["speech_end_wall_time"] > 0


@pytest.mark.asyncio
async def test_streaming_session_begin_captures_server_request_id() -> None:
    stream = _make_stream()

    await stream._handle_message(
        {
            "event": "session.begin",
            "request_id": "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68",
        }
    )

    assert stream._session_id == ""
    assert stream._request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"


@pytest.mark.asyncio
async def test_streaming_request_id_stores_raw_value_without_format_assumptions() -> None:
    stream = _make_stream()

    await stream._handle_message(
        {
            "event": "session.begin",
            "request_id": "srv_custom-id_v9",
        }
    )

    assert stream._session_id == ""
    assert stream._request_id == "srv_custom-id_v9"


@pytest.mark.asyncio
async def test_streaming_session_begin_without_request_id_leaves_request_id_empty() -> None:
    stream = _make_stream()

    await stream._handle_message({"event": "session.begin"})

    assert stream._session_id == ""
    assert stream._request_id == ""


@pytest.mark.asyncio
async def test_streaming_emits_end_of_speech_before_final_transcript() -> None:
    stream = _make_stream()

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
    ]

    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]


@pytest.mark.asyncio
async def test_streaming_final_after_speech_end_includes_audio_end_time() -> None:
    stream = _make_stream()
    stream._audio_position = 1.25

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    stream._audio_position = 1.75
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})
    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    eos_event = stream._event_ch.events[1]
    final_event = stream._event_ch.events[2]
    assert final_event.alternatives[0].end_time == 1.75
    assert eos_event.alternatives[0].end_time == 1.75
    assert final_event.alternatives[0].metadata["speech_end_wall_time"] > 0


@pytest.mark.asyncio
async def test_streaming_final_transcript_waits_for_speech_end_for_end_time() -> None:
    stream = _make_stream()
    stream._audio_position = 1.25

    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    assert stream._event_ch.events == []

    stream._audio_position = 2.0
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    eos_event = stream._event_ch.events[0]
    final_event = stream._event_ch.events[1]
    assert final_event.alternatives[0].end_time == 2.0
    assert eos_event.alternatives[0].end_time == 2.0


@pytest.mark.asyncio
async def test_streaming_final_transcript_uses_current_audio_position_without_vad() -> None:
    stream = _make_stream(endpointing="manual")
    stream._audio_position = 1.25

    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    final_event = stream._event_ch.events[0]
    assert final_event.alternatives[0].end_time == 1.25


@pytest.mark.asyncio
async def test_streaming_final_transcript_uses_contract_timestamps_when_present() -> None:
    stream = _make_stream(endpointing="manual")

    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "hello",
            "language": "en-IN",
            "language_confidence": 0.99,
            "start_s": 0.7,
            "end_s": 1.2,
        }
    )

    final_event = stream._event_ch.events[0]
    assert final_event.alternatives[0].start_time == 0.7
    assert final_event.alternatives[0].end_time == 1.2


@pytest.mark.asyncio
async def test_streaming_logs_include_server_request_id_after_session_begin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()

    with caplog.at_level(logging.DEBUG, logger=stt_streaming.logger.name):
        await stream._handle_message(
            {
                "event": "session.begin",
                "session_id": "sess_4594d4503cd4",
                "request_id": "srv_custom-id_v9",
            }
        )
        await stream._handle_message(
            {
                "event": "transcript.partial",
                "utterance_idx": 0,
                "text": "नमस्ते",
                "confidence": 0.91,
            }
        )

    partial_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Sarvam realtime STT transcript.partial"
    ]
    assert len(partial_records) == 1
    assert partial_records[0].request_id == "srv_custom-id_v9"
    assert partial_records[0].session_id == "sess_4594d4503cd4"
    assert partial_records[0].raw_data["text"] == "नमस्ते"


@pytest.mark.asyncio
async def test_streaming_logs_partial_and_final_transcripts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    stream._audio_position = 1.0

    with caplog.at_level(logging.DEBUG, logger=stt_streaming.logger.name):
        await stream._handle_message(
            {
                "event": "transcript.partial",
                "utterance_idx": 0,
                "text": "नमस्ते",
                "confidence": 0.91,
            }
        )
        await stream._handle_message(
            {
                "event": "transcript.final",
                "utterance_idx": 0,
                "text": "नमस्ते आप कैसे हैं",
                "language": "hi-IN",
                "language_confidence": 0.99,
            }
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "Sarvam realtime STT transcript.partial" in messages
    assert "Sarvam realtime STT transcript.final" in messages


@pytest.mark.asyncio
async def test_streaming_info_logs_essential_data_without_raw_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream(endpointing="manual")
    final_payload = {
        "event": "transcript.final",
        "utterance_idx": 0,
        "text": "hello",
        "language": "en-IN",
        "language_confidence": 0.99,
    }

    with caplog.at_level(logging.INFO, logger=stt_streaming.logger.name):
        await stream._handle_message(final_payload)

    info_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Sarvam realtime STT transcript.final"
    ]
    assert len(info_records) == 1
    assert info_records[0].text == "hello"
    assert not hasattr(info_records[0], "raw_data")

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=stt_streaming.logger.name):
        await stream._handle_message(final_payload)

    debug_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Sarvam realtime STT raw event"
    ]
    assert len(debug_records) == 1
    assert debug_records[0].raw_data == final_payload


@pytest.mark.asyncio
async def test_streaming_emits_pending_end_of_speech_when_final_never_arrives() -> None:
    stream = _make_stream()

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})
    await stream._emit_pending_eos_after_timeout(0)

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
    ]


@pytest.mark.asyncio
async def test_new_speech_start_emits_pending_end_of_speech_for_previous_utterance() -> None:
    stream = _make_stream()
    stream._audio_position = 1.0

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    stream._audio_position = 1.5
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})
    stream._audio_position = 2.0
    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 1})

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
    ]
    eos_event = stream._event_ch.events[1]
    assert eos_event.alternatives[0].end_time == 1.5
    assert eos_event.alternatives[0].metadata["speech_end_wall_time"] > 0
    assert stream._pending_eos is False
    assert stream._eos_emitted_for_utterance is False
    assert stream._utterance_start_audio_pos == 2.0


@pytest.mark.asyncio
async def test_streaming_safe_send_str_ignores_closed_transport() -> None:
    stream = _make_stream()
    calls = []

    closed_ws = SimpleNamespace(closed=True, send_str=lambda payload: calls.append(payload))
    await stream._safe_send_str(closed_ws, {"event": "end"})

    assert calls == []

    async def _raise_reset(payload: str) -> None:
        raise stt_streaming.aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    closing_ws = SimpleNamespace(closed=False, send_str=_raise_reset)
    await stream._safe_send_str(closing_ws, {"event": "end"})


@pytest.mark.asyncio
async def test_streaming_usage_metrics_emit_server_authoritative_session_end() -> None:
    stream = _make_stream()
    stream._request_id = "sess_123"

    stream._on_audio_duration_report(1.5)
    await stream._handle_message(
        {
            "event": "session.end",
            "session_id": "sess_123",
            "audio_duration_s": 2.25,
        }
    )

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [2.25]
    assert all(event.request_id == "sess_123" for event in usage_events)
    assert stream._session_ended is True
    assert stream._server_audio_duration_reported is True


@pytest.mark.asyncio
async def test_streaming_usage_accepts_server_duration_smaller_than_local_estimate() -> None:
    stream = _make_stream()

    stream._on_audio_duration_report(5.0)
    await stream._handle_message(
        {
            "event": "session.end",
            "session_id": "sess_123",
            "audio_duration_s": 2.25,
        }
    )
    await stream._handle_message(
        {
            "event": "session.end",
            "session_id": "sess_123",
            "audio_duration_s": 2.25,
        }
    )

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [2.25]


@pytest.mark.asyncio
async def test_streaming_usage_falls_back_to_local_duration_on_clean_close() -> None:
    stream = _make_stream()

    stream._on_audio_duration_report(1.5)
    stream._emit_local_usage_fallback()
    stream._emit_local_usage_fallback()

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [1.5]


@pytest.mark.asyncio
async def test_streaming_usage_event_is_converted_to_livekit_stt_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _idle_run(self: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(stt_streaming.StreamingSpeechStream, "_run", _idle_run)
    stt_impl = stt_streaming.STTStreaming(api_key="sk_test")
    metrics = []
    stt_impl.on("metrics_collected", metrics.append)
    stream = stt_impl.stream()

    stream._event_ch.send_nowait(
        stt_streaming.stt.SpeechEvent(
            type=stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE,
            request_id="sess_123",
            recognition_usage=stt_streaming.stt.RecognitionUsage(audio_duration=3.0),
        )
    )
    await asyncio.sleep(0)
    await stream.aclose()
    await stt_impl.aclose()

    assert len(metrics) == 1
    assert metrics[0].request_id == "sess_123"
    assert metrics[0].audio_duration == 3.0
    assert metrics[0].streamed is True
    assert metrics[0].metadata.model_name == "saaras:v3-realtime"
    assert metrics[0].metadata.model_provider == "Sarvam"


@pytest.mark.asyncio
async def test_streaming_error_handling_distinguishes_fatal_and_non_fatal() -> None:
    stream = _make_stream()
    stream._request_id = "sess_123"

    await stream._handle_message(
        {
            "event": "error",
            "code": "chunk_too_large",
            "message": "chunk too large",
            "is_fatal": False,
        }
    )

    with pytest.raises(APIStatusError) as excinfo:
        await stream._handle_message(
            {
                "event": "error",
                "code": "model_unavailable",
                "message": "backend saturated",
                "is_fatal": True,
                "status_code": 503,
            }
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.body["code"] == "model_unavailable"
    assert excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_streaming_error_logs_include_raw_sarvam_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    stream._request_id = "req_123"
    raw_error = {
        "event": "error",
        "code": "invalid_subscription_key",
        "message": "Invalid subscription key",
        "is_fatal": True,
        "status_code": 1003,
    }

    with caplog.at_level(logging.DEBUG, logger=stt_streaming.logger.name):
        with pytest.raises(APIStatusError):
            await stream._handle_message(raw_error)

    info_records = [
        record for record in caplog.records if record.getMessage() == "Sarvam realtime STT error"
    ]
    assert len(info_records) == 1
    assert info_records[0].error_code == "invalid_subscription_key"
    assert info_records[0].status_code == 1003
    assert not hasattr(info_records[0], "raw_data")

    raw_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Sarvam realtime STT raw event"
    ]
    assert len(raw_records) == 1
    assert raw_records[0].raw_data == raw_error

    error_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Fatal Sarvam realtime STT error"
    ]
    assert len(error_records) == 1
    assert error_records[0].request_id == "req_123"
    assert error_records[0].error_code == "invalid_subscription_key"
    assert error_records[0].raw_message == raw_error


def test_reset_connection_state_clears_session_and_utterance_fields() -> None:
    class _FakeFallbackTask:
        def __init__(self) -> None:
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled = True

    stream = _make_stream()
    stream._request_id = "req_old"
    stream._session_id = "sess_old"
    stream._session_ended = True
    stream._utterance_idx = 7
    stream._manual_speech_started = True
    stream._pending_eos = True
    stream._pending_eos_time = 1.0
    stream._pending_final_data = {"text": "hello"}
    stream._utterance_start_audio_pos = 3.0
    stream._utterance_speech_end_audio_pos = 4.0
    stream._utterance_speech_end_wall = 5.0
    stream._final_received_for_utterance = True
    stream._eos_emitted_for_utterance = True
    stream._local_audio_duration = 10.0
    stream._total_reported_audio_duration = 50.0
    stream._server_audio_duration_reported = True
    fallback_task = _FakeFallbackTask()
    stream._eos_fallback_task = fallback_task

    stream._reset_connection_state()

    assert fallback_task.cancelled is True

    assert stream._request_id == ""
    assert stream._session_id == ""
    assert stream._session_ended is False
    assert stream._utterance_idx is None
    assert stream._manual_speech_started is False
    assert stream._pending_eos is False
    assert stream._pending_eos_time is None
    assert stream._pending_final_data is None
    assert stream._utterance_start_audio_pos == 0.0
    assert stream._utterance_speech_end_audio_pos is None
    assert stream._utterance_speech_end_wall is None
    assert stream._final_received_for_utterance is False
    assert stream._eos_emitted_for_utterance is False
    assert stream._local_audio_duration == 0.0
    assert stream._total_reported_audio_duration == 0.0
    assert stream._server_audio_duration_reported is False
    assert stream._eos_fallback_task is None


@pytest.mark.asyncio
async def test_session_end_delta_after_connection_reset() -> None:
    stream = _make_stream()
    stream._total_reported_audio_duration = 50.0

    stream._reset_connection_state()

    await stream._handle_message(
        {
            "event": "session.end",
            "session_id": "sess_new",
            "audio_duration_s": 12.0,
        }
    )

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert len(usage_events) == 1
    assert usage_events[0].recognition_usage.audio_duration == 12.0


@pytest.mark.asyncio
async def test_collector_flush_before_reset_emits_pending_usage() -> None:
    stream = _make_stream()
    stream._audio_duration_collector.push(2.5)

    stream._reset_connection_state()

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert len(usage_events) == 1
    assert usage_events[0].recognition_usage.audio_duration == 2.5
    assert stream._total_reported_audio_duration == 0.0


@pytest.mark.asyncio
async def test_reset_connection_state_allows_new_request_id_capture() -> None:
    stream = _make_stream()
    stream._request_id = "req_old"
    stream._session_id = "sess_old"

    stream._reset_connection_state()

    await stream._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_new",
            "request_id": "req_new",
        }
    )

    assert stream._request_id == "req_new"
    assert stream._session_id == "sess_new"


@pytest.mark.asyncio
async def test_streaming_process_messages_raises_on_realtime_rejection_close() -> None:
    stream = _make_stream()

    ws = SimpleNamespace(
        receive=lambda: asyncio.sleep(
            0,
            result=SimpleNamespace(
                type=stt_streaming.aiohttp.WSMsgType.CLOSE,
                data=4000,
                extra="beta access denied",
            ),
        ),
        close_code=4000,
    )

    with pytest.raises(APIStatusError) as excinfo:
        await stream._process_messages(ws)

    assert excinfo.value.status_code == 4000
    assert "beta access denied" in excinfo.value.message


@pytest.mark.asyncio
async def test_streaming_process_messages_stops_after_session_end() -> None:
    stream = _make_stream()

    class _SessionEndWS:
        close_code = None

        def __init__(self) -> None:
            self.receive_count = 0

        async def receive(self) -> SimpleNamespace:
            self.receive_count += 1
            if self.receive_count > 1:
                raise AssertionError("process_messages should stop after session.end")
            return SimpleNamespace(
                type=stt_streaming.aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "event": "session.end",
                        "request_id": "req_123",
                        "audio_duration_s": 1.25,
                    }
                ),
            )

    ws = _SessionEndWS()

    await stream._process_messages(ws)

    assert ws.receive_count == 1
    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [1.25]


@pytest.mark.asyncio
async def test_streaming_process_messages_logs_realtime_rejection_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    ws = SimpleNamespace(
        receive=lambda: asyncio.sleep(
            0,
            result=SimpleNamespace(
                type=stt_streaming.aiohttp.WSMsgType.CLOSE,
                data=4000,
                extra="beta access denied",
            ),
        ),
        close_code=4000,
    )

    with caplog.at_level(logging.ERROR, logger=stt_streaming.logger.name):
        with pytest.raises(APIStatusError):
            await stream._process_messages(ws)

    close_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Sarvam realtime STT WebSocket closed unexpectedly"
    ]
    assert len(close_records) == 1
    assert close_records[0].close_code == 4000
    assert close_records[0].close_reason == "beta access denied"


@pytest.mark.asyncio
async def test_streaming_connect_logs_handshake_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    response_error = stt_streaming.aiohttp.ClientResponseError(
        request_info=None,
        history=(),
        status=403,
        message="Forbidden",
    )

    async def _raise_response_error(*args: object, **kwargs: object) -> None:
        raise response_error

    stream._session = SimpleNamespace(ws_connect=_raise_response_error)

    with caplog.at_level(logging.ERROR, logger=stt_streaming.logger.name):
        with pytest.raises(stt_streaming.aiohttp.ClientResponseError):
            await stream._connect_ws()

    records = [
        record
        for record in caplog.records
        if record.getMessage() == "Sarvam realtime STT WebSocket handshake failed"
    ]
    assert len(records) == 1
    assert records[0].status_code == 403
    assert "API-SUBSCRIPTION-KEY" not in records[0].url
