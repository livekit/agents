from __future__ import annotations

import asyncio
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
    stream._total_reported_audio_duration = 0.0
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
        vad_smoothing_alpha=0.5,
    )

    url = stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)

    assert url.startswith(stt_streaming.SARVAM_STT_REALTIME_URL)
    assert _parse_ws_url(url) == {
        "language-code": "hi-IN",
        "stream-type": "fast",
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": "16000",
        "model": "saaras:v3-realtime",
        "vad_sot_threshold": "0.4",
        "vad_min_speech_ms": "300",
        "vad_min_silence_ms": "800",
        "vad_smoothing_alpha": "0.5",
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
    assert "vad_sot_threshold" not in params
    assert "vad_min_speech_ms" not in params


def test_streaming_options_validate_realtime_contract() -> None:
    with pytest.raises(ValueError, match="language auto is only supported"):
        stt_streaming.StreamingSTTOptions(language="auto", api_key="sk_test", stream_type="fast")

    with pytest.raises(ValueError, match="sample_rate must be one of"):
        stt_streaming.StreamingSTTOptions(language="hi-IN", api_key="sk_test", sample_rate=44100)

    with pytest.raises(ValueError, match="language od-IN is not supported"):
        stt_streaming.StreamingSTTOptions(language="od-IN", api_key="sk_test")


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

    assert params["language-code"] == "auto"
    assert params["stream-type"] == "simulated"
    assert params["mode"] == "translate"


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
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
    ]
    final_event = stream._event_ch.events[2]
    assert stream._session_id == "sess_123"
    assert stream._request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
    assert final_event.request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
    assert final_event.alternatives[0].text == "नमस्ते आप कैसे हैं"
    assert final_event.alternatives[0].language == "hi-IN"
    assert final_event.alternatives[0].confidence == 0.99
    assert final_event.alternatives[0].end_time == 1.75

    eos_event = stream._event_ch.events[3]
    assert eos_event.alternatives[0].end_time == 1.75
    assert eos_event.alternatives[0].metadata["utterance_idx"] == 0


@pytest.mark.asyncio
async def test_streaming_session_begin_captures_server_request_id() -> None:
    stream = _make_stream()

    await stream._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_4594d4503cd4",
            "request_id": "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68",
        }
    )

    assert stream._session_id == "sess_4594d4503cd4"
    assert stream._request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"


@pytest.mark.asyncio
async def test_streaming_request_id_stores_raw_value_without_format_assumptions() -> None:
    stream = _make_stream()

    await stream._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_xyz",
            "request_id": "srv_custom-id_v9",
        }
    )

    assert stream._session_id == "sess_xyz"
    assert stream._request_id == "srv_custom-id_v9"


@pytest.mark.asyncio
async def test_streaming_session_begin_without_request_id_leaves_request_id_empty() -> None:
    stream = _make_stream()

    await stream._handle_message({"event": "session.begin", "session_id": "sess_only"})

    assert stream._session_id == "sess_only"
    assert stream._request_id == ""


@pytest.mark.asyncio
async def test_streaming_defers_end_of_speech_until_final_transcript() -> None:
    stream = _make_stream()

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
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
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
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

    final_event = stream._event_ch.events[1]
    eos_event = stream._event_ch.events[2]
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

    final_event = stream._event_ch.events[0]
    eos_event = stream._event_ch.events[1]
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
async def test_streaming_logs_include_server_request_id_after_session_begin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()

    with caplog.at_level(logging.INFO, logger=stt_streaming.logger.name):
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


@pytest.mark.asyncio
async def test_streaming_logs_partial_and_final_transcripts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    stream._audio_position = 1.0

    with caplog.at_level(logging.INFO, logger=stt_streaming.logger.name):
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
async def test_streaming_usage_metrics_emit_periodic_and_session_end_deltas() -> None:
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
    assert [event.recognition_usage.audio_duration for event in usage_events] == [1.5, 0.75]
    assert all(event.request_id == "sess_123" for event in usage_events)
    assert stream._session_ended is True


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
    stream._manual_speech_started = True
    stream._pending_eos = True
    stream._pending_eos_time = 1.0
    stream._pending_final_data = {"text": "hello"}
    stream._utterance_start_audio_pos = 3.0
    stream._utterance_speech_end_audio_pos = 4.0
    stream._utterance_speech_end_wall = 5.0
    stream._final_received_for_utterance = True
    stream._eos_emitted_for_utterance = True
    stream._total_reported_audio_duration = 50.0
    fallback_task = _FakeFallbackTask()
    stream._eos_fallback_task = fallback_task

    stream._reset_connection_state()

    assert fallback_task.cancelled is True

    assert stream._request_id == ""
    assert stream._session_id == ""
    assert stream._session_ended is False
    assert stream._manual_speech_started is False
    assert stream._pending_eos is False
    assert stream._pending_eos_time is None
    assert stream._pending_final_data is None
    assert stream._utterance_start_audio_pos == 0.0
    assert stream._utterance_speech_end_audio_pos is None
    assert stream._utterance_speech_end_wall is None
    assert stream._final_received_for_utterance is False
    assert stream._eos_emitted_for_utterance is False
    assert stream._total_reported_audio_duration == 0.0
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
