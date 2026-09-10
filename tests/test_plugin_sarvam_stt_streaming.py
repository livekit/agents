from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from livekit.agents import APIStatusError
from livekit.plugins import sarvam
from livekit.plugins.sarvam import stt_streaming, tts

pytestmark = pytest.mark.plugin("sarvam")


class _FakeEventChannel:
    def __init__(self) -> None:
        self.events = []

    def send_nowait(self, event: object) -> None:
        self.events.append(event)


def _make_stream(
    *,
    endpointing: str = "vad",
) -> stt_streaming.RealtimeSpeechStream:
    stream = object.__new__(stt_streaming.RealtimeSpeechStream)
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
    stream._opts = stt_streaming.RealtimeSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        endpointing=endpointing,
    )
    stream._logger = stt_streaming.logger.getChild("RealtimeSpeechStream")
    stream._build_log_context = stt_streaming.RealtimeSpeechStream._build_log_context.__get__(
        stream, stt_streaming.RealtimeSpeechStream
    )
    stream._pending_final_data = None
    stream._utterance_start_audio_pos = 0.0
    stream._utterance_speech_end_audio_pos = None
    stream._utterance_speech_end_wall = None
    stream._final_received_for_utterance = False
    stream._eos_emitted_for_utterance = False
    stream._manual_speech_started = False
    stream._flush_observed = False
    stream._stream_started_at = stt_streaming.time.time()
    stream._audio_position = 0.0
    stream._audio_duration_collector = stt_streaming.PeriodicCollector(
        callback=stt_streaming.RealtimeSpeechStream._on_audio_duration_report.__get__(
            stream, stt_streaming.RealtimeSpeechStream
        ),
        duration=5.0,
    )
    return stream


async def _flush_pending_config_update(
    stream: stt_streaming.RealtimeSpeechStream,
) -> list[dict[str, object]]:
    """Send the queued config update the way the audio pump does."""
    sent: list[dict[str, object]] = []
    ws = SimpleNamespace(
        closed=False,
        send_str=lambda payload: asyncio.sleep(0, result=sent.append(json.loads(payload))),
    )
    await stream._send_pending_config_update(ws)
    return sent


def _parse_ws_url(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return {key: value[0] for key, value in qs.items()}


def test_realtime_stt_exports_and_legacy_aliases() -> None:
    assert sarvam.STTRealtime is stt_streaming.STTRealtime
    assert sarvam.RealtimeSpeechStream is stt_streaming.RealtimeSpeechStream
    assert sarvam.STTStreaming is sarvam.STTRealtime
    assert sarvam.StreamingSpeechStream is sarvam.RealtimeSpeechStream
    assert "STTStreaming" in sarvam.__all__
    assert "StreamingSpeechStream" in sarvam.__all__
    assert stt_streaming.StreamingSTTOptions is stt_streaming.RealtimeSTTOptions


def test_streaming_disables_connection_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturedStream:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        stt_streaming,
        "RealtimeSpeechStream",
        _CapturedStream,
    )
    stt = stt_streaming.STTRealtime(
        api_key="sk_test",
        http_session=object(),  # type: ignore[arg-type]
    )
    conn_options = stt_streaming.APIConnectOptions(
        max_retry=3,
        retry_interval=1.5,
        timeout=12.0,
    )

    stt.stream(conn_options=conn_options)

    stream_conn_options = captured["conn_options"]
    assert isinstance(
        stream_conn_options,
        stt_streaming.APIConnectOptions,
    )
    assert stream_conn_options.max_retry == 0
    assert stream_conn_options.retry_interval == 1.5
    assert stream_conn_options.timeout == 12.0


def test_realtime_ws_url_includes_core_and_vad_params() -> None:
    opts = stt_streaming.RealtimeSTTOptions(
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
    opts = stt_streaming.RealtimeSTTOptions(
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
    opts = stt_streaming.RealtimeSTTOptions(
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


def test_realtime_ws_url_includes_connection_only_vad_controls() -> None:
    opts = stt_streaming.RealtimeSTTOptions(
        language="auto",
        api_key="sk_test",
        vad_prefix_padding_ms=320,
    )

    params = _parse_ws_url(
        stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    )

    assert params["prefix_padding_ms"] == "320"


def test_realtime_ws_url_omits_prefix_padding_for_manual_endpointing() -> None:
    opts = stt_streaming.RealtimeSTTOptions(
        language="en-IN",
        api_key="sk_test",
        endpointing="manual",
        vad_prefix_padding_ms=320,
    )

    params = _parse_ws_url(
        stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    )

    assert "prefix_padding_ms" not in params


def test_streaming_options_validate_realtime_contract() -> None:
    with pytest.raises(ValueError, match="sample_rate must be one of"):
        stt_streaming.RealtimeSTTOptions(language="hi-IN", api_key="sk_test", sample_rate=44100)

    with pytest.raises(ValueError, match="language od-IN is not supported"):
        stt_streaming.RealtimeSTTOptions(language="od-IN", api_key="sk_test")

    with pytest.raises(ValueError, match="vad_prefix_padding_ms"):
        stt_streaming.RealtimeSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            vad_prefix_padding_ms=-1,
        )

    with pytest.raises(ValueError, match="mode must be one of"):
        stt_streaming.RealtimeSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            mode="indic-en",
        )


def test_streaming_options_reject_server_tuned_vad_smoothing() -> None:
    with pytest.raises(TypeError, match="vad_smoothing_alpha"):
        stt_streaming.RealtimeSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            vad_smoothing_alpha=0.5,
        )


@pytest.mark.parametrize("encoding", ["linear16", "linear32", "mulaw", "alaw"])
def test_streaming_options_accept_all_contract_encodings(encoding: str) -> None:
    opts = stt_streaming.RealtimeSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        encoding=encoding,
    )

    assert opts.encoding == encoding


@pytest.mark.parametrize("encoding", ["mulaw", "alaw"])
def test_realtime_pcm_telephony_encoders_round_trip(encoding: str) -> None:
    pcm = np.array([-30000, -1000, 0, 1000, 30000], dtype="<i2").tobytes()

    encoded = stt_streaming._encode_pcm_for_wire(encoding, pcm)
    decoded = np.frombuffer(tts._decode_telephony(encoding, encoded), dtype="<i2")

    assert len(encoded) == 5
    assert np.max(np.abs(decoded.astype(np.int32) - np.frombuffer(pcm, dtype="<i2"))) < 1500


def test_realtime_pcm_linear32_encoder_preserves_full_scale() -> None:
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype="<i2").tobytes()

    encoded = stt_streaming._encode_pcm_for_wire("linear32", pcm)

    assert np.frombuffer(encoded, dtype="<i4").tolist() == [
        -2147483648,
        -65536,
        0,
        65536,
        2147418112,
    ]


@pytest.mark.asyncio
async def test_session_begin_records_resolved_config() -> None:
    stream = _make_stream()
    resolved_config = {
        "encoding": "mulaw",
        "sample_rate": 8000,
        "prefix_padding_ms": 320,
    }

    await stream._handle_message(
        {
            "event": "session.begin",
            "request_id": "req_123",
            "config": resolved_config,
        }
    )
    resolved_config["sample_rate"] = 16000

    assert stream.resolved_config == {
        "encoding": "mulaw",
        "sample_rate": 8000,
        "prefix_padding_ms": 320,
    }


@pytest.mark.parametrize("endpointing", ["vad", "manual"])
def test_auto_language_is_valid_for_all_contract_endpointing_modes(endpointing: str) -> None:
    opts = stt_streaming.RealtimeSTTOptions(
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
    opts = stt_streaming.RealtimeSTTOptions(
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
    stream = _make_stream()
    stream._ws = None

    stream.update_options(
        language="en-IN",
        stream_type="fast",
        mode="translate",
        endpointing="vad",
        prompt="LiveKit",
        vad_sot_threshold=0.6,
    )

    assert stream._pending_config_update == {
        "event": "config.update",
        "language_code": "en-IN",
        "stream_type": "fast",
        "mode": "translate",
        "prompt": "LiveKit",
        "threshold": 0.6,
    }


def test_active_stream_keeps_connection_only_options_on_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _make_stream()
    stream._pending_config_update = None

    with caplog.at_level(logging.WARNING, logger=stt_streaming.logger.name):
        stream.update_options(
            sample_rate=8000,
            return_timestamps=True,
            prompt="LiveKit",
        )

    assert stream._opts.sample_rate == 16000
    assert stream._opts.return_timestamps is False
    assert stream._pending_config_update == {
        "event": "config.update",
        "prompt": "LiveKit",
    }
    assert "only apply to new streams" in caplog.text


def test_streaming_option_updates_merge_before_the_next_audio_frame() -> None:
    stream = _make_stream()
    stream._pending_config_update = None
    stream.update_options(prompt="LiveKit")
    stream.update_options(prompt="LiveKit", mode="translate")

    assert stream._pending_config_update == {
        "event": "config.update",
        "prompt": "LiveKit",
        "mode": "translate",
    }


def test_streaming_option_update_clears_prompt_with_empty_string() -> None:
    previous = stt_streaming.RealtimeSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        prompt="LiveKit",
    )
    current = stt_streaming.RealtimeSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        prompt=None,
    )

    assert stt_streaming.RealtimeSpeechStream._config_update_payload(previous, current) == {
        "event": "config.update",
        "prompt": "",
    }


@pytest.mark.asyncio
async def test_instance_update_preserves_per_stream_language_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated instance-level update must not clobber a per-stream override.

    ``stream(language=...)`` is a per-stream override, so pushing the instance
    defaults wholesale would silently reset it and queue a server-side language
    change the caller never asked for.
    """

    async def _idle_run(self: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(stt_streaming.RealtimeSpeechStream, "_run", _idle_run)

    stt_impl = stt_streaming.STTRealtime(
        api_key="sk_test",
        language="hi-IN",
        http_session=object(),  # type: ignore[arg-type]
    )
    stream = stt_impl.stream(language="ta-IN")

    stt_impl.update_options(prompt="LiveKit")

    assert stream._opts.language == "ta-IN"
    assert stt_impl._opts.language == "hi-IN"
    assert stream._pending_config_update == {
        "event": "config.update",
        "prompt": "LiveKit",
    }

    await stream.aclose()
    await stt_impl.aclose()


def test_stream_update_options_without_arguments_is_a_no_op() -> None:
    stream = _make_stream()
    stream._pending_config_update = None

    stream.update_options()

    assert stream._pending_config_update is None
    assert stream._opts.language == "hi-IN"


@pytest.mark.asyncio
async def test_active_stream_defers_endpointing_until_config_acknowledgement() -> None:
    stream = _make_stream()
    stream._active_endpointing = "vad"
    stream._utterance_in_progress = True
    stream._pending_endpointing = None
    stream._endpointing_update_acknowledged = False
    stream._pending_config_update = None
    stream.update_options(endpointing="manual")

    assert stream._active_endpointing == "vad"
    assert stream._pending_endpointing == "manual"

    assert await _flush_pending_config_update(stream) == [
        {"event": "config.update", "endpointing": "manual"}
    ]
    stream._handle_config_updated({"applied": ["endpointing=manual"]})
    assert stream._active_endpointing == "vad"

    stream._utterance_in_progress = False
    stream._apply_pending_endpointing()
    assert stream._active_endpointing == "manual"


@pytest.mark.asyncio
async def test_endpointing_ignores_ack_for_an_update_not_yet_sent() -> None:
    """An earlier update's acknowledgement must not promote a still-queued switch.

    The payload is only flushed on the next audio frame, so a ``config.updated`` for a
    previous change can arrive first. Promoting on it would put the client in manual
    mode, sending client-delimited boundaries, while the server is still in ``vad``.
    """
    stream = _make_stream()
    # An earlier prompt update that has already been sent and is awaiting its ack.
    stream.update_options(prompt="LiveKit")
    await _flush_pending_config_update(stream)
    # The endpointing switch is queued but not yet on the wire.
    stream.update_options(endpointing="manual")
    assert stream._pending_config_update == {
        "event": "config.update",
        "endpointing": "manual",
    }

    await stream._handle_message({"event": "config.updated", "applied": ["prompt=LiveKit"]})

    assert stream._active_endpointing == "vad"
    assert stream._pending_endpointing == "manual"
    assert stream._endpointing_update_acknowledged is False

    # Once it is actually sent and acknowledged, the switch promotes.
    await _flush_pending_config_update(stream)
    await stream._handle_message({"event": "config.updated", "applied": ["endpointing=manual"]})
    assert stream._active_endpointing == "manual"


@pytest.mark.asyncio
async def test_endpointing_ignores_ack_that_does_not_list_endpointing() -> None:
    """Acks are per message, so a later unrelated ack can arrive after ours is sent."""
    stream = _make_stream()
    stream.update_options(endpointing="manual")
    await _flush_pending_config_update(stream)

    await stream._handle_message({"event": "config.updated", "applied": ["mode=translate"]})

    assert stream._active_endpointing == "vad"
    assert stream._pending_endpointing == "manual"

    # A multi-key ack that does include endpointing is accepted.
    await stream._handle_message(
        {"event": "config.updated", "applied": ["mode=translate", "endpointing=manual"]}
    )
    assert stream._active_endpointing == "manual"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "applied",
    [None, "endpointing=manual", [42]],
    ids=["missing", "not_a_list", "non_string_entries"],
)
async def test_endpointing_falls_back_when_applied_list_is_unusable(
    applied: object,
) -> None:
    """An unexpected `applied` shape must not stall the switch forever."""
    stream = _make_stream()
    stream.update_options(endpointing="manual")
    await _flush_pending_config_update(stream)

    payload: dict[str, object] = {"event": "config.updated"}
    if applied is not None:
        payload["applied"] = applied
    await stream._handle_message(payload)

    assert stream._active_endpointing == "manual"


@pytest.mark.asyncio
async def test_deferred_endpointing_ack_is_accepted() -> None:
    """The server suffixes the entry when it defers to the next utterance boundary."""
    stream = _make_stream()
    stream.update_options(endpointing="manual")
    await _flush_pending_config_update(stream)

    await stream._handle_message(
        {
            "event": "config.updated",
            "applied": ["endpointing=manual (pending: applies next boundary)"],
        }
    )

    assert stream._active_endpointing == "manual"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final_payload",
    [
        None,
        {"event": "transcript.final", "utterance_idx": 0, "text": ""},
        {"event": "transcript.final", "utterance_idx": 0, "text": "   "},
    ],
    ids=["no_final", "empty_final", "whitespace_final"],
)
async def test_speech_end_promotes_pending_endpointing_without_a_valid_final(
    final_payload: dict[str, object] | None,
) -> None:
    """A turn with no usable final must still count as an utterance boundary.

    Otherwise `_utterance_in_progress` stays set, `_apply_pending_endpointing` refuses
    to promote, and the stream is stranded in ``vad`` while the server has already
    acknowledged the switch to ``manual`` and stopped sending ``vad.*`` events.
    """
    stream = _make_stream()
    stream.update_options(endpointing="manual")
    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await _flush_pending_config_update(stream)
    await stream._handle_message({"event": "config.updated", "applied": ["endpointing=manual"]})

    assert stream._utterance_in_progress is True
    assert stream._active_endpointing == "vad"

    if final_payload is not None:
        await stream._handle_message(final_payload)
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert stream._utterance_in_progress is False
    assert stream._active_endpointing == "manual"
    assert stream._pending_endpointing is None


@pytest.mark.asyncio
@pytest.mark.parametrize("endpointing", ["vad", "manual"])
@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
async def test_blank_final_transcript_is_not_emitted(endpointing: str, text: str) -> None:
    """A blank final has no words, so committing it would trigger a reply on nothing."""
    stream = _make_stream(endpointing=endpointing)
    stream._audio_position = 1.0

    await stream._handle_message({"event": "transcript.final", "utterance_idx": 0, "text": text})

    finals = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == []
    assert stream._pending_final_data is None


@pytest.mark.asyncio
async def test_repeated_speech_end_still_completes_the_utterance() -> None:
    stream = _make_stream()
    stream.update_options(endpointing="manual")
    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await _flush_pending_config_update(stream)
    await stream._handle_message({"event": "config.updated", "applied": ["endpointing=manual"]})
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})
    # A duplicate speech end takes the already-emitted branch; it must not reopen or
    # strand the utterance.
    stream._utterance_in_progress = True
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert stream._utterance_in_progress is False
    eos_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.END_OF_SPEECH
    ]
    assert len(eos_events) == 1


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
    # The endpoint sends no recognition confidence, so it falls back to 1.0 and the
    # language-identification score stays in metadata.
    assert final_event.alternatives[0].confidence == 1.0
    assert final_event.alternatives[0].metadata["language_confidence"] == 0.99
    assert final_event.alternatives[0].end_time == 1.75

    eos_event = stream._event_ch.events[2]
    assert eos_event.alternatives == []
    assert final_event.alternatives[0].metadata["speech_end_wall_time"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload_confidence", "expected"),
    [
        ({"confidence": 0.42}, 0.42),
        ({}, 1.0),
        ({"confidence": None}, 1.0),
        ({"confidence": "high"}, 1.0),
        # bool is a subclass of int; False must not read as zero confidence.
        ({"confidence": False}, 1.0),
    ],
)
async def test_transcript_confidence_uses_recognition_score_with_absent_fallback(
    payload_confidence: dict[str, object],
    expected: float,
) -> None:
    """`language_confidence` is a language-ID score and must not stand in for it.

    An absent value falls back to 1.0 rather than 0.0, matching `_extract_confidence`
    in stt.py, so livekit-agents' confidence averaging is not dragged toward zero.
    """
    stream = _make_stream(endpointing="manual")

    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते",
            "language": "hi-IN",
            "language_confidence": 0.2,
            **payload_confidence,
        }
    )

    final_event = stream._event_ch.events[0]
    assert final_event.alternatives[0].confidence == expected
    assert final_event.alternatives[0].metadata["language_confidence"] == 0.2


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
    assert eos_event.alternatives == []
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
    assert eos_event.alternatives == []


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
    assert not hasattr(info_records[0], "text")
    assert info_records[0].text_length == len("hello")
    assert info_records[0].language == "en-IN"
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
    assert debug_records[0].raw_data["text"] == "hello"


@pytest.mark.asyncio
async def test_streaming_emits_end_of_speech_when_final_never_arrives() -> None:
    """Speech end is reported immediately, without waiting on a final transcript."""
    stream = _make_stream()

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._eos_emitted_for_utterance is True


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
    assert eos_event.alternatives == []
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
    # The 1.5s already billed incrementally is topped up to the server total of 2.25s.
    assert [event.recognition_usage.audio_duration for event in usage_events] == [1.5, 0.75]
    assert sum(event.recognition_usage.audio_duration for event in usage_events) == 2.25
    assert all(event.request_id == "sess_123" for event in usage_events)
    assert stream._session_ended is True
    assert stream._server_audio_duration_reported is True


@pytest.mark.asyncio
async def test_streaming_session_end_falls_back_to_local_duration_without_server_usage() -> None:
    stream = _make_stream()
    stream._request_id = "req_123"
    stream._on_audio_duration_report(1.5)

    await stream._handle_message({"event": "session.end", "request_id": "req_123"})
    await stream._handle_message({"event": "session.end", "request_id": "req_123"})

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [1.5]
    assert stream._server_audio_duration_reported is False
    assert stream._session_ended is True


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
    # 5.0s was already billed incrementally, so a smaller server total adds nothing
    # rather than double counting or emitting a negative delta.
    assert [event.recognition_usage.audio_duration for event in usage_events] == [5.0]


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

    monkeypatch.setattr(stt_streaming.RealtimeSpeechStream, "_run", _idle_run)
    stt_impl = stt_streaming.STTRealtime(api_key="sk_test")
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


@pytest.mark.asyncio
async def test_session_end_commits_final_buffered_without_speech_end() -> None:
    stream = _make_stream()
    stream._audio_position = 3.5

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
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
    ]

    await stream._handle_message(
        {
            "event": "session.end",
            "request_id": "req_123",
            "audio_duration_s": 3.5,
        }
    )

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    final_event = stream._event_ch.events[2]
    assert final_event.alternatives[0].text == "नमस्ते आप कैसे हैं"
    assert final_event.alternatives[0].end_time == 3.5
    assert stream._pending_final_data is None


@pytest.mark.asyncio
async def test_clean_close_commits_final_buffered_without_speech_end() -> None:
    stream = _make_stream()
    stream._audio_position = 2.0
    ws = SimpleNamespace(
        receive=lambda: asyncio.sleep(
            0,
            result=SimpleNamespace(
                type=stt_streaming.aiohttp.WSMsgType.CLOSE,
                data=1000,
                extra=None,
            ),
        ),
        close_code=1000,
    )

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "hello",
            "language": "en-IN",
            "language_confidence": 0.99,
        }
    )

    await stream._process_messages(ws)

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert stream._event_ch.events[2].alternatives[0].end_time == 2.0


@pytest.mark.asyncio
async def test_terminal_flush_is_idempotent_across_session_end_and_close() -> None:
    stream = _make_stream()
    stream._audio_position = 1.0

    await stream._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "hello",
            "language": "en-IN",
        }
    )
    await stream._handle_message({"event": "session.end", "request_id": "req_123"})
    stream._flush_terminal_utterance()

    finals = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    eos_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.END_OF_SPEECH
    ]
    assert len(finals) == 1
    assert len(eos_events) == 1


@pytest.mark.asyncio
async def test_audio_duration_collector_emits_usage_incrementally() -> None:
    stream = _make_stream()

    stream._on_audio_duration_report(5.0)
    stream._on_audio_duration_report(5.0)

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [5.0, 5.0]
    assert stream._total_reported_audio_duration == 10.0
    assert stream._local_audio_duration == 10.0


@pytest.mark.asyncio
async def test_session_end_tops_up_pending_collector_audio_to_server_total() -> None:
    stream = _make_stream()
    stream._audio_duration_collector.push(1.0)

    await stream._handle_message(
        {
            "event": "session.end",
            "request_id": "req_123",
            "audio_duration_s": 2.5,
        }
    )

    usage_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.recognition_usage.audio_duration for event in usage_events] == [1.0, 1.5]
    assert stream._total_reported_audio_duration == 2.5


@pytest.mark.asyncio
async def test_aclose_flushes_pending_audio_duration_into_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _idle_run(self: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(stt_streaming.RealtimeSpeechStream, "_run", _idle_run)

    stt_impl = stt_streaming.STTRealtime(
        api_key="sk_test",
        http_session=object(),  # type: ignore[arg-type]
    )
    metrics = []
    stt_impl.on("metrics_collected", metrics.append)
    stream = stt_impl.stream()
    stream._request_id = "req_123"
    stream._audio_duration_collector.push(1.75)

    await stream.aclose()
    await stt_impl.aclose()

    assert [metric.audio_duration for metric in metrics] == [1.75]
    assert metrics[0].request_id == "req_123"


@pytest.mark.asyncio
async def test_aclose_skips_usage_flush_when_stream_already_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_run(self: object) -> None:
        raise APIStatusError("boom", status_code=1006)

    monkeypatch.setattr(stt_streaming.RealtimeSpeechStream, "_run", _failing_run)

    stt_impl = stt_streaming.STTRealtime(
        api_key="sk_test",
        http_session=object(),  # type: ignore[arg-type]
    )
    stream = stt_impl.stream()
    stream._audio_duration_collector.push(1.0)

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass

    # The event channel closes with the failed run, so the flush must not raise.
    await stream.aclose()
    await stt_impl.aclose()


@pytest.mark.asyncio
async def test_manual_endpointing_emits_boundary_events_and_resets_each_turn() -> None:
    stream = _make_stream(endpointing="manual")
    sent: list[dict[str, object]] = []
    ws = SimpleNamespace(
        closed=False,
        send_str=lambda payload: asyncio.sleep(0, result=sent.append(json.loads(payload))),
        send_bytes=lambda payload: asyncio.sleep(0),
    )
    frame = stt_streaming.rtc.AudioFrame(
        data=bytes(16000),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=8000,
    )

    async def _input() -> object:
        yield frame
        yield stream._FlushSentinel()
        yield frame
        yield stream._FlushSentinel()

    stream._input_ch = _input()
    stream._FlushSentinel = stt_streaming.stt.RecognizeStream._FlushSentinel

    async def _run_turns() -> None:
        await stream._process_audio(ws)

    await _run_turns()

    assert [payload["event"] for payload in sent] == [
        "speech_start",
        "speech_end",
        "speech_start",
        "speech_end",
        "end",
    ]
    # The collector is flushed before the turn boundary is sent, so usage lands
    # between the speech events of each turn.
    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.RECOGNITION_USAGE,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
    ]
    # A second end-of-speech only appears because the turn reset cleared
    # _eos_emitted_for_utterance; without it _emit_end_of_speech returns early.
    first_eos, second_eos = (
        stream._event_ch.events[2],
        stream._event_ch.events[5],
    )
    assert first_eos.alternatives == []
    assert second_eos.alternatives == []
    # The speech-end position is re-anchored to the second turn.
    assert stream._utterance_speech_end_audio_pos == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_manual_final_uses_current_turn_timings_after_first_turn() -> None:
    stream = _make_stream(endpointing="manual")
    stream._audio_position = 4.0
    stream._utterance_speech_end_audio_pos = 1.0
    stream._utterance_speech_end_wall = 100.0
    stream._eos_emitted_for_utterance = True

    stream._begin_manual_utterance()
    stream._audio_position = 6.0
    stream._end_manual_utterance()
    await stream._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 1,
            "text": "second turn",
            "language": "en-IN",
        }
    )

    assert [event.type for event in stream._event_ch.events] == [
        stt_streaming.stt.SpeechEventType.START_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.END_OF_SPEECH,
        stt_streaming.stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    final_event = stream._event_ch.events[2]
    assert final_event.alternatives[0].end_time == 6.0
    assert final_event.alternatives[0].metadata["speech_end_wall_time"] > 100.0


@pytest.mark.asyncio
@pytest.mark.parametrize("endpointing", ["vad", "manual"])
async def test_end_of_speech_is_emitted_as_a_bare_pipeline_sentinel(endpointing: str) -> None:
    """END_OF_SPEECH must carry no alternatives.

    livekit-agents treats an event without alternatives as a sentinel it holds until a
    concrete transcript releases it (``_should_hold_stt_event``) and always flushes from
    (``_flush_held_transcripts``). An alternative with ``start_time == 0`` would bypass
    the hold check and be timestamp-compared like a transcript instead.
    """
    stream = _make_stream(endpointing=endpointing)
    stream._audio_position = 2.0
    stream._utterance_speech_end_audio_pos = 2.0
    stream._utterance_speech_end_wall = stt_streaming.time.time()

    stream._emit_end_of_speech()

    eos_events = [
        event
        for event in stream._event_ch.events
        if event.type == stt_streaming.stt.SpeechEventType.END_OF_SPEECH
    ]
    assert len(eos_events) == 1
    assert eos_events[0].alternatives == []
    assert eos_events[0].request_id == stream._request_id


@pytest.mark.asyncio
async def test_streaming_safe_send_bytes_ignores_closed_transport() -> None:
    stream = _make_stream()
    calls = []

    closed_ws = SimpleNamespace(closed=True, send_bytes=lambda payload: calls.append(payload))
    await stream._safe_send_bytes(closed_ws, b"\x00\x01")

    assert calls == []

    async def _raise_reset(payload: bytes) -> None:
        raise stt_streaming.aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    closing_ws = SimpleNamespace(closed=False, send_bytes=_raise_reset)
    await stream._safe_send_bytes(closing_ws, b"\x00\x01")


@pytest.mark.asyncio
async def test_process_audio_stops_pumping_after_session_end() -> None:
    """The audio pump must end cleanly once the server has closed the session.

    ``_run`` gathers the audio and message pumps, so an exception here would fail the
    whole stream rather than letting it finish.
    """
    stream = _make_stream()
    stream._session_ended = True
    sent: list[object] = []

    async def _raise_reset(payload: object) -> None:
        raise stt_streaming.aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    ws = SimpleNamespace(
        closed=False,
        send_str=lambda payload: asyncio.sleep(0, result=sent.append(payload)),
        send_bytes=_raise_reset,
    )
    frame = stt_streaming.rtc.AudioFrame(
        data=bytes(16000),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=8000,
    )

    async def _input() -> object:
        yield frame
        yield frame

    stream._input_ch = _input()
    stream._FlushSentinel = stt_streaming.stt.RecognizeStream._FlushSentinel

    await stream._process_audio(ws)

    # No audio was written and no redundant "end" was sent for an ended session.
    assert sent == []
    assert stream._audio_position == 0.0


@pytest.mark.asyncio
async def test_process_audio_survives_reset_race_on_peer_close() -> None:
    """A reset raised mid-send must not fail the stream.

    The peer can close between the loop's closed check and the write, so the audio
    send has to tolerate a reset the same way the JSON control sends do.
    """
    stream = _make_stream()
    sent: list[str] = []

    async def _raise_reset(payload: object) -> None:
        raise stt_streaming.aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    ws = SimpleNamespace(
        closed=False,
        send_str=lambda payload: asyncio.sleep(0, result=sent.append(payload)),
        send_bytes=_raise_reset,
    )
    frame = stt_streaming.rtc.AudioFrame(
        data=bytes(16000),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=8000,
    )

    async def _input() -> object:
        yield frame
        yield frame

    stream._input_ch = _input()
    stream._FlushSentinel = stt_streaming.stt.RecognizeStream._FlushSentinel

    await stream._process_audio(ws)

    assert [json.loads(payload)["event"] for payload in sent] == ["end"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_type", ["fast", "balanced"])
async def test_process_audio_forwards_small_frames_regardless_of_stream_type(
    stream_type: str,
) -> None:
    """Audio must reach the server in small frames.

    ``stream_type`` is the server's flush profile, not a client send cadence, so
    buffering 500-1000 ms locally would delay server VAD boundaries and partials by
    that much.
    """
    stream = _make_stream()
    stream._opts = stt_streaming.replace(stream._opts, stream_type=stream_type)
    sent: list[bytes] = []

    ws = SimpleNamespace(
        closed=False,
        send_str=lambda payload: asyncio.sleep(0),
        send_bytes=lambda payload: asyncio.sleep(0, result=sent.append(payload)),
    )
    # One second of 16 kHz mono audio.
    frame = stt_streaming.rtc.AudioFrame(
        data=bytes(32000),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=16000,
    )

    async def _input() -> object:
        yield frame

    stream._input_ch = _input()
    stream._FlushSentinel = stt_streaming.stt.RecognizeStream._FlushSentinel

    await stream._process_audio(ws)

    expected_bytes = int(16000 * stt_streaming.AUDIO_CHUNK_MS / 1000) * 2
    assert stt_streaming.AUDIO_CHUNK_MS <= 100
    assert len(sent) == 1000 // stt_streaming.AUDIO_CHUNK_MS
    assert {len(payload) for payload in sent} == {expected_bytes}
