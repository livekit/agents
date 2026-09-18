from __future__ import annotations

from dataclasses import replace
from typing import Any, get_args

import pytest

from livekit.agents import APIStatusError
from livekit.agents.types import APIConnectOptions
from livekit.plugins.sarvam import tts as sarvam_tts

pytestmark = pytest.mark.unit


def _make_tts(**kwargs: Any) -> sarvam_tts.TTS:
    return sarvam_tts.TTS(api_key="sk_test", http_session=object(), **kwargs)  # type: ignore[arg-type]


def test_synthesize_honors_explicit_conn_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturedChunkedStream:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(sarvam_tts, "ChunkedStream", _CapturedChunkedStream)
    tts = _make_tts()
    conn_options = APIConnectOptions(max_retry=5, retry_interval=1.5, timeout=12.0)

    tts.synthesize("hello", conn_options=conn_options)

    stream_conn_options = captured["conn_options"]
    assert stream_conn_options is conn_options
    assert stream_conn_options.max_retry == 5
    assert stream_conn_options.retry_interval == 1.5
    assert stream_conn_options.timeout == 12.0


def test_stream_honors_explicit_conn_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _CapturedSynthesizeStream:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(sarvam_tts, "SynthesizeStream", _CapturedSynthesizeStream)
    tts = _make_tts()
    conn_options = APIConnectOptions(max_retry=5, retry_interval=1.5, timeout=12.0)

    tts.stream(conn_options=conn_options)

    stream_conn_options = captured["conn_options"]
    assert stream_conn_options is conn_options
    assert stream_conn_options.max_retry == 5
    assert stream_conn_options.retry_interval == 1.5
    assert stream_conn_options.timeout == 12.0


def test_v4_defaults_to_its_own_speaker_and_ws_v2_endpoint() -> None:
    tts = _make_tts(model="bulbul:v4-flash")

    assert tts._opts.speaker == "shubh_en_narration_gentle"
    assert sarvam_tts._websocket_url(tts._opts) == (
        "wss://api.sarvam.ai/text-to-speech/ws/v2?model=bulbul:v4-flash&send_completion_event=True"
    )

    pinned = _make_tts(model="bulbul:v4-flash", ws_url="wss://example.test/text-to-speech/ws/v2")
    assert sarvam_tts._websocket_url(pinned._opts).startswith(
        "wss://example.test/text-to-speech/ws/v2?"
    )


def test_v3_keeps_v1_websocket_endpoint_and_speaker() -> None:
    tts = _make_tts(model="bulbul:v3")

    assert tts._opts.speaker == "shubh"
    assert sarvam_tts._websocket_url(tts._opts) == (
        "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=True"
    )


def test_v4_rejects_v3_speaker_names() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        _make_tts(model="bulbul:v4-flash", speaker="shubh")

    tts = _make_tts(model="bulbul:v4-flash", speaker="ritu_hi_medical")
    assert tts._opts.speaker == "ritu_hi_medical"


def test_v4_request_fields() -> None:
    tts = _make_tts(model="bulbul:v4-flash", dict_id="dict-1", enable_cached_responses=True)

    # caching is ignored by v4, so it is never sent
    assert sarvam_tts._model_extra_fields(tts._opts) == {
        "pitch": 0.0,
        "loudness": 1.0,
        "enable_preprocessing": False,
        "temperature": 0.6,
        "dict_id": "dict-1",
    }
    # the websocket config adds the buffering knobs for the shared v3/v4 pipeline
    assert "bulbul:v4-flash" in sarvam_tts._V3_PIPELINE_MODELS


def test_v3_request_fields_unchanged() -> None:
    tts = _make_tts(model="bulbul:v3", dict_id="dict-1")

    assert sarvam_tts._model_extra_fields(tts._opts) == {
        "temperature": 0.6,
        "dict_id": "dict-1",
    }


def test_v4_parameter_bounds() -> None:
    assert _make_tts(model="bulbul:v4-flash", loudness=2.5)._opts.loudness == 2.5
    assert _make_tts(model="bulbul:v4-flash", pitch=0.7)._opts.pitch == 0.5  # clamped, not rejected

    with pytest.raises(ValueError, match="loudness"):
        _make_tts(model="bulbul:v4-flash", loudness=3.0)
    with pytest.raises(ValueError, match="pace"):
        _make_tts(model="bulbul:v4-flash", pace=0.3)
    with pytest.raises(ValueError, match="temperature"):
        _make_tts(model="bulbul:v4-flash", temperature=1.5)

    # v3 keeps the wider ranges
    v3 = _make_tts(model="bulbul:v3", pace=0.3, temperature=1.5)
    assert (v3._opts.pace, v3._opts.temperature) == (0.3, 1.5)


def test_v4_streaming_sample_rate_limits() -> None:
    too_high = _make_tts(model="bulbul:v4-flash", speech_sample_rate=48000)
    with pytest.raises(ValueError, match="speech_sample_rate"):
        sarvam_tts._websocket_url(too_high._opts)

    opus = _make_tts(model="bulbul:v4-flash", speech_sample_rate=22050, output_audio_codec="opus")
    with pytest.raises(ValueError, match="speech_sample_rate"):
        sarvam_tts._websocket_url(opus._opts)

    v3 = _make_tts(model="bulbul:v3", speech_sample_rate=48000)
    assert "/text-to-speech/ws?model=bulbul:v3" in sarvam_tts._websocket_url(v3._opts)


def test_v4_flash_is_the_only_accepted_v4_wire_name() -> None:
    """The API rejects `bulbul:v4` with a 400; `bulbul:v4-flash` is the only valid spelling."""
    accepted = set(get_args(sarvam_tts.SarvamTTSModels))
    assert "bulbul:v4-flash" in accepted
    assert "bulbul:v4" not in accepted
    assert "bulbul:v4" not in sarvam_tts.MODEL_SPEAKER_COMPATIBILITY

    tts = _make_tts(model="bulbul:v4-flash")
    # `_opts.model` is the value sent as the REST body's "model" field
    assert tts._opts.model == "bulbul:v4-flash"
    assert "model=bulbul:v4-flash&" in sarvam_tts._websocket_url(tts._opts)


def test_rejected_update_options_leaves_live_options_untouched() -> None:
    tts = _make_tts(model="bulbul:v3")
    before = replace(tts._opts)

    # `shubh` is a v3-only speaker, so switching model alone cannot succeed
    with pytest.raises(ValueError, match="incompatible"):
        tts.update_options(model="bulbul:v4-flash")

    assert (tts._opts.model, tts._opts.speaker) == (before.model, before.speaker)
    assert sarvam_tts._websocket_url(tts._opts) == sarvam_tts._websocket_url(before)


def test_update_options_revalidates_stored_params_against_the_new_model() -> None:
    # pace 0.3 and temperature 1.5 are valid on v3 but out of range on v4-flash
    tts = _make_tts(model="bulbul:v3", pace=0.3, temperature=1.5)

    with pytest.raises(ValueError, match="pace"):
        tts.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical")
    assert tts._opts.model == "bulbul:v3"

    tts.update_options(pace=1.0, temperature=0.6)
    tts.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical")
    assert tts._opts.model == "bulbul:v4-flash"

    # v3 pitch 0.7 is clamped to the tighter v4-flash bound on the switch
    wide = _make_tts(model="bulbul:v3", pitch=0.7)
    assert wide._opts.pitch == 0.7
    wide.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical")
    assert wide._opts.pitch == 0.5


def test_update_options_invalidates_the_pool_only_for_handshake_fields() -> None:
    tts = _make_tts(model="bulbul:v3")
    invalidated: list[bool] = []
    tts._pool.invalidate = lambda: invalidated.append(True)  # type: ignore[method-assign]

    # pace rides in the per-request config, so the pooled socket stays usable
    tts.update_options(pace=1.2)
    assert invalidated == []

    # model and send_completion_event are pinned in the handshake URL
    tts.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical")
    assert invalidated == [True]

    tts.update_options(send_completion_event=False)
    assert invalidated == [True, True]


def test_stream_config_follows_the_socket_it_was_handed() -> None:
    """A stream created before update_options must not send its old model's config."""
    tts = _make_tts(model="bulbul:v3", speaker="shubh", temperature=1.5)

    stream = object.__new__(sarvam_tts.SynthesizeStream)
    stream._tts = tts
    stream._opts = replace(tts._opts)

    tts.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical", temperature=0.6)

    # stand in for the socket the pool would hand over after the switch
    ws = object()
    tts._ws_handshake_opts[id(ws)] = replace(tts._opts)
    stream._adopt_handshake_opts(ws)  # type: ignore[arg-type]

    assert (stream._opts.model, stream._opts.speaker) == ("bulbul:v4-flash", "ritu_hi_medical")
    # the emitter was already initialized from these, so they stay snapshotted
    assert stream._opts.speech_sample_rate == tts._opts.speech_sample_rate
    assert stream._opts.output_audio_codec == tts._opts.output_audio_codec

    # an untracked socket leaves the snapshot alone rather than guessing
    before = replace(stream._opts)
    stream._adopt_handshake_opts(object())  # type: ignore[arg-type]
    assert stream._opts == before


def test_config_only_update_survives_a_reused_socket() -> None:
    """Speaker and tuning ride in the config frame, so the socket must not revert them."""
    tts = _make_tts(model="bulbul:v3", speaker="shubh", pace=1.0)

    # a socket handshaken before the update; its model is unchanged, so the pool
    # legitimately keeps reusing it
    ws = object()
    tts._ws_handshake_opts[id(ws)] = replace(tts._opts)

    tts.update_options(speaker="ritu", pace=1.2)
    stream = object.__new__(sarvam_tts.SynthesizeStream)
    stream._tts = tts
    stream._opts = replace(tts._opts)

    stream._adopt_handshake_opts(ws)  # type: ignore[arg-type]

    assert (stream._opts.speaker, stream._opts.pace) == ("ritu", 1.2)


def _error_stream() -> sarvam_tts.SynthesizeStream:
    """A SynthesizeStream carrying only the attributes `_handle_error_message` reads.

    The real ``__init__`` spawns a task that connects to the API, which a unit test
    must not do.
    """
    stream = object.__new__(sarvam_tts.SynthesizeStream)
    stream._opts = _make_tts(model="bulbul:v4-flash")._opts
    stream._session_id = 0
    stream._connection_state = sarvam_tts.ConnectionState.CONNECTED
    stream._client_request_id = None
    stream._server_request_id = None
    return stream


@pytest.mark.parametrize(
    ("frame", "status_code", "retryable"),
    [
        # schema rejections carry an integer `code`
        ({"code": 422, "message": "Input parameters has to be a valid dictionary"}, 422, False),
        (
            {"code": 400, "message": "Speech sample rate can only be 8000, 16000, 22050, 24000 Hz"},
            400,
            False,
        ),
        # speaker and codec rejections omit `code` and prefix the message instead
        (
            {"message": "400: Speaker 'shubh' is not compatible with model bulbul:v4-flash"},
            400,
            False,
        ),
        # transient failures stay retryable
        ({"code": 429, "message": "rate limit exceeded"}, 429, True),
        ({"code": 503, "message": "model unavailable"}, 503, True),
        # an unrecognized frame keeps the previous retry-by-default behaviour
        ({"message": "something we cannot classify"}, -1, True),
    ],
)
async def test_error_frame_status_code_drives_retryability(
    frame: dict[str, Any], status_code: int, retryable: bool
) -> None:
    resp = {"type": "error", "data": frame}

    with pytest.raises(APIStatusError) as exc:
        await _error_stream()._handle_error_message(resp)

    assert exc.value.status_code == status_code
    assert exc.value.retryable is retryable
    # __str__ renders message and body, and the framework logs it with %s when it
    # retries, so no provider-written text may be reachable through it
    assert frame["message"] not in str(exc.value)


async def test_error_frame_keeps_provider_text_in_redactable_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    speech = "my card number is 4111 1111 1111 1111"
    resp = {"type": "error", "data": {"code": 422, "message": f"cannot synthesize: {speech}"}}

    with caplog.at_level("ERROR", logger=sarvam_tts.logger.name):
        with pytest.raises(APIStatusError) as exc:
            await _error_stream()._handle_error_message(resp)

    record = next(r for r in caplog.records if r.name == sarvam_tts.logger.name)
    # provider text reaches the log only under lk.pii.* keys, which collectors redact
    assert speech not in record.getMessage()
    assert speech in record.__dict__["lk.pii.error_message"]
    assert "error_message" not in record.__dict__
    # this record is the only place the frame survives in full, so it has to
    assert record.__dict__["lk.pii.raw_message"] == resp
    # the exception the framework logs with %s carries none of it
    assert speech not in str(exc.value)


async def test_error_frame_forwards_request_id() -> None:
    with pytest.raises(APIStatusError) as exc:
        await _error_stream()._handle_error_message(
            {
                "type": "error",
                "data": {"request_id": "20260918_abc", "code": 422, "message": "bad input"},
            }
        )

    assert exc.value.request_id == "20260918_abc"
