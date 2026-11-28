from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock

import aiohttp
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    tts,
)
from livekit.agents.tts import AudioEmitter
from livekit.plugins.sarvam import models as sarvam_models, tts as sarvam_tts
from livekit.plugins.sarvam.tts import (
    ALLOWED_OUTPUT_AUDIO_CODECS,
    _codec_to_mime_type,
    _decode_telephony,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
async def _cleanup_leaked_keepalive_tasks() -> AsyncIterator[None]:
    """Cancel keepalive tasks left by tests that never close their TTS."""
    yield
    for task in list(asyncio.all_tasks()):
        if task.get_name() == "sarvam-tts-ws-keepalive" and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


def _make_tts(**kwargs: Any) -> sarvam_tts.TTS:
    return sarvam_tts.TTS(api_key="sk_test", http_session=object(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "cls"), [("synthesize", "ChunkedStream"), ("stream", "SynthesizeStream")]
)
def test_explicit_conn_options_reach_the_stream(
    monkeypatch: pytest.MonkeyPatch, method: str, cls: str
) -> None:
    captured: dict[str, object] = {}

    class _Captured:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(sarvam_tts, cls, _Captured)
    conn_options = APIConnectOptions(max_retry=5, retry_interval=1.5, timeout=12.0)

    t = _make_tts()
    if method == "synthesize":
        t.synthesize("hello", conn_options=conn_options)
    else:
        t.stream(conn_options=conn_options)

    assert captured["conn_options"] is conn_options


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
    tts = _make_tts(model="bulbul:v4-flash", dict_id="dict-1")

    # no temperature: bulbul:v4-flash ignores it server-side
    assert sarvam_tts._model_extra_fields(tts._opts) == {
        "pitch": 0.0,
        "loudness": 1.0,
        "enable_preprocessing": False,
        "dict_id": "dict-1",
    }


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
    accepted = set(get_args(sarvam_models.SarvamTTSModels))
    assert "bulbul:v4-flash" in accepted
    assert "bulbul:v4" not in accepted
    assert "bulbul:v4" not in sarvam_models.MODEL_SPEAKER_COMPATIBILITY

    tts = _make_tts(model="bulbul:v4-flash")
    # `_opts.model` is the value sent as the REST body's "model" field
    assert tts._opts.model == "bulbul:v4-flash"
    assert "model=bulbul:v4-flash&" in sarvam_tts._websocket_url(tts._opts)


def test_model_types_stay_importable_from_tts() -> None:
    """These were public in `tts` before moving to `models`, so callers may import them there."""
    for name in (
        "SarvamTTSModels",
        "SarvamTTSOutputAudioBitrate",
        "SarvamTTSLanguages",
        "SarvamTTSSpeakers",
        "MODEL_SPEAKER_COMPATIBILITY",
    ):
        assert getattr(sarvam_tts, name) is getattr(sarvam_models, name)


def test_rejected_update_options_leaves_live_options_untouched() -> None:
    tts = _make_tts(model="bulbul:v3")
    before = replace(tts._opts)

    # `shubh` is a v3-only speaker, so switching model alone cannot succeed
    with pytest.raises(ValueError, match="incompatible"):
        tts.update_options(model="bulbul:v4-flash")

    assert (tts._opts.model, tts._opts.speaker) == (before.model, before.speaker)
    assert sarvam_tts._websocket_url(tts._opts) == sarvam_tts._websocket_url(before)


def test_update_options_revalidates_stored_params_against_the_new_model() -> None:
    # pace 0.3 is valid on v3 but out of range on v4-flash
    tts = _make_tts(model="bulbul:v3", pace=0.3)

    with pytest.raises(ValueError, match="pace"):
        tts.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical")
    assert tts._opts.model == "bulbul:v3"

    tts.update_options(pace=1.0)
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


class _FakeSocket:
    """A Sarvam TTS websocket that records the frames sent to it and answers each flush."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.frames: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self._replies: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        frame = json.loads(data)
        self.frames.append(frame)
        if frame["type"] == "flush":
            self._replies.put_nowait({"type": "audio", "data": {"audio": "AAAA"}})
            self._replies.put_nowait({"type": "event", "data": {"event_type": "final"}})

    async def receive(self, timeout: float | None = None) -> SimpleNamespace:
        # between requests this blocks like an idle socket, which is what a keepalive sees
        reply = await self._replies.get()
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(reply), extra=None)

    async def close(self) -> None:
        self.closed = True

    def configs(self) -> list[dict[str, Any]]:
        return [frame["data"] for frame in self.frames if frame["type"] == "config"]


class _FakeSession:
    """Opens a `_FakeSocket` for every websocket handshake."""

    def __init__(self) -> None:
        self.sockets: list[_FakeSocket] = []

    async def ws_connect(self, url: str, **kwargs: Any) -> _FakeSocket:
        self.sockets.append(_FakeSocket(url))
        return self.sockets[-1]


_SocketTTS = Callable[..., tuple[sarvam_tts.TTS, _FakeSession]]


@pytest.fixture
async def socket_tts() -> AsyncIterator[_SocketTTS]:
    """Build TTS instances over a `_FakeSession`, closing them even if the test fails."""
    made: list[sarvam_tts.TTS] = []

    def make(**kwargs: Any) -> tuple[sarvam_tts.TTS, _FakeSession]:
        session = _FakeSession()
        made.append(sarvam_tts.TTS(api_key="sk_test", http_session=session, **kwargs))  # type: ignore[arg-type]
        return made[-1], session

    yield make
    for instance in made:
        await instance.aclose()


def _ws_stream(tts: sarvam_tts.TTS) -> sarvam_tts.SynthesizeStream:
    """A SynthesizeStream carrying what `_run_ws` reads, without the task `__init__` starts."""
    stream = object.__new__(sarvam_tts.SynthesizeStream)
    stream._tts = tts
    stream._opts = replace(tts._opts)
    stream._conn_options = APIConnectOptions(max_retry=0, timeout=5.0)
    stream._session_id = 0
    stream._connection_state = sarvam_tts.ConnectionState.DISCONNECTED
    stream._client_request_id = None
    stream._server_request_id = None
    stream._send_task = None
    stream._recv_task = None
    stream._ws_conn = None
    stream._mark_started = lambda: None  # type: ignore[method-assign]
    return stream


async def _speak(stream: sarvam_tts.SynthesizeStream) -> None:
    async def sentences() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(token="Namaste.")

    await stream._run_ws(sentences(), MagicMock(spec=AudioEmitter))  # type: ignore[arg-type]


async def test_stream_created_before_a_model_switch_speaks_with_the_new_options(
    socket_tts: _SocketTTS,
) -> None:
    """A stream adopts model, speaker and language together, but keeps its own rate and codec."""
    tts, session = socket_tts(
        model="bulbul:v4-flash",
        target_language_code="as-IN",
        speaker="kangkana_as_conversational",
        speech_sample_rate=24000,
        output_audio_codec="opus",
    )
    stream = _ws_stream(tts)

    tts.update_options(
        model="bulbul:v3", speaker="shubh", target_language_code="hi-IN", output_audio_codec="mp3"
    )
    await _speak(stream)

    (ws,) = session.sockets
    assert ws.url == (
        "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=True"
    )
    (config,) = ws.configs()
    assert (config["model"], config["speaker"], config["target_language_code"]) == (
        "bulbul:v3",
        "shubh",
        "hi-IN",
    )
    assert (config["output_audio_codec"], config["speech_sample_rate"]) == ("opus", 24000)


async def test_model_switch_during_the_pool_handshake_keeps_config_and_socket_in_step(
    socket_tts: _SocketTTS,
) -> None:
    """update_options can land while the pool is connecting for a stream."""
    tts, session = socket_tts()
    stream = _ws_stream(tts)
    connect = session.ws_connect

    async def connect_then_switch(url: str, **kwargs: Any) -> _FakeSocket:
        ws = await connect(url, **kwargs)
        if len(session.sockets) == 1:
            tts.update_options(model="bulbul:v4-flash", speaker="ritu_hi_medical")
        return ws

    session.ws_connect = connect_then_switch  # type: ignore[method-assign]
    await _speak(stream)

    # the pool drops the socket opened with the old options and connects again
    (used,) = [ws for ws in session.sockets if ws.frames]
    assert "?model=bulbul:v4-flash&" in used.url
    assert [(config["model"], config["speaker"]) for config in used.configs()] == [
        ("bulbul:v4-flash", "ritu_hi_medical")
    ]


async def test_config_only_update_applies_to_a_pending_stream_on_the_same_socket(
    socket_tts: _SocketTTS,
) -> None:
    """Speaker and tuning ride in the config frame, so changing them keeps the socket."""
    tts, session = socket_tts()

    await _speak(_ws_stream(tts))
    pending = _ws_stream(tts)
    tts.update_options(speaker="ritu", pace=1.2)
    await _speak(pending)

    (ws,) = session.sockets
    assert [(config["speaker"], config["pace"]) for config in ws.configs()] == [
        ("shubh", 1.0),
        ("ritu", 1.2),
    ]


_TO_V4_FLASH = {"model": "bulbul:v4-flash", "speaker": "ritu_hi_medical"}
_BACK_TO_V3 = {"model": "bulbul:v3", "speaker": "shubh"}


@pytest.mark.virtual_time
@pytest.mark.parametrize(
    "switches",
    [
        # nothing changed, so the pool hands this socket out again and keeps it alive
        [],
        [_TO_V4_FLASH],
        # the handshake matches the TTS again, but the pool still retired this socket
        [_TO_V4_FLASH, _BACK_TO_V3],
    ],
    ids=["unchanged", "switched", "switched-and-back"],
)
async def test_socket_retired_mid_utterance_is_closed_when_returned(
    socket_tts: _SocketTTS,
    monkeypatch: pytest.MonkeyPatch,
    switches: list[dict[str, str]],
) -> None:
    """The pool only closes a retired socket on its next acquisition, which may not come."""
    monkeypatch.setattr(sarvam_tts, "_KEEPALIVE_INTERVAL", 0.01)
    tts, session = socket_tts()
    connect = session.ws_connect

    async def connect_and_switch_on_flush(url: str, **kwargs: Any) -> _FakeSocket:
        ws = await connect(url, **kwargs)
        send_str = ws.send_str

        async def send_and_switch(data: str) -> None:
            await send_str(data)
            if json.loads(data)["type"] == "flush":
                for switch in switches:
                    tts.update_options(**switch)

        ws.send_str = send_and_switch  # type: ignore[method-assign]
        return ws

    session.ws_connect = connect_and_switch_on_flush  # type: ignore[method-assign]
    await _speak(_ws_stream(tts))
    await asyncio.sleep(0.05)

    (ws,) = session.sockets
    reused = not switches
    assert ws.closed is not reused
    assert any(frame["type"] == "ping" for frame in ws.frames) is reused


def _error_stream() -> sarvam_tts.SynthesizeStream:
    """A SynthesizeStream carrying only what `_handle_error_message` reads."""
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
        # a status sent as a digit string is still a status
        ({"code": "400", "message": "invalid speaker"}, 400, False),
        # transient failures stay retryable
        ({"code": 429, "message": "rate limit exceeded"}, 429, True),
        ({"code": 503, "message": "model unavailable"}, 503, True),
        # an unrecognized frame keeps the previous retry-by-default behaviour
        ({"message": "something we cannot classify"}, -1, True),
        ({"code": "invalid_request_error", "message": "bad input"}, -1, True),
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


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == sarvam_tts.logger.name and r.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    ("model", "replacement"),
    [("bulbul:v2", "bulbul:v4-flash"), ("bulbul:v3-beta", "bulbul:v3")],
)
def test_removed_model_warns_with_its_replacement(
    model: str, replacement: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts = _make_tts(model=model)

    assert any(model in w and f"'{replacement}'" in w for w in _warnings(caplog))
    # sent as given rather than swapped for the replacement; the Sarvam API has the last word
    assert tts._opts.model == model


def test_update_options_to_removed_model_warns(caplog: pytest.LogCaptureFixture) -> None:
    tts = _make_tts(model="bulbul:v3")

    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts.update_options(model="bulbul:v2")

    assert any("'bulbul:v4-flash'" in w for w in _warnings(caplog))
    assert tts._opts.model == "bulbul:v2"


@pytest.mark.parametrize(
    ("name", "value"),
    [("pitch", 0.5), ("loudness", 1.5), ("enable_preprocessing", True)],
)
def test_v4_flash_only_arg_on_v3_warns_instead_of_failing(
    name: str, value: object, caplog: pytest.LogCaptureFixture
) -> None:
    """A bulbul:v2 config moved to bulbul:v3 may still set these."""
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts = _make_tts(model="bulbul:v3", **{name: value})

    assert any(f"`{name}`" in w and "bulbul:v4-flash" in w for w in _warnings(caplog))
    assert name not in sarvam_tts._model_extra_fields(tts._opts)


def test_update_options_v4_flash_only_arg_on_v3_warns(caplog: pytest.LogCaptureFixture) -> None:
    tts = _make_tts(model="bulbul:v3")

    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts.update_options(loudness=1.5)

    assert any("`loudness`" in w for w in _warnings(caplog))


@pytest.mark.parametrize(
    "configure",
    [
        lambda: _make_tts(model="bulbul:v4-flash", temperature=0.3),
        lambda: _with_update(_make_tts(model="bulbul:v4-flash"), temperature=0.3),
        lambda: _with_update(
            _make_tts(model="bulbul:v3", speaker="ritu"),
            model="bulbul:v4-flash",
            speaker="ritu_hi_medical",
            temperature=0.3,
        ),
    ],
    ids=["init", "update_options", "update_options_model_switch"],
)
def test_v4_flash_temperature_warns_and_is_not_sent(
    configure: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """bulbul:v4-flash serves its own tuned sampling and drops temperature server-side."""
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts = configure()

    assert "temperature" not in sarvam_tts._model_extra_fields(tts._opts)
    assert any("`temperature`" in w and "bulbul:v4-flash" in w for w in _warnings(caplog))


def test_v4_flash_does_not_bound_the_temperature_it_ignores() -> None:
    # 1.5 is outside the API's 0.01-1.0, but it is never sent, so it cannot be rejected
    tts = _make_tts(model="bulbul:v4-flash", temperature=1.5)
    assert "temperature" not in sarvam_tts._model_extra_fields(tts._opts)

    tts.update_options(temperature=1.5)
    assert tts._opts.temperature == 1.5


def test_v3_temperature_is_still_sent_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts = _make_tts(model="bulbul:v3", temperature=0.3)
        tts.update_options(temperature=0.4)

    assert _warnings(caplog) == []
    assert sarvam_tts._model_extra_fields(tts._opts)["temperature"] == 0.4


def _with_update(tts: sarvam_tts.TTS, **options: Any) -> sarvam_tts.TTS:
    tts.update_options(**options)
    return tts


def test_v3_defaults_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        _make_tts(model="bulbul:v3")

    assert _warnings(caplog) == []


def test_v4_flash_applies_its_args_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        tts = _make_tts(model="bulbul:v4-flash", pitch=0.3, loudness=1.5, enable_preprocessing=True)

    assert _warnings(caplog) == []
    fields = sarvam_tts._model_extra_fields(tts._opts)
    assert (fields["pitch"], fields["loudness"], fields["enable_preprocessing"]) == (
        0.3,
        1.5,
        True,
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: _make_tts(enable_cached_responses=True),
        lambda: _make_tts().update_options(enable_cached_responses=True),
    ],
    ids=["init", "update_options"],
)
def test_enable_cached_responses_warns_instead_of_raising(
    call: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger=sarvam_tts.logger.name):
        call()

    assert any("`enable_cached_responses`" in w for w in _warnings(caplog))


@pytest.mark.parametrize(
    "call",
    [
        lambda: _make_tts(pich=0.2),
        lambda: _make_tts().update_options(pich=0.2),
    ],
    ids=["init", "update_options"],
)
def test_unknown_argument_still_raises(call: Any) -> None:
    with pytest.raises(TypeError, match="pich"):
        call()


_CODEC_MIMES = {
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
    # pcm and the telephony codecs are pushed as linear PCM
    "linear16": "audio/pcm",
    "mulaw": "audio/pcm",
    "alaw": "audio/pcm",
}


def test_codec_to_mime_type_covers_every_allowed_codec() -> None:
    assert set(_CODEC_MIMES) == ALLOWED_OUTPUT_AUDIO_CODECS
    assert {codec: _codec_to_mime_type(codec) for codec in _CODEC_MIMES} == _CODEC_MIMES
    with pytest.raises(ValueError, match="Unsupported output_audio_codec"):
        _codec_to_mime_type("unknown_codec")


@pytest.mark.parametrize("codec", ["linear16", "mulaw", "alaw"])
def test_pcm_and_telephony_codecs_are_accepted(codec: str) -> None:
    assert _make_tts(output_audio_codec=codec)._opts.output_audio_codec == codec
    t = _make_tts()
    t.update_options(output_audio_codec=codec)
    assert t._opts.output_audio_codec == codec


def test_unknown_codec_is_rejected() -> None:
    with pytest.raises(ValueError, match="output_audio_codec must be one of"):
        _make_tts(output_audio_codec="ogg")
    with pytest.raises(ValueError, match="output_audio_codec must be one of"):
        _make_tts().update_options(output_audio_codec="ogg")


def _text(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload), extra=None)


_AUDIO = _text({"type": "audio", "data": {"audio": "AAAA"}})
_FINAL = _text({"type": "event", "data": {"event_type": "final"}})


class _FakeWS:
    """Replays queued frames, then blocks like an idle socket until closed."""

    def __init__(
        self,
        recv: list[SimpleNamespace] | None = None,
        *,
        send_error: BaseException | None = None,
        recv_error: BaseException | None = None,
    ) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self._recv = list(recv or [])
        self._send_error = send_error
        self._recv_error = recv_error
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(json.loads(data))

    async def receive(self, timeout: float | None = None) -> SimpleNamespace:
        if self._recv_error is not None:
            raise self._recv_error
        if self._recv:
            return self._recv.pop(0)
        await self._closed.wait()
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _FakePool:
    """Hands out one socket and records evictions."""

    last_acquire_time = 0.0
    last_connection_reused = False

    def __init__(self, ws: _FakeWS | None = None, *, remove_error: bool = False) -> None:
        self.ws = ws
        self.removed: list[object] = []
        self._remove_error = remove_error

    @contextlib.asynccontextmanager
    async def connection(self, *, timeout: float) -> AsyncIterator[_FakeWS | None]:
        yield self.ws

    def remove(self, conn: object) -> None:
        if self._remove_error:
            raise RuntimeError("pool already shut down")
        self.removed.append(conn)


async def _run_on(ws: _FakeWS, **tts_kwargs: Any) -> sarvam_tts.SynthesizeStream:
    stream = _ws_stream(_make_tts(**tts_kwargs))
    stream._tts._pool = _FakePool(ws)  # type: ignore[assignment]

    async def no_words() -> AsyncIterator[SimpleNamespace]:
        return
        yield

    await stream._run_ws(no_words(), MagicMock(spec=tts.AudioEmitter))  # type: ignore[arg-type]
    return stream


async def test_ws_config_carries_codec_and_v3_fields() -> None:
    ws = _FakeWS([_AUDIO, _FINAL])
    stream = await _run_on(ws, model="bulbul:v3", output_audio_codec="opus")
    await stream._tts._stop_keepalive(ws)  # type: ignore[arg-type]

    config = ws.sent[0]
    assert config["type"] == "config"
    data = config["data"]
    assert data["output_audio_codec"] == "opus"
    assert {"output_audio_bitrate", "min_buffer_size", "max_chunk_length", "temperature"} <= set(
        data
    )
    assert "pitch" not in data and "loudness" not in data
    # the keepalive is paused while the socket is in use
    assert all(frame["type"] in {"config", "text", "flush"} for frame in ws.sent)


async def test_ws_config_v4_flash_sends_pitch_but_not_temperature() -> None:
    ws = _FakeWS([_AUDIO, _FINAL])
    stream = await _run_on(ws, model="bulbul:v4-flash", temperature=0.3)
    await stream._tts._stop_keepalive(ws)  # type: ignore[arg-type]

    data = ws.sent[0]["data"]
    assert {"pitch", "loudness"} <= set(data)
    assert "temperature" not in data


@pytest.mark.parametrize(
    ("error", "match"),
    [
        # a dead transport must raise so the pool evicts the socket instead of reusing it
        (ConnectionResetError("Cannot write to closing transport"), "transport closed before send"),
        (RuntimeError("WebSocket connection is closed"), "transport closed before send"),
        (ConnectionResetError("peer reset connection unexpectedly"), "Send task failed"),
        # recv_task waits on the send side, so a send failure must still release it
        (ValueError("simulated config error"), "Send task failed"),
    ],
)
async def test_send_failure_raises_connection_error(error: BaseException, match: str) -> None:
    with pytest.raises(APIConnectionError, match=match):
        await _run_on(_FakeWS([_FINAL], send_error=error))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # a dead transport is evicted and retried; any other reset fails the session
        (
            aiohttp.ClientConnectionResetError("Cannot write to closing transport"),
            APIConnectionError,
        ),
        (ConnectionResetError("peer reset connection"), APIStatusError),
    ],
)
async def test_recv_connection_reset(error: BaseException, expected: type[Exception]) -> None:
    with pytest.raises(expected):
        await _run_on(_FakeWS(recv_error=error))


@pytest.mark.parametrize(
    ("send_error", "resumed"), [(None, True), (ValueError("bad config"), False)]
)
async def test_keepalive_resumes_only_after_a_clean_session(
    send_error: BaseException | None, resumed: bool
) -> None:
    ws = _FakeWS([_AUDIO, _FINAL], send_error=send_error)
    t = _make_tts()
    t._start_keepalive(ws)  # type: ignore[arg-type]
    stream = _ws_stream(t)
    t._pool = _FakePool(ws)  # type: ignore[assignment]

    async def no_words() -> AsyncIterator[SimpleNamespace]:
        return
        yield

    with contextlib.suppress(APIConnectionError):
        await stream._run_ws(no_words(), MagicMock(spec=tts.AudioEmitter))  # type: ignore[arg-type]

    # a failed socket is discarded by the pool, so a keepalive on it would be orphaned
    assert (id(ws) in t._ws_keepalive_tasks) is resumed
    await t._stop_keepalive(ws)  # type: ignore[arg-type]


async def test_aclose_closes_segments_channel() -> None:
    stream = _make_tts().stream()
    stream.end_input()
    await stream.aclose()
    assert stream._segments_ch.closed


@pytest.mark.parametrize(
    ("message", "keep_reading"),
    [
        (json.dumps({"type": "event", "data": {"event_type": "final"}}), False),
        ("not json{", True),
        (json.dumps({"type": "ping", "data": {}}), True),
    ],
    ids=["final", "invalid-json", "unknown-type"],
)
async def test_websocket_message_ends_only_on_final(message: str, keep_reading: bool) -> None:
    emitter = MagicMock(spec=tts.AudioEmitter)
    assert await _ws_stream(_make_tts())._handle_websocket_message(message, emitter) is keep_reading
    assert emitter.end_input.called is not keep_reading


@pytest.mark.parametrize(
    ("codec", "data", "samples"),
    [
        # ITU-T G.711 reference values
        ("mulaw", b"\xff\x80\x00\x7f", [0, 32124, -32124, 0]),
        ("alaw", b"\xd5\xaa\x2a", [8, 32256, -32256]),
        ("mulaw", b"", []),
        ("alaw", b"", []),
    ],
)
def test_decode_telephony(codec: str, data: bytes, samples: list[int]) -> None:
    out = _decode_telephony(codec, data)
    assert [
        int.from_bytes(out[i : i + 2], "little", signed=True) for i in range(0, len(out), 2)
    ] == (samples)


def test_decode_telephony_rejects_other_codecs() -> None:
    with pytest.raises(ValueError, match="does not support codec"):
        _decode_telephony("mp3", b"\x00")


@pytest.mark.parametrize("codec", ["mulaw", "alaw", "mp3", "linear16"])
async def test_audio_message_decodes_only_telephony_codecs(codec: str) -> None:
    raw = b"\xd5\xaa\x2a\x00"
    emitter = MagicMock(spec=tts.AudioEmitter)
    stream = _ws_stream(_make_tts(output_audio_codec=codec))
    resp = {"type": "audio", "data": {"audio": base64.b64encode(raw).decode()}}

    assert await stream._handle_audio_message(resp, emitter) is True
    expected = _decode_telephony(codec, raw) if codec in {"mulaw", "alaw"} else raw
    emitter.push.assert_called_once_with(expected)


async def test_empty_audio_is_skipped() -> None:
    emitter = MagicMock(spec=tts.AudioEmitter)
    stream = _ws_stream(_make_tts(output_audio_codec="mulaw"))
    assert await stream._handle_audio_message({"type": "audio", "data": {"audio": ""}}, emitter)
    emitter.push.assert_not_called()


@pytest.fixture
def keepalive_tts(monkeypatch: pytest.MonkeyPatch) -> sarvam_tts.TTS:
    monkeypatch.setattr(sarvam_tts, "_KEEPALIVE_INTERVAL", 1.0)
    return _make_tts()


@pytest.mark.virtual_time
async def test_keepalive_start_and_stop(keepalive_tts: sarvam_tts.TTS) -> None:
    ws = _FakeWS()
    keepalive_tts._start_keepalive(ws)  # type: ignore[arg-type]
    task = keepalive_tts._ws_keepalive_tasks[id(ws)]

    await keepalive_tts._stop_keepalive(ws)  # type: ignore[arg-type]
    await keepalive_tts._stop_keepalive(ws)  # type: ignore[arg-type]
    assert task.done() and id(ws) not in keepalive_tts._ws_keepalive_tasks


def test_keepalive_disabled_with_zero_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sarvam_tts, "_KEEPALIVE_INTERVAL", 0.0)
    t = _make_tts()
    t._start_keepalive(_FakeWS())  # type: ignore[arg-type]
    assert t._ws_keepalive_tasks == {}


@pytest.mark.virtual_time
async def test_keepalive_drains_then_pings_an_idle_socket(keepalive_tts: sarvam_tts.TTS) -> None:
    pool = _FakePool()
    keepalive_tts._pool = pool  # type: ignore[assignment]
    # draining lets aiohttp's heartbeat see PONGs; a ping only goes out once reads time out
    ws = _FakeWS([_text({}) for _ in range(3)])
    keepalive_tts._start_keepalive(ws)  # type: ignore[arg-type]

    await asyncio.sleep(0.5)
    assert ws.sent == []
    await asyncio.sleep(3)
    await keepalive_tts._stop_keepalive(ws)  # type: ignore[arg-type]

    assert len(ws.sent) >= 2 and all(frame == {"type": "ping"} for frame in ws.sent)
    assert pool.removed == []


@pytest.mark.virtual_time
@pytest.mark.parametrize(
    "msg_type", [aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR]
)
async def test_keepalive_evicts_a_socket_the_server_closed(
    keepalive_tts: sarvam_tts.TTS, msg_type: aiohttp.WSMsgType
) -> None:
    pool = _FakePool()
    keepalive_tts._pool = pool  # type: ignore[assignment]
    ws = _FakeWS([SimpleNamespace(type=msg_type, data=1006, extra=None)])
    keepalive_tts._start_keepalive(ws)  # type: ignore[arg-type]

    await keepalive_tts._ws_keepalive_tasks[id(ws)]
    assert pool.removed == [ws]


@pytest.mark.virtual_time
@pytest.mark.parametrize("remove_error", [False, True])
async def test_keepalive_evicts_and_exits_on_failed_ping(
    keepalive_tts: sarvam_tts.TTS, remove_error: bool
) -> None:
    pool = _FakePool(remove_error=remove_error)
    keepalive_tts._pool = pool  # type: ignore[assignment]
    ws = _FakeWS(send_error=ConnectionResetError("Cannot write to closing transport"))
    keepalive_tts._start_keepalive(ws)  # type: ignore[arg-type]

    task = keepalive_tts._ws_keepalive_tasks[id(ws)]
    await task
    assert task.exception() is None
    assert pool.removed == ([] if remove_error else [ws])


@pytest.mark.virtual_time
async def test_close_ws_stops_keepalive_and_closes_socket(keepalive_tts: sarvam_tts.TTS) -> None:
    ws = _FakeWS()
    keepalive_tts._start_keepalive(ws)  # type: ignore[arg-type]
    await keepalive_tts._close_ws(ws)  # type: ignore[arg-type]
    assert ws.closed and id(ws) not in keepalive_tts._ws_keepalive_tasks

    unwatched = _FakeWS()
    await keepalive_tts._close_ws(unwatched)  # type: ignore[arg-type]
    assert unwatched.closed


def test_keepalive_intervals_beat_sarvam_idle_timeout() -> None:
    # Sarvam drops a socket idle for 60 s; the protocol heartbeat is the first line of defense
    assert 0 < sarvam_tts._WS_HEARTBEAT_INTERVAL <= sarvam_tts._KEEPALIVE_INTERVAL < 60


async def test_connect_ws_passes_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sarvam_tts, "_WS_HEARTBEAT_INTERVAL", 5.0)
    captured: dict[str, Any] = {}

    class _Session:
        async def ws_connect(self, url: str, **kwargs: Any) -> _FakeWS:
            captured.update(kwargs)
            return _FakeWS()

    t = _make_tts()
    t._session = _Session()  # type: ignore[assignment]
    ws = await t._connect_ws(timeout=5.0)
    await t._stop_keepalive(ws)  # type: ignore[arg-type]

    assert captured["heartbeat"] == 5.0
