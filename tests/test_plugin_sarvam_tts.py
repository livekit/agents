from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock

import aiohttp
import pytest

from livekit.agents import APIStatusError
from livekit.agents.tts import AudioEmitter
from livekit.agents.types import APIConnectOptions
from livekit.plugins.sarvam import models as sarvam_models, tts as sarvam_tts

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
    """A stream adopts the model-coupled options its socket was opened with, all together.

    update_options validates model, speaker and language as one set, so taking only some
    of them could pair a model with a speaker or language it rejects. Sample rate and
    codec stay the stream's own: its output emitter is already initialized from them.
    """
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
