from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.plugin("deepgram")


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_str(self, data: str) -> None:
        self.sent.append(data)


def _make_flux_stream(*, ws=None, **opts_kwargs):
    # exercise SpeechStreamv2's update logic without starting its connection task
    from livekit.agents.types import NOT_GIVEN
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2, STTOptions

    opts = STTOptions(
        model="flux-general-en",
        sample_rate=16000,
        keyterm=[],
        endpoint_url="wss://api.deepgram.com/v2/listen",
        eot_threshold=opts_kwargs.get("eot_threshold", NOT_GIVEN),
        eager_eot_threshold=opts_kwargs.get("eager_eot_threshold", NOT_GIVEN),
        eot_timeout_ms=opts_kwargs.get("eot_timeout_ms", NOT_GIVEN),
        language_hint=opts_kwargs.get("language_hint", []),
        numerals=opts_kwargs.get("numerals", False),
        profanity_filter=opts_kwargs.get("profanity_filter", False),
        redact=opts_kwargs.get("redact", NOT_GIVEN),
    )
    opts.keyterm = opts_kwargs.get("keyterm", [])
    stream = SimpleNamespace(
        _opts=opts,
        _reconnect_event=asyncio.Event(),
        _reconfigure_atask=None,
        _ws=ws,
    )
    stream._send_configure = SpeechStreamv2._send_configure.__get__(stream)
    return stream


async def test_update_options_uses_stored_language_for_model_validation():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key", language="fr")
    stt.update_options(model="nova-2-meeting")
    assert stt._opts.model == "nova-2-general"


async def test_update_options_explicit_language_overrides_stored():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key", language="fr")
    stt.update_options(model="nova-2-meeting", language="en-US")
    assert stt._opts.model == "nova-2-meeting"


async def test_update_options_no_language_set_keeps_en_only_model():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key")
    stt.update_options(model="nova-2-meeting")
    assert stt._opts.model == "nova-2-meeting"


async def test_flux_live_fields_reconfigure_without_reconnect():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, eot_threshold=0.7)
    SpeechStreamv2.update_options(
        stream, eot_threshold=0.85, keyterm=["LiveKit"], language_hint=["en"]
    )

    # a Configure send is scheduled in-band, no reconnect
    assert not stream._reconnect_event.is_set()
    assert stream._reconfigure_atask is not None
    await stream._reconfigure_atask

    assert stream._opts.eot_threshold == 0.85
    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0]) == {
        "type": "Configure",
        "thresholds": {"eot_threshold": 0.85},
        "keyterms": ["LiveKit"],
        "language_hints": ["en"],
    }


async def test_flux_reconnect_fields_skip_inband_configure():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws)
    SpeechStreamv2.update_options(stream, model="flux-general-multi", eot_threshold=0.8)

    # model can't be tuned in-band; a reconnect carries every option instead
    assert stream._reconnect_event.is_set()
    assert stream._reconfigure_atask is None
    assert ws.sent == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("numerals", True), ("profanity_filter", True), ("redact", "numbers")],
)
async def test_flux_connection_time_fields_trigger_reconnect_not_configure(field, value):
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws)
    SpeechStreamv2.update_options(stream, **{field: value})

    # Flux can't toggle these via Configure, only at connection time
    assert getattr(stream._opts, field) == value
    assert stream._reconnect_event.is_set()
    assert stream._reconfigure_atask is None
    assert ws.sent == []


async def test_flux_connection_config_includes_formatting_fields():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    config = SpeechStreamv2._live_config(
        _make_flux_stream(numerals=True, profanity_filter=True, redact="aggressive_numbers")
    )
    assert config["numerals"] is True
    assert config["profanity_filter"] is True
    assert config["redact"] == "aggressive_numbers"

    default_config = SpeechStreamv2._live_config(_make_flux_stream())
    assert not {"numerals", "profanity_filter", "redact"} & default_config.keys()


async def test_flux_configure_sends_only_changed_fields():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    # stream already has a threshold and keyterms configured
    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, eot_threshold=0.7, keyterm=["existing"])

    # only keyterms change: the Configure delta must omit the unchanged threshold
    SpeechStreamv2.update_options(stream, keyterm=["LiveKit", "Deepgram"])
    await stream._reconfigure_atask

    assert json.loads(ws.sent[0]) == {"type": "Configure", "keyterms": ["LiveKit", "Deepgram"]}


async def test_flux_configure_thresholds_only_delta():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, keyterm=["existing"])

    SpeechStreamv2.update_options(stream, eot_timeout_ms=5000)
    await stream._reconfigure_atask

    assert json.loads(ws.sent[0]) == {"type": "Configure", "thresholds": {"eot_timeout_ms": 5000}}


async def test_flux_configure_sends_are_ordered():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, eot_threshold=0.7)

    # rapid successive updates chain off each other and reach the server in order
    SpeechStreamv2.update_options(stream, eot_threshold=0.8)
    SpeechStreamv2.update_options(stream, eot_threshold=0.9)
    await stream._reconfigure_atask

    assert [json.loads(m)["thresholds"]["eot_threshold"] for m in ws.sent] == [0.8, 0.9]


async def test_flux_configure_noop_when_disconnected():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    stream = _make_flux_stream(ws=None)
    # no active connection: the next reconnect carries the latest options instead
    SpeechStreamv2.update_options(stream, keyterm=["LiveKit"])
    await stream._reconfigure_atask

    assert stream._opts.keyterm == ["LiveKit"]


class _LiveWS:
    """Records what the send loop writes. receive() parks so recv_task stays alive."""

    def __init__(self) -> None:
        # a single ordered log, so "Finalize came after the audio" is assertable
        self.wire: list[str] = []
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        self.wire.append(json.loads(data)["type"])

    async def send_bytes(self, data: bytes) -> None:
        self.wire.append("audio")

    async def receive(self):
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self._closed.set()

    def sent(self) -> list[str]:
        """The wire log without the periodic KeepAlive noise."""
        return [msg for msg in self.wire if msg != "KeepAlive"]


def _live_stream(ws: _LiveWS):
    """A real SpeechStream running its real _run loop against a fake socket."""
    import dataclasses
    from typing import Any, cast

    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.deepgram.stt import STT, SpeechStream

    instance = STT(api_key="test-key", language="en-US", sample_rate=16000)
    stream = SpeechStream(
        stt=instance,
        opts=dataclasses.replace(instance._opts, sample_rate=16000),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        http_session=cast(Any, SimpleNamespace(closed=False)),
        base_url="wss://api.deepgram.com/v1/listen",
    )

    async def _fake_connect() -> Any:
        return ws

    # patched before the _run task gets its first tick, so no real socket is opened
    stream._connect_ws = _fake_connect
    return stream


def _frame(ms: int, sample_rate: int = 16000):
    from livekit import rtc

    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream to send"
        await asyncio.sleep(0.01)


async def test_flush_finalizes_the_turn_when_no_audio_is_left_to_send():
    # 50ms is exactly one repack chunk, so AudioByteStream.flush() returns no frames.
    # Finalize must still go out, otherwise the turn waits on Deepgram's own endpointing
    # and has_ended leaks into the next utterance.
    ws = _LiveWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: ws.sent() == ["audio"])

        stream.flush()
        await _wait_until(lambda: ws.sent() == ["audio", "Finalize"])
    finally:
        await stream.aclose()


async def test_flush_finalizes_after_the_buffered_audio():
    # a partial chunk is still pending: it has to reach the server before Finalize,
    # otherwise the tail of the turn is transcribed against the next one
    ws = _LiveWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(30))
        stream.flush()
        await _wait_until(lambda: ws.sent() == ["audio", "Finalize"])
    finally:
        await stream.aclose()


class _RecordingSession:
    """Stands in for the aiohttp session so the connect kwargs are assertable."""

    def __init__(self) -> None:
        self.closed = False
        self.kwargs: dict = {}
        self._ws = _LiveWS()

    async def ws_connect(self, url: str, **kwargs):
        self.kwargs = kwargs
        return self._ws


class _DeadSocket:
    """A socket that has silently gone away: writes fail, the read side never notices.

    This is the half-open case (no FIN, no RST). recv_task can only park, so the
    keepalive write is the one place the drop can surface.
    """

    def __init__(self) -> None:
        self.closed = False
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        import aiohttp

        raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    async def send_bytes(self, data: bytes) -> None:
        import aiohttp

        raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    async def receive(self):
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self._closed.set()


def _v1_stream(*, http_session=None, connect=None):
    """A real v1 SpeechStream running its real _run loop."""
    import dataclasses
    from typing import Any, cast

    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.deepgram.stt import STT, SpeechStream

    instance = STT(api_key="test-key", language="en-US", sample_rate=16000)
    stream = SpeechStream(
        stt=instance,
        opts=dataclasses.replace(instance._opts, sample_rate=16000),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        http_session=cast(Any, http_session or SimpleNamespace(closed=False)),
        base_url="wss://api.deepgram.com/v1/listen",
    )
    if connect is not None:
        stream._connect_ws = connect
    return stream


async def test_v1_socket_is_opened_with_a_heartbeat():
    # aiohttp defaults heartbeat and receive_timeout to None, so without this the
    # read side of a half-open socket parks forever and the reconnect loop in _run,
    # which only runs when something raises, never gets a turn. stt_v2 already does
    # this; v1 is the Nova-3 path and was the only Deepgram stream left unbounded.
    session = _RecordingSession()
    stream = _v1_stream(http_session=session)
    try:
        await _wait_until(lambda: "heartbeat" in session.kwargs)
        assert session.kwargs["heartbeat"] == 30.0
    finally:
        await stream.aclose()


async def test_keepalive_write_drop_reconnects_instead_of_stalling():
    # the keepalive used to swallow every exception and return, which left send_task
    # parked on the input channel and recv_task parked on receive(): _run stayed
    # alive on a socket that was gone, and the session went quiet with no error.
    sockets: list[_DeadSocket] = []

    async def _connect():
        ws = _DeadSocket()
        sockets.append(ws)
        return ws

    stream = _v1_stream(connect=_connect)
    try:
        await _wait_until(lambda: len(sockets) > 1)
    finally:
        await stream.aclose()


class _HeartbeatTimeoutSocket:
    """A socket in the state aiohttp leaves it in when a ping goes unanswered.

    The heartbeat closes the connection itself, so this arrives as WSMsgType.ERROR
    rather than as a close frame, and the reason lives only on ws.exception().
    """

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_str(self, data: str) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    def exception(self):
        import aiohttp

        return aiohttp.ServerTimeoutError("No PONG received after 15.0s")

    async def receive(self):
        import aiohttp

        self.receives += 1
        # yield, so that a recv loop which steps over this rather than ending
        # fails the test instead of starving the event loop and hanging it
        await asyncio.sleep(0)
        return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), None)

    async def close(self) -> None:
        self.closed = True


async def test_heartbeat_timeout_reconnects_without_spinning():
    # the ERROR has to end the recv loop. logging it as an unexpected type and
    # continuing only works because aiohttp happens to report CLOSED next, and it
    # throws away the one value that says why the socket went away.
    sockets: list[_HeartbeatTimeoutSocket] = []

    async def _connect():
        ws = _HeartbeatTimeoutSocket()
        sockets.append(ws)
        return ws

    stream = _v1_stream(connect=_connect)
    try:
        await _wait_until(lambda: len(sockets) > 1)
        assert sockets[0].receives == 1
    finally:
        await stream.aclose()


def _live_flux_stream(ws: _LiveWS, *, speaking: bool):
    """A real SpeechStreamv2 running its real _run loop against a fake socket."""
    import dataclasses
    from typing import Any, cast

    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2, STTv2

    instance = STTv2(api_key="test-key", sample_rate=16000)
    stream = SpeechStreamv2(
        stt=instance,
        opts=dataclasses.replace(instance._opts, sample_rate=16000),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        http_session=cast(Any, SimpleNamespace(closed=False)),
        base_url="wss://api.deepgram.com/v2/listen",
    )
    # whether Flux has an open turn; normally set by StartOfTurn on the recv path,
    # which never runs here because _LiveWS.receive() parks
    stream._speaking = speaking

    async def _fake_connect() -> Any:
        return ws

    # patched before the _run task gets its first tick, so no real socket is opened
    stream._connect_ws = _fake_connect
    return stream


async def test_flux_flush_forces_the_end_of_turn():
    # flush() means "end the current segment", and ForceEndTurn is how Flux says that,
    # exactly as Finalize is on v1. without it a flushed turn sits until eot_threshold
    # is met or eot_timeout_ms elapses.
    ws = _LiveWS()
    stream = _live_flux_stream(ws, speaking=True)
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: ws.sent() == ["audio"])

        stream.flush()
        await _wait_until(lambda: ws.sent() == ["audio", "ForceEndTurn"])
    finally:
        await stream.aclose()


async def test_flux_flush_is_quiet_when_no_turn_is_open():
    # end_input() flushes on every close, and forcing a turn that never started only
    # earns a FORCE_END_TURN_NO_ACTIVE_TURN warning back
    ws = _LiveWS()
    stream = _live_flux_stream(ws, speaking=False)
    try:
        stream.push_frame(_frame(50))
        stream.flush()
        # the second chunk pins the ordering: a ForceEndTurn would land between them
        stream.push_frame(_frame(50))
        await _wait_until(lambda: ws.sent() == ["audio", "audio"])
    finally:
        await stream.aclose()


class _EventCh:
    def __init__(self) -> None:
        self.events: list = []

    def send_nowait(self, event) -> None:
        self.events.append(event)


def _make_flux_event_stream():
    """Drives _process_stream_event without a connection or the _run loop."""
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2, STTOptions

    stream = SimpleNamespace(
        _opts=STTOptions(
            model="flux-general-en",
            sample_rate=16000,
            keyterm=[],
            endpoint_url="wss://api.deepgram.com/v2/listen",
        ),
        _event_ch=_EventCh(),
        _request_id="",
        _speaking=False,
        start_time_offset=0.0,
    )
    stream._send_transcript_event = SpeechStreamv2._send_transcript_event.__get__(stream)
    stream._process_stream_event = SpeechStreamv2._process_stream_event.__get__(stream)
    return stream


async def test_flux_manual_end_of_turn_finalizes_the_turn():
    from livekit.agents import stt

    stream = _make_flux_event_stream()

    stream._process_stream_event({"type": "TurnInfo", "event": "StartOfTurn", "words": []})
    # trigger=manual is the reply to ForceEndTurn; it has to finalize exactly like
    # a model-detected EndOfTurn, otherwise a forced turn never reaches the LLM
    stream._process_stream_event(
        {
            "type": "TurnInfo",
            "event": "EndOfTurn",
            "trigger": "manual",
            "transcript": "cancel my subscription",
            "words": [{"word": "cancel", "confidence": 0.9, "start": 0.0, "end": 0.5}],
        }
    )

    assert [e.type for e in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._event_ch.events[1].alternatives[0].text == "cancel my subscription"
    assert stream._speaking is False


async def test_flux_warning_is_surfaced_without_ending_the_stream(caplog):
    import logging

    stream = _make_flux_event_stream()

    # forcing a turn that never started is recoverable, unlike an Error
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.deepgram"):
        stream._process_stream_event(
            {
                "type": "Warning",
                "code": "FORCE_END_TURN_NO_ACTIVE_TURN",
                "description": "no active turn to end",
            }
        )

    assert "deepgram sent a warning" in caplog.text
    assert stream._event_ch.events == []
