from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

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
        model=opts_kwargs.get("model", "flux-general-en"),
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
        _language_hint_warned=False,
    )
    stream._send_configure = SpeechStreamv2._send_configure.__get__(stream)
    stream._sends_language_hint = SpeechStreamv2._sends_language_hint.__get__(stream)
    stream._configure_language_hint = SpeechStreamv2._configure_language_hint.__get__(stream)
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
    # the multilingual model is the one Deepgram takes a hint for
    stream = _make_flux_stream(ws=ws, model="flux-general-multi", eot_threshold=0.7)
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


def test_flux_turn_resumed_reaches_the_pipeline_as_an_interim():
    # the pipeline drops a preemptive reply when an interim moves past the preflight
    # transcript, so TurnResumed has to arrive as one, carrying what the user went on to say
    from livekit.agents.stt import SpeechEvent, SpeechEventType
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    events: list[SpeechEvent] = []
    stream = SimpleNamespace(
        _opts=SimpleNamespace(language="en"),
        _request_id="",
        _speaking=False,
        start_time_offset=0.0,
        _event_ch=SimpleNamespace(send_nowait=events.append),
    )
    stream._send_transcript_event = SpeechStreamv2._send_transcript_event.__get__(stream)

    # the payloads of https://developers.deepgram.com/docs/flux/state, trimmed
    for event, transcript in [
        ("StartOfTurn", "I need to"),
        ("EagerEndOfTurn", "I need to cancel my subscription."),
        ("TurnResumed", "I need to cancel my subscription please"),
    ]:
        words = [{"word": w, "confidence": 0.9} for w in transcript.split()]
        SpeechStreamv2._process_stream_event(
            stream,
            {"type": "TurnInfo", "event": event, "transcript": transcript, "words": words},
        )

    assert [(e.type, [a.text for a in e.alternatives]) for e in events] == [
        (SpeechEventType.START_OF_SPEECH, []),
        (SpeechEventType.INTERIM_TRANSCRIPT, ["I need to"]),
        (SpeechEventType.PREFLIGHT_TRANSCRIPT, ["I need to cancel my subscription."]),
        (SpeechEventType.INTERIM_TRANSCRIPT, ["I need to cancel my subscription please"]),
    ]


class _LiveWS:
    """Records what the send loop writes. receive() parks so recv_task stays alive."""

    def __init__(self) -> None:
        # a single ordered log, so "Finalize came after the audio" is assertable
        self.wire: list[str] = []
        # the raw sends, for the tests that assert a payload
        self.frames: list[str] = []
        # the connect path reads the response headers for its debug log, and the
        # v2 Configure path checks `closed` before writing
        self._response = SimpleNamespace(headers={})
        self.closed = False
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        self.frames.append(data)
        self.wire.append(json.loads(data)["type"])

    async def send_bytes(self, data: bytes) -> None:
        self.wire.append("audio")

    async def receive(self):
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self.closed = True
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


@pytest.mark.parametrize("audio_chunk_duration_ms", [0, -1, 1.5, True])
def test_flux_rejects_invalid_chunk_size(audio_chunk_duration_ms):
    from livekit.plugins.deepgram import STTv2

    with pytest.raises(ValueError, match="audio_chunk_duration_ms"):
        STTv2(api_key="test-key", audio_chunk_duration_ms=audio_chunk_duration_ms)


def test_flux_rejects_chunk_size_smaller_than_one_sample():
    from livekit.plugins.deepgram import STTv2

    with pytest.raises(ValueError, match="audio_chunk_duration_ms"):
        STTv2(api_key="test-key", sample_rate=100, audio_chunk_duration_ms=1)


@pytest.mark.parametrize("audio_chunk_duration_ms", [None, 20, 80])
@pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
@pytest.mark.parametrize("duration_ms", [400, 410])
async def test_flux_chunk_size_preserves_audio_and_flushes_tail(
    audio_chunk_duration_ms, sample_rate, duration_ms
):
    from typing import Any, cast

    from livekit import rtc
    from livekit.plugins.deepgram import STTv2

    class AudioWS(_LiveWS):
        def __init__(self) -> None:
            super().__init__()
            self.audio: list[bytes] = []

        async def send_bytes(self, data: bytes) -> None:
            self.audio.append(data)
            await super().send_bytes(data)

    ws = AudioWS()
    # Omitting the argument exercises the public constructor's existing default.
    options = (
        {}
        if audio_chunk_duration_ms is None
        else {"audio_chunk_duration_ms": audio_chunk_duration_ms}
    )
    instance = STTv2(
        api_key="test-key",
        sample_rate=sample_rate,
        http_session=cast(Any, SimpleNamespace(closed=False)),
        **options,
    )
    stream = instance.stream()

    async def connect():
        return ws

    # Replace only the network connection; exercise the real stream and send loop.
    stream._connect_ws = connect
    try:
        assert "audio_chunk_duration_ms" not in stream._live_config()
        input_samples = sample_rate // 100  # 10 ms RTC frames
        input_bytes = input_samples * 2
        pcm = bytes(i % 256 for i in range(sample_rate * duration_ms // 1000 * 2))
        for offset in range(0, len(pcm), input_bytes):
            stream.push_frame(
                rtc.AudioFrame(
                    data=pcm[offset : offset + input_bytes],
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=input_samples,
                )
            )
        stream.end_input()
        await _wait_until(lambda: "CloseStream" in ws.wire)

        chunk_bytes = sample_rate * (audio_chunk_duration_ms or 50) // 1000 * 2
        full_chunks, remainder = divmod(len(pcm), chunk_bytes)
        expected_sizes = [chunk_bytes] * full_chunks
        if remainder:
            expected_sizes.append(remainder)
        assert [len(data) for data in ws.audio] == expected_sizes
        assert b"".join(ws.audio) == pcm
        assert ws.sent() == ["audio"] * len(expected_sizes) + ["CloseStream"]
    finally:
        await stream.aclose()
        await instance.aclose()


class _RecordingSession:
    """Stands in for the aiohttp session so the connect kwargs are assertable."""

    def __init__(self) -> None:
        self.closed = False
        self.kwargs: dict = {}
        self._ws = _LiveWS()

    async def ws_connect(self, url: str, **kwargs):
        self.kwargs = kwargs
        return self._ws


class _URLRecordingSession:
    """Hands out a fresh socket per connect and keeps every URL it was asked to open."""

    def __init__(self) -> None:
        self.closed = False
        self.urls: list[str] = []

    async def ws_connect(self, url: str, **kwargs):
        self.urls.append(url)
        ws = _LiveWS()
        ws._response = SimpleNamespace(headers={})  # read for the dg- response headers
        return ws


async def test_stt_update_options_reaches_open_streams_diarization_and_tags():
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.deepgram import STT

    session = _URLRecordingSession()
    instance = STT(api_key="test-key", language="en-US", http_session=session)
    stream = instance.stream()
    try:
        await _wait_until(lambda: len(session.urls) == 1)
        assert "diarize" not in parse_qs(urlparse(session.urls[0]).query)

        instance.update_options(enable_diarization=True, tags=["prod"])

        await _wait_until(lambda: len(session.urls) == 2)
        query = parse_qs(urlparse(session.urls[1]).query)
        assert query["diarize"] == ["true"]
        assert query["tag"] == ["prod"]
    finally:
        await stream.aclose()
        await instance.aclose()


async def test_update_options_keeps_diarization_capability_in_sync():
    from livekit.agents.stt import MultiSpeakerAdapter
    from livekit.plugins.deepgram import STT

    instance = STT(api_key="test-key")
    try:
        assert not instance.capabilities.diarization

        instance.update_options(enable_diarization=True)
        assert instance.capabilities.diarization
        MultiSpeakerAdapter(stt=instance)

        instance.update_options(enable_diarization=False)
        assert not instance.capabilities.diarization
    finally:
        await instance.aclose()


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


async def test_v1_socket_exposes_control_frames_to_monitor():
    # Control frames must reach the shared monitor to reset its silence deadline.
    session = _RecordingSession()
    stream = _v1_stream(http_session=session)
    try:
        await _wait_until(lambda: "autoping" in session.kwargs)
        assert session.kwargs["autoping"] is False
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


class _ConnectRecordingSession:
    """Records every connect URL and hands out a fresh socket per connection.

    The connect URL is where both Deepgram plugins carry their options, and a settings
    change reconnects, so tests read the URLs in order.
    """

    def __init__(self) -> None:
        self.closed = False
        self.urls: list[str] = []
        self.sockets: list[_LiveWS] = []

    async def ws_connect(self, url: str, **kwargs):
        self.urls.append(url)
        ws = _LiveWS()
        self.sockets.append(ws)
        return ws

    def last_params(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.urls[-1]).query)


def _v1_stt_and_stream(**kwargs):
    """A real STT and the stream it hands out, on a socket that never dials out."""
    from typing import Any, cast

    from livekit.plugins.deepgram import STT

    session = _ConnectRecordingSession()
    stt = STT(api_key="test-key", http_session=cast(Any, session), **kwargs)
    return stt, stt.stream(), session


def _v2_stt_and_stream(*, model: str, **kwargs):
    """A real STTv2 and the stream it hands out, on a socket that never dials out."""
    from typing import Any, cast

    from livekit.plugins.deepgram import STTv2

    session = _ConnectRecordingSession()
    stt = STTv2(api_key="test-key", model=model, http_session=cast(Any, session), **kwargs)
    return stt, stt.stream(), session


_PRERECORDED_RESPONSE = {
    "metadata": {"request_id": "batch-request"},
    "results": {
        "channels": [
            {
                "detected_language": "en",
                "alternatives": [
                    {
                        "transcript": "hello",
                        "confidence": 0.99,
                        "words": [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.99}],
                    }
                ],
            }
        ]
    },
}


class _BatchResponse:
    """The prerecorded response; only `json()` is read off it."""

    async def __aenter__(self) -> _BatchResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self) -> dict:
        return _PRERECORDED_RESPONSE


class _BatchSession:
    """Records every batch request URL. `_ConnectRecordingSession`'s other half."""

    def __init__(self) -> None:
        self.closed = False
        self.urls: list[str] = []

    def post(self, *, url: str, **_: object) -> _BatchResponse:
        self.urls.append(url)
        return _BatchResponse()

    def last_params(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.urls[-1]).query)


def _v1_stt_and_http_session(**kwargs):
    """A real STT and the session its batch requests post through, which never dials out."""
    from typing import Any, cast

    from livekit.plugins.deepgram import STT

    session = _BatchSession()
    stt = STT(api_key="test-key", http_session=cast(Any, session), **kwargs)
    return stt, session


async def test_flux_keeps_the_language_hint_its_model_cannot_send(caplog):
    # Deepgram answers a hint on any model but flux-general-multi with a 400, so it stays
    # off the wire — and stays on the options, so a later model switch can send it.
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    with caplog.at_level(logging.WARNING):
        stt, stream, session = _v2_stt_and_stream(model="flux-general-en", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert "language_hint" not in SpeechStreamv2._live_config(stream)
        assert "language_hint" not in session.last_params()
        assert stt._opts.language_hint == ["en"]
        assert stream._opts.language_hint == ["en"]
        assert "not sending `language_hint`" in caplog.text
        assert "will be ignored" not in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_flux_unsent_language_hint_never_reaches_a_configure_frame(caplog):
    stt, stream, session = _v2_stt_and_stream(model="flux-general-en")
    await _wait_until(lambda: stream._ws is not None)
    try:
        with caplog.at_level(logging.WARNING):
            stt.update_options(language_hint=["en"])
        await asyncio.sleep(0)

        # a hint the model cannot take is not worth a Configure either, and the
        # suppressed send leaves the caller's copy alone
        assert stream._reconfigure_atask is None
        assert session.sockets[-1].frames == []
        assert stt._opts.language_hint == ["en"]
        assert stream._opts.language_hint == ["en"]
        assert "not sending `language_hint`" in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_flux_multilingual_model_still_sends_the_language_hint(caplog):
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v2_stt_and_stream(model="flux-general-multi", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]

        stt.update_options(language_hint=["en", "es"])
        await stream._reconfigure_atask

        assert json.loads(session.sockets[-1].frames[0]) == {
            "type": "Configure",
            "language_hints": ["en", "es"],
        }
        assert "not sending `language_hint`" not in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_flux_clears_the_language_hint_with_an_explicit_empty_list(caplog):
    # Flux keeps omitted fields, so a clear has to reach the connection as an empty
    # array — the one value Deepgram reads as "clear all" — or the language bias from
    # the connect-time hint survives until the socket is rebuilt.
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v2_stt_and_stream(model="flux-general-multi", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]

        stt.update_options(language_hint=[])
        assert stream._reconfigure_atask is not None
        await stream._reconfigure_atask

        assert len(session.sockets[-1].frames) == 1
        assert json.loads(session.sockets[-1].frames[0]) == {
            "type": "Configure",
            "language_hints": [],
        }
        # the caller asked for exactly this, so there is nothing to warn about
        assert "not sending `language_hint`" not in caplog.text
        assert stt._opts.language_hint == []
        assert stream._opts.language_hint == []
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_flux_clear_stays_off_a_model_that_takes_no_hint(caplog):
    # hints never reach the wire here, so a clear has nothing to clear
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v2_stt_and_stream(model="flux-general-en", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        with caplog.at_level(logging.WARNING):
            stt.update_options(language_hint=[])
            await asyncio.sleep(0)

        assert stream._reconfigure_atask is None
        assert session.sockets[-1].frames == []
        assert stt._opts.language_hint == []
        assert stream._opts.language_hint == []
        # only the connect-time warning, none from the clear
        assert caplog.text.count("not sending `language_hint`") == 1
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_flux_model_round_trip_sends_the_hint_again(caplog):
    # the model switch reconnects, so the hint must not ride along to a model that rejects
    # it — and must come back when the stream returns to the multilingual model
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v2_stt_and_stream(model="flux-general-multi", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]

        stt.update_options(model="flux-general-en")
        await _wait_until(lambda: len(session.urls) == 2)
        assert "language_hint" not in session.last_params()
        assert stt._opts.language_hint == ["en"]

        stt.update_options(model="flux-general-multi")
        await _wait_until(lambda: len(session.urls) == 3)
        assert session.last_params()["language_hint"] == ["en"]

        # a stream created after the round trip carries the hint too
        second = stt.stream()
        try:
            await _wait_until(lambda: len(session.urls) == 4)
            assert session.last_params()["language_hint"] == ["en"]
        finally:
            await second.aclose()
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_flux_warns_once_per_stream_across_reconnects(caplog):
    # the warning belongs to the stream, not to the connection: reconnecting rebuilds the
    # config, and the caller must not be told again
    stt, stream, session = _v2_stt_and_stream(model="flux-general-en", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        stt.update_options(model="flux-general-en")
        await _wait_until(lambda: len(session.urls) == 2)
        stt.update_options(model="flux-general-en")
        await _wait_until(lambda: len(session.urls) == 3)

        assert caplog.text.count("not sending `language_hint`") == 1
        assert stt._opts.language_hint == ["en"]
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_multi_language_sends_the_hint_as_repeated_params(caplog):
    # nova-3's multilingual mode is the only place Deepgram takes the hint
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3", language="multi", language_hint=["en", "es"]
    )
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en", "es"]
        assert stream._opts.language_hint == ["en", "es"]
        assert "not sending `language_hint`" not in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_nova3_general_multi_language_sends_the_hint(caplog):
    # the second member of the set the gateway accepts
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3-general", language="multi", language_hint=["en"]
    )
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]
        assert stream._opts.language_hint == ["en"]
        assert "not sending `language_hint`" not in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


@pytest.mark.parametrize("model", ["NOVA-3", "  nova-3-general  "])
async def test_v1_hint_model_match_ignores_case_and_whitespace(caplog, model):
    # the accepted set is matched the way the gateway matches it
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v1_stt_and_stream(model=model, language="multi", language_hint=["en"])
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]
        assert "not sending `language_hint`" not in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_nova2_multi_language_leaves_the_hint_off_the_wire(caplog):
    # a hint is not a nova-2 feature, even though the language alone looks right
    stt, stream, session = _v1_stt_and_stream(
        model="nova-2-general", language="multi", language_hint=["en"]
    )
    try:
        with caplog.at_level(logging.WARNING):
            await _wait_until(lambda: len(session.urls) == 1)

        assert "language_hint" not in session.last_params()
        assert stream._opts.language_hint == ["en"]
        assert "not sending `language_hint`" in caplog.text
        assert "model 'nova-2-general'" in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_nova3_medical_multi_language_leaves_the_hint_off_the_wire(caplog):
    # nova-3-medical is a nova-3 model but not one of the two ids the gateway's gate names,
    # so it takes no hint until someone widens that set deliberately
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3-medical", language="multi", language_hint=["en"]
    )
    try:
        with caplog.at_level(logging.WARNING):
            await _wait_until(lambda: len(session.urls) == 1)

        assert "language_hint" not in session.last_params()
        assert stream._opts.language_hint == ["en"]
        assert "not sending `language_hint`" in caplog.text
        assert "model 'nova-3-medical'" in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


@pytest.mark.parametrize("model", ["nova-3-multilingual", "flux-general-en", "nova-9-future"])
async def test_v1_multi_language_leaves_the_hint_off_the_wire_outside_the_nova3_set(caplog, model):
    # neither another nova-3 alias nor an unknown model is in the accepted set
    stt, stream, session = _v1_stt_and_stream(model=model, language="multi", language_hint=["en"])
    try:
        with caplog.at_level(logging.WARNING):
            await _wait_until(lambda: len(session.urls) == 1)

        assert "language_hint" not in session.last_params()
        assert stream._opts.language_hint == ["en"]
        assert "not sending `language_hint`" in caplog.text
        assert f"model '{model}'" in caplog.text
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_language_hint_is_not_sent_outside_multi_mode(caplog):
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3", language="en-US", language_hint=["en"]
    )
    try:
        with caplog.at_level(logging.WARNING):
            await _wait_until(lambda: len(session.urls) == 1)

        assert "language_hint" not in session.last_params()
        assert "not sending `language_hint`" in caplog.text
        assert "model 'nova-3'" in caplog.text
        assert "language is 'en-US'" in caplog.text
        # the suppressed send leaves the caller's hint on both copies
        assert stream._opts.language_hint == ["en"]
        assert stt._opts.language_hint == ["en"]
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_language_change_out_of_multi_stops_sending_the_hint(caplog):
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3", language="multi", language_hint=["en"]
    )
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]

        with caplog.at_level(logging.WARNING):
            stt.update_options(language="en-US")
            await _wait_until(lambda: len(session.urls) == 2)

        assert "language_hint" not in session.last_params()
        assert "not sending `language_hint`" in caplog.text
        assert stream._opts.language_hint == ["en"]
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_language_round_trip_sends_the_hint_again(caplog):
    # multi -> en -> multi on one stream: every language change reconnects, so the hint
    # leaves the wire and comes back with the language
    caplog.set_level(logging.WARNING)
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3", language="multi", language_hint=["en"]
    )
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        assert session.last_params()["language_hint"] == ["en"]

        stt.update_options(language="en-US")
        await _wait_until(lambda: len(session.urls) == 2)
        assert "language_hint" not in session.last_params()
        assert stream._opts.language_hint == ["en"]
        assert stt._opts.language_hint == ["en"]

        stt.update_options(language="multi")
        await _wait_until(lambda: len(session.urls) == 3)
        assert session.last_params()["language_hint"] == ["en"]
    finally:
        await stream.aclose()
        await stt.aclose()


async def test_v1_warns_once_per_stream_across_reconnects(caplog):
    stt, stream, session = _v1_stt_and_stream(
        model="nova-3", language="en-US", language_hint=["en"]
    )
    await _wait_until(lambda: len(session.urls) == 1)
    try:
        stt.update_options(language="en-GB")
        await _wait_until(lambda: len(session.urls) == 2)
        stt.update_options(language="en-AU")
        await _wait_until(lambda: len(session.urls) == 3)

        assert caplog.text.count("not sending `language_hint`") == 1
        assert stream._opts.language_hint == ["en"]
    finally:
        await stream.aclose()
        await stt.aclose()


@pytest.mark.parametrize("model", ["nova-3", "nova-3-general", "NOVA-3"])
async def test_v1_batch_request_carries_the_hint_the_gate_allows(caplog, model):
    # Deepgram's hint covers the prerecorded API too, one param per language
    caplog.set_level(logging.WARNING)
    stt, session = _v1_stt_and_http_session(
        model=model, language="multi", language_hint=["en", "es"]
    )
    await stt.recognize([_frame(160)])

    assert session.last_params()["language_hint"] == ["en", "es"]
    assert session.last_params()["language"] == ["multi"]
    assert stt._opts.language_hint == ["en", "es"]
    assert "not sending `language_hint`" not in caplog.text


@pytest.mark.parametrize(
    ("model", "language"),
    [("nova-2-general", "multi"), ("nova-3-medical", "multi"), ("nova-3", "en-US")],
)
async def test_v1_batch_request_drops_a_hint_the_gate_excludes_and_warns_once(
    caplog, model, language
):
    stt, session = _v1_stt_and_http_session(
        model=model, language=language, language_hint=["en", "es"]
    )
    with caplog.at_level(logging.WARNING):
        await stt.recognize([_frame(160)])
        await stt.recognize([_frame(160)])

    assert len(session.urls) == 2
    for url in session.urls:
        assert "language_hint" not in parse_qs(urlsplit(url).query)

    # the warning belongs to the instance: the second request does not repeat it
    assert caplog.text.count("not sending `language_hint`") == 1
    assert f"model '{model}'" in caplog.text
    # the suppressed send leaves the caller's hint on the options
    assert stt._opts.language_hint == ["en", "es"]


async def test_v1_batch_hint_survives_an_option_update(caplog):
    # update_options keeps the hint the same way, so a later allowed request sends it
    caplog.set_level(logging.WARNING)
    stt, session = _v1_stt_and_http_session(model="nova-3", language="en-US", language_hint=["en"])
    with caplog.at_level(logging.WARNING):
        await stt.recognize([_frame(160)])
        assert "language_hint" not in session.last_params()

        stt.update_options(language="multi")
        await stt.recognize([_frame(160)])

    assert session.last_params()["language_hint"] == ["en"]
    assert stt._opts.language_hint == ["en"]
