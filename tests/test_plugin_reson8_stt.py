"""
Hermetic tests for the Reson8 STT plugin.

Two layers, no credentials and no external network:

* pure checks on the option sections, the query string they produce and the
  turn state machine in ``SpeechStream._process_message``;
* a loopback ``aiohttp`` stand-in for the Reson8 API, so the real request,
  websocket-upgrade and error-mapping paths run end to end rather than against
  a mock.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import reson8
from livekit.plugins.reson8._utils import (
    ERROR_MESSAGE_HEADER,
    INTEGRATION_HEADER,
    PRERECORDED_PATH,
    TURNS_PATH,
    _confidence,
    _word_time,
    build_speech_data,
    build_url,
    normalize_languages,
    problem_message,
    status_error,
)
from livekit.plugins.reson8.stt import (
    AudioOptions,
    BiasingOptions,
    SpeechStream,
    STTOptions,
    TranscriptOptions,
    TurnOptions,
)

pytestmark = pytest.mark.unit

SpeechEventType = stt.SpeechEventType
NO_RETRY = APIConnectOptions(max_retry=0)
INTEGRATION = f"livekit-python:{reson8.__version__}"

HOSTED = "https://api.reson8.dev"
SELF_HOSTED = "https://stt.internal.example"
OTHER = "https://other.example"


def _frame(num_channels: int = 1, samples: int = 1600) -> rtc.AudioFrame:
    """0.1s of silent 16 kHz audio."""

    return rtc.AudioFrame(
        data=b"\x00\x00" * samples * num_channels,
        sample_rate=16000,
        num_channels=num_channels,
        samples_per_channel=samples,
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class FakeChan:
    """Stand-in for the stream's event channel that just records."""

    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


class MakeStream(Protocol):
    def __call__(self, **overrides: object) -> SpeechStream: ...


@pytest.fixture
def make_stream() -> MakeStream:
    """
    Build a ``SpeechStream`` without running its base ``__init__``.

    ``RecognizeStream.__init__`` spawns background tasks that open a websocket.
    ``_process_message`` is synchronous and touches only these attributes, so
    bypassing ``__init__`` keeps the state-machine tests network- and loop-free.
    """

    def _make(**overrides: object) -> SpeechStream:
        stream = SpeechStream.__new__(SpeechStream)
        stream._opts = STTOptions(**overrides)  # type: ignore[arg-type]
        stream._request_id = "req-test"
        stream._speaking = False
        stream._candidate = None
        stream._pending_reconnect = False
        stream._turn_settled = asyncio.Event()
        stream._start_time_offset = 0.0
        stream._speech_duration = 0.0
        stream._event_ch = FakeChan()  # type: ignore[assignment]
        return stream

    return _make


def emitted(stream: SpeechStream) -> list[stt.SpeechEvent]:
    return cast(FakeChan, stream._event_ch).events


@dataclass
class Recorded:
    """What the fake Reson8 server saw, and how to talk back."""

    base_url: str
    handshake_headers: dict[str, str] = field(default_factory=dict)
    post_headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)
    audio: list[bytes] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    connections: int = 0
    _ws: web.WebSocketResponse | None = None

    async def send(self, message: object) -> None:
        await asyncio.wait_for(self.connected.wait(), timeout=5)
        assert self._ws is not None
        await self._ws.send_str(json.dumps(message))

    async def wait_for_connections(self, count: int) -> None:
        async def _poll() -> None:
            while self.connections < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_poll(), timeout=5)

    async def wait_for_text(self, count: int = 1) -> None:
        async def _poll() -> None:
            while len(self.text) < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_poll(), timeout=5)


class StartServer(Protocol):
    async def __call__(
        self,
        *,
        post_status: int = ...,
        post_body: str = ...,
        ws_status: int | None = ...,
        ws_error_message: str | None = ...,
    ) -> Recorded: ...


@pytest.fixture
async def reson8_server() -> AsyncIterator[StartServer]:
    runners: list[web.AppRunner] = []

    async def _start(
        *,
        post_status: int = 200,
        post_body: str = '{"text": "hello"}',
        ws_status: int | None = None,
        ws_error_message: str | None = None,
    ) -> Recorded:
        rec = Recorded(base_url="")

        async def prerecorded(request: web.Request) -> web.StreamResponse:
            rec.post_headers = dict(request.headers)
            rec.query = dict(request.query)
            await request.read()
            return web.Response(status=post_status, text=post_body, content_type="application/json")

        async def accept_turns(request: web.Request) -> web.StreamResponse:
            rec.handshake_headers = dict(request.headers)
            rec.query = dict(request.query)

            ws = web.WebSocketResponse()
            await ws.prepare(request)
            rec._ws = ws
            rec.connections += 1
            rec.connected.set()

            async for msg in ws:
                if msg.type is aiohttp.WSMsgType.BINARY:
                    rec.audio.append(msg.data)
                elif msg.type is aiohttp.WSMsgType.TEXT:
                    rec.text.append(msg.data)

            return ws

        async def reject_turns(request: web.Request) -> web.StreamResponse:
            rec.handshake_headers = dict(request.headers)
            rec.query = dict(request.query)
            headers = {ERROR_MESSAGE_HEADER: ws_error_message} if ws_error_message else {}
            # a rejected upgrade carries no body, only this header
            return web.Response(status=ws_status or 500, headers=headers, text="")

        app = web.Application()
        app.router.add_post(PRERECORDED_PATH, prerecorded)
        app.router.add_get(TURNS_PATH, reject_turns if ws_status is not None else accept_turns)

        runner = web.AppRunner(app)
        await runner.setup()

        runners.append(runner)
        await web.TCPSite(runner, "127.0.0.1", 0).start()

        rec.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        return rec

    yield _start

    for runner in runners:
        await runner.cleanup()


@pytest.fixture
async def finalizing_server() -> AsyncIterator[Callable[[], Awaitable[str]]]:
    """
    A Reson8 stand-in that answers only after it sees ``flush_request``.

    That is what asking Reson8 to finalise looks like from the server side: the
    turn events arrive in response to the flush, not before it.
    """

    runners: list[web.AppRunner] = []

    async def _start() -> str:
        saw_flush = asyncio.Event()

        async def turns(request: web.Request) -> web.StreamResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)

            async def answer() -> None:
                await saw_flush.wait()
                await asyncio.sleep(0.05)  # finalising is not instant
                for message in (
                    {"type": "turn_start"},
                    {"type": "turn_end_candidate", "text": "hello world"},
                    {"type": "turn_end"},
                ):
                    await ws.send_str(json.dumps(message))

            task = asyncio.create_task(answer())
            try:
                async for msg in ws:
                    if msg.type is aiohttp.WSMsgType.TEXT and "flush_request" in msg.data:
                        saw_flush.set()
            finally:
                await utils.aio.cancel_and_wait(task)

            return ws

        app = web.Application()
        app.router.add_get(TURNS_PATH, turns)

        runner = web.AppRunner(app)
        await runner.setup()

        runners.append(runner)
        await web.TCPSite(runner, "127.0.0.1", 0).start()

        return f"http://127.0.0.1:{runner.addresses[0][1]}"

    yield _start

    for runner in runners:
        await runner.cleanup()


@pytest.fixture
async def client_session() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session


class EventLog:
    """Drains a live stream in the background and records what it emitted."""

    def __init__(self, stream: SpeechStream) -> None:
        self.events: list[stt.SpeechEvent] = []
        self._stream = stream
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for event in self._stream:
            self.events.append(event)

    async def wait_for(self, count: int, timeout: float = 5.0) -> list[stt.SpeechEvent]:
        async def _poll() -> None:
            while len(self.events) < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_poll(), timeout=timeout)
        return self.events

    async def assert_quiet(self, seconds: float = 0.5) -> None:
        before = len(self.events)
        await asyncio.sleep(seconds)
        assert len(self.events) == before, f"unexpected events: {self.events[before:]}"

    async def aclose(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


def _stt(base_url: str, session: aiohttp.ClientSession, **kwargs: object) -> reson8.STT:
    return reson8.STT(
        api_key="secret",
        base_url=base_url,
        http_session=session,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# construction: credentials and endpoint resolution
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESON8_API_KEY", raising=False)
    monkeypatch.delenv("RESON8_BASE_URL", raising=False)
    monkeypatch.delenv("RESON8_API_URL", raising=False)


def test_a_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="RESON8_API_KEY"):
        reson8.STT()


def test_the_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESON8_API_KEY", "from-env")
    assert reson8.STT()._api_key == "from-env"


def test_base_url_defaults_to_the_hosted_api() -> None:
    assert reson8.STT(api_key="k")._base_url == HOSTED


def test_base_url_argument_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESON8_BASE_URL", OTHER)
    assert reson8.STT(api_key="k", base_url=SELF_HOSTED)._base_url == SELF_HOSTED


def test_base_url_reads_its_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESON8_BASE_URL", SELF_HOSTED)
    assert reson8.STT(api_key="k")._base_url == SELF_HOSTED


def test_the_renamed_env_var_is_honoured_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A self-hosted deployment on the old name must not be sent to the public API.

    ``RESON8_API_URL`` was the name in the standalone 0.2.x releases of this
    plugin. Without the fallback the endpoint would silently become
    api.reson8.dev, sending audio and the API key to the wrong host.
    """

    monkeypatch.setenv("RESON8_API_URL", SELF_HOSTED)

    with caplog.at_level("WARNING"):
        instance = reson8.STT(api_key="k")

    assert instance._base_url == SELF_HOSTED
    assert "RESON8_API_URL is deprecated" in caplog.text


def test_the_current_env_var_wins_over_the_renamed_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESON8_BASE_URL", SELF_HOSTED)
    monkeypatch.setenv("RESON8_API_URL", OTHER)
    assert reson8.STT(api_key="k")._base_url == SELF_HOSTED


def test_a_trailing_slash_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESON8_BASE_URL", f"{SELF_HOSTED}/")
    assert reson8.STT(api_key="k")._base_url == SELF_HOSTED


def test_capabilities_track_the_transcript_options() -> None:
    plain = reson8.STT(api_key="k")
    assert plain.capabilities.streaming
    assert plain.capabilities.offline_recognize
    assert plain.capabilities.aligned_transcript is False

    # word timings are what make an aligned transcript possible
    aligned = reson8.STT(api_key="k", transcript=TranscriptOptions(words=True))
    assert aligned.capabilities.aligned_transcript == "word"


def test_model_and_provider() -> None:
    assert reson8.STT(api_key="k").provider == "Reson8"
    assert reson8.STT(api_key="k").model == "default"
    biasing = BiasingOptions(custom_model_id="m1")
    assert reson8.STT(api_key="k", biasing=biasing).model == "m1"


# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ([], None),
        ("nl", "nl"),
        ("NL", "nl"),
        ("nl,de", "nl,de"),
        (["nl", "de"], "nl,de"),
        (" nl , de ", "nl,de"),
        (["nl", "", "de"], "nl,de"),
    ],
)
def test_normalize_languages(value: str | list[str] | None, expected: str | None) -> None:
    assert normalize_languages(value) == expected


@pytest.mark.parametrize("value", ["xx", "nl,xx", ["nl", "xx"], "english"])
def test_an_unsupported_language_raises(value: str | list[str]) -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        reson8.STT(api_key="k", language=value)


def test_every_advertised_language_is_accepted() -> None:
    for code in reson8.SUPPORTED_LANGUAGES:
        assert normalize_languages(code) == code


# ---------------------------------------------------------------------------
# option validation, all before any request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(lambda: AudioOptions(num_channels=0), "between 1 and 10", id="no-channels"),
        pytest.param(lambda: AudioOptions(num_channels=11), "between 1 and 10", id="too-many"),
        pytest.param(lambda: AudioOptions(sample_rate=0), "must be positive", id="zero-rate"),
        pytest.param(lambda: AudioOptions(sample_rate=-16000), "must be positive", id="negative"),
    ],
)
def test_bad_audio_options_are_rejected(build: Callable[[], AudioOptions], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("num_channels", [1, 10])
def test_the_documented_channel_bounds_are_accepted(num_channels: int) -> None:
    assert AudioOptions(num_channels=num_channels).num_channels == num_channels


def test_pcm_is_the_only_encoding() -> None:
    """
    rtc frames are signed 16-bit PCM and we forward them unchanged.

    Accepting mulaw/alaw would only let a caller mislabel what is on the wire:
    the batch path overrides the encoding anyway, and the streaming path would
    tell Reson8 to decode PCM bytes as companded ones.
    """

    assert AudioOptions().encoding == "pcm_s16le"

    for encoding in ("mulaw", "alaw", "auto", "flac"):
        with pytest.raises(ValueError, match=r"(?i)unsupported encoding"):
            AudioOptions(encoding=encoding)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_an_out_of_range_probability_raises(value: float) -> None:
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        reson8.STT(api_key="k", turn=TurnOptions(final_probability=value))


@pytest.mark.parametrize(
    ("eager", "final"),
    [
        (0.5, 0.4),
        (0.9, 0.7),
        (0.95, None),  # above the server's 0.92 default
    ],
)
def test_inverted_thresholds_raise(eager: float, final: float | None) -> None:
    with pytest.raises(ValueError, match="must be below"):
        TurnOptions(eager_probability=eager, final_probability=final)


@pytest.mark.parametrize(
    ("eager", "final"),
    [
        (0.5, 0.5),  # both explicit, identical
        (None, 0.5),  # 0.5 == the server's default eager
    ],
)
def test_equal_thresholds_warn(
    eager: float | None, final: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Equal thresholds are legal but pointless: the preflight gets no lead."""

    with caplog.at_level("WARNING"):
        TurnOptions(eager_probability=eager, final_probability=final)
    assert caplog.records


@pytest.mark.parametrize(("eager", "final"), [(None, None), (0.35, 0.7)])
def test_sane_thresholds_are_quiet(
    eager: float | None, final: float | None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        TurnOptions(eager_probability=eager, final_probability=final)
    assert not caplog.records


def test_too_many_phrases_raises() -> None:
    with pytest.raises(ValueError, match="at most 250 entries"):
        BiasingOptions(phrases=[f"p{i}" for i in range(251)])


def test_the_documented_phrase_maximum_is_accepted() -> None:
    phrases = [f"p{i}" for i in range(250)]
    assert BiasingOptions(phrases=phrases).phrases == phrases


def test_a_comma_inside_a_phrase_raises() -> None:
    with pytest.raises(ValueError, match="may contain a comma"):
        BiasingOptions(phrases=["fine", "not,fine"])


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda values: BiasingOptions(phrases=values), id="phrases"),
        pytest.param(lambda values: BiasingOptions(patterns=values), id="patterns"),
    ],
)
def test_an_empty_entry_raises(build: Callable[[Sequence[str]], BiasingOptions]) -> None:
    with pytest.raises(ValueError, match="empty entry"):
        build(["ok", "  "])


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: BiasingOptions(phrases="Reson8"), id="phrases"),
        pytest.param(lambda: BiasingOptions(patterns="[0-9]{4}"), id="patterns"),
    ],
)
def test_a_bare_string_is_rejected(build: Callable[[], BiasingOptions]) -> None:
    """A bare ``str`` satisfies ``Sequence[str]``, so it would be split per character."""
    with pytest.raises(ValueError, match="takes a sequence of strings"):
        build()


@pytest.mark.parametrize(
    "pattern",
    ["[0-9]{4,6}", "(INV)?[0-9]{4,5}", "AMZ[0-9]{6}", "[A-Z]{2}[0-9]{2} [A-Z]{3}"],
)
def test_a_braced_repeat_range_survives_validation(pattern: str) -> None:
    """The comma in ``{m,n}`` is not a separator on the wire, so it must be allowed."""

    assert BiasingOptions(patterns=[pattern]).patterns == [pattern]


@pytest.mark.parametrize("pattern", ["a,b", "AMZ[0-9]{6},X"])
def test_a_comma_outside_braces_still_raises(pattern: str) -> None:
    with pytest.raises(ValueError, match="outside"):
        BiasingOptions(patterns=[pattern])


def test_negative_strength_raises() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        BiasingOptions(strength=-0.1)


def test_an_unsupported_filler_mode_raises() -> None:
    with pytest.raises(ValueError, match=r"(?i)unsupported filler_mode"):
        TranscriptOptions(filler_mode="loud")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# query string
# ---------------------------------------------------------------------------


def test_query_params_defaults() -> None:
    assert STTOptions().query_params(streaming=True) == {
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "channels": "1",
        "include_timestamps": "true",
        "include_language": "true",
    }


@pytest.mark.parametrize(
    ("language", "expected"),
    [("nl", "nl"), ("nl,de", "nl,de")],
)
def test_a_pinned_language_reaches_the_query_string(language: str, expected: str) -> None:
    params = STTOptions(language=language).query_params(streaming=True)
    assert params["language"] == expected


def test_language_is_omitted_for_auto_detection() -> None:
    assert "language" not in STTOptions().query_params(streaming=True)


def test_confidence_is_batch_only() -> None:
    """/turns reports no confidence, so the flag would be meaningless there."""

    opts = STTOptions(transcript=TranscriptOptions(confidence=True))
    assert "include_confidence" not in opts.query_params(streaming=True)
    assert opts.query_params(streaming=False)["include_confidence"] == "true"


def test_the_turn_thresholds_are_streaming_only() -> None:
    opts = STTOptions(turn=TurnOptions(eager_probability=0.35, final_probability=0.7))

    streaming = opts.query_params(streaming=True)
    assert streaming["eager_turn_probability"] == "0.35"
    assert streaming["final_turn_probability"] == "0.7"

    batch = opts.query_params(streaming=False)
    assert "eager_turn_probability" not in batch
    assert "final_turn_probability" not in batch


def test_the_audio_shape_is_described_in_reson8s_own_words() -> None:
    opts = STTOptions(audio=AudioOptions(sample_rate=8000, num_channels=2))
    params = opts.query_params(streaming=True)

    assert params["encoding"] == "pcm_s16le"
    assert params["sample_rate"] == "8000"
    # Reson8 spells it "channels"; num_channels is the framework's word
    assert params["channels"] == "2"


@pytest.mark.parametrize("streaming", [True, False])
def test_biasing_applies_to_both_endpoints(streaming: bool) -> None:
    opts = STTOptions(
        biasing=BiasingOptions(
            custom_model_id="m1",
            phrases=["reson8", "livekit"],
            patterns=["AMZ[0-9]{6}", "[A-Z]{2} [0-9]{3}"],
            strength=0.8,
        ),
        transcript=TranscriptOptions(words=True, filler_mode="verbatim"),
    )
    params = opts.query_params(streaming=streaming)

    assert params["custom_model_id"] == "m1"
    assert params["phrases"] == "reson8,livekit"
    assert params["patterns"] == "AMZ[0-9]{6},[A-Z]{2} [0-9]{3}"
    assert params["bias_strength"] == "0.8"
    assert params["filler_mode"] == "verbatim"
    assert params["include_words"] == "true"


def test_zero_strength_is_sent_rather_than_treated_as_unset() -> None:
    """0 is a value the server accepts; ``if self.strength`` would drop it."""

    opts = STTOptions(biasing=BiasingOptions(strength=0))
    assert opts.query_params(streaming=True)["bias_strength"] == "0"


def test_omitted_biasing_sends_nothing() -> None:
    params = STTOptions().query_params(streaming=True)
    for key in ("phrases", "patterns", "bias_strength", "filler_mode", "custom_model_id"):
        assert key not in params


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.reson8.dev", "wss://api.reson8.dev/turns?a=1"),
        ("http://localhost:8080", "ws://localhost:8080/turns?a=1"),
        ("https://api.reson8.dev/", "wss://api.reson8.dev/turns?a=1"),
    ],
)
def test_build_url_swaps_the_scheme_for_websockets(base_url: str, expected: str) -> None:
    assert build_url(base_url, "/turns", {"a": "1"}, websocket=True) == expected


def test_build_url_encodes_params() -> None:
    url = build_url(HOSTED, "/turns", {"language": "nl,de"})
    assert url == "https://api.reson8.dev/turns?language=nl%2Cde"


# ---------------------------------------------------------------------------
# transcript payload mapping
# ---------------------------------------------------------------------------


def test_confidence_passes_through_the_documented_range() -> None:
    """Reson8 reports a probability in (0, 1]; it must not be transformed."""

    assert _confidence({"confidence": 0.99}) == pytest.approx(0.99)
    assert _confidence({"confidence": 1.5}) == 1.0


@pytest.mark.parametrize("word", [{"text": "hi"}, {"confidence": 0.0}, {"confidence": -0.5}])
def test_an_absent_or_impossible_confidence_is_not_given(word: dict[str, object]) -> None:
    assert _confidence(word) is NOT_GIVEN


def test_word_times_apply_the_stream_offset() -> None:
    word = {"start_ms": 1000, "duration_ms": 500}
    assert _word_time(word, "start", offset=2.0) == pytest.approx(3.0)
    assert _word_time(word, "end", offset=2.0) == pytest.approx(3.5)
    assert _word_time({}, "start", offset=1.0) is NOT_GIVEN


def test_build_speech_data_minimal() -> None:
    data = build_speech_data({"text": "hello"}, language="en")
    assert data.text == "hello"
    assert data.language == "en"
    assert data.words == []


@pytest.mark.parametrize(
    ("msg_language", "configured", "expected"),
    [
        ("fr", "en", "fr"),  # what the server detected wins
        (None, "es", "es"),  # a single pinned language is a safe fallback
        ("", "nl", "nl"),
        (None, None, ""),  # auto-detection with nothing reported
        (None, "nl,de", ""),  # no single dominant language to assume
        ("nl", "nl,de", "nl"),
    ],
)
def test_build_speech_data_language(
    msg_language: str | None, configured: str | None, expected: str
) -> None:
    msg: dict[str, object] = {"text": "hi"}
    if msg_language is not None:
        msg["language"] = msg_language

    assert build_speech_data(msg, language=configured).language == expected


def test_build_speech_data_words_carry_timings_and_confidence() -> None:
    msg = {
        "text": "hi",
        "words": [{"text": "hi", "start_ms": 0, "duration_ms": 200, "confidence": 0.9}],
    }
    data = build_speech_data(msg, language="en", start_time_offset=1.0)

    assert data.words is not None
    word = data.words[0]
    assert word == "hi"  # TimedString subclasses str
    assert word.start_time == pytest.approx(1.0)
    assert word.end_time == pytest.approx(1.2)
    assert word.confidence == pytest.approx(0.9)


def test_build_speech_data_confidence_is_the_mean_of_the_words() -> None:
    msg = {
        "text": "hi there",
        "words": [
            {"text": "hi", "confidence": 0.99},
            {"text": "there", "confidence": 0.97},
        ],
    }
    assert build_speech_data(msg, language="en").confidence == pytest.approx(0.98)


def test_build_speech_data_applies_the_offset_to_the_turn() -> None:
    msg = {"text": "hi", "start_ms": 1000, "duration_ms": 500}
    data = build_speech_data(msg, language="en", start_time_offset=2.0)
    assert data.start_time == pytest.approx(3.0)
    assert data.end_time == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# the turn state machine
# ---------------------------------------------------------------------------


def test_start_of_speech_is_emitted_once(make_stream: MakeStream) -> None:
    stream = make_stream()

    stream._process_message({"type": "turn_start"})
    stream._process_message({"type": "turn_start"})

    starts = [e for e in emitted(stream) if e.type == SpeechEventType.START_OF_SPEECH]
    assert len(starts) == 1


def test_an_empty_candidate_is_not_surfaced_as_a_preflight(make_stream: MakeStream) -> None:
    stream = make_stream()

    stream._process_message({"type": "turn_start"})
    stream._process_message({"type": "turn_end_candidate", "text": ""})

    assert all(e.type != SpeechEventType.PREFLIGHT_TRANSCRIPT for e in emitted(stream))


def test_an_unhandled_message_type_produces_no_events(make_stream: MakeStream) -> None:
    stream = make_stream()

    stream._process_message({"type": "something_new"})

    assert emitted(stream) == []


def test_a_repeated_candidate_does_not_re_emit_a_preflight(make_stream: MakeStream) -> None:
    """
    A preflight cancels the generation running on the previous one.

    Re-sending unchanged text would restart the agent from scratch and throw
    away the head start it already had.
    """

    stream = make_stream(language="en")

    stream._process_message({"type": "turn_start"})
    stream._process_message({"type": "turn_end_candidate", "text": "order lunch"})
    stream._process_message({"type": "turn_end_candidate", "text": "order lunch"})

    preflights = [e for e in emitted(stream) if e.type == SpeechEventType.PREFLIGHT_TRANSCRIPT]
    assert len(preflights) == 1


def test_a_revised_candidate_does_re_emit_a_preflight(make_stream: MakeStream) -> None:
    stream = make_stream(language="en")

    stream._process_message({"type": "turn_start"})
    stream._process_message({"type": "turn_end_candidate", "text": "order"})
    stream._process_message({"type": "turn_end_candidate", "text": "order lunch"})

    preflights = [e for e in emitted(stream) if e.type == SpeechEventType.PREFLIGHT_TRANSCRIPT]
    assert [e.alternatives[0].text for e in preflights] == ["order", "order lunch"]


def test_the_last_candidate_still_becomes_the_final(make_stream: MakeStream) -> None:
    """Deduping the preflight must not stop turn_end promoting the candidate."""

    stream = make_stream(language="en")

    stream._process_message({"type": "turn_start"})
    stream._process_message({"type": "turn_end_candidate", "text": "order lunch"})
    stream._process_message({"type": "turn_end_candidate", "text": "order lunch"})
    stream._process_message({"type": "turn_end"})

    finals = [e for e in emitted(stream) if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["order lunch"]


# ---------------------------------------------------------------------------
# error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"code": "session_rejected"}', "session_rejected"),
        ('{"detail": "Credit limit exceeded"}', "Credit limit exceeded"),
        (
            '{"code": "invalid_query_parameter", "detail": "channels must be between 1 and 10"}',
            "invalid_query_parameter: channels must be between 1 and 10",
        ),
        ("", None),
        ("not json", None),
        ("[]", None),
        ("{}", None),
        ('{"code": 402, "detail": null}', None),
    ],
)
def test_problem_message(body: str, expected: str | None) -> None:
    assert problem_message(body) == expected


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [
        (400, False),
        (401, False),  # a bad key will keep being a bad key
        (402, False),  # so will an empty balance
        (429, True),  # concurrency frees up
        (500, True),
    ],
)
def test_status_error_retryability(status_code: int, retryable: bool) -> None:
    assert status_error(status_code).retryable is retryable


def test_status_error_adds_an_actionable_hint() -> None:
    assert "RESON8_API_KEY" in status_error(401).message
    assert "https://docs.reson8.dev/limits/" in status_error(402).message
    assert "https://docs.reson8.dev/limits/" in status_error(429).message


def test_status_error_keeps_the_server_reason_without_repeating_it() -> None:
    message = status_error(402, detail="Credit limit exceeded").message
    assert message.count("Credit limit exceeded") == 1
    assert "https://docs.reson8.dev/limits/" in message


def test_status_error_renders_an_unmapped_status() -> None:
    err = status_error(500)
    assert err.status_code == 500
    assert "500" in err.message


# ---------------------------------------------------------------------------
# over the wire, against a loopback Reson8
# ---------------------------------------------------------------------------


async def test_requests_are_authenticated_and_attributed(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    instance = _stt(server.base_url, client_session)

    await instance.recognize(_frame(), conn_options=NO_RETRY)
    assert server.post_headers["Authorization"] == "ApiKey secret"
    assert server.post_headers[INTEGRATION_HEADER] == INTEGRATION
    assert server.post_headers["Content-Type"] == "application/octet-stream"

    stream = instance.stream(conn_options=NO_RETRY)
    try:
        await asyncio.wait_for(server.connected.wait(), timeout=5)
    finally:
        await stream.aclose()

    assert server.handshake_headers["Authorization"] == "ApiKey secret"
    assert server.handshake_headers[INTEGRATION_HEADER] == INTEGRATION


async def test_a_turn_over_the_wire(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """
    A whole turn, in the order an ``AgentSession`` depends on.

    FINAL must land before END_OF_SPEECH: in ``turn_detection="stt"`` mode
    END_OF_SPEECH commits the user turn, so the transcript has to be
    accumulated by then.
    """

    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    stream.push_frame(_frame())
    stream.flush()
    await server.wait_for_text()

    await server.send({"type": "turn_start"})
    await server.send({"type": "turn_end_candidate", "text": "hello world"})
    await server.send({"type": "turn_end"})

    try:
        events = await log.wait_for(5)
        assert [e.type for e in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.PREFLIGHT_TRANSCRIPT,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
            SpeechEventType.RECOGNITION_USAGE,
        ]
        assert events[1].alternatives[0].text == "hello world"
        assert events[2].alternatives[0].text == "hello world"
        assert events[2].alternatives[0].language == "en"

        usage = events[4].recognition_usage
        assert usage is not None
        assert usage.audio_duration == pytest.approx(0.1)
    finally:
        await log.aclose()
        await stream.aclose()

    # the flush sentinel is what lets a caller commit a turn early, without
    # waiting for final_turn_probability to be crossed
    assert json.loads(server.text[0]) == {"type": "flush_request"}
    assert server.audio, "no audio frame reached the server"
    assert server.query["language"] == "en"


async def test_end_input_waits_for_the_final_transcript(
    finalizing_server: Callable[[], Awaitable[str]], client_session: aiohttp.ClientSession
) -> None:
    """
    ``end_input`` queues a flush; the transcript answers it.

    Closing the socket as soon as the input channel ends discards the turn_end
    the flush asked for, and the caller sees the stream finish empty.
    """

    base_url = await finalizing_server()
    stream = _stt(base_url, client_session, language="en").stream(conn_options=NO_RETRY)

    stream.push_frame(_frame())
    stream.end_input()

    events: list[stt.SpeechEvent] = []
    try:
        async with asyncio.timeout(10):
            async for event in stream:
                events.append(event)
    finally:
        await stream.aclose()

    types = [e.type for e in events]
    assert SpeechEventType.FINAL_TRANSCRIPT in types, f"stream ended with {types}"
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == "hello world"


async def test_end_input_does_not_wait_when_nothing_was_sent(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """
    A stream that carried no audio has nothing to wait for.

    The drain is bounded by the connect timeout, so gating it on outstanding
    audio is what keeps an empty teardown from stalling for that long.
    """

    server = await reson8_server()
    # a long drain deadline, so a wrong gate shows up as a hang not a pass
    stream = _stt(server.base_url, client_session).stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=30)
    )

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    stream.end_input()

    async with asyncio.timeout(5):
        async for _ in stream:
            pass

    await stream.aclose()


async def test_aclose_stays_immediate_with_a_turn_outstanding(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """``aclose`` cancels the run task, so it must not sit in the drain."""

    server = await reson8_server()
    # a long drain deadline, so a wrong gate shows up as a hang not a pass
    stream = _stt(server.base_url, client_session).stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=30)
    )

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    stream.push_frame(_frame())
    stream.flush()
    await server.wait_for_text()

    # audio is outstanding and the server will never answer it
    async with asyncio.timeout(5):
        await stream.aclose()


async def test_usage_is_not_reported_without_audio(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """A turn that carried no audio must not emit a zero-duration usage event."""

    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    await server.send({"type": "turn_start"})
    await server.send({"type": "turn_end_candidate", "text": "hi"})
    await server.send({"type": "turn_end"})

    try:
        events = await log.wait_for(4)
        assert SpeechEventType.RECOGNITION_USAGE not in [e.type for e in events]
    finally:
        await log.aclose()
        await stream.aclose()


async def test_update_options_waits_for_the_turn_to_end(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """
    New options must not redial in the middle of an utterance.

    Reson8 holds the turn server-side, so a redial mid-turn abandons the audio
    already sent and starts the replacement connection from whatever is spoken
    next -- splitting one utterance across two sessions and transcribing
    neither in full. The update waits for the turn to close instead.
    """

    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    await server.send({"type": "turn_start"})
    await server.send({"type": "turn_end_candidate", "text": "stale text"})
    opening = await log.wait_for(2)
    assert opening[1].type == SpeechEventType.PREFLIGHT_TRANSCRIPT

    try:
        stream.update_options(transcript=TranscriptOptions(words=True))

        # the turn is still open, so the connection is left alone
        await asyncio.sleep(0.3)
        assert server.connections == 1, "redialled in the middle of a turn"

        # closing the turn releases the held update
        await server.send({"type": "turn_end"})
        events = await log.wait_for(4)
        assert [e.type for e in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.PREFLIGHT_TRANSCRIPT,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        ]
        assert events[2].alternatives[0].text == "stale text"

        await server.wait_for_connections(2)
        assert server.query["include_words"] == "true"

        # and the replacement session starts with no turn state carried over:
        # a leftover candidate would be promoted by an unrelated turn_end, and
        # a leftover speaking flag would swallow the next START_OF_SPEECH
        await server.send({"type": "turn_end"})
        await log.assert_quiet()

        await server.send({"type": "turn_start"})
        await server.send({"type": "turn_end_candidate", "text": "fresh text"})
        await server.send({"type": "turn_end"})

        events = await log.wait_for(8)
        assert [e.type for e in events[4:]] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.PREFLIGHT_TRANSCRIPT,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        ]
        assert events[6].alternatives[0].text == "fresh text"
    finally:
        await log.aclose()
        await stream.aclose()


async def test_update_options_reconnects_at_once_when_no_turn_is_open(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    try:
        stream.update_options(language="de")
        await server.wait_for_connections(2)
        assert server.query["language"] == "de"
    finally:
        await stream.aclose()


async def test_a_reconnect_keeps_audio_that_has_not_been_sent_yet(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """
    Audio is chunked to 100ms before it goes out.

    A shorter frame sits in the chunker until enough arrives, and those bytes
    have already been taken off the input channel -- so a chunker rebuilt for
    the replacement connection would drop them silently.
    """

    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    try:
        # half a chunk: consumed from the input channel, not yet on the wire
        stream.push_frame(_frame(samples=800))
        await asyncio.sleep(0.1)
        assert not server.audio, "a partial chunk should not have been sent"

        stream.update_options(language="de")
        await server.wait_for_connections(2)

        # the other half completes the chunk on the new connection
        stream.push_frame(_frame(samples=800))
        stream.flush()
        await server.wait_for_text()
    finally:
        await stream.aclose()

    assert sum(len(chunk) for chunk in server.audio) == 800 * 2 * 2, (
        "the buffered half-chunk was dropped by the reconnect"
    )


async def test_a_rejected_upgrade_surfaces_the_status_and_the_reason(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """
    Reson8 rejects the upgrade with a status and an X-Error-Message header.

    Reporting it as a connection error instead would make an exhausted credit
    balance look like a network blip, and retry it with backoff.
    """

    server = await reson8_server(ws_status=402, ws_error_message="organization out of credits")
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    with pytest.raises(APIStatusError) as excinfo:
        await stream._run()

    assert excinfo.value.status_code == 402
    assert excinfo.value.retryable is False
    assert "organization out of credits" in excinfo.value.message
    await stream.aclose()


async def test_a_rejected_upgrade_without_a_reason_still_explains_itself(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server(ws_status=401)
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    with pytest.raises(APIStatusError, match="RESON8_API_KEY"):
        await stream._run()

    await stream.aclose()


async def test_an_unreachable_host_is_a_connection_error(
    client_session: aiohttp.ClientSession,
) -> None:
    stream = _stt("http://127.0.0.1:1", client_session).stream(conn_options=NO_RETRY)

    with pytest.raises(APIConnectionError, match="Failed to connect to Reson8"):
        await stream._run()

    await stream.aclose()


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (402, '{"code": "session_rejected"}', "session_rejected"),
        (400, '{"code": "invalid_query_parameter", "detail": "Invalid encoding: mp3"}', "mp3"),
        (413, "", "413"),
    ],
)
async def test_a_rejected_batch_request_reports_why(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    status: int,
    body: str,
    expected: str,
) -> None:
    server = await reson8_server(post_status=status, post_body=body)

    with pytest.raises(APIStatusError) as excinfo:
        await _stt(server.base_url, client_session).recognize(_frame(), conn_options=NO_RETRY)

    assert excinfo.value.status_code == status
    assert expected in excinfo.value.message


async def test_a_batch_timeout_is_a_timeout_error(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = await reson8_server()

    def timing_out_post(*args: object, **kwargs: object) -> None:
        raise asyncio.TimeoutError

    monkeypatch.setattr(client_session, "post", timing_out_post)

    with pytest.raises(APITimeoutError):
        await _stt(server.base_url, client_session).recognize(_frame(), conn_options=NO_RETRY)


async def test_streaming_rejects_a_frame_of_the_wrong_shape(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """
    The channel count is in the query string before any frame arrives.

    The base class resamples to the declared rate but never remixes, so a frame
    with a different channel count would be re-chunked under the wrong layout
    and silently mislabelled.
    """

    server = await reson8_server()
    stream = _stt(server.base_url, client_session, audio=AudioOptions(num_channels=1)).stream(
        conn_options=NO_RETRY
    )

    with pytest.raises(ValueError, match="expected 1-channel frames, got 2"):
        stream.push_frame(_frame(num_channels=2))

    await stream.aclose()
    assert not server.audio, "nothing should have reached the server"


async def test_streaming_accepts_the_configured_shape(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, audio=AudioOptions(num_channels=2)).stream(
        conn_options=NO_RETRY
    )

    stream.push_frame(_frame(num_channels=2))
    stream.flush()

    try:
        await server.wait_for_text()
    finally:
        await stream.aclose()

    assert server.audio, "no audio frame reached the server"
    assert server.query["channels"] == "2"


async def test_batch_describes_the_buffer_it_actually_posts(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """recognize() reports the audio it sends, not the configured shape."""

    server = await reson8_server()
    instance = _stt(
        server.base_url,
        client_session,
        audio=AudioOptions(sample_rate=48000, num_channels=1),
    )

    await instance.recognize(_frame(num_channels=2), conn_options=NO_RETRY)

    assert server.query["sample_rate"] == "16000"
    assert server.query["channels"] == "2"
    assert server.query["encoding"] == "pcm_s16le"


async def test_batch_rejects_a_buffer_with_too_many_channels(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """The buffer's own shape is validated too, before anything is posted."""

    server = await reson8_server()

    with pytest.raises(ValueError, match="between 1 and 10"):
        await _stt(server.base_url, client_session).recognize(
            _frame(num_channels=11), conn_options=NO_RETRY
        )

    assert not server.post_headers, "nothing should have been posted"
