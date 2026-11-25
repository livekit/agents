"""Hermetic tests for the Reson8 STT plugin, against a loopback aiohttp stand-in for the API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Protocol, cast, get_args

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
from livekit.plugins.reson8 import stt as reson8_stt
from livekit.plugins.reson8._utils import (
    ERROR_MESSAGE_HEADER,
    INTEGRATION_HEADER,
    MODEL_LANGUAGES,
    PRERECORDED_PATH,
    TURNS_PATH,
    _confidence,
    build_speech_data,
    build_url,
    normalize_languages,
    problem_parts,
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
ONE_RETRY = APIConnectOptions(max_retry=1, timeout=5)
TURN = [
    {"type": "turn_start"},
    {"type": "turn_end_candidate", "text": "hello world"},
    {"type": "turn_end"},
]
TURN_TYPES = [
    SpeechEventType.START_OF_SPEECH,
    SpeechEventType.PREFLIGHT_TRANSCRIPT,
    SpeechEventType.FINAL_TRANSCRIPT,
    SpeechEventType.END_OF_SPEECH,
]

HOSTED = "https://api.reson8.dev"
SELF_HOSTED = "https://stt.internal.example"
OTHER = "https://other.example"
LIMITS_URL = "https://docs.reson8.dev/limits/"
REJECTED = "Reson8 rejected the request"


def _frame(num_channels: int = 1, samples: int = 1600) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples * num_channels,
        sample_rate=16000,
        num_channels=num_channels,
        samples_per_channel=samples,
    )


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.fixture(autouse=True)
def _env_and_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RESON8_API_KEY", "RESON8_BASE_URL", "RESON8_API_URL"):
        monkeypatch.delenv(name, raising=False)
    # the production ceilings are tuned for real latency; a loopback answers in microseconds
    monkeypatch.setattr(reson8_stt, "_FINAL_TURN_TIMEOUT", 0.1)
    monkeypatch.setattr(reson8_stt, "_FINAL_TURN_GRACE", 0.02)
    # the framework's first retry otherwise waits a fixed 0.1s
    monkeypatch.setattr(APIConnectOptions, "_interval_for_retry", lambda self, n: 0.0)


class FakeChan:
    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


@pytest.fixture
def offline_stream() -> SpeechStream:
    """A stream built without ``__init__``, so ``_process_message`` runs with no socket."""
    stream = SpeechStream.__new__(SpeechStream)
    stream._opts = STTOptions(language="en")
    stream._request_id = "req-test"
    stream._speaking = False
    stream._candidate = None
    stream._turn_settled = asyncio.Event()
    stream._turn_ended = asyncio.Event()
    stream._input_ch = SimpleNamespace(start_time_offset=0.0)  # type: ignore[assignment]
    stream._speech_duration = 0.0
    stream._event_ch = FakeChan()  # type: ignore[assignment]
    return stream


@dataclass
class Recorded:
    """What the fake Reson8 server saw, and how to talk back."""

    base_url: str = ""
    handshake_headers: dict[str, str] = field(default_factory=dict)
    post_headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)
    audio: list[bytes] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    connections: int = 0
    ws: web.WebSocketResponse | None = None

    async def send(self, *messages: object) -> None:
        await asyncio.wait_for(self.connected.wait(), timeout=5)
        assert self.ws is not None
        for message in messages:
            await self.ws.send_str(json.dumps(message))

    async def hang_up(self) -> None:
        assert self.ws is not None
        await self.ws.close()


class StartServer(Protocol):
    async def __call__(
        self,
        *,
        post_status: int = ...,
        post_body: str = ...,
        ws_status: int | None = ...,
        ws_error_message: str | None = ...,
        answer_flush: list[dict[str, object]] | None = ...,
        hang_up_after_flush: bool = ...,
    ) -> Recorded: ...


@pytest.fixture
async def reson8_server() -> AsyncIterator[StartServer]:
    runners: list[web.AppRunner] = []
    tasks: list[asyncio.Task[None]] = []

    async def _start(
        *,
        post_status: int = 200,
        post_body: str = '{"text": "hello"}',
        ws_status: int | None = None,
        ws_error_message: str | None = None,
        answer_flush: list[dict[str, object]] | None = None,
        hang_up_after_flush: bool = False,
    ) -> Recorded:
        rec = Recorded()

        async def prerecorded(request: web.Request) -> web.StreamResponse:
            rec.post_headers = dict(request.headers)
            rec.query = dict(request.query)
            await request.read()
            return web.Response(status=post_status, text=post_body, content_type="application/json")

        async def turns(request: web.Request) -> web.StreamResponse:
            rec.handshake_headers = dict(request.headers)
            rec.query = dict(request.query)
            if ws_status is not None:
                # a rejected upgrade carries no body, only this header
                headers = {ERROR_MESSAGE_HEADER: ws_error_message} if ws_error_message else {}
                return web.Response(status=ws_status, headers=headers, text="")

            ws = web.WebSocketResponse()
            await ws.prepare(request)
            rec.ws = ws
            rec.connections += 1
            rec.connected.set()

            async def answer() -> None:
                # finalising is not instant
                await asyncio.sleep(0.02)
                for message in answer_flush or []:
                    await ws.send_str(json.dumps(message))
                if hang_up_after_flush:
                    await ws.close()

            async for msg in ws:
                if msg.type is aiohttp.WSMsgType.BINARY:
                    rec.audio.append(msg.data)
                elif msg.type is aiohttp.WSMsgType.TEXT:
                    rec.text.append(msg.data)
                    if answer_flush is not None or hang_up_after_flush:
                        tasks.append(asyncio.create_task(answer()))
            return ws

        app = web.Application()
        app.router.add_post(PRERECORDED_PATH, prerecorded)
        app.router.add_get(TURNS_PATH, turns)
        runner = web.AppRunner(app)
        await runner.setup()
        runners.append(runner)
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        rec.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        return rec

    yield _start

    await utils.aio.cancel_and_wait(*tasks)
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
        self._task = asyncio.create_task(self._read(stream))

    async def _read(self, stream: SpeechStream) -> None:
        async for event in stream:
            self.events.append(event)

    async def wait_for(self, count: int) -> list[stt.SpeechEvent]:
        await _until(lambda: len(self.events) >= count)
        return self.events

    async def aclose(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


def _stt(base_url: str, session: aiohttp.ClientSession, **kwargs: object) -> reson8.STT:
    return reson8.STT(api_key="secret", base_url=base_url, http_session=session, **kwargs)  # type: ignore[arg-type]


async def _drain(stream: SpeechStream) -> list[stt.SpeechEvent]:
    events: list[stt.SpeechEvent] = []
    try:
        async with asyncio.timeout(5):
            async for event in stream:
                events.append(event)
    finally:
        await stream.aclose()
    return events


def _finals(events: list[stt.SpeechEvent]) -> list[str]:
    return [e.alternatives[0].text for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]


# ---------------------------------------------------------------------------
# construction and options
# ---------------------------------------------------------------------------


def test_a_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="RESON8_API_KEY"):
        reson8.STT()


def test_the_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESON8_API_KEY", "from-env")
    assert reson8.STT()._api_key == "from-env"


@pytest.mark.parametrize(
    ("env", "argument", "expected"),
    [
        ({}, None, HOSTED),
        ({"RESON8_BASE_URL": OTHER}, SELF_HOSTED, SELF_HOSTED),
        ({"RESON8_BASE_URL": SELF_HOSTED}, None, SELF_HOSTED),
        ({"RESON8_BASE_URL": SELF_HOSTED, "RESON8_API_URL": OTHER}, None, SELF_HOSTED),
        ({"RESON8_BASE_URL": f"{SELF_HOSTED}/"}, None, SELF_HOSTED),
    ],
)
def test_base_url_resolution(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], argument: str | None, expected: str
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    kwargs = {"base_url": argument} if argument else {}
    assert reson8.STT(api_key="k", **kwargs)._base_url == expected


def test_the_renamed_env_var_is_honoured_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A self-hosted deployment on the 0.2.x name must not be sent to the public API."""
    monkeypatch.setenv("RESON8_API_URL", SELF_HOSTED)
    with caplog.at_level("WARNING"):
        instance = reson8.STT(api_key="k")

    assert instance._base_url == SELF_HOSTED
    assert "RESON8_API_URL is deprecated" in caplog.text


@pytest.mark.parametrize(
    ("base_url", "warns"),
    [
        ("http://reson8.internal.example", True),
        ("http://10.0.0.5:8080", True),
        ("https://api.reson8.dev", False),
        ("http://127.0.0.1:8080", False),
        ("http://localhost:8080", False),
        ("http://[::1]:8080", False),
    ],
)
def test_a_plaintext_remote_base_url_warns(
    base_url: str, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        reson8.STT(api_key="k", base_url=base_url)
    assert ("not encrypted" in caplog.text) is warns


@pytest.mark.parametrize("words", [True, False])
def test_the_aligned_transcript_capability_follows_the_options(words: bool) -> None:
    instance = reson8.STT(api_key="k", transcript=TranscriptOptions(words=not words))
    assert instance.capabilities.aligned_transcript == (False if words else "word")

    instance.update_options(language="de")
    assert instance.capabilities.aligned_transcript == (False if words else "word")

    instance.update_options(transcript=TranscriptOptions(words=words))
    assert instance.capabilities.aligned_transcript == ("word" if words else False)


def test_model_and_provider() -> None:
    assert reson8.STT(api_key="k").provider == "Reson8"
    assert reson8.STT(api_key="k").model == "resonant-1"
    assert reson8.STT(api_key="k", model="rumble-2").model == "rumble-2"

    biasing = BiasingOptions(custom_model_id="m1")
    assert reson8.STT(api_key="k", biasing=biasing).model == "resonant-1"


def test_an_unlisted_model_is_sent_without_local_checks() -> None:
    """A model newer than the plugin is left for Reson8 to check."""

    biasing = BiasingOptions(patterns=["AMZ[0-9]{6}"], strength=1.2)
    instance = reson8.STT(api_key="k", model="rumble-3", language="xx", biasing=biasing)
    assert instance.model == "rumble-3"

    params = STTOptions(model="rumble-3", language="XX", biasing=biasing).query_params(
        streaming=True
    )
    assert params["model"] == "rumble-3"
    assert params["language"] == "xx"
    assert params["patterns"] == "AMZ[0-9]{6}"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ([], None),
        ("NL", "nl"),
        ("nl,de", "nl,de"),
        (["nl", "de"], "nl,de"),
        (" nl , de ", "nl,de"),
        (["nl", "", "de"], "nl,de"),
        ("ja,yue", "ja,yue"),
    ],
)
def test_normalize_languages(value: str | list[str] | None, expected: str | None) -> None:
    assert normalize_languages(value, "rumble-2") == expected


@pytest.mark.parametrize("value", ["xx", "nl,xx", ["nl", "xx"], "english"])
def test_an_unsupported_language_raises(value: str | list[str]) -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        reson8.STT(api_key="k", language=value)


def test_every_listed_language_is_accepted() -> None:
    for model, languages in MODEL_LANGUAGES.items():
        for code in languages:
            assert normalize_languages(code, model) == code


def test_every_model_language_is_advertised() -> None:
    """A listed model missing from the table would skip its language check."""

    assert set(MODEL_LANGUAGES) == set(get_args(reson8.STTModel))
    for languages in MODEL_LANGUAGES.values():
        assert set(languages) <= set(reson8.SUPPORTED_LANGUAGES)


@pytest.mark.parametrize("model", [None, "resonant-1"])
def test_resonant_1_rejects_a_rumble_2_only_language(model: reson8.STTModel | None) -> None:
    """Reson8 uses resonant-1 when no model is sent."""

    with pytest.raises(ValueError, match="resonant-1 supports"):
        reson8.STT(api_key="k", model=model, language="ja")


def test_rumble_2_accepts_its_own_languages() -> None:
    reson8.STT(api_key="k", model="rumble-2", language="ja")


def test_a_language_override_is_checked_against_the_model() -> None:
    instance = reson8.STT(api_key="k", model="resonant-1")

    with pytest.raises(ValueError, match="unsupported language"):
        instance.update_options(language="ja")

    with pytest.raises(ValueError, match="unsupported language"):
        instance.stream(language="ja")


async def test_a_recognize_language_is_checked_before_any_request() -> None:
    instance = reson8.STT(api_key="k", model="resonant-1")

    with pytest.raises(ValueError, match="unsupported language"):
        await instance.recognize(_frame(), language="ja", conn_options=NO_RETRY)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: AudioOptions(num_channels=0), "between 1 and 10"),
        (lambda: AudioOptions(num_channels=11), "between 1 and 10"),
        (lambda: AudioOptions(sample_rate=0), "must be positive"),
        (lambda: AudioOptions(sample_rate=-16000), "must be positive"),
        # rtc frames are PCM, so any other label would mislabel the bytes on the wire
        *(
            (lambda e=e: AudioOptions(encoding=e), "(?i)unsupported encoding")
            for e in ("mulaw", "alaw", "auto", "flac")
        ),
        (lambda: TurnOptions(final_probability=-0.1), "must be between 0 and 1"),
        (lambda: TurnOptions(final_probability=1.1), "must be between 0 and 1"),
        (lambda: TurnOptions(eager_probability=0.5, final_probability=0.4), "must be below"),
        (lambda: TurnOptions(eager_probability=0.9, final_probability=0.7), "must be below"),
        (lambda: BiasingOptions(phrases=[f"p{i}" for i in range(251)]), "at most 250 entries"),
        (lambda: BiasingOptions(phrases=["fine", "not,fine"]), "cannot contain a comma"),
        (lambda: BiasingOptions(phrases=["ok", "  "]), "empty entry"),
        (lambda: BiasingOptions(patterns=["ok", "  "]), "empty entry"),
        # a bare str is a Sequence[str] that would be split per character
        (lambda: BiasingOptions(patterns="[0-9]{4}"), "takes a sequence of strings"),
        (lambda: BiasingOptions(patterns=["a,b"]), "outside"),
        (lambda: BiasingOptions(patterns=["AMZ[0-9]{6},X"]), "outside"),
        (lambda: BiasingOptions(strength=-0.1), "must be non-negative"),
        (lambda: TranscriptOptions(filler_mode="loud"), "(?i)unsupported filler_mode"),
    ],
)
def test_invalid_options_raise(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: AudioOptions(num_channels=1),
        lambda: AudioOptions(num_channels=10),
        lambda: BiasingOptions(phrases=[f"p{i}" for i in range(250)]),
        # the comma in a {m,n} repeat is not a separator on the wire
        *(
            (lambda p=p: BiasingOptions(patterns=[p]))
            for p in ("[0-9]{4,6}", "(INV)?[0-9]{4,5}", "[A-Z]{2}[0-9]{2} [A-Z]{3}")
        ),
    ],
)
def test_boundary_options_are_accepted(build: Callable[[], object]) -> None:
    build()


@pytest.mark.parametrize(
    ("eager", "final", "warns"),
    [
        (0.5, 0.5, True),
        (None, None, False),
        (0.35, 0.7, False),
        # defaults differ per model, so a lone threshold is not checked against one
        (0.95, None, False),
        (None, 0.4, False),
        (None, 0.5, False),
    ],
)
def test_equal_thresholds_warn(
    eager: float | None, final: float | None, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        TurnOptions(eager_probability=eager, final_probability=final)
    assert bool(caplog.records) is warns


@pytest.mark.parametrize(
    "build",
    [
        lambda v: BiasingOptions(phrases=[v]),
        lambda v: BiasingOptions(patterns=[f"{v},x"]),
        lambda v: BiasingOptions(phrases=v),
    ],
)
def test_rejected_vocabulary_is_not_quoted_back(build: Callable[[str], BiasingOptions]) -> None:
    """Customer vocabulary in a ValueError message escapes redaction."""
    with pytest.raises(ValueError) as excinfo:
        build("Jan, Willem de Vries")

    message = str(excinfo.value)
    assert "Jan" not in message and "Willem" not in message
    assert "phrases" in message or "patterns" in message


@pytest.mark.parametrize(
    ("biasing", "message"),
    [
        pytest.param(BiasingOptions(patterns=["AMZ[0-9]{6}"]), "patterns", id="patterns"),
        pytest.param(BiasingOptions(strength=1.2), "at most 1", id="strength"),
    ],
)
def test_rumble_2_rejects_biasing_it_cannot_apply(biasing: BiasingOptions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reson8.STT(api_key="k", model="rumble-2", biasing=biasing)

    instance = reson8.STT(api_key="k", model="rumble-2")
    with pytest.raises(ValueError, match=message):
        instance.update_options(biasing=biasing)

    with pytest.raises(ValueError, match=message):
        STTOptions(model="rumble-2", biasing=biasing)


def test_rumble_2_accepts_a_strength_of_1() -> None:
    reson8.STT(api_key="k", model="rumble-2", biasing=BiasingOptions(strength=1))


@pytest.mark.parametrize("model", [None, "resonant-1"])
def test_patterns_and_a_strong_bias_pass_without_rumble_2(model: reson8.STTModel | None) -> None:
    biasing = BiasingOptions(patterns=["AMZ[0-9]{6}"], strength=1.2)
    reson8.STT(api_key="k", model=model, biasing=biasing)


# ---------------------------------------------------------------------------
# query string and payload mapping
# ---------------------------------------------------------------------------


def test_query_params_defaults() -> None:
    assert STTOptions().query_params(streaming=True) == {
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "channels": "1",
        "include_timestamps": "true",
        "include_language": "true",
    }


@pytest.mark.parametrize("streaming", [True, False])
def test_the_model_reaches_both_endpoints(streaming: bool) -> None:
    params = STTOptions(model="rumble-2").query_params(streaming=streaming)
    assert params["model"] == "rumble-2"


def test_query_params_by_endpoint() -> None:
    opts = STTOptions(
        language="nl,de",
        audio=AudioOptions(sample_rate=8000, num_channels=2),
        transcript=TranscriptOptions(confidence=True),
        turn=TurnOptions(eager_probability=0.35, final_probability=0.7),
    )

    streaming = opts.query_params(streaming=True)
    assert streaming["language"] == "nl,de"
    assert (streaming["sample_rate"], streaming["channels"]) == ("8000", "2")
    assert streaming["eager_turn_probability"] == "0.35"
    assert streaming["final_turn_probability"] == "0.7"
    # /turns reports no confidence
    assert "include_confidence" not in streaming

    batch = opts.query_params(streaming=False)
    assert batch["include_confidence"] == "true"
    assert "eager_turn_probability" not in batch and "final_turn_probability" not in batch


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
    opts = STTOptions(biasing=BiasingOptions(strength=0))
    assert opts.query_params(streaming=True)["bias_strength"] == "0"


@pytest.mark.parametrize(
    ("base_url", "params", "expected"),
    [
        (HOSTED, {"a": "1"}, "https://api.reson8.dev/turns?a=1"),
        (f"{HOSTED}/", {"a": "1"}, "https://api.reson8.dev/turns?a=1"),
        ("http://localhost:8080", {"a": "1"}, "http://localhost:8080/turns?a=1"),
        (HOSTED, {"language": "nl,de"}, "https://api.reson8.dev/turns?language=nl%2Cde"),
    ],
)
def test_build_url(base_url: str, params: dict[str, str], expected: str) -> None:
    assert build_url(base_url, "/turns", params) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ({"confidence": 0.99}, 0.99),
        ({"confidence": 1.5}, 1.0),
        ({"text": "hi"}, NOT_GIVEN),
        ({"confidence": 0.0}, NOT_GIVEN),
        ({"confidence": -0.5}, NOT_GIVEN),
    ],
)
def test_confidence(word: dict[str, object], expected: object) -> None:
    assert _confidence(word) == expected


@pytest.mark.parametrize(
    ("msg_language", "configured", "expected"),
    [
        ("fr", "en", "fr"),  # what the server detected wins
        (None, "es", "es"),  # a single pinned language is a safe fallback
        ("", "nl", "nl"),
        (None, None, ""),
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


def test_build_speech_data_timings_and_confidence() -> None:
    msg = {
        "text": "hi there",
        "start_ms": 1000,
        "duration_ms": 500,
        "words": [
            {"text": "hi", "start_ms": 1000, "duration_ms": 200, "confidence": 0.99},
            {"text": "there", "confidence": 0.97},
        ],
    }
    data = build_speech_data(msg, language="en", start_time_offset=2.0)

    assert (data.start_time, data.end_time) == (pytest.approx(3.0), pytest.approx(3.5))
    assert data.confidence == pytest.approx(0.98)
    assert data.words is not None
    hi, there = data.words
    assert hi == "hi"
    assert (hi.start_time, hi.end_time) == (pytest.approx(3.0), pytest.approx(3.2))
    assert hi.confidence == pytest.approx(0.99)
    assert there.start_time is NOT_GIVEN


# ---------------------------------------------------------------------------
# the turn state machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidates", "preflights"),
    [
        # a repeated preflight would restart the agent's generation for nothing
        (["order lunch", "order lunch"], ["order lunch"]),
        (["order", "order lunch"], ["order", "order lunch"]),
        (["", "order lunch"], ["order lunch"]),
    ],
)
def test_candidates_become_preflights_then_the_final(
    offline_stream: SpeechStream, candidates: list[str], preflights: list[str]
) -> None:
    offline_stream._process_message({"type": "turn_start"})
    offline_stream._process_message({"type": "turn_start"})
    offline_stream._process_message({"type": "something_new"})
    for text in candidates:
        offline_stream._process_message({"type": "turn_end_candidate", "text": text})
    offline_stream._process_message({"type": "turn_end"})

    events = cast(FakeChan, offline_stream._event_ch).events
    assert [e.type for e in events] == [
        SpeechEventType.START_OF_SPEECH,
        *[SpeechEventType.PREFLIGHT_TRANSCRIPT] * len(preflights),
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    assert texts == [*preflights, candidates[-1]]


@pytest.mark.parametrize(
    "payload",
    [
        {"words": ["hi"]},
        {"start_ms": "abc"},
        {"words": [{"text": "hi", "confidence": "high"}]},
        {"words": [{"text": "hi", "start_ms": "x", "duration_ms": 1}]},
        {"start_ms": int("9" * 4000)},
        {"words": [{"text": "hi", "start_ms": int("9" * 4000)}]},
    ],
)
def test_a_malformed_turn_payload_is_skipped(
    offline_stream: SpeechStream, payload: dict[str, object]
) -> None:
    """A mapping error would escape the recv task and end the stream unretried."""
    offline_stream._process_message({"type": "turn_end_candidate", "text": "hi", **payload})
    offline_stream._process_message({"type": "turn_end_candidate", "text": "recovered"})
    offline_stream._process_message({"type": "turn_end"})

    events = cast(FakeChan, offline_stream._event_ch).events
    assert _finals(events) == ["recovered"]


def test_provider_event_content_stays_out_of_log_bodies(
    offline_stream: SpeechStream, caplog: pytest.LogCaptureFixture
) -> None:
    """A log body cannot be redacted, so provider fields go under ``lk.pii.*`` keys."""
    smuggled = "turn_end; patient Jan Willem de Vries"
    with caplog.at_level("DEBUG", logger="livekit.plugins.reson8"):
        offline_stream._process_message({"type": smuggled, "text": "secret words"})

    assert smuggled not in caplog.text
    assert "secret words" not in caplog.text
    for record in caplog.records:
        for key, value in record.__dict__.items():
            if value in (smuggled, "secret words"):
                assert key.startswith("lk.pii."), f"{key} is not a redactable key"
    assert any(getattr(r, "lk.pii.type", None) == smuggled for r in caplog.records)


# ---------------------------------------------------------------------------
# error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"code": "session_rejected"}', ("session_rejected", None)),
        ('{"detail": "Credit limit exceeded"}', (None, "Credit limit exceeded")),
        (
            '{"code": "invalid_query_parameter", "detail": "channels must be 1 to 10"}',
            ("invalid_query_parameter", "channels must be 1 to 10"),
        ),
        ("", (None, None)),
        ("not json", (None, None)),
        ("[]", (None, None)),
        ("{}", (None, None)),
        ('{"code": 402, "detail": null}', (None, None)),
    ],
)
def test_problem_parts(body: str, expected: tuple[str | None, str | None]) -> None:
    assert problem_parts(body) == expected


def test_the_server_free_text_stays_off_the_exception(caplog: pytest.LogCaptureFixture) -> None:
    """The framework logs the exception's str, so free text there escapes redaction."""
    phrase = "Jan Willem de Vries"
    with caplog.at_level("WARNING"):
        err = status_error(400, code="invalid_query_parameter", detail=f"bad phrase: {phrase}")

    assert phrase not in str(err) and phrase not in err.message and phrase not in str(err.body)
    record = next(r for r in caplog.records if r.message == REJECTED)
    assert getattr(record, "lk.pii.detail") == f"bad phrase: {phrase}"
    assert getattr(record, "lk.pii.code") == "invalid_query_parameter"


@pytest.mark.parametrize(
    ("code", "identifier"),
    [
        ("invalid_query_parameter", True),
        ("a", True),
        ("code_9", True),
        ("phrase rejected: Jan Willem de Vries", False),
        ("UPPER_SNAKE", False),
        ("has-a-hyphen", False),
        ("trailing space ", False),
        ("x" * 65, False),
    ],
)
def test_only_an_identifier_shaped_code_reaches_the_exception(
    code: str, identifier: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        err = status_error(400, code=code)

    assert (code in err.message) is identifier
    record = next(r for r in caplog.records if r.message == REJECTED)
    assert getattr(record, "lk.pii.code") == code
    if not identifier:
        assert code not in caplog.text


@pytest.mark.parametrize(
    ("status_code", "code", "retryable", "hint"),
    [
        (400, None, False, "400"),
        (401, None, False, "RESON8_API_KEY"),
        (402, "session_rejected", False, LIMITS_URL),
        (429, None, True, LIMITS_URL),
        (500, None, True, "500"),
    ],
)
def test_status_error(status_code: int, code: str | None, retryable: bool, hint: str) -> None:
    err = status_error(status_code, code=code)
    assert err.status_code == status_code
    assert err.retryable is retryable
    assert hint in err.message
    if code:
        assert err.message.count(code) == 1


# ---------------------------------------------------------------------------
# streaming, against a loopback Reson8
# ---------------------------------------------------------------------------


async def test_requests_are_authenticated_and_attributed(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    instance = _stt(server.base_url, client_session)
    integration = f"livekit-python:{reson8.__version__}"

    await instance.recognize(_frame(), conn_options=NO_RETRY)
    assert server.post_headers["Authorization"] == "ApiKey secret"
    assert server.post_headers[INTEGRATION_HEADER] == integration
    assert server.post_headers["Content-Type"] == "application/octet-stream"

    stream = instance.stream(conn_options=NO_RETRY)
    try:
        await asyncio.wait_for(server.connected.wait(), timeout=5)
    finally:
        await stream.aclose()
    assert server.handshake_headers["Authorization"] == "ApiKey secret"
    assert server.handshake_headers[INTEGRATION_HEADER] == integration


async def test_closing_the_recognizer_closes_its_streams(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    instance = _stt(server.base_url, client_session)

    first = instance.stream(conn_options=NO_RETRY)
    second = instance.stream(conn_options=NO_RETRY)
    await _until(lambda: server.connections == 2)
    # one stream already closed must not stop the rest from closing
    await first.aclose()
    await instance.aclose()

    with pytest.raises(RuntimeError, match="input ended|is closed"):
        second.push_frame(_frame())
    # the session belongs to the caller
    assert not client_session.closed


async def test_a_turn_over_the_wire(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """FINAL lands before END_OF_SPEECH, which commits the user turn in stt mode."""
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    stream.push_frame(_frame())
    stream.flush()
    await _until(lambda: bool(server.text))
    await server.send(*TURN)

    try:
        events = await log.wait_for(5)
        assert [e.type for e in events] == [*TURN_TYPES, SpeechEventType.RECOGNITION_USAGE]
        assert events[2].alternatives[0].text == "hello world"
        assert events[2].alternatives[0].language == "en"
        assert events[4].recognition_usage is not None
        assert events[4].recognition_usage.audio_duration == pytest.approx(0.1)
    finally:
        await log.aclose()
        await stream.aclose()

    assert json.loads(server.text[0]) == {"type": "flush_request"}
    assert server.audio
    assert server.query["language"] == "en"


async def test_unusable_frames_are_skipped_and_no_audio_means_no_usage(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    try:
        assert server.ws is not None
        for frame in ("[]", "null", "5", '"text"', "not json"):
            await server.ws.send_str(frame)
        await server.send(*TURN)

        events = await log.wait_for(4)
        await asyncio.sleep(0.02)
        assert [e.type for e in events] == TURN_TYPES
    finally:
        await log.aclose()
        await stream.aclose()


@pytest.mark.parametrize("hang_up", [False, True])
async def test_end_input_waits_for_the_final_transcript(
    reson8_server: StartServer, client_session: aiohttp.ClientSession, hang_up: bool
) -> None:
    """``end_input`` queues a flush, and its answer must not be discarded by closing early."""
    server = await reson8_server(answer_flush=TURN, hang_up_after_flush=hang_up)
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)

    stream.push_frame(_frame())
    stream.end_input()

    assert _finals(await _drain(stream)) == ["hello world"]


async def test_an_unanswered_flush_closes_cleanly(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server(answer_flush=[])
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    stream.push_frame(_frame())
    stream.end_input()

    assert await _drain(stream) == []


async def test_a_close_inside_the_final_turn_hold_is_clean(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reson8_stt, "_FINAL_TURN_GRACE", 0.1)
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    stream.push_frame(_frame())
    stream.end_input()
    await _until(lambda: bool(server.text))

    async def answer_then_hang_up() -> None:
        await server.send({"type": "turn_end_candidate", "text": "delivered"}, {"type": "turn_end"})
        # let the waiter start its hold, so the close lands while the notification is cleared
        await asyncio.sleep(0.03)
        await server.hang_up()

    answering = asyncio.create_task(answer_then_hang_up())
    try:
        assert _finals(await _drain(stream)) == ["delivered"]
    finally:
        await utils.aio.cancel_and_wait(answering)


async def test_end_input_waits_past_a_stale_confirmation(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reson8_stt, "_FINAL_TURN_GRACE", 0.1)
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    # turn A is outstanding, then audio for turn B goes out
    await server.send({"type": "turn_start"})
    stream.push_frame(_frame())
    stream.end_input()
    await _until(lambda: bool(server.text))

    async def confirm_a_then_b() -> None:
        # A's confirmation lands alone first, carrying no transcript for B
        await server.send({"type": "turn_end"})
        await asyncio.sleep(0.03)
        await server.send({"type": "turn_end_candidate", "text": "turn B"}, {"type": "turn_end"})

    answering = asyncio.create_task(confirm_a_then_b())
    try:
        assert _finals(await _drain(stream)) == ["turn B"]
    finally:
        await utils.aio.cancel_and_wait(answering)


async def test_a_close_with_a_turn_open_is_an_error(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    stream.push_frame(_frame())
    stream.end_input()
    await _until(lambda: bool(server.text))
    await server.send({"type": "turn_start"})
    await asyncio.sleep(0.01)
    await server.hang_up()

    with pytest.raises(APIConnectionError, match="turn still open") as excinfo:
        await _drain(stream)
    assert excinfo.value.retryable is False


async def test_trailing_silence_does_not_fail_the_stream(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    stream.push_frame(_frame())
    await server.send(*TURN)
    await asyncio.sleep(0.02)
    # the caller keeps pushing silence after the turn closed
    stream.push_frame(_frame())
    stream.end_input()

    started = time.monotonic()
    assert _finals(await _drain(stream)) == ["hello world"]
    assert time.monotonic() - started < reson8_stt._FINAL_TURN_TIMEOUT * 3


@pytest.mark.parametrize("push_audio", [False, True])
async def test_teardown_does_not_wait_for_a_turn(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    monkeypatch: pytest.MonkeyPatch,
    push_audio: bool,
) -> None:
    """With no audio sent, end_input has nothing to wait for; aclose never waits at all."""
    server = await reson8_server()
    # a long drain deadline, so a wrong gate shows up as a hang not a pass
    monkeypatch.setattr(reson8_stt, "_FINAL_TURN_TIMEOUT", 30.0)
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    async with asyncio.timeout(2):
        if push_audio:
            stream.push_frame(_frame())
            stream.flush()
            await _until(lambda: bool(server.text))
            await stream.aclose()
        else:
            stream.end_input()
            await _drain(stream)


async def test_an_internal_redial_is_not_counted_twice(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    created = time.time()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=ONE_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    try:
        # an internal redial, which contributes to the offset
        await asyncio.sleep(0.05)
        stream.update_options(language="de")
        await _until(lambda: server.connections == 2)

        # then a failure, so the framework retries the whole run
        await server.hang_up()
        await _until(lambda: server.connections == 3)

        await server.send(
            {"type": "turn_start"},
            {"type": "turn_end_candidate", "text": "after", "start_ms": 0, "duration_ms": 100},
            {"type": "turn_end"},
        )
        events = await log.wait_for(2)
        final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
        # the offset can never exceed wall clock; counting the redial twice does
        assert final.alternatives[0].start_time <= time.time() - created
    finally:
        await log.aclose()
        await stream.aclose()


async def test_a_reconnect_does_not_rewind_transcript_timing(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """Each connection restarts start_ms, but transcripts share one audio timeline."""
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="en").stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    async def turn(text: str, events: int) -> None:
        stream.push_frame(_frame())
        stream.flush()
        await _until(lambda: len(server.text) == server.connections)
        await server.send(
            {"type": "turn_start"},
            {"type": "turn_end_candidate", "text": text, "start_ms": 0, "duration_ms": 20},
            {"type": "turn_end"},
        )
        await log.wait_for(events)

    try:
        await asyncio.wait_for(server.connected.wait(), timeout=5)
        await turn("first", 5)
        gap = 0.05
        await asyncio.sleep(gap)
        stream.update_options(language="de")
        await _until(lambda: server.connections == 2)
        await turn("second", 10)

        first, second = [
            e.alternatives[0] for e in log.events if e.type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert (first.start_time, first.end_time) == (pytest.approx(0.0), pytest.approx(0.02))
        assert second.start_time >= gap * 0.8, "the second session's timeline restarted at zero"
        assert second.start_time >= first.end_time
    finally:
        await log.aclose()
        await stream.aclose()


async def test_a_mutated_sequence_still_triggers_a_reconnect(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """Options copy the caller's list, so a later change to it is a real update."""
    server = await reson8_server()
    shared = ["LiveKit"]
    stream = _stt(server.base_url, client_session, biasing=BiasingOptions(phrases=shared)).stream(
        conn_options=NO_RETRY
    )

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    assert server.query["phrases"] == "LiveKit"
    try:
        shared.append("Reson8")
        stream.update_options(biasing=BiasingOptions(phrases=shared))
        await _until(lambda: server.connections == 2)
        assert server.query["phrases"] == "LiveKit,Reson8"
    finally:
        await stream.aclose()


async def test_an_unchanged_update_does_not_reconnect(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, language="nl").stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    await server.send(*TURN[:2])
    await log.wait_for(2)
    try:
        stream.update_options(language="nl", transcript=TranscriptOptions())
        await asyncio.sleep(0.05)
        assert server.connections == 1

        await server.send(TURN[2])
        events = await log.wait_for(4)
        assert _finals(events) == ["hello world"]
    finally:
        await log.aclose()
        await stream.aclose()


async def test_update_options_mid_turn_abandons_that_turn(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """A redial loses the turn in progress, and nothing of it leaks into the new session."""
    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    await server.send({"type": "turn_start"}, {"type": "turn_end_candidate", "text": "stale"})
    await log.wait_for(2)
    try:
        stream.update_options(transcript=TranscriptOptions(words=True))
        await _until(lambda: server.connections == 2)
        # the interrupted turn is closed out rather than left open
        events = await log.wait_for(3)
        assert events[2].type == SpeechEventType.END_OF_SPEECH

        # a leftover turn_end would promote the stale candidate
        await server.send(TURN[2])
        await server.send(*TURN)
        events = await log.wait_for(7)
        await asyncio.sleep(0.02)
        assert [e.type for e in events[3:]] == TURN_TYPES
        assert _finals(events) == ["hello world"]
    finally:
        await log.aclose()
        await stream.aclose()


async def test_a_reconnect_keeps_audio_that_has_not_been_sent_yet(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """A partial 100ms chunk already taken off the input must survive the redial."""
    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    try:
        stream.push_frame(_frame(samples=800))
        await asyncio.sleep(0.02)
        assert not server.audio

        stream.update_options(language="de")
        await _until(lambda: server.connections == 2)
        stream.push_frame(_frame(samples=800))
        stream.flush()
        await _until(lambda: bool(server.text))
    finally:
        await stream.aclose()

    assert sum(len(chunk) for chunk in server.audio) == 800 * 2 * 2


async def test_a_dropped_socket_closes_out_the_open_turn(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """A framework retry mid-turn emits END_OF_SPEECH so the session is not left waiting."""
    server = await reson8_server()
    stream = _stt(server.base_url, client_session).stream(conn_options=ONE_RETRY)
    log = EventLog(stream)

    await asyncio.wait_for(server.connected.wait(), timeout=5)
    try:
        stream.push_frame(_frame())
        await server.send({"type": "turn_start"})
        await log.wait_for(1)

        await server.hang_up()
        await _until(lambda: server.connections == 2)
        await server.send({"type": "turn_start"})
        events = await log.wait_for(3)
        assert [e.type for e in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.END_OF_SPEECH,
            SpeechEventType.START_OF_SPEECH,
        ]
    finally:
        await log.aclose()
        await stream.aclose()


async def test_a_rejected_upgrade_surfaces_the_status_and_the_reason(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected upgrade is a status error, not a retried network blip."""
    server = await reson8_server(ws_status=402, ws_error_message="organization out of credits")
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    with caplog.at_level("WARNING"), pytest.raises(APIStatusError) as excinfo:
        await stream._run()
    await stream.aclose()

    assert excinfo.value.status_code == 402
    assert excinfo.value.retryable is False
    assert LIMITS_URL in excinfo.value.message
    # the header is free text, so it only reaches a redactable key
    assert "organization out of credits" not in str(excinfo.value)
    assert "organization out of credits" not in caplog.text
    record = next(r for r in caplog.records if r.message == REJECTED)
    assert getattr(record, "lk.pii.detail") == "organization out of credits"


async def test_a_send_failure_is_retryable(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = await reson8_server()
    real = aiohttp.ClientWebSocketResponse.send_bytes
    failures = 0

    async def fail_once(self: object, *args: object, **kwargs: object) -> None:
        nonlocal failures
        if failures == 0:
            failures += 1
            raise aiohttp.ClientConnectionResetError("cannot write to closing transport")
        await real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_bytes", fail_once)
    instance = _stt(server.base_url, client_session)
    errors: list[stt.STTError] = []
    instance.on("error", errors.append)
    stream = instance.stream(conn_options=ONE_RETRY)
    try:
        stream.push_frame(_frame())
        await _until(lambda: server.connections == 2)
    finally:
        await stream.aclose()

    [error] = errors
    assert isinstance(error.error, APIConnectionError)
    assert "Failed to send audio" in str(error.error)
    assert error.error.retryable and error.recoverable
    assert error.error.__cause__ is None


async def test_a_websocket_error_frame_surfaces_its_cause(
    reson8_server: StartServer,
    client_session: aiohttp.ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ERROR frame's cause must not be lost to the CLOSED that follows it."""
    server = await reson8_server()
    sent = False

    async def error_frame(self: object, *args: object, **kwargs: object) -> aiohttp.WSMessage:
        nonlocal sent
        if not sent:
            sent = True
            return aiohttp.WSMessage(
                aiohttp.WSMsgType.ERROR, aiohttp.WebSocketError(1002, "invalid frame"), None
            )
        await asyncio.sleep(0)
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "receive", error_frame)
    stream = _stt(server.base_url, client_session).stream(conn_options=NO_RETRY)

    with pytest.raises(APIConnectionError, match="WebSocketError") as excinfo:
        await _drain(stream)
    assert excinfo.value.__cause__ is None


async def test_an_unreachable_host_is_a_connection_error(
    client_session: aiohttp.ClientSession,
) -> None:
    stream = _stt("http://127.0.0.1:1", client_session).stream(conn_options=NO_RETRY)
    with pytest.raises(APIConnectionError, match="Failed to connect to Reson8"):
        await stream._run()
    await stream.aclose()


async def test_streaming_frame_shape_must_match_the_configured_channels(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    """The base class never remixes, so a mismatched frame would be silently mislabelled."""
    server = await reson8_server()
    stream = _stt(server.base_url, client_session, audio=AudioOptions(num_channels=2)).stream(
        conn_options=NO_RETRY
    )
    try:
        with pytest.raises(ValueError, match="expected 2-channel frames, got 1"):
            stream.push_frame(_frame(num_channels=1))
        stream.push_frame(_frame(num_channels=2))
        stream.flush()
        await _until(lambda: bool(server.text))
    finally:
        await stream.aclose()

    assert sum(len(chunk) for chunk in server.audio) == 1600 * 2 * 2
    assert server.query["channels"] == "2"


# ---------------------------------------------------------------------------
# batch recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (402, '{"code": "session_rejected"}', "session_rejected"),
        (400, '{"code": "invalid_query_parameter", "detail": "x"}', "invalid_query_parameter"),
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


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        "null",
        "5",
        '"text"',
        "not json",
        "",
        '{"text": "hi", "words": ["hi"]}',
        '{"text": "hi", "start_ms": "abc"}',
        '{"text": "hi", "words": [{"text": "hi", "confidence": "high"}]}',
        '{"text": "hi", "start_ms": ' + "9" * 4000 + "}",
    ],
)
async def test_an_unusable_batch_body_is_a_retryable_connection_error(
    reson8_server: StartServer, client_session: aiohttp.ClientSession, body: str
) -> None:
    """A bare mapping error would escape as a non-API exception the framework never retries."""
    server = await reson8_server(post_body=body)

    with pytest.raises(APIConnectionError, match="malformed response body") as excinfo:
        await _stt(server.base_url, client_session).recognize(_frame(), conn_options=NO_RETRY)

    assert excinfo.value.retryable is True
    assert excinfo.value.__cause__ is None


async def test_an_empty_batch_body_is_an_empty_transcript(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server(post_body="{}")
    event = await _stt(server.base_url, client_session).recognize(_frame(), conn_options=NO_RETRY)

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == ""


async def test_a_batch_timeout_is_a_timeout_error(
    client_session: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timing_out_post(*args: object, **kwargs: object) -> None:
        raise asyncio.TimeoutError

    monkeypatch.setattr(client_session, "post", timing_out_post)
    with pytest.raises(APITimeoutError):
        await _stt(HOSTED, client_session).recognize(_frame(), conn_options=NO_RETRY)


async def test_batch_describes_the_buffer_it_actually_posts(
    reson8_server: StartServer, client_session: aiohttp.ClientSession
) -> None:
    server = await reson8_server()
    instance = _stt(
        server.base_url, client_session, audio=AudioOptions(sample_rate=48000, num_channels=1)
    )

    await instance.recognize(_frame(num_channels=2), conn_options=NO_RETRY)
    assert server.query["sample_rate"] == "16000"
    assert server.query["channels"] == "2"
    assert server.query["encoding"] == "pcm_s16le"

    with pytest.raises(ValueError, match="between 1 and 10"):
        await instance.recognize(_frame(num_channels=11), conn_options=NO_RETRY)
