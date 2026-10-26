"""p99lab Audrey STT against a scripted stand-in server on localhost (no network, no credentials)."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.plugins.p99lab import STT
from livekit.plugins.p99lab.stt import DEFAULT_BASE_URL

pytestmark = pytest.mark.unit

KEY = "p99_test_key"
FAST_RETRY = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=5.0)
BILLING = "https://console.p99lab.com/dashboard/billing"
E = stt.SpeechEventType


def tone(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)


def frames(
    samples: np.ndarray, sample_rate: int = 16000, frame_ms: int = 20
) -> list[rtc.AudioFrame]:
    n = sample_rate * frame_ms // 1000
    return [
        rtc.AudioFrame(
            data=samples[i : i + n].tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(samples[i : i + n]),
        )
        for i in range(0, len(samples), n)
    ]


async def normal_session(ws: web.WebSocketResponse, conn: dict) -> None:
    """A partial once audio arrives, a final at each flush, a final and done at end."""
    partial_sent = False
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            conn["audio"] += msg.data
            if not partial_sent and len(conn["audio"]) >= 3200:
                partial_sent = True
                await ws.send_json({"type": "partial", "text": "hello", "start": 0.0})
        elif msg.type == aiohttp.WSMsgType.TEXT:
            kind = json.loads(msg.data)["type"]
            conn["texts"].append(kind)
            if kind == "flush":
                await ws.send_json({"type": "final", "text": "hello world", "start": 0, "end": 1})
            elif kind == "end":
                await ws.send_json({"type": "final", "text": "the end", "start": 1, "end": 2})
                await ws.send_json({"type": "done", "audio_seconds": 2.0})
                return


def refuse(msg: dict):
    async def script(ws: web.WebSocketResponse, conn: dict) -> None:
        await ws.send_json({"type": "error", **msg})

    return script


class FakeAudrey:
    def __init__(self) -> None:
        self.connections: list[dict] = []
        self.scripts = [normal_session]
        self.requests: list[dict] = []
        self.reply: tuple[int, dict] = (200, {"text": "hello there", "audio_seconds": 1.0})

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = {"auth": request.headers.get("Authorization"), "audio": bytearray(), "texts": []}
        self.connections.append(conn)
        script = self.scripts[min(len(self.connections) - 1, len(self.scripts) - 1)]
        conn["config"] = json.loads(await ws.receive_str())
        await script(ws, conn)
        if not ws.closed:
            await ws.close()
        return ws

    async def transcribe_handler(self, request: web.Request) -> web.Response:
        body = await request.read()
        self.requests.append(
            {
                "auth": request.headers.get("Authorization"),
                "query": dict(request.query),
                "body": body,
            }
        )
        status, reply = self.reply
        return web.json_response(reply, status=status)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/stream", self.ws_handler)
        app.router.add_post("/v1/transcribe", self.transcribe_handler)
        return app


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("P99LAB_API_KEY", "P99LAB_API_URL", "AUDREY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
async def fake():
    fake = FakeAudrey()
    server = TestServer(fake.app())
    await server.start_server()
    fake.base_url = str(server.make_url("")).rstrip("/")
    yield fake
    await server.close()


@pytest.fixture
async def http():
    session = aiohttp.ClientSession()
    yield session
    await session.close()


def make(fake: FakeAudrey, http: aiohttp.ClientSession, **kw) -> STT:
    return STT(api_key=KEY, base_url=fake.base_url, http_session=http, **kw)


async def collect(stream) -> list[stt.SpeechEvent]:
    async def _all():
        return [ev async for ev in stream]

    return await asyncio.wait_for(_all(), 10)


def test_defaults_and_env(monkeypatch: pytest.MonkeyPatch, http) -> None:
    with pytest.raises(ValueError, match="P99LAB_API_KEY"):
        STT(http_session=http)
    monkeypatch.setenv("P99LAB_API_KEY", "p99_env")
    s = STT(http_session=http)
    assert s._api_key == "p99_env" and (s.model, s.provider) == ("audrey", "p99lab")
    assert s._opts.base_url == DEFAULT_BASE_URL
    assert s._opts.stream_url == DEFAULT_BASE_URL.replace("https://", "wss://") + "/v1/stream"
    monkeypatch.setenv("P99LAB_API_URL", "https://staging.example.test/")
    s = STT(http_session=http)
    assert s._opts.stream_url == "wss://staging.example.test/v1/stream"


@pytest.mark.parametrize(
    "kwargs", [{"sample_rate": 44100}, {"endpoint_ms": 100}, {"prompt": "x" * 501}]
)
def test_invalid_options(kwargs: dict, http) -> None:
    with pytest.raises(ValueError):
        STT(api_key=KEY, http_session=http, **kwargs)


async def test_stream_config_audio_and_events(fake, http) -> None:
    stream = make(fake, http, prompt="Acme").stream()
    for f in frames(tone(0.5)):
        stream.push_frame(f)
    stream.flush()
    stream.end_input()
    events = await collect(stream)
    (conn,) = fake.connections
    assert conn["auth"] == f"Bearer {KEY}"
    assert conn["config"] == {"language": "en", "sample_rate": 16000, "prompt": "Acme"}
    assert conn["texts"] == ["flush", "flush", "end"]  # flush(), then end_input() flushes and ends
    finals = [ev.alternatives[0].text for ev in events if ev.type == E.FINAL_TRANSCRIPT]
    assert finals == ["hello world", "hello world", "the end"]
    assert E.INTERIM_TRANSCRIPT in [ev.type for ev in events]


async def test_busy_and_503_reconnect(fake, http) -> None:
    fake.scripts = [
        refuse({"message": "server busy: too much audio waiting for transcription"}),
        refuse({"message": "billing is unavailable; retry shortly", "code": 503}),
        normal_session,
    ]
    stream = make(fake, http).stream(conn_options=FAST_RETRY)
    for f in frames(tone(0.3)):
        stream.push_frame(f)
    stream.end_input()
    await collect(stream)
    assert len(fake.connections) == 3


@pytest.mark.parametrize(
    "msg,status,reason",
    [
        ({"message": "unknown or revoked API key", "code": 401}, 401, None),
        ({"message": "unknown API key"}, 401, None),
        (
            {
                "message": f"your credits are used up; choose a plan to continue at {BILLING}",
                "code": 402,
                "top_up_url": BILLING,
            },
            402,
            "no_plan",
        ),
        (
            {
                "message": "your last plan payment failed; update your card under Manage billing to continue",
                "code": 402,
            },
            402,
            "payment_failed",
        ),
        (
            {
                "message": "this API key reached its daily audio limit; it resets at midnight Pacific time",
                "code": 429,
            },
            429,
            None,
        ),
    ],
)
async def test_stream_refusals_are_not_retried(
    fake, http, msg: dict, status: int, reason: str | None
) -> None:
    fake.scripts = [refuse(msg)]
    stream = make(fake, http).stream(conn_options=FAST_RETRY)
    for f in frames(tone(0.2)):
        stream.push_frame(f)
    with pytest.raises(APIStatusError) as info:
        await collect(stream)
    err = info.value
    assert err.status_code == status and not err.retryable
    assert len(fake.connections) == 1
    assert KEY not in repr(err)
    if reason:
        assert err.body["reason"] == reason and err.body["console_url"] == BILLING


async def test_out_of_credit_after_the_last_final(fake, http) -> None:
    async def out_of_credit(ws: web.WebSocketResponse, conn: dict) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data)["type"] == "end":
                await ws.send_json({"type": "final", "text": "last words", "start": 0, "end": 0.5})
                await ws.send_json(
                    {
                        "type": "error",
                        "code": 402,
                        "message": "your credits are used up; choose a plan or add credits to continue",
                        "top_up_url": BILLING,
                    }
                )
                return

    fake.scripts = [out_of_credit]
    stream = make(fake, http).stream(conn_options=FAST_RETRY)
    for f in frames(tone(0.3)):
        stream.push_frame(f)
    stream.end_input()
    events: list[stt.SpeechEvent] = []
    with pytest.raises(APIStatusError) as info:
        async for ev in stream:
            events.append(ev)
    assert [ev.alternatives[0].text for ev in events if ev.type == E.FINAL_TRANSCRIPT] == [
        "last words"
    ]
    assert info.value.status_code == 402 and info.value.body["reason"] == "no_credit"


async def test_recognize_posts_wav(fake, http) -> None:
    event = await make(fake, http, prompt="Acme").recognize(frames(tone(1.0)))
    assert event.type == E.FINAL_TRANSCRIPT and event.alternatives[0].text == "hello there"
    (req,) = fake.requests
    assert req["auth"] == f"Bearer {KEY}"
    assert req["query"] == {"language": "en", "prompt": "Acme"}
    assert req["body"][:4] == b"RIFF"


@pytest.mark.parametrize(
    "status,detail,code,reason",
    [
        (
            401,
            {"code": "unknown_key", "message": "unknown or revoked API key"},
            "unknown_key",
            None,
        ),
        (401, "missing API key", "unknown_key", None),
        (
            402,
            {
                "code": "insufficient_balance",
                "message": "your credits are used up; choose a plan or add credits to continue",
                "top_up_url": BILLING,
            },
            "insufficient_balance",
            "no_credit",
        ),
        (
            429,
            {
                "code": "key_daily_limit",
                "message": "this API key reached its daily limit of 20 audio hours; it resets at midnight Pacific time",
            },
            "key_daily_limit",
            None,
        ),
    ],
)
async def test_recognize_refusals(
    fake, http, status: int, detail, code: str, reason: str | None
) -> None:
    fake.reply = (status, {"detail": detail})
    with pytest.raises(APIStatusError) as info:
        await make(fake, http).recognize(frames(tone(0.5)), conn_options=FAST_RETRY)
    err = info.value
    assert err.status_code == status and not err.retryable and err.body["code"] == code
    if reason:
        assert err.body["reason"] == reason and err.body["console_url"] == BILLING
    assert len(fake.requests) == 1
