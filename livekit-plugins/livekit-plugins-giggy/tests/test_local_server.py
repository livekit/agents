"""Tests that run real aiohttp traffic against an offline loopback server.

They need no Giggy account or credentials, but they open local sockets and wait on real
timers (heartbeats, read timeouts), so they sit in the giggy plugin category rather than
the hermetic unit suite.
"""

import asyncio
import io
import json
import unittest
import wave
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

import livekit.plugins.giggy as provider_module
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    utils,
)
from livekit.plugins import giggy

pytestmark = pytest.mark.plugin("giggy")

KEY = "offline-test-key"
VOICE = "00000000-0000-0000-0000-000000000000"
GENERATION = "00000000-0000-4000-8000-000000000001"
RESULT_URL = "https://00000000000000000000000000000000.r2.cloudflarestorage.com/transcript.json?signature=offline"


class WarmConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutated_caller_credentials_are_not_sent_by_health_checks(self):
        calls = []

        async def health(request):
            calls.append(dict(request.headers))
            return web.Response(body=b"ok")

        app = web.Application()
        app.router.add_get("/__ingress/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]) + "/v1"
        try:
            for credential in (
                "cookie_jar",
                "unrelated_cookie",
                "Authorization",
                "X-API-Key",
                "Cookie",
            ):
                with self.subTest(credential=credential):
                    calls.clear()
                    async with aiohttp.ClientSession(
                        cookie_jar=aiohttp.CookieJar(unsafe=True)
                    ) as session:
                        provider = provider_module.TTS(
                            api_key="test-key", voice="voice", http_session=session, base_url=base
                        )
                        if credential == "cookie_jar":
                            session.cookie_jar.update_cookies({"sid": "caller-private"}, URL(base))
                            self.assertTrue(session.cookie_jar.filter_cookies(URL(base)))
                        elif credential == "unrelated_cookie":
                            session.cookie_jar.update_cookies(
                                {"sid": "other-provider"}, URL("https://other.example/")
                            )
                        else:
                            session.headers[credential] = "caller-private-default"
                        try:
                            if credential == "unrelated_cookie":
                                self.assertTrue(await provider.connect())
                                self.assertIs(provider._ensure_session(), session)
                                self.assertEqual(len(calls), 1)
                                self.assertNotIn("Cookie", calls[0])
                            else:
                                self.assertFalse(await provider.connect())
                                self.assertEqual(calls, [])
                                with self.assertRaisesRegex(ValueError, "credential-free"):
                                    provider._ensure_session()
                        finally:
                            await provider.aclose()
                        self.assertFalse(session.closed)
        finally:
            await runner.cleanup()

    async def test_dev_health_route_fallback(self):
        app = web.Application()

        async def health(request):
            return web.Response(body=b"ok")

        app.router.add_get("/api/internal/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]) + "/v1"
        try:
            async with aiohttp.ClientSession() as session:
                provider = provider_module.TTS(
                    api_key="test-key", voice="voice", http_session=session, base_url=base
                )
                self.assertTrue(await provider.connect())
                await provider.aclose()
                self.assertFalse(session.closed)
        finally:
            await runner.cleanup()

    async def test_connect_reuses_socket_and_close_preserves_caller_session(self):
        calls = []

        async def handle(request):
            calls.append((request.method, request.path, id(request.transport)))
            return web.Response(body=b"pcm!")

        app = web.Application()
        app.router.add_get("/__ingress/health", handle)
        app.router.add_post("/v1/audio/speech", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]) + "/v1"
        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(keepalive_timeout=0.15)
            ) as session:
                provider = provider_module.TTS(
                    api_key="test-key", voice="voice", http_session=session, base_url=base
                )
                self.assertTrue(
                    callable(getattr(provider, "connect", None)),
                    "Provider needs a connection lifecycle",
                )
                with patch("livekit.plugins.giggy._connection.HEARTBEAT_SECONDS", 0.04):
                    self.assertTrue(await provider.connect())
                    deadline = asyncio.get_running_loop().time() + 2
                    while (
                        provider._connection.checks < 3
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        await asyncio.sleep(0.01)
                    self.assertGreaterEqual(len(calls), 3)
                    async with session.post(base + "/audio/speech", data=b"{}") as response:
                        await response.read()
                    self.assertEqual(len({r[2] for r in calls}), 1)
                    await provider.aclose()
                    self.assertFalse(session.closed)
                    count = len(calls)
                    await asyncio.sleep(0.1)
                    self.assertEqual(len(calls), count)

            # LiveKit does not own provider shutdown. Automatic standalone
            # prewarming must finish without an explicit provider.aclose().
            for borrowed in (True, False):
                async with utils.http_context.open() as session:
                    provider = provider_module.TTS(
                        api_key="test-key",
                        voice="voice",
                        http_session=session if borrowed else None,
                        base_url=base,
                    )
                    agent_session = AgentSession(tts=provider)
                    with patch("livekit.plugins.giggy._connection.HEARTBEAT_SECONDS", 0.02):
                        try:
                            await agent_session.start(Agent(instructions="Offline lifecycle test"))
                            await asyncio.wait_for(provider._connection._checked.wait(), 2)
                            await agent_session.aclose()
                            task = provider._connection._task
                            count = len(calls)
                            await asyncio.sleep(0.08)
                            self.assertEqual(len(calls), count)
                            self.assertTrue(task.done())
                            self.assertFalse(session.closed)
                        finally:
                            await agent_session.aclose()
                            await provider.aclose()

            # A worker-owned heartbeat must be awaited during job shutdown,
            # even when the application never explicitly closes the provider.
            async with aiohttp.ClientSession() as session:
                provider = provider_module.TTS(
                    api_key="test-key", voice="voice", http_session=session, base_url=base
                )
                job = MagicMock()
                with (
                    patch("livekit.agents.job.get_job_context", return_value=job),
                    patch("livekit.plugins.giggy._connection.HEARTBEAT_SECONDS", 0.02),
                ):
                    try:
                        provider.prewarm()
                        provider.prewarm()
                        await asyncio.wait_for(provider._connection._checked.wait(), 2)
                        task = provider._connection._task
                        job.add_shutdown_callback.assert_called_once()
                        await job.add_shutdown_callback.call_args.args[0]()
                        self.assertTrue(task.done())
                        count = len(calls)
                        await asyncio.sleep(0.08)
                        self.assertEqual(len(calls), count)
                        self.assertFalse(session.closed)
                    finally:
                        await provider.aclose()

            # Capturing the shared session once prevents its context factory
            # reopening a client after that context has closed.
            provider = provider_module.TTS(api_key="test-key", voice="voice", base_url=base)
            with patch("livekit.plugins.giggy._connection.HEARTBEAT_SECONDS", 0.02):
                try:
                    async with utils.http_context.open():
                        self.assertTrue(await provider.connect())
                        task = provider._connection._task
                    count = len(calls)
                    await asyncio.sleep(0.08)
                    self.assertEqual(len(calls), count)
                    self.assertTrue(task.done())
                    self.assertFalse(provider._connection.ready)
                finally:
                    await provider.aclose()
        finally:
            await runner.cleanup()

    async def test_health_failure_recovers_without_synthesis(self):
        calls = []

        async def health(request):
            calls.append(request.method)
            return web.Response(status=503 if len(calls) == 1 else 200)

        app = web.Application()
        app.router.add_get("/__ingress/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]) + "/v1"
        try:
            async with aiohttp.ClientSession() as session:
                provider = provider_module.TTS(
                    api_key="test-key", voice="voice", http_session=session, base_url=base
                )
                with patch("livekit.plugins.giggy._connection.HEARTBEAT_SECONDS", 0.03):
                    self.assertFalse(await provider.connect())
                    deadline = asyncio.get_running_loop().time() + 2
                    while (
                        not provider._connection.ready
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        await asyncio.sleep(0.01)
                    self.assertTrue(provider._connection.ready)
                    self.assertEqual(provider._connection.failures, 1)
                    self.assertTrue(all(method == "GET" for method in calls))
                    await provider.aclose()
                    self.assertFalse(session.closed)
        finally:
            await runner.cleanup()


class Response:
    """A canned aiohttp response for the mocked status and signed-artifact reads."""

    def __init__(self, body):
        self.status = 200
        self.headers = {"Content-Type": "application/json", "x-request-id": "stt-request"}
        self.body = json.dumps(body).encode()
        if isinstance(body, list):
            self.headers["Content-Type"] = "text/event-stream"
            self.body = b"".join(
                f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
                for event, payload in body
            )
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_chunked(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


def transcript_events():
    return [
        ("generation_created", {"generation_uuid": GENERATION}),
        (
            "transcript_chunk",
            {
                "generation_uuid": GENERATION,
                "attempt": 1,
                "sequence": 0,
                "text": "Hello LiveKit.",
                "replace": True,
                "complete": True,
            },
        ),
        (
            "generation_updated",
            {
                "generation_uuid": GENERATION,
                "status": "completed",
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": RESULT_URL},
                },
            },
        ),
    ]


def audio(duration=0.5):
    return rtc.AudioFrame(
        data=b"\x01\x02" * int(16000 * duration),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=int(16000 * duration),
    )


@pytest.fixture
def session(monkeypatch):
    import livekit.plugins.giggy.stt as giggy_stt

    monkeypatch.setattr(giggy_stt, "uuid4", lambda: GENERATION)
    client = MagicMock(spec=aiohttp.ClientSession)
    client.get.side_effect = [
        Response(
            {
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": RESULT_URL},
                }
            }
        ),
        Response({"text": "Hello LiveKit.", "languages": ["English"], "audio_duration_ms": 500}),
    ]
    return client


@pytest.mark.parametrize("case", ["success", "headers_timeout", "body_timeout", "wrong_attempt"])
async def test_stt_real_http(session, monkeypatch, case):
    import livekit.plugins.giggy.stt as giggy_stt

    requests = []

    async def handler(request):
        requests.append(
            {
                "headers": dict(request.headers),
                "query": dict(request.query),
                "audio": await request.read(),
            }
        )
        if case == "headers_timeout":
            await asyncio.sleep(0.2)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if case == "body_timeout":
            await asyncio.sleep(0.2)
            return response
        if request.transport is None or request.transport.is_closing():
            return response
        events = transcript_events()
        if case == "wrong_attempt":
            partial = dict(events[1][1], complete=False, sequence=0)
            events[1][1].update(attempt=2, sequence=1)
            events.insert(1, ("transcript_chunk", partial))
        wire = Response(events).body.replace(b"\n", b"\r\n")
        for index in range(0, len(wire), 17):
            await response.write(wire[index : index + 17])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        monkeypatch.setattr(giggy_stt, "GIGGY_ORIGIN", f"http://127.0.0.1:{port}")
        async with aiohttp.ClientSession() as client:
            # Only signed-artifact/status reads are mocked; admission, framing,
            # connection cleanup and read timeouts exercise actual aiohttp.
            monkeypatch.setattr(client, "get", session.get)
            provider = giggy.STT(api_key=KEY, http_session=client)
            options = APIConnectOptions(timeout=0.05 if "timeout" in case else 2, max_retry=5)
            if case == "success":
                event = await provider.recognize(audio(), conn_options=options)
                assert event.alternatives[0].text == "Hello LiveKit."
            else:
                expected = APITimeoutError if "timeout" in case else APIConnectionError
                with pytest.raises(expected) as caught:
                    await provider.recognize(audio(), conn_options=options)
                assert caught.value.retryable is False
                assert caught.value.__cause__ is None
            await provider.aclose()
            assert not client.closed
        assert len(requests) == 1
        assert requests[0]["headers"]["xi-api-key"] == KEY
        assert requests[0]["query"]["mode"] == "streaming"
        with wave.open(io.BytesIO(requests[0]["audio"]), "rb") as wav:
            assert wav.getnframes() == 8000
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("read_timeout", [False, True])
async def test_real_http_error_is_single_post(
    monkeypatch: pytest.MonkeyPatch, read_timeout: bool
) -> None:
    """Exercise actual aiohttp transport against an offline loopback server."""
    requests: list[dict[str, object]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        if read_timeout:
            response = web.StreamResponse(headers={"Content-Type": "application/octet-stream"})
            await response.prepare(request)
            await asyncio.sleep(0.3)
            return response
        return web.Response(status=503, text=KEY)

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        base_url = f"http://127.0.0.1:{port}/v1"
        async with aiohttp.ClientSession() as session:
            provider = giggy.TTS(api_key=KEY, voice=VOICE, http_session=session, base_url=base_url)
            expected = APITimeoutError if read_timeout else APIStatusError
            with pytest.raises(expected) as caught:
                async with provider.synthesize(
                    "Hi",
                    conn_options=APIConnectOptions(max_retry=5, retry_interval=0, timeout=0.05),
                ) as stream:
                    await asyncio.wait_for(stream.collect(), 2)
            assert KEY not in str(caught.value)
            assert caught.value.retryable is False
            await provider.aclose()
            assert not session.closed
        assert len(requests) == 1
        assert requests[0]["headers"]["Authorization"] == "Bearer " + KEY
        assert "Idempotency-Key" not in requests[0]["headers"]
    finally:
        await runner.cleanup()
