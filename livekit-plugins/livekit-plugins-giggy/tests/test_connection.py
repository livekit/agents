import asyncio
import unittest
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

import livekit.plugins.giggy as provider_module
from livekit.agents import Agent, AgentSession, utils

pytestmark = pytest.mark.unit


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
            async with aiohttp.ClientSession() as session:
                provider = provider_module.TTS(
                    api_key="test-key", voice="voice", http_session=session, base_url=base
                )
                session.headers["Authorization"] = "Bearer caller-private-default"
                self.assertFalse(await provider.connect())
                self.assertEqual(calls, [])
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
