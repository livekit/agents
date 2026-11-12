"""Gladia startup and configuration regression contracts."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError, http
from livekit.plugins.gladia import STT


async def test_startup_timeout_reaches_consumer() -> None:
    """Startup failure reaches fallback callers. https://github.com/livekit/agents/issues/2645"""
    release = asyncio.Event()

    async def initialize(request: web.Request) -> web.StreamResponse:
        await request.read()
        await release.wait()
        return web.Response(status=503)

    app = web.Application()
    app.router.add_post("/live", initialize)
    async with TestServer(app) as server, http.open() as session:
        model = STT(api_key="test", http_session=session, base_url=str(server.make_url("/live")))
        try:
            with pytest.raises(APIConnectionError) as caught:
                async with model.stream(
                    conn_options=APIConnectOptions(max_retry=0, read_timeout=0.03)
                ) as stream:
                    await asyncio.wait_for(anext(stream), 2)
            assert caught.value.retryable
            assert isinstance(caught.value.__cause__, APITimeoutError)
        finally:
            release.set()
            await model.aclose()


# Any: request JSON, read by Gladia's documented shape
Json = dict[str, Any]
Region = Literal["us-west", "eu-west"]


@pytest.mark.parametrize(
    "initial,updated", [("eu-west", None), ("us-west", None), ("us-west", "eu-west")]
)
async def test_region_is_in_initialization_query(initial: Region, updated: Region | None) -> None:
    """Region updates reach the initialization URL. https://github.com/livekit/agents/pull/3018 https://github.com/livekit/agents/pull/7523"""
    requests: list[tuple[dict[str, str], Json]] = []
    initialized = asyncio.Event()

    async def initialize(request: web.Request) -> web.StreamResponse:
        requests.append((dict(request.query), await request.json()))
        initialized.set()
        return web.json_response({"id": "session", "url": str(request.url.with_path("/ws"))})

    async def peer(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_post("/live", initialize)
    app.router.add_get("/ws", peer)
    async with TestServer(app) as server, http.open() as session:
        model = STT(
            api_key="test",
            http_session=session,
            base_url=str(server.make_url("/live")),
            region=initial,
        )
        if updated:
            model.update_options(region=updated)
        try:
            async with model.stream(conn_options=APIConnectOptions(max_retry=0)):
                await asyncio.wait_for(initialized.wait(), 2)
            query, body = requests[0]
            assert query["region"] == (updated or initial)
            assert "region" not in body
        finally:
            await model.aclose()


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8091 reproduces: update_options accepts enabled translation without target languages",
)
async def test_translation_update_rejects_missing_targets_without_mutation() -> None:
    """Invalid translation updates preserve valid state. https://github.com/livekit/agents/pull/8091"""
    requests: list[Json] = []
    initialized = asyncio.Event()

    async def initialize(request: web.Request) -> web.StreamResponse:
        requests.append(await request.json())
        initialized.set()
        return web.json_response({"id": "session", "url": str(request.url.with_path("/ws"))})

    async def peer(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_post("/live", initialize)
    app.router.add_get("/ws", peer)
    async with TestServer(app) as server, http.open() as session:
        model = STT(
            api_key="test",
            http_session=session,
            base_url=str(server.make_url("/live")),
            translation_enabled=True,
            translation_target_languages=["fr"],
        )
        try:
            with pytest.raises(ValueError, match="translation_target_languages"):
                model.update_options(translation_target_languages=[])
            async with model.stream(conn_options=APIConnectOptions(max_retry=0)):
                await asyncio.wait_for(initialized.wait(), 2)
            assert requests[0]["translation_config"]["target_languages"] == ["fr"]
            model.update_options(translation_enabled=False, translation_target_languages=[])
            initialized.clear()
            async with model.stream(conn_options=APIConnectOptions(max_retry=0)):
                await asyncio.wait_for(initialized.wait(), 2)
            assert requests[1]["translation"] is False
        finally:
            await model.aclose()
