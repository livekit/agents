"""Shared helpers for local xAI realtime repro probes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_MODEL = os.environ.get("XAI_REALTIME_MODEL", "grok-voice-latest")


def require_xai_api_key() -> str:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        print("FAIL: set XAI_API_KEY in the environment", file=sys.stderr)
        raise SystemExit(1)
    return key


def pass_fail(ok: bool, message: str) -> int:
    label = "PASS" if ok else "FAIL"
    print(f"{label}: {message}")
    return 0 if ok else 1


async def open_realtime_ws(
    api_key: str,
) -> tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]:
    session = aiohttp.ClientSession()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "LiveKit Agents xAI repro",
    }
    try:
        ws = await session.ws_connect(XAI_REALTIME_URL, headers=headers)
    except Exception:
        await session.close()
        raise
    return session, ws


async def send_event(ws: aiohttp.ClientWebSocketResponse, event: dict[str, Any]) -> None:
    await ws.send_str(json.dumps(event))


async def recv_json(
    ws: aiohttp.ClientWebSocketResponse,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
    if msg.type == aiohttp.WSMsgType.TEXT:
        data = json.loads(msg.data)
        if not isinstance(data, dict):
            raise TypeError(f"expected JSON object, got {type(data).__name__}")
        return data
    if msg.type == aiohttp.WSMsgType.ERROR:
        raise RuntimeError(f"websocket error: {ws.exception()}")
    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
        raise RuntimeError("websocket closed")
    raise RuntimeError(f"unexpected websocket message type: {msg.type}")


async def wait_for_event(
    ws: aiohttp.ClientWebSocketResponse,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 45.0,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for matching event")
        event = await recv_json(ws, timeout=remaining)
        if on_event is not None:
            on_event(event)
        if predicate(event):
            return event


async def drain_until(
    ws: aiohttp.ClientWebSocketResponse,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 45.0,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _capture(event: dict[str, Any]) -> None:
        seen.append(event)
        if on_event is not None:
            on_event(event)

    await wait_for_event(ws, predicate, timeout=timeout, on_event=_capture)
    return seen


def event_error_message(event: dict[str, Any]) -> str | None:
    if event.get("type") != "error":
        return None
    err = event.get("error") or {}
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)
    return str(err)


def run_async(main: Callable[[], Awaitable[int]]) -> None:
    raise SystemExit(asyncio.run(main()))
