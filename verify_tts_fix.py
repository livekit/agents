"""Verify the TTS close/flush coordination fix at the WebSocket wire level.

Reproduces the reviewer's exact scenario:  stream.push_text("hello"); stream.end_input()
and asserts that close_context is sent only AFTER flush_completed (and after all audio).
"""

import asyncio
import json

import websockets

from livekit.plugins._60db import TTS

events: list[tuple[str, str]] = []


def classify(data) -> str:
    if not isinstance(data, str):
        return "binary-audio?"
    try:
        msg = json.loads(data)
    except Exception:
        return "raw"
    if "audio_chunk" in msg:
        return "audio_chunk"
    if msg.get("flush_completed"):
        return "flush_completed"
    if msg.get("context_closed"):
        return "context_closed"
    if msg.get("connection_established"):
        return "connection_established"
    if "error" in msg or msg.get("type") == "error":
        return f"SERVER-ERROR: {json.dumps(msg)[:200]}"
    if msg.get("context_created"):
        return "context_created"
    if "create_context" in msg:
        return "SEND create_context"
    if "send_text" in msg:
        return "SEND send_text"
    if "flush_context" in msg:
        return "SEND flush_context"
    if "close_context" in msg:
        return "SEND close_context"
    return f"other:{sorted(msg)[:2]}"


class WsProxy:
    def __init__(self, ws) -> None:
        self._ws = ws

    async def send(self, data):
        events.append(("SEND", classify(data)))
        return await self._ws.send(data)

    async def recv(self):
        data = await self._ws.recv()
        events.append(("RECV", classify(data)))
        return data

    def __getattr__(self, name):
        return getattr(self._ws, name)


class ConnProxy:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return WsProxy(await self._conn.__aenter__())

    async def __aexit__(self, *args):
        return await self._conn.__aexit__(*args)


_orig_connect = websockets.connect
websockets.connect = lambda *a, **kw: ConnProxy(_orig_connect(*a, **kw))


async def main() -> None:
    tts = TTS()

    # --- reviewer's scenario (single push + end_input) ---
    # ("hello" alone makes the provider return empty_audio, so use a real phrase)
    audio_bytes = 0
    async with tts.stream() as stream:
        stream.push_text("Hello there, this is a coordination check.")
        stream.end_input()
        async for ev in stream:
            audio_bytes += len(bytes(ev.frame.data))

    print("wire trace:")
    for direction, kind in events:
        print(f"  {direction:4} {kind}")

    send_idx = {e[1]: i for i, e in enumerate(events) if e[0] == "SEND"}
    recv_idx = {e[1]: i for i, e in enumerate(events) if e[0] == "RECV"}

    assert "SEND send_text" in send_idx, "text was never sent"
    assert "SEND flush_context" in send_idx, "flush was never sent"
    assert "SEND close_context" in send_idx, "close was never sent"
    assert "flush_completed" in recv_idx, "server never confirmed the flush"
    assert "context_closed" in recv_idx, "context never closed"
    assert "audio_chunk" in recv_idx, "no audio received"

    # the core assertions from the review:
    assert send_idx["SEND flush_context"] < recv_idx["flush_completed"], (
        "close ran before the flush was even answered"
    )
    assert recv_idx["flush_completed"] < send_idx["SEND close_context"], (
        "BUG: close_context sent before flush_completed — truncated speech risk"
    )
    assert recv_idx["audio_chunk"] < send_idx["SEND close_context"], (
        "BUG: context closed before any audio arrived"
    )
    assert recv_idx["context_closed"] > send_idx["SEND close_context"]

    print(f"\naudio received: {audio_bytes} bytes")
    print("CONFIRMED: close_context is sent only after flush_completed (and after audio)")


try:
    asyncio.run(main())
except Exception:
    print("\nwire trace (at failure):")
    for direction, kind in events:
        print(f"  {direction:4} {kind}")
    raise
