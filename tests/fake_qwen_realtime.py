"""In-process stand-ins for Alibaba Cloud Model Studio's realtime WebSocket endpoints.

The Qwen STT/TTS plugin is a protocol translator, so its tests drive it against a real
aiohttp WebSocket server that speaks the documented event flow rather than a mocked
socket. That keeps the framing, JSON and asyncio interleaving real; only the model
behind it is fake.

Protocol references:
  - https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-client-events
  - https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-server-events
  - https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-client-events
  - https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-server-events
"""

from __future__ import annotations

import base64
import json
from typing import Any

from aiohttp import WSMsgType, web

REALTIME_PATH = "/api-ws/v1/realtime"


class _FakeRealtimeServer:
    """Serves REALTIME_PATH on an ephemeral localhost port."""

    def __init__(self) -> None:
        self.client_events: list[dict[str, Any]] = []
        self.query: dict[str, str] = {}
        self.headers: dict[str, str] = {}
        self.url: str = ""
        self._script: list[dict[str, Any]] = []
        self._runner: web.AppRunner | None = None
        self._close_after_script = False
        self._ignore_finish = False
        # By default the script plays right after session.updated. A trigger holds it
        # until the client has sent `count` events of `event_type`, so a test can model
        # "the transcript arrives after the audio did".
        self._script_trigger: tuple[str, int] | None = None
        self._script_played = False

    def script(self, *events: dict[str, Any]) -> None:
        """Queue server events to emit once the session is configured."""
        self._script.extend(events)

    def emit_script_after(self, event_type: str, count: int = 1) -> None:
        """Play the script only once `count` client events of `event_type` have arrived."""
        self._script_trigger = (event_type, count)

    def close_after_script(self) -> None:
        """Drop the connection after the scripted events, without a finish handshake."""
        self._close_after_script = True

    def ignore_finish(self) -> None:
        """Never answer session.finish, holding the socket open (a wedged upstream)."""
        self._ignore_finish = True

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get(REALTIME_PATH, self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"ws://127.0.0.1:{port}{REALTIME_PATH}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def events_of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.client_events if e.get("type") == event_type]

    def _script_due(self, latest: str) -> bool:
        if self._script_played:
            return False
        if self._script_trigger is None:
            return latest == "session.update"
        event_type, count = self._script_trigger
        return latest == event_type and len(self.events_of_type(event_type)) >= count

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.query = dict(request.query)
        self.headers = dict(request.headers)
        await ws.send_json({"event_id": "srv_0", "type": "session.created", "session": {}})

        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            self.client_events.append(event)
            await self._on_client_event(ws, event)
            if ws.closed:
                break
        return ws

    async def _on_client_event(self, ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        raise NotImplementedError

    async def _emit_script(self, ws: web.WebSocketResponse) -> None:
        self._script_played = True
        for out in self._script:
            await ws.send_json(out)
        if self._close_after_script:
            await ws.close()

    async def _maybe_emit_script(self, ws: web.WebSocketResponse, latest: str) -> None:
        if self._script_due(latest):
            await self._emit_script(ws)


class FakeASRServer(_FakeRealtimeServer):
    """qwen3-asr-flash-realtime: audio in, transcription events out."""

    def __init__(self) -> None:
        super().__init__()
        self.audio = bytearray()

    async def _on_client_event(self, ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "session.update":
            await ws.send_json(
                {"event_id": "srv_1", "type": "session.updated", "session": event.get("session")}
            )
        elif kind == "input_audio_buffer.append":
            self.audio.extend(base64.b64decode(event["audio"]))
        elif kind == "input_audio_buffer.commit":
            await ws.send_json({"event_id": "srv_2", "type": "input_audio_buffer.committed"})
        elif kind == "session.finish":
            if self._ignore_finish:
                return
            await ws.send_json({"event_id": "srv_9", "type": "session.finished"})
            await ws.close()
            return
        await self._maybe_emit_script(ws, kind or "")


class FakeTTSServer(_FakeRealtimeServer):
    """qwen3-tts-flash-realtime: text in, base64 audio deltas out."""

    def __init__(self, *, audio_on_finish: list[bytes] | None = None) -> None:
        super().__init__()
        self.text: str = ""
        # Audio the server flushes only after session.finish, modelling the documented
        # "server flushes remaining audio then finishes" behaviour.
        self._audio_on_finish = audio_on_finish or []

    async def _on_client_event(self, ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "session.update":
            await ws.send_json(
                {"event_id": "srv_1", "type": "session.updated", "session": event.get("session")}
            )
            await self._emit_script(ws)
        elif kind == "input_text_buffer.append":
            self.text += event.get("text", "")
        elif kind == "input_text_buffer.commit":
            await ws.send_json({"event_id": "srv_2", "type": "input_text_buffer.committed"})
        elif kind == "session.finish":
            if self._ignore_finish:
                return
            for chunk in self._audio_on_finish:
                await ws.send_json(audio_delta(chunk))
            await ws.send_json({"event_id": "srv_9", "type": "session.finished"})
            await ws.close()


def pcm(sample_count: int, *, value: int = 1) -> bytes:
    """A little-endian 16-bit PCM payload of `sample_count` samples."""
    return int(value).to_bytes(2, "little", signed=True) * sample_count


def audio_delta(payload: bytes) -> dict[str, Any]:
    """A response.audio.delta carrying `payload` the way the server sends it."""
    return {
        "event_id": "srv_a",
        "type": "response.audio.delta",
        "delta": base64.b64encode(payload).decode(),
    }
