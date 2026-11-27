"""Deepdub TTS and STT plugin tests: offline, against a local fake of the API (``WS /ws`` for
streaming text-to-speech, ``WS /transcribe-websocket`` for speech-to-text).

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import patch

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
)

pytestmark = pytest.mark.unit

_CONN = APIConnectOptions(max_retry=0, timeout=5)
_PCM_CHUNK = b"\x01\x00" * 2400  # 100 ms of 24 kHz s16le


def _frame(ms: int = 100, rate: int = 16000) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(b"\x10\x00" * n, sample_rate=rate, num_channels=1, samples_per_channel=n)


async def _serve(
    path: str, handler: Callable[[web.Request], Awaitable[Any]]
) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}"


def _audio(generation_id: str, index: int, **flags: Any) -> bytes:
    """An audio chunk, sent as a binary frame of JSON the way Deepdub sends it."""
    message = {"generationId": generation_id, "index": index, "isFinished": False, **flags}
    if not flags.get("isCancelled"):
        message["data"] = base64.b64encode(_PCM_CHUNK).decode()
    return json.dumps(message).encode()


class _FakeTTSServer:
    """Speaks each turn as two generations once ``end-stream`` arrives."""

    def __init__(
        self,
        *,
        hold_turn: bool = False,
        mute: bool = False,
        error: dict[str, Any] | None = None,
    ) -> None:
        self.messages: list[dict[str, Any]] = []
        self.api_keys: list[str | None] = []
        self.turn_started = asyncio.Event()
        self._hold_turn = hold_turn
        self._mute = mute
        self._error = error

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        self.api_keys.append(request.headers.get("x-api-key"))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps({"action": "status", "message": "connected"}))
        turn = 0
        async for msg in ws:
            message = json.loads(msg.data)
            self.messages.append(message)
            action = message["action"]
            if action == "stream-text":
                self.turn_started.set()
                if self._hold_turn:
                    await ws.send_bytes(_audio(f"held-{turn}", 0))
            elif action == "end-stream":
                if self._mute:
                    continue
                if self._error is not None:
                    await ws.send_str(json.dumps(self._error))
                    continue
                turn += 1
                await ws.send_bytes(_audio(f"t{turn}-a", 0))
                await ws.send_bytes(_audio(f"t{turn}-a", 1, isFinished=True))
                await ws.send_str(json.dumps({"action": "pong"}))
                await ws.send_bytes(_audio(f"t{turn}-b", 0))
                await ws.send_bytes(_audio(f"t{turn}-b", 1, isFinished=True, isFinal=True))
            elif action == "cancel":
                await ws.send_bytes(_audio(f"held-{turn}", 1, isFinished=True, isCancelled=True))
                await ws.send_str(json.dumps({"action": "status", "message": "cancelled"}))
        return ws

    def actions(self) -> list[str]:
        return [m["action"] for m in self.messages]


def test_tts_requires_api_key():
    from livekit.plugins.deepdub import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_properties_and_validation():
    from livekit.plugins.deepdub import TTS

    t = TTS(api_key="dd-test")
    assert t.model == "dd-etts-4.0"
    assert t.provider == "Deepdub"
    assert t.sample_rate == 24000
    assert t.capabilities.streaming is True
    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="dd-test", sample_rate=12345)


def test_tts_locale_normalization():
    from livekit.plugins.deepdub.tts import _resolve_locale

    assert _resolve_locale("en-US") == "en-US"
    assert _resolve_locale("en_us") == "en-US"
    assert _resolve_locale("he") == "he-IL"
    assert _resolve_locale("english") == "en-US"
    assert _resolve_locale("es-MX") == "es-MX"


async def test_tts_stream_forwards_tokens_and_ends_the_turn_on_flush():
    from livekit.plugins.deepdub import TTS

    server = _FakeTTSServer()
    runner, base = await _serve("/ws", server.handler)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(
                api_key="dd-test",
                base_url=base,
                locale="he",
                voice_prompt_id="voice-1",
                sample_rate=24000,
                enable_logging=False,
                http_session=session,
            )
            replies: list[list[Any]] = []
            for tokens in ([" ", "Hello ", "there."], ["Second ", "reply."]):
                stream = tts.stream(conn_options=_CONN)
                for token in tokens:
                    stream.push_text(token)
                stream.flush()
                stream.end_input()
                replies.append([ev async for ev in stream])
            await tts.aclose()
    finally:
        await runner.cleanup()

    # Both replies ran on one pooled connection, each configured and closed as a turn.
    assert server.api_keys == ["dd-test"]
    assert server.actions() == [
        "stream-config",
        "stream-text",
        "stream-text",
        "end-stream",
        "stream-config",
        "stream-text",
        "stream-text",
        "end-stream",
    ]
    assert server.messages[0]["data"] == {
        "model": "dd-etts-4.0",
        "voicePromptId": "voice-1",
        "locale": "he-IL",
        "format": "s16le",
        "sampleRate": 24000,
        "enableLogging": False,
    }
    assert [m["data"]["text"] for m in server.messages if m["action"] == "stream-text"] == [
        "Hello ",
        "there.",
        "Second ",
        "reply.",
    ]
    # Each turn is two generations of two 100 ms chunks.
    for events in replies:
        assert sum(ev.frame.duration for ev in events) == pytest.approx(0.4, abs=0.03)
        assert len({ev.segment_id for ev in events}) == 1


async def test_tts_stream_without_text_sends_no_turn():
    from livekit.plugins.deepdub import TTS

    server = _FakeTTSServer()
    runner, base = await _serve("/ws", server.handler)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(api_key="dd-test", base_url=base, http_session=session)
            stream = tts.stream(conn_options=_CONN)
            stream.push_text("  ")
            stream.flush()
            stream.end_input()
            events = [ev async for ev in stream]
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert events == []
    assert server.actions() == ["stream-config"]


async def test_tts_synthesize_speaks_one_turn():
    from livekit.plugins.deepdub import TTS

    server = _FakeTTSServer()
    runner, base = await _serve("/ws", server.handler)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(api_key="dd-test", base_url=base, http_session=session)
            frames = [ev.frame async for ev in tts.synthesize("Hello world.", conn_options=_CONN)]
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert server.actions() == ["stream-config", "stream-text", "end-stream"]
    assert server.messages[1]["data"]["text"] == "Hello world."
    assert sum(f.duration for f in frames) == pytest.approx(0.4, abs=0.03)


async def test_tts_session_error_is_not_retryable():
    from livekit.plugins.deepdub import TTS

    error = {"action": "error", "message": "queue route not found for model: x", "time": 1}
    server = _FakeTTSServer(error=error)
    runner, base = await _serve("/ws", server.handler)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(api_key="dd-test", base_url=base, http_session=session)
            with pytest.raises(APIStatusError) as exc_info:
                async for _ in tts.synthesize("Hello.", conn_options=_CONN):
                    pass
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


async def test_tts_closing_an_interrupted_stream_sends_cancel():
    from livekit.plugins.deepdub import TTS

    server = _FakeTTSServer(hold_turn=True)
    runner, base = await _serve("/ws", server.handler)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(api_key="dd-test", base_url=base, http_session=session)
            stream = tts.stream(conn_options=_CONN)
            stream.push_text("A long reply the user interrupts.")
            await asyncio.wait_for(server.turn_started.wait(), 5)
            await stream.aclose()
            for _ in range(100):
                if "cancel" in server.actions():
                    break
                await asyncio.sleep(0.01)
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert server.actions() == ["stream-config", "stream-text", "cancel"]


async def test_tts_stream_times_out_when_the_turn_never_finishes():
    from livekit.plugins.deepdub import TTS

    server = _FakeTTSServer(mute=True)
    runner, base = await _serve("/ws", server.handler)
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(api_key="dd-test", base_url=base, http_session=session)
            stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
            stream.push_text("Hello.")
            # The receiver is already waiting, untimed, when end-stream goes out.
            await asyncio.wait_for(server.turn_started.wait(), 5)
            stream.end_input()
            with pytest.raises(APITimeoutError):
                await asyncio.wait_for(_drain(stream), 5)
            for _ in range(100):
                if "cancel" in server.actions():
                    break
                await asyncio.sleep(0.01)
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert server.actions() == ["stream-config", "stream-text", "end-stream", "cancel"]


def _token(text: str, start: int | None, end: int | None, *, final: bool, lang: str = "en") -> dict:
    token: dict[str, Any] = {"text": text, "is_final": final, "language": lang}
    if start is not None:
        token.update(start_ms=start, end_ms=end, confidence=0.9)
    return token


def _stt_handler(
    seen: dict[str, Any],
    script: Callable[[web.WebSocketResponse, dict[str, Any]], Awaitable[None]],
) -> Callable[[web.Request], Awaitable[web.WebSocketResponse]]:
    async def handler(request: web.Request) -> web.WebSocketResponse:
        seen["auth"] = request.headers.get("Authorization")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        seen["config"] = json.loads((await ws.receive()).data)
        seen["audio"] = 0
        seen["control"] = []
        await script(ws, seen)
        return ws

    return handler


async def _collect(
    stream: stt.SpeechStream, frames: int, *, flush: bool = False
) -> list[stt.SpeechEvent]:
    for _ in range(frames):
        stream.push_frame(_frame())
    if flush:
        stream.flush()
    stream.end_input()
    return [ev async for ev in stream]


async def test_stt_stream_emits_interim_preflight_and_final():
    from livekit.plugins.deepdub import STT

    async def script(ws: web.WebSocketResponse, seen: dict[str, Any]) -> None:
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["audio"] += len(msg.data)
                if not sent:
                    sent = True
                    await ws.send_str(
                        json.dumps({"tokens": [_token("Hel", 400, 480, final=False)]})
                    )
                    await ws.send_str(
                        json.dumps(
                            {
                                "tokens": [_token("Hello,", 400, 900, final=True)],
                                "total_audio_proc_ms": 1000,
                            }
                        )
                    )
                    await ws.send_str(
                        json.dumps(
                            {
                                "tokens": [
                                    _token(" world.", 1000, 1500, final=True),
                                    _token("<end>", None, None, final=True),
                                ],
                                "total_audio_proc_ms": 2000,
                            }
                        )
                    )
            elif msg.type == aiohttp.WSMsgType.TEXT and msg.data == "":
                await ws.send_str(json.dumps({"tokens": [], "finished": True}))
                await ws.close()

    seen: dict[str, Any] = {}
    runner, base = await _serve("/transcribe-websocket", _stt_handler(seen, script))
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(
                api_key="dd-test",
                base_url=base,
                language="en-US",
                context_terms=["Deepdub"],
                max_endpoint_delay_ms=700,
                http_session=session,
            )
            events = await _collect(s.stream(conn_options=_CONN), 5)
    finally:
        await runner.cleanup()

    assert seen["auth"] == "Bearer dd-test"
    assert seen["config"] == {
        "model": "stt-egf",
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "enable_endpoint_detection": True,
        "language_hints": ["en"],
        "language_hints_strict": False,
        "max_endpoint_delay_ms": 700,
        "context": {"terms": ["Deepdub"]},
    }
    assert seen["audio"] == 5 * 3200  # 500 ms, sent in 100 ms chunks

    types = [ev.type for ev in events if ev.type != stt.SpeechEventType.RECOGNITION_USAGE]
    assert types == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = next(ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    data = final.alternatives[0]
    assert data.text == "Hello, world."
    assert data.language == "en"
    assert data.confidence == pytest.approx(0.9)
    assert data.end_time - data.start_time == pytest.approx(1.1)
    usage = [ev.recognition_usage for ev in events if ev.recognition_usage is not None]
    assert sum(u.audio_duration for u in usage) == pytest.approx(2.0)


async def test_stt_flush_finalizes_and_closes_the_utterance():
    from livekit.plugins.deepdub import STT

    async def script(ws: web.WebSocketResponse, seen: dict[str, Any]) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and msg.data:
                seen["control"].append(json.loads(msg.data)["type"])
                if json.loads(msg.data)["type"] == "finalize":
                    await ws.send_str(
                        json.dumps(
                            {
                                "tokens": [
                                    _token("שלום", 100, 400, final=True, lang="he"),
                                    _token("<fin>", None, None, final=True),
                                ]
                            }
                        )
                    )
            elif msg.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(json.dumps({"tokens": [], "finished": True}))
                await ws.close()

    seen: dict[str, Any] = {}
    runner, base = await _serve("/transcribe-websocket", _stt_handler(seen, script))
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="dd-test", base_url=base, http_session=session)
            events = await _collect(s.stream(conn_options=_CONN), 2, flush=True)
    finally:
        await runner.cleanup()

    assert "language_hints" not in seen["config"]
    assert seen["control"] == ["finalize"]
    # The first words arrive together with <fin>, so speech still starts before it ends.
    types = [ev.type for ev in events if ev.type != stt.SpeechEventType.RECOGNITION_USAGE]
    assert types == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = events[types.index(stt.SpeechEventType.FINAL_TRANSCRIPT)]
    assert final.alternatives[0].text == "שלום"
    assert final.alternatives[0].language == "he"


async def test_stt_downmixes_stereo_input():
    from livekit.plugins.deepdub import STT

    async def script(ws: web.WebSocketResponse, seen: dict[str, Any]) -> None:
        seen["pcm"] = b""
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["pcm"] += msg.data
            elif msg.type == aiohttp.WSMsgType.TEXT and msg.data == "":
                await ws.send_str(json.dumps({"tokens": [], "finished": True}))
                await ws.close()

    n = 1600  # 100 ms per channel at 16 kHz
    stereo = rtc.AudioFrame(
        b"\x10\x00\x30\x00" * n, sample_rate=16000, num_channels=2, samples_per_channel=n
    )
    seen: dict[str, Any] = {}
    runner, base = await _serve("/transcribe-websocket", _stt_handler(seen, script))
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="dd-test", base_url=base, http_session=session)
            stream = s.stream(conn_options=_CONN)
            stream.push_frame(stereo)
            stream.end_input()
            async for _ in stream:
                pass
    finally:
        await runner.cleanup()

    assert seen["pcm"] == b"\x20\x00" * n


async def test_stt_disconnect_mid_session_raises():
    from livekit.plugins.deepdub import STT

    async def script(ws: web.WebSocketResponse, seen: dict[str, Any]) -> None:
        await ws.receive()
        await ws.close()

    seen: dict[str, Any] = {}
    runner, base = await _serve("/transcribe-websocket", _stt_handler(seen, script))
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="dd-test", base_url=base, http_session=session)
            stream = s.stream(conn_options=_CONN)
            stream.push_frame(_frame())
            with pytest.raises(APIConnectionError):
                await asyncio.wait_for(_drain(stream), 5)
            await stream.aclose()
    finally:
        await runner.cleanup()


async def _drain(stream: AsyncIterator[Any]) -> None:
    async for _ in stream:
        pass


async def test_stt_endpoint_without_final_words_still_ends_speech():
    from livekit.plugins.deepdub import STT

    async def script(ws: web.WebSocketResponse, seen: dict[str, Any]) -> None:
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and not sent:
                sent = True
                await ws.send_str(json.dumps({"tokens": [_token("uh", 100, 200, final=False)]}))
                await ws.send_str(json.dumps({"tokens": [_token("<end>", None, None, final=True)]}))
            elif msg.type == aiohttp.WSMsgType.TEXT and msg.data == "":
                await ws.send_str(json.dumps({"tokens": [], "finished": True}))
                await ws.close()

    seen: dict[str, Any] = {}
    runner, base = await _serve("/transcribe-websocket", _stt_handler(seen, script))
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="dd-test", base_url=base, http_session=session)
            events = await _collect(s.stream(conn_options=_CONN), 2)
    finally:
        await runner.cleanup()

    types = [ev.type for ev in events if ev.type != stt.SpeechEventType.RECOGNITION_USAGE]
    assert types == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_stt_server_error_raises_status_error():
    from livekit.plugins.deepdub import STT

    async def script(ws: web.WebSocketResponse, seen: dict[str, Any]) -> None:
        await ws.send_str(
            json.dumps(
                {"error_code": 401, "error_type": "unauthorized", "error_message": "invalid key"}
            )
        )
        await ws.close()

    seen: dict[str, Any] = {}
    runner, base = await _serve("/transcribe-websocket", _stt_handler(seen, script))
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="dd-test", base_url=base, http_session=session)
            stream = s.stream(conn_options=_CONN)
            stream.push_frame(_frame())
            with pytest.raises(APIStatusError) as exc_info:
                async for _ in stream:
                    pass
    finally:
        await runner.cleanup()

    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False


def test_stt_options_validation():
    from livekit.plugins.deepdub import STT

    s = STT(api_key="dd-test")
    assert s.model == "stt-egf"
    assert s.provider == "Deepdub"
    assert s.capabilities.offline_recognize is False
    assert s.capabilities.manual_flush is True
    with pytest.raises(ValueError, match="supports"):
        STT(api_key="dd-test", language="fr")
    with pytest.raises(ValueError, match="max_endpoint_delay_ms"):
        STT(api_key="dd-test", max_endpoint_delay_ms=100)
