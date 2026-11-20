"""Hamsa TTS plugin tests: offline, against a local fake of ``WS /v1/realtime/ws``.

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import (
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
)

pytestmark = pytest.mark.unit

_AUDIO = b"\x10\x00" * 800  # 50 ms at 16 kHz


async def _serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}"


async def _hamsa_server(seen: dict[str, Any], *, reply: str = "audio") -> tuple[web.AppRunner, str]:
    """A fake Hamsa realtime socket.

    ``reply`` picks the behavior per request: ``"audio"`` (info, ack, audio, end), ``"error"``
    (an error frame), ``"close-auth"`` (close with 4001) or ``"silent"`` (never answer).
    """
    seen.setdefault("requests", [])
    seen.setdefault("connections", 0)

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        seen["connections"] += 1
        seen["key"] = request.headers.get("x-api-key")
        seen["query"] = dict(request.query)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps({"type": "info", "payload": {"message": "connected"}}))
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            seen["requests"].append(json.loads(msg.data))
            if reply == "audio":
                await ws.send_str(json.dumps({"type": "ack", "payload": {"message": "ok"}}))
                await ws.send_bytes(_AUDIO)
                await ws.send_bytes(_AUDIO)
                await ws.send_str(json.dumps({"type": "end", "payload": {"message": "done"}}))
            elif reply == "error":
                await ws.send_str(
                    json.dumps({"type": "error", "payload": {"message": "bad speaker"}})
                )
            elif reply == "close-auth":
                await ws.close(code=4001, message=b"Authentication failed")
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime/ws", ws_handler)
    return await _serve(app)


# --- configuration ---


def test_tts_requires_api_key():
    from livekit.plugins.hamsa import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_reads_api_key_from_env():
    from livekit.plugins.hamsa import TTS

    with patch.dict("os.environ", {"HAMSA_API_KEY": "env-key"}):
        assert TTS()._opts.api_key == "env-key"


def test_tts_properties():
    from livekit.plugins.hamsa import TTS

    t = TTS(api_key="k")
    assert t.model == "hamsa-tts"
    assert t.provider == "Hamsa"
    assert t.capabilities.streaming is True
    assert t.sample_rate == 16000
    assert t.num_channels == 1


def test_tts_rejects_bad_settings():
    from livekit.plugins.hamsa import TTS

    with pytest.raises(ValueError, match="dialect"):
        TTS(api_key="k", dialect="xx")
    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="k", sample_rate=24000)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expressiveness"):
        TTS(api_key="k", expressiveness=3)


def test_update_options_validates_and_keeps_settings_on_error():
    from livekit.plugins.hamsa import TTS

    t = TTS(api_key="k", speaker="Mariam", dialect="egy")
    t.update_options(speaker="Carla", dialect="leb", expressiveness=1.5)
    assert (t._opts.speaker, t._opts.dialect, t._opts.expressiveness) == ("Carla", "leb", 1.5)
    with pytest.raises(ValueError):
        t.update_options(speaker="Dalal", dialect="nope")
    assert (t._opts.speaker, t._opts.dialect) == ("Carla", "leb")


def test_build_request():
    from livekit.plugins.hamsa.tts import TTS, _build_request

    opts = TTS(api_key="k", speaker="Mariam", dialect="egy")._opts
    assert _build_request(opts, "أهلا") == {
        "type": "tts",
        "payload": {
            "text": "أهلا",
            "speaker": "Mariam",
            "dialect": "egy",
            "languageId": "ar",
            "mulaw": False,
            "sampleRate": "16k",
        },
    }

    opts = TTS(api_key="k", speaker="Emma", dialect="en", sample_rate=8000, expressiveness=0)._opts
    payload = _build_request(opts, "hi")["payload"]
    assert payload["languageId"] == "en"
    assert payload["sampleRate"] == "8k"
    assert payload["expressiveness"] == 0


# --- synthesize() ---


async def _synthesize(seen: dict[str, Any], *, reply: str = "audio", timeout: float = 5.0) -> Any:
    from livekit.plugins.hamsa import TTS

    runner, base = await _hamsa_server(seen, reply=reply)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            try:
                return await t.synthesize(
                    "مرحبا بك", conn_options=APIConnectOptions(max_retry=0, timeout=timeout)
                ).collect()
            finally:
                await t.aclose()
    finally:
        await runner.cleanup()


async def test_synthesize_sends_one_request_with_the_key_in_a_header():
    seen: dict[str, Any] = {}
    frame = await _synthesize(seen)
    assert seen["key"] == "sk-test"
    assert seen["query"] == {}  # never in the URL
    assert [r["payload"]["text"] for r in seen["requests"]] == ["مرحبا بك"]
    assert frame.sample_rate == 16000
    # Both audio chunks arrive (the emitter may pad the last frame with silence).
    assert frame.samples_per_channel >= 2 * len(_AUDIO) // 2


async def test_an_error_frame_is_a_non_retryable_error_without_provider_text():
    with pytest.raises(APIError) as exc:
        await _synthesize({}, reply="error")
    assert exc.value.retryable is False
    assert "bad speaker" not in str(exc.value)


async def test_close_4001_is_an_authentication_error():
    with pytest.raises(APIStatusError) as exc:
        await _synthesize({}, reply="close-auth")
    assert exc.value.status_code == 401
    assert exc.value.retryable is False


async def test_a_silent_server_times_out():
    with pytest.raises(APITimeoutError):
        await _synthesize({}, reply="silent", timeout=0.05)


# --- stream() ---


async def _synthesize_stream(pushes: list[str | None], seen: dict[str, Any]) -> list[Any]:
    from livekit.plugins.hamsa import TTS

    runner, base = await _hamsa_server(seen)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
            for p in pushes:
                stream.flush() if p is None else stream.push_text(p)
            stream.end_input()
            try:
                return [ev async for ev in stream]
            finally:
                await stream.aclose()
                await t.aclose()
    finally:
        await runner.cleanup()


async def test_llm_token_deltas_become_sentence_requests_on_one_connection():
    seen: dict[str, Any] = {}
    deltas = [
        "مرحباً بك في مركز",
        " الرياض الصحي، معك ليلى. كيف ",
        "يمكنني مساعدتك اليوم؟ العيادات ",
        "تعمل من السبت إلى الخميس.",
    ]
    events = await _synthesize_stream(deltas, seen)
    texts = [r["payload"]["text"] for r in seen["requests"]]
    assert 1 < len(texts) < len(deltas)
    assert " ".join(texts) == "".join(deltas).strip()
    assert seen["connections"] == 1
    assert sum(len(ev.frame.data) for ev in events) > 0
    assert len({ev.segment_id for ev in events}) == 1


async def test_a_stream_with_no_text_opens_no_connection():
    seen: dict[str, Any] = {}
    events = await _synthesize_stream([None], seen)
    assert events == []
    assert seen["connections"] == 0


async def test_the_connection_is_reused_across_requests():
    from livekit.plugins.hamsa import TTS

    seen: dict[str, Any] = {}
    runner, base = await _hamsa_server(seen)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            opts = APIConnectOptions(max_retry=0, timeout=5.0)
            await t.synthesize("أهلا", conn_options=opts).collect()
            await t.synthesize("وسهلا", conn_options=opts).collect()
            await t.aclose()
    finally:
        await runner.cleanup()
    assert len(seen["requests"]) == 2
    assert seen["connections"] == 1


async def test_prewarm_opens_a_connection_before_the_first_request():
    from livekit.plugins.hamsa import TTS

    seen: dict[str, Any] = {}
    runner, base = await _hamsa_server(seen)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            t.prewarm()
            for _ in range(50):
                if seen["connections"]:
                    break
                await asyncio.sleep(0.002)
            assert seen["connections"] == 1
            await t.synthesize("أهلا", conn_options=APIConnectOptions(max_retry=0)).collect()
            await t.aclose()
    finally:
        await runner.cleanup()
    assert seen["connections"] == 1


# --- request size limit and shutdown ---


def test_split_text_keeps_requests_within_the_limit():
    from livekit.plugins.hamsa.tts import _split_text

    assert _split_text("  أهلا وسهلا  ") == ["أهلا وسهلا"]
    assert _split_text("   ") == []
    words = ["كلمة"] * 30  # 4 chars + spaces
    pieces = _split_text(" ".join(words), limit=20)
    assert all(len(p) <= 20 for p in pieces)
    assert " ".join(pieces) == " ".join(words)
    assert _split_text("x" * 45, limit=20) == ["x" * 20, "x" * 20, "x" * 5]


async def test_a_streamed_sentence_over_the_limit_becomes_several_requests():
    seen: dict[str, Any] = {}
    long_sentence = " ".join(["مرحبا"] * 500) + "."  # about 3,000 characters, one sentence
    events = await _synthesize_stream([long_sentence], seen)
    texts = [r["payload"]["text"] for r in seen["requests"]]
    assert len(texts) >= 2
    assert all(len(t) <= 2000 for t in texts)
    assert " ".join(texts) == long_sentence
    assert seen["connections"] == 1
    assert len({ev.segment_id for ev in events}) == 1


async def test_synthesize_splits_text_over_the_limit():
    from livekit.plugins.hamsa import TTS

    seen: dict[str, Any] = {}
    runner, base = await _hamsa_server(seen)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            text = " ".join(["أهلا"] * 900)  # about 4,500 characters
            await t.synthesize(text, conn_options=APIConnectOptions(max_retry=0)).collect()
            await t.aclose()
    finally:
        await runner.cleanup()
    texts = [r["payload"]["text"] for r in seen["requests"]]
    assert len(texts) == 3
    assert all(len(x) <= 2000 for x in texts)
    assert seen["connections"] == 1


async def test_synthesize_with_empty_text_opens_no_connection():
    from livekit.plugins.hamsa import TTS

    seen: dict[str, Any] = {}
    runner, base = await _hamsa_server(seen)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            events = [ev async for ev in t.synthesize("  ")]
            await t.aclose()
    finally:
        await runner.cleanup()
    assert events == []
    assert seen["connections"] == 0


async def test_aclose_stops_an_in_flight_synthesize():
    from livekit.plugins.hamsa import TTS

    seen: dict[str, Any] = {}
    runner, base = await _hamsa_server(seen, reply="silent")
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            stream = t.synthesize("أهلا", conn_options=APIConnectOptions(max_retry=3, timeout=30))
            task = asyncio.create_task(stream.collect())
            for _ in range(100):
                if seen["requests"]:
                    break
                await asyncio.sleep(0.001)
            await t.aclose()
            await asyncio.gather(task, return_exceptions=True)
            assert task.done()
    finally:
        await runner.cleanup()
    assert seen["connections"] == 1  # no retry reopened a connection after shutdown
