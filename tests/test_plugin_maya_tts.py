from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIError
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import maya
from livekit.plugins.maya import tts as tts_module

pytestmark = pytest.mark.plugin("maya")

SAMPLE_RATE = 24000


def _pcm(num_samples: int) -> bytes:
    return b"\x01\x00" * num_samples


class _MayaServer:
    """A stand-in for Maya's v2 websocket.

    Speaks enough of the protocol to exercise the handshake, the turn model and
    barge-in: sentences accumulate under a ``context_id`` and the turn is
    terminated by exactly one ``end`` or ``cancelled``.
    """

    def __init__(
        self,
        *,
        reject_start: bool = False,
        error_on_text: str | None = None,
        samples_per_text: int = 480,
        never_ends: bool = False,
    ) -> None:
        self.reject_start = reject_start
        self.error_on_text = error_on_text
        self.samples_per_text = samples_per_text
        self.never_ends = never_ends

        self.start_frames: list[dict[str, Any]] = []
        self.text_frames: list[dict[str, Any]] = []
        self.cancels: list[dict[str, Any]] = []
        self.connections = 0

        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None
        self.base_url = ""

    async def __aenter__(self) -> _MayaServer:
        app = web.Application()
        app.router.add_get("/v1/tts/stream", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._runner is not None and self._session is not None
        await self._session.close()
        await self._runner.cleanup()

    def tts(self, **kwargs: Any) -> maya.TTS:
        return maya.TTS(api_key="k", base_url=self.base_url, http_session=self._session, **kwargs)

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1
        started = False
        open_turns: set[str] = set()

        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            frame = json.loads(msg.data)
            kind = frame.get("type")

            if kind == "start":
                self.start_frames.append(frame)
                if self.reject_start:
                    await ws.send_str(json.dumps({"type": "error", "error": "invalid 'voice'"}))
                    continue
                started = True
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "metadata",
                            "sample_rate": SAMPLE_RATE,
                            "channels": 1,
                            "encoding": "pcm_s16le",
                            "session_id": "test-session",
                        }
                    )
                )
                continue

            if not started:
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "error",
                            "error": 'v2 frames require {"type":"start","v2":true} first',
                        }
                    )
                )
                continue

            if kind == "text":
                self.text_frames.append(frame)
                ctx = frame["context_id"]
                text = frame.get("text", "")

                if self.error_on_text is not None and self.error_on_text in text:
                    await ws.send_str(
                        json.dumps({"type": "error", "error": "origin_error", "context_id": ctx})
                    )
                    continue

                if text:
                    open_turns.add(ctx)
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "audio",
                                "context_id": ctx,
                                "audio": base64.b64encode(_pcm(self.samples_per_text)).decode(),
                            }
                        )
                    )

                if not frame.get("continue", False) and ctx in open_turns:
                    if self.never_ends:
                        continue  # still generating, as during a long turn
                    open_turns.discard(ctx)
                    await ws.send_str(json.dumps({"type": "end", "context_id": ctx}))
            elif kind == "cancel":
                self.cancels.append(frame)
                ctx = frame.get("context_id")
                if ctx in open_turns:
                    open_turns.discard(ctx)
                    await ws.send_str(json.dumps({"type": "cancelled", "context_id": ctx}))

        return ws


async def _collect(stream: Any) -> bytes:
    audio = bytearray()
    async for ev in stream:
        audio += bytes(ev.frame.data)
    return bytes(audio)


_NO_RETRY = APIConnectOptions(max_retry=0, timeout=10.0)

# The default tokenizer breaks on western punctuation, so these exercise the
# turn model as three separate sentences.
_THREE_SENTENCES = [
    "There once was a tortoise living deep in the forest.",
    "He walked very slowly but he never stopped moving forward.",
    "In the end it was the tortoise who won that race.",
]


async def _synthesize(tts: maya.TTS, text: str) -> bytes:
    return await _collect(tts.synthesize(text, conn_options=_NO_RETRY))


async def _stream(tts: maya.TTS, sentences: list[str]) -> bytes:
    stream = tts.stream()
    for sentence in sentences:
        stream.push_text(sentence + " ")
    stream.end_input()
    try:
        return await _collect(stream)
    finally:
        await stream.aclose()


# ---------------------------------------------------------------- construction


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAYA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MAYA_API_KEY"):
        maya.TTS()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAYA_API_KEY", "from-env")
    assert maya.TTS()._opts.api_key == "from-env"


def test_api_key_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAYA_API_KEY", "from-env")
    assert maya.TTS(api_key="explicit")._opts.api_key == "explicit"


def test_provider_and_default_model() -> None:
    tts = maya.TTS(api_key="k")
    assert tts.provider == "Maya"
    assert tts.model == "Maya 2 Native"


def test_supports_streaming() -> None:
    assert maya.TTS(api_key="k").capabilities.streaming


def test_native_sample_rate() -> None:
    assert maya.TTS(api_key="k").sample_rate == SAMPLE_RATE


def test_has_a_default_voice() -> None:
    assert maya.TTS(api_key="k")._opts.voice == tts_module.DEFAULT_VOICE


def test_any_voice_name_is_passed_through() -> None:
    # Voices are not enumerated here: Maya validates the name, so one added
    # later works without a release.
    frame = maya.TTS(api_key="k", voice="SomeNewVoice")._opts.start_frame()
    assert frame["voice"] == "SomeNewVoice"


def test_language_is_unset_by_default() -> None:
    # No language means Maya auto-detects, which suits code-switched text.
    assert maya.TTS(api_key="k")._opts.language is NOT_GIVEN


# ------------------------------------------------------------------ handshake


def test_start_frame_selects_v2_and_carries_settings() -> None:
    tts = maya.TTS(api_key="k", voice="Arjun", language="hi", model="Maya 2 Native Emotional")
    assert tts._opts.start_frame() == {
        "type": "start",
        "v2": True,
        "voice": "Arjun",
        "language": "hi",
        "model": "Maya 2 Native Emotional",
    }


def test_start_frame_omits_unset_fields() -> None:
    frame = maya.TTS(api_key="k")._opts.start_frame()
    assert frame == {"type": "start", "v2": True, "voice": "Ananya"}


def test_unverified_language_is_still_sent() -> None:
    # The literal type documents what is verified; Maya validates the value, so
    # a language it adds later works without a release here.
    frame = maya.TTS(api_key="k", language="as")._opts.start_frame()
    assert frame["language"] == "as"


def test_ws_url_derives_from_base_url() -> None:
    tts = maya.TTS(api_key="k")
    assert tts._opts.get_ws_url() == "wss://tts.mayaresearch.ai/v1/tts/stream"


def test_ws_url_honours_a_self_hosted_base_url() -> None:
    tts = maya.TTS(api_key="k", base_url="http://localhost:8080")
    assert tts._opts.get_ws_url() == "ws://localhost:8080/v1/tts/stream"


# -------------------------------------------------------------------- options


def test_update_options() -> None:
    tts = maya.TTS(api_key="k")
    tts.update_options(voice="Arjun", language="ta")
    assert tts._opts.voice == "Arjun"
    assert tts._opts.language == "ta"


def test_update_options_preserves_unset_fields() -> None:
    tts = maya.TTS(api_key="k", voice="Arjun", language="hi")
    tts.update_options(voice="Ananya")
    assert tts._opts.voice == "Ananya"
    assert tts._opts.language == "hi"


# ------------------------------------------------------------------- protocol


async def test_synthesize_runs_one_turn() -> None:
    async with _MayaServer() as server:
        tts = server.tts()
        audio = await _synthesize(tts, "नमस्ते।")
        await tts.aclose()

    # the emitter pads its final frame, so the payload is a prefix of the output
    assert audio.startswith(_pcm(480))
    assert server.start_frames[0]["v2"] is True
    assert len(server.text_frames) == 1
    assert server.text_frames[0]["continue"] is False


async def test_a_turn_shares_one_context_and_ends_once() -> None:
    async with _MayaServer() as server:
        tts = server.tts()
        audio = await _stream(tts, _THREE_SENTENCES)
        await tts.aclose()

    sentences = [f for f in server.text_frames if f.get("text")]
    closer = [f for f in server.text_frames if not f.get("text")]

    assert len(sentences) == 3
    assert len({f["context_id"] for f in server.text_frames}) == 1
    assert all(f["continue"] for f in sentences)
    assert len(closer) == 1 and closer[0]["continue"] is False
    assert audio.startswith(_pcm(480) * 3)


async def test_turns_reuse_one_connection() -> None:
    async with _MayaServer() as server:
        tts = server.tts()
        await _stream(tts, ["पहला।"])
        await _stream(tts, ["दूसरा।"])
        await tts.aclose()

    assert server.connections == 1
    assert len({f["context_id"] for f in server.text_frames}) == 2


async def test_interruption_cancels_the_turn() -> None:
    # A turn still generating when the user barges in has to be dropped, or the
    # pooled connection keeps streaming it into whatever comes next.
    async with _MayaServer(never_ends=True) as server:
        tts = server.tts()
        stream = tts.stream()
        for sentence in _THREE_SENTENCES:
            stream.push_text(sentence + " ")
        stream.end_input()

        async def _drain() -> None:
            with contextlib.suppress(Exception):
                async for _ in stream:
                    pass

        task = asyncio.create_task(_drain())
        await asyncio.sleep(0.3)
        await stream.aclose()
        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await tts.aclose()

    assert len(server.cancels) == 1
    assert server.cancels[0]["context_id"] == server.text_frames[0]["context_id"]


async def test_a_finished_turn_is_not_cancelled() -> None:
    async with _MayaServer() as server:
        tts = server.tts()
        await _stream(tts, ["नमस्ते।"])
        await tts.aclose()

    assert server.cancels == []


async def test_rejected_handshake_raises() -> None:
    async with _MayaServer(reject_start=True) as server:
        tts = server.tts(voice="Nobody")
        with pytest.raises(APIError, match="rejected the connection settings"):
            await _synthesize(tts, "नमस्ते।")
        await tts.aclose()


async def test_error_frame_raises() -> None:
    async with _MayaServer(error_on_text="बुरा") as server:
        tts = server.tts()
        with pytest.raises(APIError, match="origin_error"):
            await _synthesize(tts, "बुरा वाक्य।")
        await tts.aclose()


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAYA_BASE_URL", "http://localhost:9999")
    assert maya.TTS(api_key="k")._opts.get_ws_url() == "ws://localhost:9999/v1/tts/stream"


def test_base_url_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAYA_BASE_URL", "http://from-env:1")
    tts = maya.TTS(api_key="k", base_url="http://explicit:2")
    assert tts._opts.get_ws_url() == "ws://explicit:2/v1/tts/stream"
