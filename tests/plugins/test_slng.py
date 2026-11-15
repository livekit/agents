"""SLNG plugin: TTS streaming text privacy, per-connection init and option updates, and STT
reading only the bridge's messages it knows, against a local peer."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, http, stt, tokenize
from livekit.plugins.slng import STT, TTS
from tests.fakes import until

PCM = b"\x00\x01" * 2400
BRIDGE = "/v1/bridges/unmute/tts/deepgram/aura:2"
DOWN_BRIDGE = "/v1/bridges/unmute/tts/cartesia/sonic:3"


@dataclass
class Slng:
    """The local SLNG bridge's endpoint and each socket's init config."""

    endpoint: str = ""
    down_endpoint: str = ""  # a bridge that answers 503
    configs: list[dict[str, object]] = field(default_factory=list)


@pytest.fixture
async def slng() -> AsyncIterator[Slng]:
    """A local SLNG bridge: answers each flush with audio; ``configs`` holds each socket's init config config."""
    peer = Slng()

    async def speak(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        peer.configs.append(json.loads((await socket.receive()).data)["config"])
        await socket.send_json({"type": "ready"})
        async for message in socket:
            if json.loads(message.data).get("type") == "flush":
                await socket.send_bytes(PCM)
                await socket.send_json({"type": "audio_end"})
        return socket

    async def unavailable(request: web.Request) -> web.Response:
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get(BRIDGE, speak)
    app.router.add_get(DOWN_BRIDGE, unavailable)
    async with TestServer(app) as server, http.open():
        peer.endpoint = str(server.make_url(BRIDGE).with_scheme("ws"))
        peer.down_endpoint = str(server.make_url(DOWN_BRIDGE).with_scheme("ws"))
        yield peer


async def _say(tts: TTS, text: str) -> bytes:
    async with tts.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
        stream.push_text(text)
        stream.end_input()
        return b"".join([bytes(event.frame.data) async for event in stream])


async def test_streaming_text_is_kept_out_of_log_messages(
    slng: Slng, caplog: pytest.LogCaptureFixture
) -> None:
    """Text being synthesized never appears in the plugin's log lines."""
    tts = TTS(api_key="key", model=slng.endpoint, voice="voice")
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.slng")

    audio = await _say(tts, "CONFIDENTIALACCOUNT")
    await tts.aclose()

    assert audio == PCM
    assert all("CONFIDENTIALACCOUNT" not in record.getMessage() for record in caplog.records)


async def test_speed_update_opens_a_connection_with_current_init(slng: Slng) -> None:
    """A speed change opens a socket whose init carries it. https://github.com/livekit/agents/issues/6174"""
    tts = TTS(api_key="key", model=slng.endpoint, voice="voice")

    for speed in (1.0, 1.2):
        tts.update_options(speed=speed)
        assert await _say(tts, "Hello.") == PCM
    await tts.aclose()

    assert [config["speed"] for config in slng.configs] == [1.0, 1.2]


async def test_connection_chain_fails_over_without_losing_audio(slng: Slng) -> None:
    """An unavailable first bridge hands the turn to the next one, which plays all of it."""
    # Catches: the chain giving up on the first bridge's 503, or the fallback leg dropping
    # the audio of the turn it took over.
    tts = TTS(api_key="key", connections=[slng.down_endpoint, slng.endpoint], voice="voice")

    options = APIConnectOptions(max_retry=0, retry_interval=0)
    async with tts.stream(conn_options=options) as stream:
        stream.push_text("Hello.")
        stream.end_input()
        audio = b"".join([bytes(event.frame.data) async for event in stream])
    await tts.aclose()

    assert audio == PCM
    assert len(slng.configs) == 1


class _Splitter(tokenize.SentenceTokenizer):
    """The default split, keeping each text it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        self.texts.append(text)
        return super().tokenize(text, language=language)


# Catches: an update refused for a blank voice that already swapped the text options, so the
# caller's error hides a tokenizer that now splits every synthesis.
async def test_a_refused_update_changes_nothing(slng: Slng) -> None:
    tts = TTS(api_key="key", model=slng.endpoint, voice="voice")
    splitter = _Splitter()

    with pytest.raises(ValueError):
        tts.update_options(voice=" ", speed=1.5, sentence_tokenizer=splitter)
    audio = await _say(tts, "Hello.")
    await tts.aclose()

    assert audio == PCM
    assert splitter.texts == []
    assert [config["speed"] for config in slng.configs] == [1.0]


async def test_a_warm_standby_initializes_the_next_segments_connection_ahead(slng: Slng) -> None:
    """With ``warm_standby``, prewarm opens and initializes a bridge connection before the first
    turn, and the next one is open before the turn after it."""
    # Catches: each segment paying the bridge's connect and init on the turn's critical path, or
    # a connection used for one segment's session handed to the next.
    tts = TTS(api_key="key", model=slng.endpoint, voice="voice", warm_standby=True)
    tts.prewarm()
    await until(lambda: len(slng.configs) == 1, "the standby is initialized before any turn")
    assert await _say(tts, "Hello.") == PCM
    await until(lambda: len(slng.configs) == 2, "the next standby is initialized ahead")
    assert await _say(tts, "Again.") == PCM
    await until(lambda: len(slng.configs) == 3, "and the one after")
    await tts.aclose()


STT_BRIDGE = "/v1/bridges/unmute/stt/deepgram/nova:3"


# Catches: a JSON frame that is not an object (or a message type the bridge added since) failing
# the stream with protocol instead of being skipped.
async def test_stt_skips_frames_it_does_not_read() -> None:
    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.receive()
        await socket.send_json({"type": "ready"})
        async for message in socket:
            if isinstance(message.data, str) and json.loads(message.data).get("type") == "close":
                await socket.send_str("[1, 2]")
                await socket.send_json({"type": "speech_started_v2"})
                await socket.send_json({"type": ["not", "a", "tag"]})
                await socket.send_json({"type": "final_transcript", "transcript": "hello"})
                await socket.close()
        return socket

    app = web.Application()
    app.router.add_get(STT_BRIDGE, transcribe)
    async with TestServer(app) as server, http.open():
        base_url = f"127.0.0.1:{server.port}"
        model = STT(model="deepgram/nova:3", api_key="key", slng_base_url=base_url)
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
        stream.push_frame(rtc.AudioFrame(b"\x00\x01" * 1600, 16000, 1, 1600))
        stream.end_input()
        events = [event async for event in stream]
        await model.aclose()

    finals = [e for e in events if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["hello"]
