"""Contracts every streaming TTS plugin keeps, each run against its provider's wire on a local
WsPeer: a read times out only while a response is owed, so a stream whose input is open and
waiting for more text never times out, while a flushed or ended input whose answer never comes
does; and closing the stream and the model ends every wait.

A case says how to build the plugin on the peer and how its provider answers each message. A
plugin whose wire the peer cannot speak is skipped with what is missing.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import msgpack  # type: ignore[import-untyped]  # msgpack ships no type information
import pytest

from livekit.agents import APIConnectOptions, APIError, APITimeoutError, _core, http, tts
from livekit.plugins import (
    asyncai,
    baseten,
    bland,
    cartesia,
    deepgram,
    elevenlabs,
    fishaudio,
    gnani,
    google,
    gradium,
    inworld,
    minimax,
    murf,
    neuphonic,
    nvidia,
    palabra,
    resemble,
    respeecher,
    rime,
    sarvam,
    slng,
    smallestai,
    soniox,
    speechify,
    telnyx,
    upliftai,
    vakyam,
    xai,
)
from tests.fakes import until
from tests.plugins.conftest import Close
from tests.plugins.test_aws_speech import TONE_MP3

if TYPE_CHECKING:
    from tests.plugins.conftest import Connection, Frame, WsPeer

PCM = b"\x01\x02" * 2400  # 100 ms of 24 kHz audio, 50 ms at 48 kHz
COMMIT = "https://github.com/livekit/agents/commit/9e79bc520b80553e3ca665884a2fac2beeb63042"


@dataclass
class TtsCase:
    """How a streaming TTS plugin meets its provider on a WsPeer."""

    make: Callable[[WsPeer], tts.TTS]
    answer: Callable[[Connection, str | bytes], list[Frame]]
    """The provider's answer to one message: audio for text, its end-of-segment marker."""
    skip: str | None = None
    setup: int = 0
    """The first messages on a socket, a session handshake, answered even when answers stop."""
    loop_deadline: str | None = None
    """Why the plugin times a read out on the event loop's clock, which only real time moves;
    None when its read deadline is the core's, which ``_core.skip_host_time`` moves."""


def _cartesia(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    audio = {"context_id": packet["context_id"], "data": base64.b64encode(PCM).decode()}
    if packet.get("continue") is False:
        return [audio, {"context_id": packet["context_id"], "done": True}]
    return [audio] if packet.get("transcript", "").strip() else []


def _deepgram(_: Connection, message: str | bytes) -> list[Frame]:
    kind = json.loads(message).get("type")
    if kind == "Speak":
        return [PCM]
    if kind == "Flush":
        return [{"type": "Flushed", "sequence_id": 0}]
    return []


# -- asyncai .. palabra answers
def _b64(audio: bytes) -> str:
    return base64.b64encode(audio).decode()


def _asyncai(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    if "transcript" not in packet:  # the session's init payload
        return []
    final = {"context_id": packet["context_id"], "final": True}
    if packet["transcript"]:
        return [{"context_id": packet["context_id"], "audio": _b64(PCM)}, final]
    return [final]


def _baseten(_: Connection, message: str | bytes) -> list[Frame]:
    if message == "__END__":
        return [Close(1000)]
    return [PCM] if isinstance(message, str) and not message.startswith("{") else []


def _baseten_qwen3(_: Connection, message: str | bytes) -> list[Frame]:
    if json.loads(message).get("type") != "input.done":
        return []
    return [{"type": "audio.start"}, PCM, {"type": "audio.done"}, {"type": "session.done"}]


def _bland(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    match packet["type"]:
        case "init":
            return [{"type": "ready", **packet["audio"]}]
        case "speak":
            return [PCM * 2]  # 100 ms at 48 kHz
        case "end_of_turn" | "cancel":
            reason = "complete" if packet["type"] == "end_of_turn" else "cancelled"
            return [{"type": "utterance_end", "context_id": packet["context_id"], "reason": reason}]
    return []


def _elevenlabs(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    context = {"contextId": packet["context_id"]}
    if packet.get("close_context"):
        return [{**context, "isFinal": True}]
    return [{**context, "audio": _b64(PCM)}] if packet.get("flush") else []


def _fishaudio(_: Connection, message: str | bytes) -> list[Frame]:
    event = msgpack.unpackb(message, raw=False)["event"]
    if event == "flush":
        return [msgpack.packb({"event": "audio", "audio": PCM}, use_bin_type=True)]
    if event == "stop":
        return [msgpack.packb({"event": "finish", "reason": "stop"}, use_bin_type=True)]
    return []


def _gnani(_: Connection, message: str | bytes) -> list[Frame]:
    return [{"type": "audio", "data": {"audio": _b64(PCM)}}, {"type": "complete"}]


def _gradium(_: Connection, message: str | bytes) -> list[Frame]:
    kind = json.loads(message)["type"]
    if kind == "setup":
        return [{"type": "ready"}]
    if kind == "text":
        return [{"type": "audio", "audio": _b64(PCM * 2)}]  # 100 ms at 48 kHz
    return [{"type": "end_of_stream"}]


def _inworld(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    result: dict[str, object] = {"contextId": packet["contextId"]}
    if "create" in packet:
        result["contextCreated"] = {}
    elif "send_text" in packet:
        result["audioChunk"] = {"audioContent": _b64(PCM)}
    elif "flush_context" in packet:
        result["flushCompleted"] = {}
    else:
        result["contextClosed"] = {}
    return [{"result": result}]


def _minimax(_: Connection, message: str | bytes) -> list[Frame]:
    match json.loads(message)["event"]:
        case "task_start":
            return [{"event": "task_started", "session_id": "session"}]
        case "task_continue":
            return [{"event": "task_continued", "data": {"audio": PCM.hex()}}]
        case "task_finish":
            return [{"event": "task_finished"}]
    return []


def _murf(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    return [{"final": True}] if packet.get("end") else [{"audio": _b64(PCM)}]


def _neuphonic(_: Connection, message: str | bytes) -> list[Frame]:
    context = json.loads(message)["context_id"]
    audio = {"data": {"audio": _b64(PCM), "context_id": context}}
    return [audio, {"data": {"stop": True, "context_id": context}}]


# -- resemble .. xai answers
class _ToPeer(http.HttpSession):
    """The test's session, opening every websocket on the peer: for a plugin whose provider URL
    is fixed, or must be wss://, the peer stands behind the provider's host."""

    def __init__(self, peer: WsPeer) -> None:
        vars(self).update(vars(http.http_session()))  # the fixture's pool, closed with it
        self._peer = urlsplit(peer.ws_url)

    async def ws_connect(self, url: str, **options: Any) -> http.WebSocket:  # Any: forwarded
        target = urlsplit(url)._replace(scheme=self._peer.scheme, netloc=self._peer.netloc)
        return await super().ws_connect(urlunsplit(target), **options)


def _resemble(_: Connection, message: str | bytes) -> list[Frame]:
    request = json.loads(message)["request_id"]
    audio = {"type": "audio", "audio_content": _b64(TONE_MP3), "request_id": request}
    return [audio, {"type": "audio_end", "request_id": request}]


def _respeecher(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    reply = {"type": "chunk", "data": _b64(PCM)} if packet["continue"] else {"type": "done"}
    return [{"context_id": packet["context_id"], **reply}]


def _rime(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    if "text" in packet:
        return [{"type": "chunk", "data": _b64(PCM)}]
    if packet["operation"] == "flush":
        return [{"type": "done"}]
    return [Close(1000)]  # eos: the server ends the session


def _rime_v1_on(peer: WsPeer) -> tts.TTS:
    peer.protocols = ("rime.v1.json",)
    peer.hello = [{"ready": {"protocol": 1, "languages": ["eng"]}}]
    url = peer.ws_url + "/coda/ws"
    return rime.TTS(api_key="key", websocket_url=url, websocket_protocol="json")


def _rime_v1(_: Connection, message: str | bytes) -> list[Frame]:
    event = json.loads(message)
    context = event["contextId"]
    if "start" in event:
        return [{"contextId": context, "started": {"requestId": "request"}}]
    if "text" in event:
        return [{"contextId": context, "audio": _b64(PCM)}]
    return [{"contextId": context, "done" if "end" in event else "cancelled": {}}]


def _sarvam(_: Connection, message: str | bytes) -> list[Frame]:
    kind = json.loads(message)["type"]
    if kind == "text":
        return [{"type": "audio", "data": {"audio": _b64(PCM)}}]
    if kind == "flush":
        return [{"type": "event", "data": {"event_type": "final"}}]
    return []  # config, ping


def _slng(_: Connection, message: str | bytes) -> list[Frame]:
    kind = json.loads(message)["type"]
    if kind == "init":
        return [{"type": "ready"}]
    if kind == "text":
        return [{"type": "audio_chunk", "data": _b64(PCM)}]
    return [{"type": "audio_end"}] if kind == "flush" else []


def _smallestai(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    if packet.get("cancel_request"):
        return []
    if packet["continue"]:
        return [{"status": "chunk", "data": {"audio": _b64(PCM)}}]
    return [{"status": "complete"}]


def _soniox(_: Connection, message: str | bytes) -> list[Frame]:
    packet = json.loads(message)
    stream = packet.get("stream_id")
    if stream is None or "model" in packet:  # keepalive, a stream's config
        return []
    if packet.get("cancel"):
        return [{"stream_id": stream, "terminated": True}]
    frames: list[Frame] = []
    if packet.get("text"):
        frames.append({"stream_id": stream, "audio": _b64(PCM)})
    if packet.get("text_end"):
        frames += [
            {"stream_id": stream, "audio_end": True},
            {"stream_id": stream, "terminated": True},
        ]
    return frames


def _telnyx(_: Connection, message: str | bytes) -> list[Frame]:
    text = json.loads(message)["text"]
    if not text:
        return [Close(1000)]  # the end of the text: the server closes once it is spoken
    return [{"audio": _b64(TONE_MP3)}] if text.strip() else []


def _vakyam_on(peer: WsPeer) -> tts.TTS:
    peer.hello = [{"type": "connected"}]
    return vakyam.TTS(api_key="key", base_url=peer.url)


def _vakyam(_: Connection, message: str | bytes) -> list[Frame]:
    kind = json.loads(message)["type"]
    if kind == "config":
        return [{"type": "configured"}]
    if kind == "text":
        end = {"type": "end_of_utterance", "characters_used": 12, "duration_seconds": 0.1}
        return [PCM, end]
    if kind == "cancel":
        return [{"type": "cancellation"}]
    return [{"type": "pong"}] if kind == "ping" else []


def _xai(_: Connection, message: str | bytes) -> list[Frame]:
    kind = json.loads(message)["type"]
    if kind == "text.delta":
        return [{"type": "audio.delta", "delta": _b64(PCM)}]
    return [{"type": "audio.done"}]


CASES: dict[str, TtsCase] = {
    "cartesia": TtsCase(
        make=lambda peer: cartesia.TTS(api_key="key", base_url=peer.url), answer=_cartesia
    ),
    "deepgram": TtsCase(
        make=lambda peer: deepgram.TTS(api_key="key", base_url=peer.url + "/v1/speak"),
        answer=_deepgram,
    ),
    # -- asyncai .. palabra
    "asyncai": TtsCase(
        make=lambda peer: asyncai.TTS(api_key="key", base_url=peer.url), answer=_asyncai
    ),
    "baseten": TtsCase(
        make=lambda peer: baseten.TTS(api_key="key", model_endpoint=peer.ws_url), answer=_baseten
    ),
    "baseten_qwen3": TtsCase(
        make=lambda peer: baseten.TTS(
            model="qwen3-tts", api_key="key", model_endpoint=peer.ws_url, voice="voice"
        ),
        answer=_baseten_qwen3,
        loop_deadline="a flushed segment is read under asyncio.timeout(read_timeout)",
    ),
    "bland": TtsCase(
        make=lambda peer: bland.TTS(api_key="key", base_url=peer.url), answer=_bland, setup=1
    ),
    "elevenlabs": TtsCase(
        make=lambda peer: elevenlabs.TTS(api_key="key", encoding="pcm_24000", base_url=peer.url),
        answer=_elevenlabs,
        loop_deadline="a context's timeout is a loop.call_later(read_timeout) timer",
    ),
    "fishaudio": TtsCase(
        make=lambda peer: fishaudio.TTS(api_key="key", output_format="pcm", base_url=peer.url),
        answer=_fishaudio,
    ),
    "gnani": TtsCase(
        make=lambda peer: gnani.TTS(api_key="key", container="raw", base_url=peer.url),
        answer=_gnani,
    ),
    "google": TtsCase(
        make=lambda _: google.TTS(),
        answer=lambda _c, _m: [],
        skip="the Cloud TTS client speaks gRPC to a fixed endpoint",
    ),
    "gradium": TtsCase(
        make=lambda peer: gradium.TTS(api_key="key", model_endpoint=peer.ws_url), answer=_gradium
    ),
    "inworld": TtsCase(
        make=lambda peer: inworld.TTS(
            api_key="key", encoding="PCM", base_url=peer.url, ws_url=peer.ws_url
        ),
        answer=_inworld,
    ),
    "minimax": TtsCase(
        make=lambda peer: minimax.TTS(api_key="key", audio_format="pcm", base_url=peer.url),
        answer=_minimax,
    ),
    "murf": TtsCase(make=lambda peer: murf.TTS(api_key="key", base_url=peer.url), answer=_murf),
    "neuphonic": TtsCase(
        make=lambda peer: neuphonic.TTS(api_key="key", base_url=peer.url), answer=_neuphonic
    ),
    "nvidia": TtsCase(
        make=lambda _: nvidia.TTS(),
        answer=lambda _c, _m: [],
        skip="Riva speaks gRPC through its own client",
    ),
    "palabra": TtsCase(
        make=lambda _: palabra.TTS(),
        answer=lambda _c, _m: [],
        skip="the Palabra SDK picks its endpoint from a region, with no base URL",
    ),
    # -- resemble .. xai
    "resemble": TtsCase(
        make=lambda peer: resemble.TTS(api_key="key", http_session=_ToPeer(peer)),
        answer=_resemble,
    ),
    "respeecher": TtsCase(
        make=lambda peer: respeecher.TTS(api_key="key", http_session=_ToPeer(peer)),
        answer=_respeecher,
    ),
    "rime": TtsCase(make=lambda peer: rime.TTS(api_key="key", base_url=peer.ws_url), answer=_rime),
    "rime_v1": TtsCase(
        make=_rime_v1_on,
        answer=_rime_v1,
        loop_deadline="the started and terminal watchdogs are asyncio.wait_for(read_timeout)",
    ),
    "sarvam": TtsCase(
        make=lambda peer: sarvam.TTS(
            api_key="key", ws_url=peer.ws_url, output_audio_codec="linear16"
        ),
        answer=_sarvam,
        loop_deadline="the receive loop reads with ws.receive(timeout=read_timeout)",
    ),
    # one bridge connection: slng.TTS chains them in a FallbackAdapter that retries each on its own
    "slng": TtsCase(
        make=lambda peer: slng.tts._Connection(
            api_key="key", model="deepgram/aura:2", voice="voice", slng_base_url=peer.url
        ),
        answer=_slng,
        setup=1,
    ),
    "smallestai": TtsCase(
        make=lambda peer: smallestai.TTS(api_key="key", ws_url=peer.ws_url), answer=_smallestai
    ),
    "soniox": TtsCase(
        make=lambda peer: soniox.TTS(api_key="key", websocket_url=peer.ws_url), answer=_soniox
    ),
    "speechify": TtsCase(
        make=lambda peer: speechify.TTS(api_key="key", base_url=peer.url),
        answer=lambda _c, _m: [],
        skip="streams over the Speechify SDK's own HTTP client, whose one whole-second timeout "
        "bounds every phase in real time (read_timeout=0.05 is refused)",
    ),
    "telnyx": TtsCase(
        make=lambda peer: telnyx.TTS(api_key="key", base_url=peer.ws_url), answer=_telnyx
    ),
    "upliftai": TtsCase(
        make=lambda peer: upliftai.TTS(api_key="key", base_url=peer.ws_url),
        answer=lambda _c, _m: [],
        skip="speaks Socket.IO (an Engine.IO handshake and packet framing), not plain frames",
    ),
    "vakyam": TtsCase(
        make=_vakyam_on,
        answer=_vakyam,
        setup=1,
        loop_deadline="each utterance is read with ws.receive(timeout=read_timeout)",
    ),
    "xai": TtsCase(
        make=lambda peer: xai.TTS(api_key="key", http_session=_ToPeer(peer)), answer=_xai
    ),
}

# (contract, plugin): what reproduces a known product bug, as a strict xfail
KNOWN: dict[tuple[str, str], str] = {
    # -- asyncai .. palabra
    ("idle_input", "gradium"): "a segment's end_of_stream answer never disarms the "
    "expect_response(final=True) deadline (gradium/tts.py:349, 372), so the next sentence's "
    "wait times out",
    # -- resemble .. xai
    ("idle_input", "slng"): "a segment's audio_end leaves the final read deadline armed, never "
    "completed (livekit-plugins-slng/livekit/plugins/slng/tts.py:1288): waiting for the next "
    "sentence times out",
    ("idle_input", "telnyx"): "a segment's close leaves the final read deadline armed, never "
    "completed (livekit-plugins-telnyx/livekit/plugins/telnyx/tts.py:181): waiting for the next "
    "sentence times out",
    ("idle_input", "xai"): "audio.done leaves the segment's final read deadline armed, never "
    "completed (livekit-plugins-xai/livekit/plugins/xai/tts.py:335): waiting for the next "
    "sentence times out",
}


# (contract, plugin): what the contract's test cannot express for that plugin
UNTESTABLE: dict[tuple[str, str], str] = {
    ("idle_input", "gnani"): "the stream speaks the text up to its first flush and ends "
    "(gnani/tts.py:435): its input never waits for more text",
    ("idle_input", "smallestai"): "the continuations protocol ends a segment only after 0.6 s of "
    "real-time silence: its answer is still owed when the test skips the core's clock",
}


def _params(contract: str) -> list[object]:
    params: list[object] = []
    for name, case in CASES.items():
        marks: list[pytest.MarkDecorator] = []
        if skip := case.skip or UNTESTABLE.get((contract, name)):
            marks.append(pytest.mark.skip(reason=skip))
        if reason := KNOWN.get((contract, name)):
            marks.append(pytest.mark.xfail(strict=True, reason=f"{COMMIT} reproduces: {reason}"))
        params.append(pytest.param(name, marks=marks))
    return params


class _Run:
    """A stream on the peer: its audio and failure as a reader sees them, and an answer switch."""

    def __init__(self, name: str, peer: WsPeer, options: APIConnectOptions) -> None:
        self.answering = True
        case = CASES[name]
        peer.reply = lambda c, m: (
            case.answer(c, m) if self.answering or len(c.received) <= case.setup else []
        )
        self.model = case.make(peer)
        self.stream = self.model.stream(conn_options=options)
        self.audio: list[tts.SynthesizedAudio] = []
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for audio in self.stream:
            self.audio.append(audio)

    def failure(self) -> BaseException | None:
        return self.reader.exception() if self.reader.done() else None

    async def aclose(self) -> None:
        await self.stream.aclose()
        await self.model.aclose()
        await asyncio.gather(self.reader, return_exceptions=True)


# Catches: a plugin reading its socket with ``receive(timeout=read_timeout)`` (or a core deadline
# left owed after the answer came) while the input waits for the LLM's next sentence: a slow
# LLM fails the turn with a timeout.
@pytest.mark.parametrize("name", _params("idle_input"))
async def test_an_open_input_waiting_for_text_never_times_out(name: str, ws_peer: WsPeer) -> None:
    loop_deadline = CASES[name].loop_deadline
    # the peer answers at once, well within the read timeout; the idle lasts longer
    read_timeout = 0.05 if loop_deadline else 5.0
    run = _Run(name, ws_peer, APIConnectOptions(max_retry=0, read_timeout=read_timeout))
    try:
        run.stream.push_text("Hello there.")
        run.stream.flush()
        await until(lambda: run.audio or run.failure(), "the first sentence is spoken")
        assert run.failure() is None

        if loop_deadline:
            for _ in range(3):  # a deadline left armed on the loop's clock fires in this idle
                await asyncio.sleep(0.05)
        _core.skip_host_time(read_timeout * 2)
        run.stream.end_input()

        # a deadline that fired fails the stream even when the stream ends first
        await until(run.reader.done, "the stream ends")
        assert run.failure() is None
    finally:
        await run.aclose()


# Catches: an answer that never comes after the input ended (or was flushed) leaving the turn
# waiting forever instead of failing with a read timeout.
@pytest.mark.parametrize("name", _params("missing_answer"))
async def test_a_missing_answer_once_the_input_ended_times_out(name: str, ws_peer: WsPeer) -> None:
    # a read timed out on the event loop's clock or on the core's both fire
    run = _Run(name, ws_peer, APIConnectOptions(max_retry=0, read_timeout=0.05))
    run.answering = False
    try:
        run.stream.push_text("Hello there.")
        run.stream.end_input()
        connection = await ws_peer.connected()
        await until(lambda: connection.received, "the plugin sends the text")
        _core.skip_host_time(11)
        await until(run.reader.done, "the stream times out")
        assert isinstance(run.failure(), APITimeoutError), run.failure()
    finally:
        await run.aclose()


# Catches: a send, receive or keepalive wait that outlives the stream and model's close,
# leaking its task (the suite's leaked-task check fails the test) or keeping the socket open.
@pytest.mark.parametrize("name", _params("close"))
async def test_closing_the_stream_and_model_ends_every_pending_wait(
    name: str, ws_peer: WsPeer
) -> None:
    run = _Run(name, ws_peer, APIConnectOptions(max_retry=0))
    run.answering = False
    run.stream.push_text("Hello there.")
    run.stream.flush()
    connection = await ws_peer.connected()
    await until(lambda: connection.received, "the plugin sends the text")

    await run.aclose()

    assert not isinstance(run.failure(), APIError)
    await until(lambda: all(c.socket.closed for c in ws_peer.connections), "sockets close")
