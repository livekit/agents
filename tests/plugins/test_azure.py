"""Azure plugin: the streaming STT on a local peer speaking the Speech service's websocket
protocol (header blocks, speech.config/context, audio framing, turns), and the REST TTS."""

from __future__ import annotations

import asyncio
import json
import struct
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    APIStatusError,
    NotGivenOr,
    http,
    stt,
)
from livekit.plugins.azure import stt as azure_stt, tts as azure_tts
from tests.plugins.conftest import Connection, Frame, WsPeer

# Speech service wire


def headers_of(message: str | bytes) -> tuple[dict[str, str], str | bytes]:
    """A client message's headers and body: text has a blank line after its headers, binary a
    2-byte header length first."""
    if isinstance(message, str):
        head, _, body = message.partition("\r\n\r\n")
        raw: str | bytes = body
    else:
        length = int.from_bytes(message[:2], "big")
        head, raw = message[2 : 2 + length].decode(), message[2 + length :]
    lines = [line.partition(":") for line in head.split("\r\n") if line]
    return {name.strip(): value.strip() for name, _, value in lines}, raw


def sent(connection: Connection, path: str) -> list[tuple[dict[str, str], str | bytes]]:
    return [m for m in map(headers_of, connection.received) if m[0]["Path"] == path]


def request_id(connection: Connection) -> str:
    """The current turn's request id: the last speech.context's."""
    return sent(connection, "speech.context")[-1][0]["X-RequestId"]


def service(path: str, connection: Connection, body: object | None = None) -> str:
    """A message the service sends for the connection's current turn."""
    return (
        f"Path: {path}\r\nX-RequestId: {request_id(connection)}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n\r\n"
        + ("" if body is None else json.dumps(body))
    )


def phrase(
    connection: Connection, text: str, *, status: str = "Success", language: str = ""
) -> str:
    body: dict[str, object] = {
        "RecognitionStatus": status,
        "DisplayText": text,
        "Offset": 0,
        "Duration": 10**7,
    }
    if language:
        body["PrimaryLanguage"] = {"Language": language, "Confidence": "High"}
    return service("speech.phrase", connection, body)


def hypothesis(connection: Connection, text: str) -> str:
    return service("speech.hypothesis", connection, {"Text": text, "Offset": 0, "Duration": 10**6})


def ends_audio(message: str | bytes) -> bool:
    headers, body = headers_of(message)
    return headers["Path"] == "audio" and "Content-Type" not in headers and body == b""


def carries_audio(message: str | bytes) -> bool:
    headers, body = headers_of(message)
    return headers["Path"] == "audio" and "Content-Type" not in headers and body != b""


def finish_with(
    *texts: str, language: str = ""
) -> Callable[[Connection, str | bytes], list[Frame]]:
    """A service that answers the end of the audio with ``texts`` as phrases, then turn.end."""

    def reply(connection: Connection, message: str | bytes) -> list[Frame]:
        if not ends_audio(message):
            return []
        return [
            *(phrase(connection, text, language=language) for text in texts),
            service("turn.end", connection),
        ]

    return reply


def query(connection: Connection) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(str(connection.request.url)).query).items()}


def context(connection: Connection) -> dict[str, Any]:
    body = sent(connection, "speech.context")[-1][1]
    assert isinstance(body, str)
    parsed: dict[str, Any] = json.loads(body)
    return parsed


def frame(duration_ms: int, sample_rate: int) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    return rtc.AudioFrame(
        data=b"\x01\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def finals(stream: azure_stt.SpeechStream) -> list[stt.SpeechData]:
    async with asyncio.timeout(2):
        return [
            event.alternatives[0]
            async for event in stream
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]


OPTIONS = APIConnectOptions(max_retry=0, connect_timeout=1, read_timeout=1)


def azure(peer: WsPeer, **kwargs: Any) -> azure_stt.STT:
    return azure_stt.STT(
        speech_key="key", speech_endpoint=peer.ws_url + "/stt/speech/universal/v2", **kwargs
    )


@pytest.mark.parametrize(
    "configured, override, expected",
    [
        ("en-US", None, ["en-US"]),
        (["fr-FR"], None, ["fr-FR"]),
        (["en-US", "fr-FR"], None, ["en-US", "fr-FR"]),
        (["en-US", "fr-FR"], "de-DE", ["de-DE"]),
    ],
)
async def test_recognizer_uses_the_requested_languages(
    ws_peer: WsPeer, configured: str | list[str], override: str | None, expected: list[str]
) -> None:
    """One language is set explicitly, several enable continuous identification, and the
    stream() and update_options() languages are honored.

    https://github.com/livekit/agents/issues/1858 https://github.com/livekit/agents/issues/4449
    """
    ws_peer.reply = finish_with("hello")
    provider = azure(ws_peer, language=configured)
    kwargs = {"language": override} if override else {}
    async with provider.stream(conn_options=OPTIONS, **kwargs) as stream:
        first = await ws_peer.connected()
        provider.update_options(language="it-IT")
        updated = await ws_peer.connected(2)
        stream.push_frame(frame(100, 16000))
        stream.end_input()
        final = await finals(stream)

    for languages, current in ((expected, first), (["it-IT"], updated)):
        detection = context(current)["phraseDetection"]
        assert isinstance(detection, dict)
        if len(languages) == 1:
            assert query(current)["language"] == languages[0]
            assert detection["language"] == languages[0]
            assert "languageId" not in context(current)
        else:
            language_id = context(current)["languageId"]
            assert isinstance(language_id, dict)
            assert (language_id["languages"], language_id["mode"]) == (
                languages,
                "DetectContinuous",
            )
            assert query(current)["lidEnabled"] == "true"
            assert "language" not in query(current)
    assert [(speech.text, speech.language) for speech in final] == [("hello", "it-IT")]


async def test_end_input_flushes_buffered_finals_before_teardown(ws_peer: WsPeer) -> None:
    """Ending input delivers every pending final before recognition stops.

    https://github.com/livekit/agents/pull/6118
    """
    ws_peer.reply = finish_with("first phrase", "second phrase")
    provider = azure(ws_peer)
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100, 16000))
        stream.end_input()
        final = await finals(stream)
    assert [speech.text for speech in final] == ["first phrase", "second phrase"]
    assert len(ws_peer.connections) == 1


@pytest.mark.parametrize("sample_rate", [8000, 48000])
async def test_recognizer_audio_matches_the_configured_sample_rate(
    ws_peer: WsPeer, sample_rate: int
) -> None:
    """The wave header declares the configured rate and the audio is sent at it, whatever the
    frames' rate.

    https://github.com/livekit/agents/pull/1072
    """
    ws_peer.reply = finish_with()
    provider = azure(ws_peer, sample_rate=sample_rate)
    async with provider.stream(conn_options=OPTIONS) as stream:
        for _ in range(10):
            stream.push_frame(frame(20, 16000))
        stream.end_input()
        await finals(stream)

    (header, wave), *audio = sent(ws_peer.connections[0], "audio")
    assert header["Content-Type"] == "audio/x-wav" and isinstance(wave, bytes)
    channels, rate = struct.unpack_from("<HI", wave, 22)
    assert (rate, channels) == (sample_rate, 1)
    seconds = sum(len(body) for _, body in audio) / 2 / sample_rate
    assert seconds == pytest.approx(0.2, abs=0.02)


async def test_stt_custom_endpoint_needs_no_region(ws_peer: WsPeer) -> None:
    """A key plus a private endpoint connects to that endpoint with the key.

    https://github.com/livekit/agents/issues/1361
    """
    ws_peer.reply = finish_with()
    provider = azure(ws_peer)
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.end_input()
        await finals(stream)
    request = ws_peer.connections[0].request
    assert request.path == "/stt/speech/universal/v2"
    assert request.headers["Ocp-Apim-Subscription-Key"] == "key"


async def test_processed_audio_is_reported_as_usage(ws_peer: WsPeer) -> None:
    """The audio a stream sent is reported as recognition usage.

    https://github.com/livekit/agents/issues/6151
    """
    ws_peer.reply = finish_with()
    provider = azure(ws_peer)
    async with provider.stream(conn_options=OPTIONS) as stream:
        for _ in range(5):
            stream.push_frame(frame(100, 16000))
        stream.end_input()
        async with asyncio.timeout(2):
            usage = [
                event.recognition_usage.audio_duration or 0.0
                async for event in stream
                if event.recognition_usage is not None
            ]

    assert sum(usage) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "status, kind", [("Error", "server"), ("TooManyRequests", "rate_limit"), ("Forbidden", "auth")]
)
async def test_a_canceled_phrase_fails_the_stream_by_its_status(
    ws_peer: WsPeer, status: str, kind: str
) -> None:
    """A speech.phrase whose status cancels the recognition ends the stream with an API error of
    its kind instead of hanging.

    https://github.com/livekit/agents/issues/5322
    """
    ws_peer.reply = lambda connection, message: (
        [phrase(connection, "", status=status)] if carries_audio(message) else []
    )
    provider = azure(ws_peer)
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100, 16000))
        with pytest.raises(APIError) as caught:
            await finals(stream)

    assert caught.value.kind == kind


# Catches: a phrase status the service added since failing the stream as a canceled
# recognition, or a message path it added since reaching the stream, instead of reading as a
# phrase that recognized nothing and a message the stream does not read.
async def test_an_unknown_phrase_status_and_message_path_do_not_fail_the_stream(
    ws_peer: WsPeer,
) -> None:
    def reply(connection: Connection, message: str | bytes) -> list[Frame]:
        if not ends_audio(message):
            return []
        return [
            service("speech.keyword.v2", connection, {"Text": 5}),
            phrase(connection, "unheard", status="SpeechEndedEarly"),
            phrase(connection, "hello"),
            service("turn.end", connection),
        ]

    ws_peer.reply = reply
    provider = azure(ws_peer)
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100, 16000))
        stream.end_input()
        final = await finals(stream)

    assert [speech.text for speech in final if speech.text] == ["hello"]


async def test_a_turn_the_service_ends_is_followed_by_a_new_one(ws_peer: WsPeer) -> None:
    """While audio still flows, a turn.end opens the next turn on the same connection: a new
    request id with its speech.context and wave header; that turn's phrases are read and a late
    one of the ended turn is dropped."""
    provider = azure(ws_peer)
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100, 16000))
        connection = await ws_peer.connected()
        await until_turns(connection, 1)
        first = request_id(connection)
        await ws_peer.send(service("turn.end", connection))
        await until_turns(connection, 2)
        stream.push_frame(frame(100, 16000))
        late = phrase(connection, "late").replace(request_id(connection), first)
        await ws_peer.send(late, phrase(connection, "second turn"))
        ws_peer.reply = finish_with()
        stream.end_input()
        final = await finals(stream)

    contexts = sent(connection, "speech.context")
    assert [h["X-RequestId"] == first for h, _ in contexts] == [True, False]
    waves = [h for h, _ in sent(connection, "audio") if h.get("Content-Type") == "audio/x-wav"]
    assert [h["X-RequestId"] for h in waves] == [h["X-RequestId"] for h, _ in contexts]
    assert [speech.text for speech in final] == ["second turn"]
    assert len(ws_peer.connections) == 1


async def until_turns(connection: Connection, count: int) -> None:
    async with asyncio.timeout(2):
        while len(sent(connection, "speech.context")) < count:
            await asyncio.sleep(0.01)


async def test_a_message_without_a_header_block_is_a_protocol_error(ws_peer: WsPeer) -> None:
    """A text frame that is not a header block and a body breaks the protocol: the stream
    fails for good."""
    ws_peer.hello = ["not a speech message"]
    provider = azure(ws_peer)
    async with provider.stream(conn_options=APIConnectOptions(max_retry=2)) as stream:
        stream.push_frame(frame(100, 16000))
        with pytest.raises(APIError) as caught:
            await finals(stream)

    assert caught.value.kind == "protocol"
    assert len(ws_peer.connections) == 1


# REST TTS


class SynthesisPeer:
    def __init__(self, mode: str = "audio") -> None:
        self.mode = mode
        self.requests: list[tuple[dict[str, str], str]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append((dict(request.headers), await request.text()))
        if self.mode == "stall":
            await asyncio.Event().wait()  # never answers
        if self.mode == "reject":
            return web.Response(status=400, text="Invalid char '&' found in SSML.")
        return web.Response(body=b"\x01\x00" * 2400, content_type="audio/pcm")


async def synthesize(
    peer: SynthesisPeer,
    text: str,
    *,
    voice: str = "en-US-JennyNeural",
    prosody: NotGivenOr[azure_tts.ProsodyConfig] = NOT_GIVEN,
) -> list[rtc.AudioFrame]:
    app = web.Application()
    app.router.add_post("/cognitiveservices/v1", peer.handle)
    async with TestServer(app) as server, http.open() as session:
        provider = azure_tts.TTS(
            speech_key="key",
            speech_endpoint=str(server.make_url("/cognitiveservices/v1")),
            http_session=session,
            voice=voice,
            prosody=prosody,
        )
        # a stalled synthesis times the read out at once
        read_timeout = 0.05 if peer.mode == "stall" else 5
        options = APIConnectOptions(max_retry=0, connect_timeout=5, read_timeout=read_timeout)
        async with asyncio.timeout(3), provider.synthesize(text, conn_options=options) as stream:
            return [event.frame async for event in stream]


@pytest.mark.parametrize(
    "text",
    [
        "Hello there.",
        pytest.param(
            "Tom & Jerry <3 \"quotes\" 'apostrophes'",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/2824 reproduces: text is "
                "inserted into the SSML unescaped",
            ),
        ),
    ],
)
async def test_ssml_wraps_text_in_voice_and_prosody_and_keeps_it_verbatim(text: str) -> None:
    """The request is valid SSML naming the voice and prosody and carrying the text unchanged.

    https://github.com/livekit/agents/issues/2824
    """
    peer = SynthesisPeer()
    frames = await synthesize(
        peer, text, voice="en-US-AvaNeural", prosody=azure_tts.ProsodyConfig(rate="fast")
    )
    assert sum(f.samples_per_channel for f in frames) == 2400
    headers, body = peer.requests[0]
    assert headers["Ocp-Apim-Subscription-Key"] == "key"
    ns = {"s": "http://www.w3.org/2001/10/synthesis"}
    voice = ET.fromstring(body).find("s:voice", ns)
    assert voice is not None and voice.get("name") == "en-US-AvaNeural"
    prosody = voice.find("s:prosody", ns)
    assert prosody is not None and prosody.get("rate") == "fast"
    assert prosody.text == text


@pytest.mark.parametrize("mode", ["stall", "reject"])
async def test_failed_synthesis_raises_a_typed_error_without_hanging(mode: str) -> None:
    """A stalled or rejected synthesis ends in an API error within the configured timeout.

    https://github.com/livekit/agents/issues/729
    """
    with pytest.raises(APIError) as caught:
        await synthesize(SynthesisPeer(mode), "Hello.")
    if mode == "reject":
        assert isinstance(caught.value, APIStatusError)
        assert caught.value.status_code == 400
