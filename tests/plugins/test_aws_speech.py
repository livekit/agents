"""AWS speech: Transcribe streaming STT over its websocket on a local peer speaking the
event-stream wire, and Polly TTS over HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, _core, stt
from livekit.plugins import aws
from livekit.plugins.aws import _eventstream, stt as aws_stt
from tests.fakes import until
from tests.plugins.conftest import Close, Connection, WsPeer

OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0, connect_timeout=0.5, read_timeout=0.5)

# 0.2 s of a 440 Hz tone, 16 kHz mono MP3 (ffmpeg, no ID3/Xing headers)
TONE_MP3 = base64.b64decode(
    "//MoxAAM8ALRv0EYAqkAZLh//+AD4Pg+D58EInB85dLg+D4P8EOD7//D/AjsoD+sHDmIAf1gQ5kAf4EOcP9Hv6F0"
    "mHf/+icy//MoxAcN6LKQAZt4AB+C/jJgQ6bAkBYgNkOQqEHJhYD2HKTkJKl9APCq7ErldB////evbWhfBUJA1+JT"
    "oNKAXbQD///1lTbv//MoxAoPIGpQH90IADWhCAQsCRgQDZiAZ5xYUR8ybxkiTRi0IJhkOo6E4cBLu0FrIn/+j//k"
    "//+3b/Tb/6v+uv///VK7y7S///MoxAgL+G4wAOe2gaFQIDAmYDFxhY3mWbcZNXlBqgjzGCMC6Y6CHDWZlZqCkZEV"
    "p0Vnr1P///zlD9vawkRAQwCGzCg6Mcoc//MoxBMP8HIsAOf4gNKZ4w0FCUMb0CMjAzwLUzOaTU0AMckQDE9Axp78"
    "S+kqH3/+n//p////////qQDSDwP///crja+EI0vD//MoxA4QiG48VV4YAAAAFhYDUwPQYTCpGtNC0G03IhxDECB+"
    "MBQHAwgBBBQGowMABkrHrfun6X/b939H///9P/Qq/2IL7lU9//MoxAYOwMqUAZp4APTrXL3uC4IUCH+LlhsasXKJ"
    "WDtc+EUXY8FKqW7/9WNbO/tGg5//8kSlY9WD+LBAT/rKMf/ohJ40jSOp//MoxAYMKIXcAc8wAZlyQUcKnEJCQnCi"
    "hbiXO06nldGlTp0srchwSUSJJbFFxQVgKYCjsTegv5VMQU1FNC4wVVVVVVVVVVVV"
)


@pytest.fixture(autouse=True)
def _environment_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "AWS_ACCESS_KEY_ID": "env-key",
        "AWS_SECRET_ACCESS_KEY": "env-secret",
        "AWS_SESSION_TOKEN": "env-token",
        "AWS_EC2_METADATA_DISABLED": "true",
    }.items():
        monkeypatch.setenv(name, value)


def transcript_event(
    text: str, language: str = "en-US", confidence: float = 0.9, *, partial: bool = False
) -> bytes:
    """A TranscriptEvent as Transcribe frames it."""
    item = {"Content": text, "StartTime": 0.1, "EndTime": 0.5, "Confidence": confidence}
    result = {
        "ResultId": "r-1",
        "StartTime": 0.1,
        "EndTime": 0.5,
        "IsPartial": partial,
        "Alternatives": [{"Transcript": text, "Items": [item]}],
        "LanguageCode": language,
    }
    headers = {":message-type": "event", ":event-type": "TranscriptEvent"}
    payload = json.dumps({"Transcript": {"Results": [result]}}).encode()
    return _eventstream.encode(headers, payload)


def exception_event(name: str, message: str) -> bytes:
    headers = {":message-type": "exception", ":exception-type": name}
    return _eventstream.encode(headers, json.dumps({"Message": message}).encode())


def events(connection: Connection) -> list[tuple[dict[str, str], bytes]]:
    """Every event-stream message the plugin sent on ``connection``."""
    return [_eventstream.decode(m) for m in connection.received if isinstance(m, bytes)]


def transcribe(peer: WsPeer, **kwargs: Any) -> aws.STT:
    return aws.STT(region="us-east-1", sample_rate=16000, base_url=peer.ws_url, **kwargs)


def frame(ms: int) -> rtc.AudioFrame:
    samples = 16 * ms
    return rtc.AudioFrame(b"\x00\x00" * samples, 16000, 1, samples)


async def next_final(stream: stt.RecognizeStream) -> stt.SpeechEvent:
    async with asyncio.timeout(2):
        async for event in stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                return event
    raise AssertionError("the stream ended without a final transcript")


def query(peer: WsPeer, connection: int = 0) -> dict[str, str]:
    return {
        k: v[0]
        for k, v in parse_qs(urlsplit(str(peer.connections[connection].request.url)).query).items()
    }


def test_event_stream_messages_round_trip_and_reject_a_bad_checksum() -> None:
    """An encoded message decodes to its headers and payload; a flipped byte fails its CRC.

    Catches a framing that Transcribe would reject or a corrupt frame read as a message."""
    message = _eventstream.encode({":event-type": "AudioEvent"}, b"pcm")

    assert _eventstream.decode(message) == ({":event-type": "AudioEvent"}, b"pcm")
    corrupt = message[:-5] + bytes([message[-5] ^ 1]) + message[-4:]
    with pytest.raises(ValueError):
        _eventstream.decode(corrupt)


async def test_audio_goes_out_as_audio_events_ended_by_an_empty_one(ws_peer: WsPeer) -> None:
    """Each audio chunk is an AudioEvent and the end of input is an empty one, after which
    Transcribe's close ends the stream cleanly."""
    ws_peer.reply = lambda connection, _: [Close(1000)] if events(connection)[-1][1] == b"" else []
    provider = transcribe(ws_peer)

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100))
        stream.end_input()
        async with asyncio.timeout(2):
            _ = [event async for event in stream]

    decoded = events(ws_peer.connections[0])
    assert all(h[":event-type"] == "AudioEvent" for h, _ in decoded)
    assert sum(len(p) for _, p in decoded) == 3200
    assert decoded[-1][1] == b""


@pytest.mark.parametrize("explicit", [True, False])
async def test_transcribe_signs_with_explicit_or_environment_credentials(
    ws_peer: WsPeer, explicit: bool
) -> None:
    """Explicit credentials win, else the environment's, session token included.

    https://github.com/livekit/agents/issues/1719 https://github.com/livekit/agents/pull/4403
    """
    credentials = aws_stt.Credentials("arg-key", "arg-secret", "arg-token")
    provider = transcribe(ws_peer, credentials=credentials) if explicit else transcribe(ws_peer)

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(20))
        await ws_peer.connected()

    signed = query(ws_peer)
    prefix = "arg" if explicit else "env"
    assert signed["X-Amz-Credential"].startswith(f"{prefix}-key/")
    assert signed["X-Amz-Credential"].endswith("/us-east-1/transcribe/aws4_request")
    assert signed["X-Amz-Security-Token"] == f"{prefix}-token"
    assert signed["X-Amz-SignedHeaders"] == "host"
    assert len(signed["X-Amz-Signature"]) == 64


@pytest.mark.parametrize("identify", [False, True])
async def test_final_keeps_the_response_language_and_confidence(
    ws_peer: WsPeer, identify: bool
) -> None:
    """Finals carry the language and confidence Transcribe returned, not defaults.

    https://github.com/livekit/agents/issues/3154 https://github.com/livekit/agents/issues/2842
    """
    ws_peer.reply = lambda connection, _: (
        [transcript_event("bonjour", "fr-FR", 0.87)] if len(connection.received) == 1 else []
    )
    provider = (
        transcribe(ws_peer, identify_language=True, language_options="en-US,fr-FR")
        if identify
        else transcribe(ws_peer, language="en-US")
    )

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100))
        final = await next_final(stream)

    request = query(ws_peer)
    expected = (None, "true") if identify else ("en-US", None)
    assert (request.get("language-code"), request.get("identify-language")) == expected
    assert "session-id" not in request
    speech = final.alternatives[0]
    assert (speech.text, speech.language, speech.confidence) == ("bonjour", "fr-FR", 0.87)


async def test_stream_language_applies_to_that_stream_only(ws_peer: WsPeer) -> None:
    """``stream(language=)`` reaches that request only. https://github.com/livekit/agents/pull/8374"""
    provider = transcribe(ws_peer, language="en-US")

    async with provider:
        for count, language in enumerate(("fr-FR", None), 1):
            kwargs = {"language": language} if language else {}
            async with provider.stream(conn_options=OPTIONS, **kwargs) as stream:
                stream.push_frame(frame(20))
                await ws_peer.connected(count)

    assert [query(ws_peer, i)["language-code"] for i in range(2)] == ["fr-FR", "en-US"]


async def test_silent_audio_reconnects_before_the_idle_timeout(ws_peer: WsPeer) -> None:
    """No audio for 14 s opens a new stream before Transcribe fails the idle one at 15 s,
    without spending a retry or reporting an error, and the new one keeps transcribing.

    https://github.com/livekit/agents/issues/5938 https://github.com/livekit/agents/issues/2099
    """
    provider = transcribe(ws_peer)
    errors: list[stt.STTError] = []
    provider.on("error", errors.append)

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100))
        first = await ws_peer.connected()
        await until(lambda: first.received, "the audio")
        _core.skip_host_time(14.0)
        await ws_peer.connected(2)
        await ws_peer.send(transcript_event("still here"))
        final = await next_final(stream)

    assert final.alternatives[0].text == "still here"
    assert errors == []


@pytest.mark.parametrize(
    "name, kind",
    [("LimitExceededException", "rate_limit"), ("BadRequestException", "invalid_request")],
)
async def test_an_exception_event_fails_the_stream_by_its_kind(
    ws_peer: WsPeer, name: str, kind: str
) -> None:
    """A Transcribe exception fails the attempt with its kind and message, not as a lost
    connection."""
    ws_peer.reply = lambda connection, _: (
        [exception_event(name, "refused")] if len(connection.received) == 1 else []
    )
    provider = transcribe(ws_peer)

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100))
        with pytest.raises(APIError) as failure:
            await next_final(stream)

    assert (failure.value.kind, failure.value.detail) == (kind, "refused")


async def test_a_corrupt_event_fails_the_stream_as_a_protocol_error(ws_peer: WsPeer) -> None:
    """A message failing its checksum breaks the protocol: the stream fails for good."""
    good = transcript_event("hello")
    corrupt = good[:-1] + bytes([good[-1] ^ 1])
    ws_peer.reply = lambda connection, _: [corrupt] if len(connection.received) == 1 else []
    provider = transcribe(ws_peer)

    async with provider, provider.stream(conn_options=APIConnectOptions(max_retry=2)) as stream:
        stream.push_frame(frame(100))
        with pytest.raises(APIError) as failure:
            await next_final(stream)

    assert failure.value.kind == "protocol"
    assert len(ws_peer.connections) == 1


async def test_polly_streams_the_whole_audio_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polly's streaming body is decoded into the complete audio.

    https://github.com/livekit/agents/issues/6538
    """
    requests: list[dict[str, object]] = []

    async def synthesize(request: web.Request) -> web.StreamResponse:
        requests.append(await request.json())
        response = web.StreamResponse(
            headers={
                "Content-Type": "audio/mpeg",
                "x-amzn-RequestCharacters": "5",
                "x-amzn-RequestId": "polly-1",
            }
        )
        await response.prepare(request)
        for offset in range(0, len(TONE_MP3), 100):
            await response.write(TONE_MP3[offset : offset + 100])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/speech", synthesize)
    async with TestServer(app) as server:
        monkeypatch.setenv("AWS_ENDPOINT_URL_POLLY", str(server.make_url("")))
        provider = aws.TTS(region="us-east-1", sample_rate=16000)
        async with provider.synthesize("hello", conn_options=OPTIONS) as stream:
            frames = [event.frame async for event in stream]
        await provider.aclose()

    assert requests[0]["Text"] == "hello"
    # MP3 decoding adds encoder padding, never less than the encoded 0.2 s
    assert 0.2 <= sum(frame.duration for frame in frames) < 0.4
    assert any(any(frame.data) for frame in frames)
