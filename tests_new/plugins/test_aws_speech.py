"""AWS speech: Transcribe streaming STT behind its SDK client, and Polly TTS over HTTP.

Transcribe runs over a CRT HTTP/2 event stream, so the tests stand in for the SDK's
``TranscribeStreamingClient``; Polly is a plain local endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from smithy_core.exceptions import CallError

from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.agents.metrics import STTMetrics
from livekit.plugins import aws
from livekit.plugins.aws import stt as aws_stt

if TYPE_CHECKING:  # the Transcribe SDK imports only on Python 3.12+
    from aws_sdk_transcribe_streaming.config import AsyncTranscribeStreamingConfig
    from aws_sdk_transcribe_streaming.models import (
        BadRequestException,
        StartStreamTranscriptionInput,
        TranscriptResultStream,
    )

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the AWS Transcribe Streaming SDK requires Python 3.12+"
)

OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0, connect_timeout=0.5, read_timeout=0.5)
IDLE_TIMEOUT = "Your request timed out because no new audio was received for 15 seconds."

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


class Transcribe:
    """StartStreamTranscription: records each client config and request, replays ``events``.

    ``first_error`` fails the first request's output, as Transcribe's idle timeout does.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.configs: list[AsyncTranscribeStreamingConfig] = []
        self.requests: list[StartStreamTranscriptionInput] = []
        self.opened = asyncio.Event()
        self.events: asyncio.Queue[TranscriptResultStream | None] = asyncio.Queue()
        self.first_error: Exception | None = None
        service = self

        class Client:
            def __init__(self, *, config: AsyncTranscribeStreamingConfig) -> None:
                service.configs.append(config)

            async def start_stream_transcription(
                self, *, input: StartStreamTranscriptionInput
            ) -> SimpleNamespace:
                service.requests.append(input)
                service.opened.set()
                error, service.first_error = service.first_error, None
                return SimpleNamespace(
                    input_stream=SimpleNamespace(send=service.audio, close=service.end),
                    await_output=lambda: service.output(error),
                )

        monkeypatch.setattr(aws_stt, "TranscribeStreamingClient", Client)

    async def audio(self, event: object) -> None:
        pass

    async def end(self) -> None:
        self.events.put_nowait(None)  # Transcribe ends its results once audio is closed

    async def output(
        self, error: Exception | None
    ) -> tuple[None, AsyncIterator[TranscriptResultStream]]:
        if error is not None:
            raise error
        return None, self.results()

    async def results(self) -> AsyncIterator[TranscriptResultStream]:
        while (event := await self.events.get()) is not None:
            yield event


@pytest.fixture
def transcribe(monkeypatch: pytest.MonkeyPatch) -> Transcribe:
    return Transcribe(monkeypatch)


def bad_request(message: str) -> BadRequestException:
    from aws_sdk_transcribe_streaming.models import BadRequestException

    return BadRequestException(message)


def transcript_event(
    text: str, language: str = "en-US", confidence: float = 0.9
) -> TranscriptResultStream:
    from aws_sdk_transcribe_streaming import models

    item = models.Item(start_time=0.1, end_time=0.5, content=text, confidence=confidence)
    result = models.Result(
        start_time=0.1,
        end_time=0.5,
        is_partial=False,
        alternatives=[models.Alternative(transcript=text, items=[item])],
        language_code=language,
    )
    event = models.TranscriptEvent(transcript=models.Transcript(results=[result]))
    return models.TranscriptResultStreamTranscriptEvent(value=event)


def frame(ms: int) -> rtc.AudioFrame:
    samples = 16 * ms
    return rtc.AudioFrame(b"\x00\x00" * samples, 16000, 1, samples)


async def next_final(stream: stt.RecognizeStream) -> stt.SpeechEvent:
    async for event in stream:
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            return event
    raise AssertionError("the stream ended without a final transcript")


async def test_usage_bills_the_audio_sent(transcribe: Transcribe) -> None:
    """Sent audio is reported as usage and as streamed STT metrics.

    https://github.com/livekit/agents/issues/6650
    """
    provider = aws.STT(region="us-east-1", sample_rate=16000)
    metrics: list[STTMetrics] = []
    provider.on("metrics_collected", metrics.append)

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100))
        stream.flush()
        event = await anext(stream)

    assert event.type == stt.SpeechEventType.RECOGNITION_USAGE
    assert event.recognition_usage is not None
    assert event.recognition_usage.audio_duration == pytest.approx(0.1)
    assert metrics[0].audio_duration == pytest.approx(0.1)
    assert metrics[0].streamed is True


async def test_each_attempt_gets_its_own_crt_transport(
    transcribe: Transcribe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reopened stream gets a fresh HTTP/2 transport; the dead one is closed.

    https://github.com/livekit/agents/pull/7094
    """
    from smithy_http.aio.crt import AWSCRTHTTPClient

    closed: list[AWSCRTHTTPClient] = []
    crt_close = AWSCRTHTTPClient.close

    async def tracking_close(self: AWSCRTHTTPClient) -> None:
        closed.append(self)
        await crt_close(self)

    monkeypatch.setattr(AWSCRTHTTPClient, "close", tracking_close)
    transcribe.first_error = bad_request(IDLE_TIMEOUT)
    provider = aws.STT(region="us-east-1", sample_rate=16000)

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100))
        transcribe.events.put_nowait(transcript_event("reopened"))
        await next_final(stream)
        first, second = transcribe.configs
        closed_while_running = list(closed)

    assert isinstance(first.transport, AWSCRTHTTPClient)
    assert second.transport is not first.transport
    assert closed_while_running == [first.transport]
    assert closed == [first.transport, second.transport]


@pytest.mark.parametrize("explicit", [True, False])
async def test_transcribe_signs_with_explicit_or_environment_credentials(
    transcribe: Transcribe, explicit: bool
) -> None:
    """Explicit credentials win, else the environment's, session token included.

    https://github.com/livekit/agents/issues/1719 https://github.com/livekit/agents/pull/4403
    """
    credentials = aws_stt.Credentials("arg-key", "arg-secret", "arg-token")
    provider = (
        aws.STT(region="us-east-1", credentials=credentials)
        if explicit
        else aws.STT(region="us-east-1")
    )

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(20))
        await transcribe.opened.wait()
        resolver = transcribe.configs[0].aws_credentials_identity_resolver
        identity = await resolver.get_identity(properties={})

    prefix = "arg" if explicit else "env"
    assert (identity.access_key_id, identity.secret_access_key, identity.session_token) == (
        f"{prefix}-key",
        f"{prefix}-secret",
        f"{prefix}-token",
    )


@pytest.mark.parametrize("identify", [False, True])
async def test_final_keeps_the_response_language_and_confidence(
    transcribe: Transcribe, identify: bool
) -> None:
    """Finals carry the language and confidence Transcribe returned, not defaults.

    https://github.com/livekit/agents/issues/3154 https://github.com/livekit/agents/issues/2842
    """
    provider = (
        aws.STT(region="us-east-1", identify_language=True, language_options="en-US,fr-FR")
        if identify
        else aws.STT(region="us-east-1", language="en-US")
    )

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(20))
        transcribe.events.put_nowait(transcript_event("bonjour", "fr-FR", 0.87))
        final = await next_final(stream)

    request = transcribe.requests[0]
    expected = (None, True) if identify else ("en-US", False)
    assert (request.language_code, request.identify_language) == expected
    assert request.session_id is None
    speech = final.alternatives[0]
    assert (speech.text, speech.language, speech.confidence) == ("bonjour", "fr-FR", 0.87)


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8374 reproduces: STT.stream(language=) "
    "is ignored",
)
async def test_stream_language_applies_to_that_stream_only(transcribe: Transcribe) -> None:
    """``stream(language=)`` reaches that request only. https://github.com/livekit/agents/pull/8374"""
    provider = aws.STT(region="us-east-1", language="en-US")

    async with provider:
        for language in ("fr-FR", None):
            transcribe.opened.clear()
            kwargs = {"language": language} if language else {}
            async with provider.stream(conn_options=OPTIONS, **kwargs) as stream:
                stream.push_frame(frame(20))
                await transcribe.opened.wait()

    assert [request.language_code for request in transcribe.requests] == ["fr-FR", "en-US"]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(lambda: bad_request(IDLE_TIMEOUT), id="bad-request"),
        pytest.param(
            lambda: CallError(
                "Unknown error for operation "
                "com.amazonaws.transcribestreaming#StartStreamTranscription - status: 400 - "
                f"id: com.amazonaws.transcribestreaming#BadRequestException - {IDLE_TIMEOUT}"
            ),
            id="call-error",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/5938 reproduces: the idle "
                "timeout surfaced as a CallError fails the stream",
            ),
        ),
    ],
)
async def test_idle_timeout_reopens_and_keeps_transcribing(
    transcribe: Transcribe, error: Callable[[], Exception]
) -> None:
    """Transcribe's 15 s idle timeout reopens the stream, which keeps transcribing.

    https://github.com/livekit/agents/issues/5938 https://github.com/livekit/agents/issues/2099
    """
    transcribe.first_error = error()
    provider = aws.STT(region="us-east-1")

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(20))
        transcribe.events.put_nowait(transcript_event("still here"))
        final = await next_final(stream)

    assert final.alternatives[0].text == "still here"
    assert len(transcribe.requests) == 2


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
