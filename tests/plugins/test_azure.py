"""Azure plugin: Speech SDK recognizer behaviour of the streaming STT, and the REST TTS."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from collections.abc import Callable
from types import SimpleNamespace

import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-untyped]  # no stubs
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

# Speech SDK recognizer seam


class Signal:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self.callbacks.append(callback)

    def fire(self, event: object = None) -> None:
        for callback in self.callbacks:
            callback(event)


class PushStream:
    def __init__(self, stream_format: SimpleNamespace) -> None:
        self.format = stream_format
        self.audio = bytearray()
        self.recognizer: Recognizer | None = None

    def write(self, data: bytes) -> None:
        self.audio += data

    def close(self) -> None:
        assert self.recognizer is not None
        self.recognizer.input_closed()


class Recognizer:
    """A Speech SDK recognizer that, like the service, finalizes buffered audio on close."""

    def __init__(
        self,
        sdk: SpeechSDK,
        speech_config: speechsdk.SpeechConfig,
        audio_config: SimpleNamespace,
        **kwargs: object,
    ) -> None:
        self.config = speech_config
        self.kwargs = kwargs
        self.push: PushStream = audio_config.stream
        self.push.recognizer = self
        self.log: list[str] = []
        self.recognizing = Signal()
        self.recognized = Signal()
        self.speech_start_detected = Signal()
        self.speech_end_detected = Signal()
        self.session_started = Signal()
        self.session_stopped = Signal()
        self.canceled = Signal()
        sdk.recognizers.append(self)
        self.sdk = sdk

    def start_continuous_recognition(self) -> None:
        self.log.append("start")
        self.session_started.fire()

    def stop_continuous_recognition(self) -> None:
        self.log.append("stop")

    def input_closed(self) -> None:
        self.log.append("input closed")
        for text in self.sdk.finals_on_close:
            result = SimpleNamespace(text=text, offset=0, duration=10**7, language="")
            self.recognized.fire(SimpleNamespace(result=result))
        self.session_stopped.fire()


class SpeechSDK:
    def __init__(self) -> None:
        self.recognizers: list[Recognizer] = []
        self.finals_on_close: list[str] = []

    async def recognizer(self, index: int) -> Recognizer:
        async with asyncio.timeout(2):
            while len(self.recognizers) <= index:
                await asyncio.sleep(0.01)
        return self.recognizers[index]


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> SpeechSDK:
    fake = SpeechSDK()
    monkeypatch.setattr(speechsdk, "SpeechRecognizer", lambda **kwargs: Recognizer(fake, **kwargs))
    monkeypatch.setattr(speechsdk.audio, "AudioStreamFormat", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(speechsdk.audio, "PushAudioInputStream", PushStream)
    monkeypatch.setattr(
        speechsdk.audio, "AudioConfig", lambda stream: SimpleNamespace(stream=stream)
    )
    monkeypatch.setattr(
        speechsdk.languageconfig,
        "AutoDetectSourceLanguageConfig",
        lambda languages: SimpleNamespace(languages=languages),
    )
    monkeypatch.setattr(
        speechsdk, "AutoDetectSourceLanguageResult", lambda result: SimpleNamespace(language="")
    )
    return fake


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


def language_mode(recognizer: Recognizer) -> str:
    return str(
        recognizer.config.get_property(speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode)
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
    sdk: SpeechSDK, configured: str | list[str], override: str | None, expected: list[str]
) -> None:
    """One language is set explicitly, several enable continuous identification, and the
    stream() and update_options() languages are honored.

    https://github.com/livekit/agents/issues/1858 https://github.com/livekit/agents/issues/4449
    """
    provider = azure_stt.STT(speech_key="key", speech_region="westus", language=configured)
    kwargs = {"language": override} if override else {}
    async with provider.stream(conn_options=OPTIONS, **kwargs) as stream:
        recognizer = await sdk.recognizer(0)
        provider.update_options(language="it-IT")
        updated = await sdk.recognizer(1)
        sdk.finals_on_close = ["hello"]
        stream.end_input()
        final = await finals(stream)

    for languages, current in ((expected, recognizer), (["it-IT"], updated)):
        if len(languages) == 1:
            assert current.kwargs == {"language": languages[0]}
            assert language_mode(current) == ""
        else:
            detection = current.kwargs["auto_detect_source_language_config"]
            assert isinstance(detection, SimpleNamespace) and detection.languages == languages
            assert "language" not in current.kwargs
            assert language_mode(current) == "Continuous"
    assert [(speech.text, speech.language) for speech in final] == [("hello", "it-IT")]


async def test_end_input_flushes_buffered_finals_before_teardown(sdk: SpeechSDK) -> None:
    """Ending input delivers every pending final before recognition stops.

    https://github.com/livekit/agents/pull/6118
    """
    sdk.finals_on_close = ["first phrase", "second phrase"]
    provider = azure_stt.STT(speech_key="key", speech_region="westus")
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame(100, 16000))
        stream.end_input()
        final = await finals(stream)
    recognizer = await sdk.recognizer(0)
    assert [speech.text for speech in final] == ["first phrase", "second phrase"]
    assert recognizer.log == ["start", "input closed", "stop"]


@pytest.mark.parametrize("sample_rate", [8000, 48000])
async def test_recognizer_audio_matches_the_configured_sample_rate(
    sdk: SpeechSDK, sample_rate: int
) -> None:
    """The push stream is declared and fed at the configured rate, whatever the frames' rate.

    https://github.com/livekit/agents/pull/1072
    """
    provider = azure_stt.STT(speech_key="key", speech_region="westus", sample_rate=sample_rate)
    async with provider.stream(conn_options=OPTIONS) as stream:
        for _ in range(10):
            stream.push_frame(frame(20, 16000))
        stream.end_input()
        await finals(stream)
    push = (await sdk.recognizer(0)).push
    assert (push.format.samples_per_second, push.format.channels) == (sample_rate, 1)
    seconds = len(push.audio) / 2 / sample_rate
    assert seconds == pytest.approx(0.2, abs=0.02)


async def test_stt_custom_endpoint_needs_no_region(sdk: SpeechSDK) -> None:
    """A key plus a private endpoint configures the recognizer for that endpoint.

    https://github.com/livekit/agents/issues/1361
    """
    endpoint = "wss://speech.private.example/stt/speech/universal/v2"
    provider = azure_stt.STT(speech_key="key", speech_endpoint=endpoint)
    async with provider.stream(conn_options=OPTIONS) as stream:
        stream.end_input()
        await finals(stream)
    config = (await sdk.recognizer(0)).config
    assert config.get_property(speechsdk.PropertyId.SpeechServiceConnection_Endpoint) == endpoint
    assert config.get_property(speechsdk.PropertyId.SpeechServiceConnection_Region) == ""


async def test_processed_audio_is_reported_as_usage(sdk: SpeechSDK) -> None:
    """The audio a stream fed the recognizer is reported as recognition usage.

    https://github.com/livekit/agents/issues/6151
    """
    provider = azure_stt.STT(speech_key="key", speech_region="westus")
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


async def test_an_error_cancellation_fails_the_stream_instead_of_hanging(sdk: SpeechSDK) -> None:
    """A service-side cancellation ends the stream with an API error naming its cause.

    https://github.com/livekit/agents/issues/5322
    """
    provider = azure_stt.STT(speech_key="key", speech_region="westus")
    details = SimpleNamespace(
        reason=speechsdk.CancellationReason.Error,
        code=speechsdk.CancellationErrorCode.ServiceTimeout,
        error_details="service timeout",
    )
    async with provider.stream(conn_options=OPTIONS) as stream:
        recognizer = await sdk.recognizer(0)

        recognizer.canceled.fire(SimpleNamespace(cancellation_details=details))

        with pytest.raises(APIError, match="service timeout"):
            await finals(stream)


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
