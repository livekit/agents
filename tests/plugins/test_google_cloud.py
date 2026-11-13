"""Google Cloud Speech-to-Text and Text-to-Speech against the SDK clients replying with recorded protos.

The plugin talks gRPC through google-cloud-speech and google-cloud-texttospeech, whose endpoints
are fixed per region, so each test installs a stand-in for the SDK client class the plugin builds.
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from types import ModuleType
from typing import TypeAlias

import google.auth
import pytest
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import Aborted, BadRequest, DeadlineExceeded
from google.auth.credentials import AnonymousCredentials, Credentials as GoogleCredentials
from google.cloud import texttospeech
from google.cloud.speech_v1.types import cloud_speech as v1
from google.cloud.speech_v2.types import cloud_speech as v2
from google.oauth2.credentials import Credentials
from google.protobuf.duration_pb2 import Duration

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
)
from livekit.agents.metrics import STTMetrics
from livekit.agents.types import TranscriptChunk
from livekit.plugins.google import STT, TTS

OPTIONS = APIConnectOptions(max_retry=0)
FRAME = rtc.AudioFrame(bytes(3200), 16000, 1, 1600)


RecognizeRequest: TypeAlias = "v1.RecognizeRequest | v2.RecognizeRequest"
StreamingRequest: TypeAlias = "v1.StreamingRecognizeRequest | v2.StreamingRecognizeRequest"
RecognizeResponse: TypeAlias = "v1.RecognizeResponse | v2.RecognizeResponse"
StreamingResponse: TypeAlias = "v1.StreamingRecognizeResponse | v2.StreamingRecognizeResponse"
StreamingRecognize: TypeAlias = (
    "Callable[[AsyncIterator[StreamingRequest]], Awaitable[AsyncIterator[StreamingResponse]]]"
)


def anonymous() -> AnonymousCredentials:
    return AnonymousCredentials()  # type: ignore[no-untyped-call]  # google-auth is untyped


def token_credentials(token: str, project: str | None = None) -> Credentials:
    return Credentials(token=token, quota_project_id=project)  # type: ignore[no-untyped-call]  # google-auth is untyped


class SpeechAPI:
    """Cloud Speech as the SDK client: records requests and answers with recorded responses.

    `replies` answer `recognize` calls in order; `streams` holds one response list per
    `streaming_recognize` call, which reads the whole request stream first. A reply that is an
    exception is raised; `error`, when set, fails every call. `serve`, when set, answers
    streaming calls instead.
    """

    def __init__(self) -> None:
        self.endpoints: list[str | None] = []
        self.credentials: list[GoogleCredentials | None] = []
        self.recognized: list[RecognizeRequest] = []
        self.streamed: list[StreamingRequest] = []
        self.replies: list[RecognizeResponse | Exception] = []
        self.streams: list[list[StreamingResponse | Exception]] = []
        self.error: Exception | None = None
        self.serve: StreamingRecognize | None = None

    def __call__(
        self,
        *,
        credentials: GoogleCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> SpeechAPI:
        self.endpoints.append(client_options.api_endpoint if client_options else None)
        self.credentials.append(credentials)
        return self

    async def recognize(self, request: RecognizeRequest, **_: object) -> RecognizeResponse:
        if self.error is not None:
            raise self.error
        self.recognized.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def streaming_recognize(
        self, *, requests: AsyncIterator[StreamingRequest], **_: object
    ) -> AsyncIterator[StreamingResponse]:
        if self.error is not None:
            raise self.error
        if self.serve is not None:
            return await self.serve(requests)
        replies = self.streams.pop(0)
        async for request in requests:
            self.streamed.append(request)

        async def respond() -> AsyncIterator[StreamingResponse]:
            for reply in replies:
                if isinstance(reply, Exception):
                    raise reply
                yield reply

        return respond()


@pytest.fixture
def speech(monkeypatch: pytest.MonkeyPatch) -> SpeechAPI:
    from livekit.plugins.google import stt as google_stt

    api = SpeechAPI()

    class Client:  # the plugin builds its client from the SDK class and unions the classes
        def __new__(  # type: ignore[misc]  # stands in for the SDK client
            cls,
            *,
            credentials: GoogleCredentials | None = None,
            client_options: ClientOptions | None = None,
        ) -> SpeechAPI:
            return api(credentials=credentials, client_options=client_options)

    monkeypatch.setattr(google_stt, "SpeechAsyncClientV1", Client)
    monkeypatch.setattr(google_stt, "SpeechAsyncClientV2", Client)
    return api


def result(
    speech_api: ModuleType = v2,
    text: str = "Hello",
    *,
    final: bool = True,
    confidence: float | None = None,
) -> StreamingResponse:
    """A streaming response carrying one result."""
    scored = {} if confidence is None else {"confidence": confidence}
    response: StreamingResponse = speech_api.StreamingRecognizeResponse(
        results=[
            speech_api.StreamingRecognitionResult(
                alternatives=[speech_api.SpeechRecognitionAlternative(transcript=text, **scored)],
                is_final=final,
                language_code="en-US",
            )
        ]
    )
    return response


def one_word(
    speech_api: ModuleType, start: Duration, end: Duration
) -> v1.SpeechRecognitionAlternative | v2.SpeechRecognitionAlternative:
    """An alternative with the single word "test", in the v1 or v2 offset field names."""
    fields = ("start_offset", "end_offset") if speech_api is v2 else ("start_time", "end_time")
    alternative: v1.SpeechRecognitionAlternative | v2.SpeechRecognitionAlternative = (
        speech_api.SpeechRecognitionAlternative(
            transcript="test",
            confidence=1.0,
            words=[
                speech_api.WordInfo(
                    word="test", confidence=1.0, **dict(zip(fields, (start, end), strict=True))
                )
            ],
        )
    )
    return alternative


async def transcribe(provider: STT) -> list[stt.SpeechEvent]:
    """Streams one frame and returns every event; the provider is closed afterwards."""
    try:
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(FRAME)
            stream.end_input()
            return [event async for event in stream]
    finally:
        await provider.aclose()


def finals(events: list[stt.SpeechEvent]) -> list[str]:
    return [
        e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]


def cloud_stt(
    model: str = "chirp_2",
    *,
    languages: str = "en-US",
    location: str = "global",
    min_confidence_threshold: float = 0.65,
    enable_voice_activity_events: bool = False,
) -> STT:
    return STT(
        credentials=anonymous(),
        project="test",
        model=model,
        languages=languages,
        location=location,
        min_confidence_threshold=min_confidence_threshold,
        enable_voice_activity_events=enable_voice_activity_events,
    )


# ---- streaming recognition


@pytest.mark.parametrize("language, text", [("ja-JP", "こんにちは"), ("fr-FR", "Bonjour")])
async def test_activity_boundaries_deliver_exactly_one_final_transcript(
    speech: SpeechAPI, language: str, text: str
) -> None:
    """Speech start and end frame exactly one final transcript.

    https://github.com/livekit/agents/issues/394 https://github.com/livekit/agents/issues/3435
    """
    events = v2.StreamingRecognizeResponse.SpeechEventType
    speech.streams = [
        [
            v2.StreamingRecognizeResponse(speech_event_type=events.SPEECH_ACTIVITY_BEGIN),
            result(text=text[:2], final=False),
            result(text=text),
            v2.StreamingRecognizeResponse(speech_event_type=events.SPEECH_ACTIVITY_END),
        ]
    ]
    provider = cloud_stt(
        languages=language, min_confidence_threshold=0, enable_voice_activity_events=True
    )

    received = await transcribe(provider)

    assert [e.type for e in received] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert finals(received) == [text]
    request = speech.streamed[0]
    assert isinstance(request, v2.StreamingRecognizeRequest)
    config = request.streaming_config
    assert config.streaming_features.enable_voice_activity_events
    assert list(config.config.language_codes) == [language]


@pytest.mark.parametrize("speech_api, model", [(v1, "latest_long"), (v2, "chirp_2")])
@pytest.mark.parametrize("confidence", [None, 0.1, 0.9])
async def test_final_transcripts_bypass_the_confidence_filter_but_interims_do_not(
    speech: SpeechAPI, speech_api: ModuleType, model: str, confidence: float | None
) -> None:
    """A final transcript is kept whatever its confidence; a low-confidence interim is dropped.

    https://github.com/livekit/agents/issues/3495 https://github.com/livekit/agents/issues/1103
    """
    speech.streams = [
        [
            result(speech_api, "Interim", final=False, confidence=confidence),
            result(speech_api, "Final", confidence=confidence),
        ]
    ]

    received = await transcribe(cloud_stt(model, min_confidence_threshold=0.5))

    assert finals(received) == ["Final"]
    interims = [e.alternatives[0].text for e in received if e.type.name == "INTERIM_TRANSCRIPT"]
    assert interims == (["Interim"] if confidence and confidence >= 0.5 else [])


async def test_a_response_with_several_results_reports_its_final_one(speech: SpeechAPI) -> None:
    """A response leads with its settled result; that one is the final transcript.

    https://github.com/livekit/agents/pull/3628
    """
    response = v2.StreamingRecognizeResponse(
        results=[
            v2.StreamingRecognitionResult(
                alternatives=[v2.SpeechRecognitionAlternative(transcript=text, confidence=1.0)],
                is_final=text == "best",
                language_code="te-ST",
            )
            for text in ("best", "test01", "test02")
        ]
    )
    speech.streams = [[response]]

    [final] = [e for e in await transcribe(cloud_stt()) if e.type.name == "FINAL_TRANSCRIPT"]

    assert (final.alternatives[0].text, final.alternatives[0].language) == ("best", "te-ST")


@pytest.mark.parametrize("speech_api, model", [(v1, "latest_long"), (v2, "chirp_2")])
async def test_word_timings_land_on_the_stream_timeline(
    speech: SpeechAPI, speech_api: ModuleType, model: str
) -> None:
    """Word offsets reach the transcript, shifted to where the stream sits on the input.

    https://github.com/livekit/agents/pull/4338
    """
    alternative = one_word(speech_api, Duration(seconds=0), Duration(seconds=1))
    speech.streams = [
        [
            speech_api.StreamingRecognizeResponse(
                results=[
                    speech_api.StreamingRecognitionResult(alternatives=[alternative], is_final=True)
                ]
            )
        ]
    ]

    [final] = [e for e in await transcribe(cloud_stt(model)) if e.type.name == "FINAL_TRANSCRIPT"]

    [word] = final.alternatives[0].words or []
    assert (word.text, word.confidence) == ("test", 1.0)
    offset = word.start_time_offset
    assert offset is not None
    assert (word.start_time, word.end_time) == pytest.approx((offset, offset + 1))


async def test_a_result_without_word_offsets_reports_unknown_timing(speech: SpeechAPI) -> None:
    """A chirp_3 result carries no words, so its timing stays 0 rather than the stream offset.

    https://github.com/livekit/agents/issues/7684
    """
    heard = result(text="hello world", final=False, confidence=0.8)
    speech.streams = [[heard, result(text="hello world", confidence=0.8)]]

    received = [e for e in await transcribe(cloud_stt("chirp_3")) if e.alternatives]

    assert [e.type.name for e in received] == ["INTERIM_TRANSCRIPT", "FINAL_TRANSCRIPT"]
    for event in received:
        data = event.alternatives[0]
        assert (data.start_time, data.end_time) == (0, 0)
        assert not data.words


async def test_chirp_metrics_count_the_audio_without_word_timestamps(speech: SpeechAPI) -> None:
    """Chirp results carry no words; the recognized audio still reaches metrics.

    https://github.com/livekit/agents/issues/4785
    """
    response = result()
    response.speech_event_offset = Duration(nanos=100_000_000)  # the audio heard so far
    speech.streams = [[response]]
    provider = cloud_stt("chirp_3")
    metrics: list[STTMetrics] = []
    provider.on("metrics_collected", metrics.append)

    assert finals(await transcribe(provider)) == ["Hello"]

    [record] = metrics
    assert record.audio_duration == pytest.approx(FRAME.duration)


async def test_a_duration_limited_rpc_restarts_without_losing_new_audio(
    speech: SpeechAPI,
) -> None:
    """When Google ends the RPC at its duration limit, the next one carries the newer audio.

    https://github.com/livekit/agents/issues/1258 https://github.com/livekit/agents/pull/2189
    """
    first_ended = asyncio.Event()
    calls = 0

    async def serve(requests: AsyncIterator[StreamingRequest]) -> AsyncIterator[StreamingResponse]:
        nonlocal calls
        calls += 1
        speech.streamed += [await anext(requests), await anext(requests)]
        if calls == 1:
            first_ended.set()
            raise Aborted("maximum stream duration reached")  # type: ignore[no-untyped-call]  # api_core is untyped
        speech.streamed += [request async for request in requests]

        async def respond() -> AsyncIterator[StreamingResponse]:
            yield result(text="After restart")

        return respond()

    speech.serve = serve
    later = rtc.AudioFrame(b"\x01\x00" * 1600, 16000, 1, 1600)
    provider = cloud_stt()

    try:
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(FRAME)
            await first_ended.wait()
            stream.push_frame(later)
            stream.end_input()
            received = [event async for event in stream]
    finally:
        await provider.aclose()

    assert finals(received) == ["After restart"]
    audio = [
        bytes(r.audio)
        for r in speech.streamed
        if isinstance(r, v2.StreamingRecognizeRequest) and r.audio
    ]
    assert audio == [bytes(FRAME.data), bytes(later.data)]


async def test_an_rpc_closed_before_audio_releases_its_request_reader(speech: SpeechAPI) -> None:
    """A recognition the server ends before any audio fails, and its request reader is released.

    https://github.com/livekit/agents/issues/5590 https://github.com/livekit/agents/issues/6181
    """
    reader_done = asyncio.Event()
    readers: list[asyncio.Task[None]] = []

    async def serve(requests: AsyncIterator[StreamingRequest]) -> AsyncIterator[StreamingResponse]:
        await anext(requests)

        async def read_rest() -> None:
            try:
                async for _ in requests:
                    pass
            finally:
                reader_done.set()

        readers.append(asyncio.create_task(read_rest()))

        async def respond() -> AsyncIterator[StreamingResponse]:  # the server closes, no result
            return
            yield

        return respond()

    speech.serve = serve
    provider = cloud_stt()

    async with provider, provider.stream(conn_options=OPTIONS) as stream:
        with pytest.raises(APIConnectionError, match="recognition ended"):
            _ = [event async for event in stream]

    async with asyncio.timeout(1):  # the reader is cancelled, not left parked on the input
        await reader_done.wait()


# ---- one-shot recognition


@pytest.mark.parametrize("speech_api, model", [(v1, "latest_long"), (v2, "chirp_2")])
@pytest.mark.parametrize(
    "empty_alternatives",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/683 reproduces: unary recognition indexes empty alternatives",
            ),
        ),
    ],
)
async def test_an_empty_recognition_does_not_prevent_later_speech(
    speech: SpeechAPI, speech_api: ModuleType, model: str, empty_alternatives: bool
) -> None:
    """An empty result is reported as no alternatives, and the next recognition still works.

    https://github.com/livekit/agents/pull/680 https://github.com/livekit/agents/issues/683
    """
    speech.replies = [
        speech_api.RecognizeResponse(
            results=[speech_api.SpeechRecognitionResult()] if empty_alternatives else []
        ),
        speech_api.RecognizeResponse(
            results=[
                speech_api.SpeechRecognitionResult(
                    alternatives=[speech_api.SpeechRecognitionAlternative(transcript="Recovered")],
                    language_code="en-US",
                )
            ]
        ),
    ]
    provider = cloud_stt(model)

    try:
        first = await provider.recognize(FRAME, conn_options=OPTIONS)
        second = await provider.recognize(FRAME, conn_options=OPTIONS)
    finally:
        await provider.aclose()

    assert first.alternatives == []
    assert [alternative.text for alternative in second.alternatives] == ["Recovered"]


@pytest.mark.parametrize("speech_api, model", [(v1, "latest_long"), (v2, "chirp_2")])
async def test_fractional_word_durations_parse(
    speech: SpeechAPI, speech_api: ModuleType, model: str
) -> None:
    """Sub-second word offsets keep their fraction. https://github.com/livekit/agents/pull/3252"""
    alternative = one_word(
        speech_api, Duration(nanos=125000000), Duration(seconds=1, nanos=750000000)
    )
    speech.replies = [
        speech_api.RecognizeResponse(
            results=[
                speech_api.SpeechRecognitionResult(alternatives=[alternative], language_code="en")
            ]
        )
    ]
    provider = cloud_stt(model)

    try:
        [data] = (await provider.recognize(FRAME, conn_options=OPTIONS)).alternatives
    finally:
        await provider.aclose()

    assert (data.start_time, data.end_time) == (0.125, 1.75)
    assert data.words is not None
    assert (data.words[0].start_time, data.words[0].end_time) == (0.125, 1.75)


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/4338: recognize() words carry no confidence or offset",
)
@pytest.mark.parametrize("speech_api, model", [(v1, "latest_long"), (v2, "chirp_2")])
async def test_recognized_words_carry_their_timing_and_confidence(
    speech: SpeechAPI, speech_api: ModuleType, model: str
) -> None:
    """One-shot recognition reports its words like streaming does.

    https://github.com/livekit/agents/pull/4338
    """
    alternative = one_word(speech_api, Duration(seconds=0), Duration(seconds=1))
    speech.replies = [
        speech_api.RecognizeResponse(
            results=[
                speech_api.SpeechRecognitionResult(
                    alternatives=[alternative], language_code="te-ST"
                )
            ]
        )
    ]
    provider = cloud_stt(model)

    try:
        event = await provider.recognize(FRAME, conn_options=OPTIONS)
    finally:
        await provider.aclose()

    assert event.alternatives == [
        stt.SpeechData(
            language=LanguageCode("te-ST"),
            start_time=0.0,
            end_time=1.0,
            text="test",
            confidence=1.0,
            words=[
                TranscriptChunk(
                    text="test", start_time=0.0, end_time=1.0, start_time_offset=0.0, confidence=1.0
                )
            ],
        )
    ]


# ---- clients, regions and credentials


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_region_change_routes_to_a_fresh_regional_client(
    speech: SpeechAPI, streaming: bool
) -> None:
    """Changing location reaches the regional endpoint and recognizer.

    https://github.com/livekit/agents/pull/1342 https://github.com/livekit/agents/pull/1889
    """
    provider = cloud_stt(location="us-central1")
    try:
        for region in ("us-central1", "europe-west4"):
            provider.update_options(location=region)
            if streaming:
                speech.streams.append([result()])
                async with provider.stream(conn_options=OPTIONS) as stream:
                    stream.push_frame(FRAME)
                    stream.end_input()
                    assert finals([event async for event in stream]) == ["Hello"]
            else:
                speech.replies.append(
                    v2.RecognizeResponse(
                        results=[
                            v2.SpeechRecognitionResult(
                                alternatives=[v2.SpeechRecognitionAlternative(transcript="Hello")]
                            )
                        ]
                    )
                )
                assert (await provider.recognize(FRAME, conn_options=OPTIONS)).alternatives
    finally:
        await provider.aclose()

    assert speech.endpoints == [
        "us-central1-speech.googleapis.com",
        "europe-west4-speech.googleapis.com",
    ]
    requests = [*speech.streamed, *speech.recognized]
    v2_requests = (v2.StreamingRecognizeRequest, v2.RecognizeRequest)
    assert [r.recognizer for r in requests if isinstance(r, v2_requests) and r.recognizer] == [
        "projects/test/locations/us-central1/recognizers/_",
        "projects/test/locations/europe-west4/recognizers/_",
    ]


@pytest.mark.parametrize("speech_api, model", [(v1, "latest_long"), (v2, "chirp_2")])
@pytest.mark.parametrize("explicit", [False, True])
async def test_workload_credentials_need_no_service_account_file(
    speech: SpeechAPI,
    monkeypatch: pytest.MonkeyPatch,
    speech_api: ModuleType,
    model: str,
    explicit: bool,
) -> None:
    """Ambient or explicit workload credentials authenticate, and their project names the recognizer.

    https://github.com/livekit/agents/issues/1428 https://github.com/livekit/agents/issues/2387
    """
    from livekit.plugins.google import stt as google_stt

    for name in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    ambient = token_credentials("ambient-token", "workload-project")
    monkeypatch.setattr(google_stt, "gauth_default", lambda: (ambient, "workload-project"))
    monkeypatch.setattr(google.auth, "default", lambda *a, **kw: (ambient, "workload-project"))
    credentials = token_credentials("explicit-token", "workload-project")
    speech.replies = [
        speech_api.RecognizeResponse(
            results=[
                speech_api.SpeechRecognitionResult(
                    alternatives=[
                        speech_api.SpeechRecognitionAlternative(transcript="Authenticated")
                    ]
                )
            ]
        )
    ]
    provider = STT(model=model, credentials=credentials) if explicit else STT(model=model)

    try:
        event = await provider.recognize(FRAME, conn_options=OPTIONS)
    finally:
        await provider.aclose()

    assert event.alternatives[0].text == "Authenticated"
    assert speech.credentials == [credentials if explicit else None]
    if speech_api is v2:
        [request] = speech.recognized
        assert isinstance(request, v2.RecognizeRequest)
        assert request.recognizer == "projects/workload-project/locations/global/recognizers/_"


async def test_an_explicit_project_names_the_recognizer(speech: SpeechAPI) -> None:
    """The project passed in wins over the one the credentials carry.

    https://github.com/livekit/agents/issues/6772
    """
    speech.replies = [v2.RecognizeResponse()]
    credentials = token_credentials("t", "credentials-project")
    provider = STT(credentials=credentials, project="explicit-project", model="chirp_2")

    try:
        await provider.recognize(FRAME, conn_options=OPTIONS)
    finally:
        await provider.aclose()

    [request] = speech.recognized
    assert isinstance(request, v2.RecognizeRequest)
    assert request.recognizer == "projects/explicit-project/locations/global/recognizers/_"


# ---- Text-to-Speech


SynthesizeRequests: TypeAlias = "AsyncIterator[texttospeech.StreamingSynthesizeRequest]"
StreamingSynthesize: TypeAlias = "Callable[[SynthesizeRequests], Awaitable[AsyncIterator[texttospeech.StreamingSynthesizeResponse]]]"


class SpeechSynthesis:
    """Cloud Text-to-Speech as the SDK client: records requests, answers with `audio`.

    `error`, when set, fails every call; `serve`, when set, answers streaming calls instead.
    """

    def __init__(self, audio: bytes = b"\x01\x00" * 480) -> None:
        self.audio = audio
        self.endpoints: list[str | None] = []
        self.inputs: list[texttospeech.SynthesisInput] = []
        self.streamed: list[texttospeech.StreamingSynthesizeRequest] = []
        self.error: Exception | None = None
        self.serve: StreamingSynthesize | None = None

    def __call__(self, *, client_options: ClientOptions, **_: object) -> SpeechSynthesis:
        self.endpoints.append(client_options.api_endpoint)
        return self

    async def synthesize_speech(
        self, *, input: texttospeech.SynthesisInput, **_: object
    ) -> texttospeech.SynthesizeSpeechResponse:
        if self.error is not None:
            raise self.error
        self.inputs.append(input)
        return texttospeech.SynthesizeSpeechResponse(audio_content=self.audio)

    async def streaming_synthesize(
        self, requests: SynthesizeRequests, **_: object
    ) -> AsyncIterator[texttospeech.StreamingSynthesizeResponse]:
        if self.error is not None:
            raise self.error
        if self.serve is not None:
            return await self.serve(requests)

        async def respond() -> AsyncIterator[texttospeech.StreamingSynthesizeResponse]:
            async for request in requests:
                self.streamed.append(request)
            for half in (self.audio[:777], self.audio[777:]):
                yield texttospeech.StreamingSynthesizeResponse(audio_content=half)

        return respond()


@pytest.fixture
def synthesis(monkeypatch: pytest.MonkeyPatch) -> SpeechSynthesis:
    api = SpeechSynthesis()

    class Client:  # the plugin builds its client from the SDK class
        def __new__(cls, *, client_options: ClientOptions, **_: object) -> SpeechSynthesis:  # type: ignore[misc]  # stands in for the SDK client
            return api(client_options=client_options)

        @classmethod
        def from_service_account_info(
            cls, info: dict[str, str], *, client_options: ClientOptions, **_: object
        ) -> SpeechSynthesis:
            return api(client_options=client_options)

    monkeypatch.setattr(texttospeech, "TextToSpeechAsyncClient", Client)
    return api


def wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return output.getvalue()


@pytest.mark.parametrize(
    "streaming, encoding",
    [
        (False, texttospeech.AudioEncoding.PCM),
        (False, texttospeech.AudioEncoding.LINEAR16),
        (True, texttospeech.AudioEncoding.PCM),
    ],
)
async def test_synthesis_delivers_all_pcm_without_wav_header_clicks(
    synthesis: SpeechSynthesis, streaming: bool, encoding: texttospeech.AudioEncoding
) -> None:
    """Every PCM sample is played at the declared rate, and a WAV header is never played.

    https://github.com/livekit/agents/issues/3043 https://github.com/livekit/agents/issues/698
    """
    pcm = b"\x01\x02" * 480 + b"\x03\x04" * 480
    synthesis.audio = wav(pcm, 24000) if encoding == texttospeech.AudioEncoding.LINEAR16 else pcm
    provider = TTS(
        credentials=anonymous(),
        audio_encoding=encoding,
        sample_rate=24000,
    )

    async with provider:
        if streaming:
            async with provider.stream(conn_options=OPTIONS) as stream:
                stream.push_text("Hello.")
                stream.flush()
                stream.push_text("Goodbye.")
                stream.end_input()
                events = [event async for event in stream]
        else:
            async with provider.synthesize("Hello.", conn_options=OPTIONS) as stream:
                events = [event async for event in stream]

    assert b"".join(bytes(e.frame.data) for e in events) == pcm * (2 if streaming else 1)
    assert {(e.frame.sample_rate, e.frame.num_channels) for e in events} == {(24000, 1)}
    assert sum(e.is_final for e in events) == (2 if streaming else 1)


async def test_consecutive_ssml_requests_carry_only_their_own_text(
    synthesis: SpeechSynthesis,
) -> None:
    """Each SSML request holds its own utterance, never the ones before it.

    https://github.com/livekit/agents/issues/4762
    """
    sentences = ["నమస్కారం.", "మీకు ఎలా సహాయం చేయగలను?"] * 40
    provider = TTS(
        credentials=anonymous(),
        voice_name="te-IN-Chirp3-HD-Erinome",
        language="te-IN",
        use_streaming=False,
        enable_ssml=True,
    )

    async with provider:
        for sentence in sentences:
            async with provider.synthesize(sentence, conn_options=OPTIONS) as stream:
                assert [event async for event in stream]

    assert [synthesized.ssml for synthesized in synthesis.inputs] == [
        f"<speak>{s}</speak>" for s in sentences
    ]


@pytest.mark.parametrize("streaming", [False, True])
async def test_tts_region_selects_the_regional_endpoint(
    synthesis: SpeechSynthesis, streaming: bool
) -> None:
    """The TTS location picks its regional endpoint. https://github.com/livekit/agents/pull/1889"""
    for region in ("us", "eu"):
        provider = (
            TTS(location=region, credentials_info={"type": "service_account"})
            if streaming
            else TTS(location=region, credentials=anonymous())
        )
        async with provider:
            async with provider.synthesize("Hello", conn_options=OPTIONS) as stream:
                assert [event async for event in stream]

    assert synthesis.endpoints == [
        "us-texttospeech.googleapis.com",
        "eu-texttospeech.googleapis.com",
    ]


async def test_closing_the_text_input_does_not_mask_the_api_error(
    synthesis: SpeechSynthesis,
) -> None:
    """gRPC reads the text input from its own task; closing that input keeps the real error.

    https://github.com/livekit/agents/issues/2951 https://github.com/livekit/agents/pull/7041
    """
    reading = asyncio.Event()
    readers: list[asyncio.Task[None]] = []

    async def serve(
        requests: SynthesizeRequests,
    ) -> AsyncIterator[texttospeech.StreamingSynthesizeResponse]:
        async def read() -> None:
            async for _ in requests:
                reading.set()

        readers.append(asyncio.create_task(read()))
        await reading.wait()  # the first request is out; the input now awaits the next sentence
        raise DeadlineExceeded("deadline exceeded")  # type: ignore[no-untyped-call]  # api_core is untyped

    synthesis.serve = serve
    provider = TTS(credentials_info={"type": "service_account"})

    async with provider:
        with pytest.raises(APITimeoutError):
            async with provider.stream(conn_options=OPTIONS) as stream:
                # the stage sends a sentence once the next one starts; the input stays open
                stream.push_text("Hello there, how are you today? The")
                _ = [event async for event in stream]
    readers[0].cancel()


# ---- errors


@pytest.mark.parametrize(
    "modality, streaming", [("stt", False), ("stt", True), ("tts", False), ("tts", True)]
)
async def test_sdk_error_details_reach_the_caller(
    speech: SpeechAPI, synthesis: SpeechSynthesis, modality: str, streaming: bool
) -> None:
    """A rejected request surfaces as APIStatusError with Google's status, details and cause."""
    details = [{"reason": "invalid_request", "domain": "speech.googleapis.com"}]
    sdk_error = BadRequest("request rejected", details=details)  # type: ignore[no-untyped-call]  # api_core is untyped
    speech.error = synthesis.error = sdk_error

    with pytest.raises(APIStatusError) as failure:
        if modality == "stt":
            recognizer = cloud_stt("chirp_3")
            async with recognizer:
                if streaming:
                    async with recognizer.stream(conn_options=OPTIONS) as recognition:
                        recognition.end_input()
                        _ = [event async for event in recognition]
                else:
                    await recognizer.recognize(FRAME, conn_options=OPTIONS)
        else:
            synthesizer = TTS(credentials=anonymous())
            async with synthesizer:
                if streaming:
                    async with synthesizer.stream(conn_options=OPTIONS) as spoken:
                        spoken.push_text("Hello.")
                        spoken.end_input()
                        _ = [event async for event in spoken]
                else:
                    async with synthesizer.synthesize("Hello", conn_options=OPTIONS) as chunked:
                        _ = [event async for event in chunked]

    assert (failure.value.status_code, failure.value.body) == (400, details)
    assert failure.value.__cause__ is sdk_error
