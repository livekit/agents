import asyncio
import contextlib
import gc

import pytest
from google.cloud.speech_v1.types import cloud_speech as cloud_speech_v1
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_v2
from google.protobuf.duration_pb2 import Duration

from livekit.agents import APIConnectOptions, LanguageCode
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import TimedString
from livekit.agents.utils.aio import ChanClosed
from livekit.plugins.google.stt import (
    SpeechStream,
    STTOptions,
    _recognize_response_to_speech_event,  # pyright: ignore[reportPrivateUsage]
    _streaming_recognize_response_to_speech_data,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.plugin("google")


@pytest.fixture
def mock_google_adc(monkeypatch):
    from livekit.plugins.google import stt as google_stt

    monkeypatch.setattr(google_stt, "gauth_default", lambda: (None, "test-project"))


class _FakeSTT:
    _label = "google.STT"
    model = "default"
    provider = "Google Cloud Platform"

    def emit(self, *_args, **_kwargs):
        pass


class _FakeStreamingCall:
    def __init__(self, requests):
        self._requests = requests
        self._consumer_task = asyncio.create_task(self._consume_requests())
        self.awaiting_audio = asyncio.Event()
        self.consumer_done = asyncio.Event()

    async def _consume_requests(self):
        try:
            await self._requests.__anext__()
            self.awaiting_audio.set()
            await self._requests.__anext__()
        finally:
            self.consumer_done.set()

    def cancel(self):
        self._consumer_task.cancel()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.awaiting_audio.wait()
        raise StopAsyncIteration


class _FakeSpeechClient:
    def __init__(self):
        self.call: _FakeStreamingCall | None = None
        self.call_ready = asyncio.Event()

    async def streaming_recognize(self, *, requests):
        self.call = _FakeStreamingCall(requests)
        self.call_ready.set()
        return self.call


class _FakeConnectionPool:
    last_acquire_time = 0.0
    last_connection_reused = False

    def __init__(self, client):
        self._client = client

    @contextlib.asynccontextmanager
    async def connection(self, *, timeout):
        yield self._client

    def remove(self, _client):
        pass


def _default_stt_options() -> STTOptions:
    return STTOptions(
        languages=[LanguageCode("en-US")],
        detect_language=True,
        interim_results=True,
        punctuate=True,
        spoken_punctuation=False,
        enable_word_time_offsets=True,
        enable_word_confidence=False,
        enable_voice_activity_events=False,
        model="default",
        sample_rate=16000,
        min_confidence_threshold=0.65,
        profanity_filter=False,
    )


async def test_google_stt_stream_cancel_does_not_leak_chanclosed_task():
    client = _FakeSpeechClient()
    stream = SpeechStream(
        stt=_FakeSTT(),
        conn_options=APIConnectOptions(max_retry=0, timeout=0.1),
        pool=_FakeConnectionPool(client),
        recognizer_cb=lambda _client: "",
        config=_default_stt_options(),
    )

    loop = asyncio.get_running_loop()
    original_exception_handler = loop.get_exception_handler()
    contexts = []
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    try:
        await asyncio.wait_for(client.call_ready.wait(), timeout=1)
        assert client.call is not None
        await asyncio.wait_for(client.call.awaiting_audio.wait(), timeout=1)

        await asyncio.wait_for(stream._task, timeout=1)
        await asyncio.wait_for(client.call.consumer_done.wait(), timeout=1)

        await stream.aclose()
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_exception_handler)

    unhandled_chanclosed = [
        context
        for context in contexts
        if context.get("message") == "Task exception was never retrieved"
        and isinstance(context.get("exception"), ChanClosed)
    ]
    assert unhandled_chanclosed == []


async def test_streaming_recognize_response_to_speech_data_01():
    srr = cloud_speech_v2.StreamingRecognizeResponse(
        results=[cloud_speech_v2.StreamingRecognitionResult()]
    )
    assert (
        _streaming_recognize_response_to_speech_data(
            srr, min_confidence_threshold=1.0, start_time_offset=0.0
        )
        is None
    )

    srr = cloud_speech_v1.StreamingRecognizeResponse(
        results=[cloud_speech_v1.StreamingRecognitionResult()]
    )
    assert (
        _streaming_recognize_response_to_speech_data(
            srr, min_confidence_threshold=1.0, start_time_offset=0.0
        )
        is None
    )


async def test_streaming_recognize_response_to_speech_data_02():
    # final result should be returned regardless of confidence
    srr = cloud_speech_v2.StreamingRecognizeResponse(
        results=[
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(confidence=0.0, transcript="test")
                ],
                is_final=True,
                language_code="te-ST",
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert type(result) is SpeechData
    assert result.text == "test"
    assert result.language == "te-ST"
    assert result.confidence == 0.0

    srr = cloud_speech_v1.StreamingRecognizeResponse(
        results=[
            cloud_speech_v1.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v1.SpeechRecognitionAlternative(confidence=0.0, transcript="test")
                ],
                is_final=True,
                language_code="te-ST",
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert type(result) is SpeechData
    assert result.text == "test"
    assert result.language == "te-ST"
    assert result.confidence == 0.0


async def test_streaming_recognize_response_to_speech_data_03():
    srr = cloud_speech_v2.StreamingRecognizeResponse(
        results=[
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(confidence=0.0, transcript="test")
                ],
                is_final=False,
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert result is None

    srr = cloud_speech_v1.StreamingRecognizeResponse(
        results=[
            cloud_speech_v1.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v1.SpeechRecognitionAlternative(confidence=0.0, transcript="test")
                ],
                is_final=False,
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert result is None


async def test_streaming_recognize_response_to_speech_data_04():
    srr = cloud_speech_v2.StreamingRecognizeResponse(
        results=[
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(
                        confidence=1.0, transcript="test01"
                    )
                ],
                is_final=False,
                language_code="te-ST",
            ),
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(
                        confidence=1.0, transcript="test02"
                    )
                ],
                is_final=False,
                language_code="te-ST",
            ),
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert type(result) is SpeechData
    assert result.text == "test01test02"
    assert result.language == "te-ST"
    assert result.confidence == 1.0


async def test_streaming_recognize_response_to_speech_data_05():
    srr = cloud_speech_v2.StreamingRecognizeResponse(
        results=[
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(
                        confidence=1.0, transcript="test01"
                    )
                ],
                is_final=False,
                language_code="te-ST",
            ),
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(
                        confidence=1.0, transcript="test02"
                    )
                ],
                is_final=False,
                language_code="te-ST",
            ),
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(confidence=1.0, transcript="best")
                ],
                is_final=True,
                language_code="te-ST",
            ),
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert type(result) is SpeechData
    assert result.text == "best"
    assert result.language == "te-ST"
    assert result.confidence == 1.0


async def test_streaming_recognize_response_to_speech_data_words():
    srr = cloud_speech_v2.StreamingRecognizeResponse(
        results=[
            cloud_speech_v2.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(
                        confidence=1.0,
                        transcript="test",
                        words=[
                            cloud_speech_v2.WordInfo(
                                word="test",
                                start_offset=Duration(seconds=0),
                                end_offset=Duration(seconds=1),
                                confidence=1.0,
                            )
                        ],
                    )
                ],
                is_final=True,
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.0
    )
    assert type(result) is SpeechData
    assert result.text == "test"
    assert result.confidence == 1.0
    assert result.words == [
        TimedString(
            text="test", start_time=0.0, end_time=1.0, start_time_offset=0.0, confidence=1.0
        )
    ]

    srr = cloud_speech_v1.StreamingRecognizeResponse(
        results=[
            cloud_speech_v1.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_v1.SpeechRecognitionAlternative(
                        confidence=1.0,
                        transcript="test",
                        words=[
                            cloud_speech_v1.WordInfo(
                                word="test",
                                start_time=Duration(seconds=0),
                                end_time=Duration(seconds=1),
                                confidence=1.0,
                            )
                        ],
                    )
                ],
                is_final=True,
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(
        srr, min_confidence_threshold=0.5, start_time_offset=0.1
    )
    assert type(result) is SpeechData
    assert result.text == "test"
    assert result.confidence == 1.0
    assert result.words == [
        TimedString(
            text="test", start_time=0.1, end_time=1.1, start_time_offset=0.1, confidence=1.0
        )
    ]


async def test_recognize_response_to_speech_event_words():
    resp = cloud_speech_v2.RecognizeResponse(
        results=[
            cloud_speech_v2.SpeechRecognitionResult(
                alternatives=[
                    cloud_speech_v2.SpeechRecognitionAlternative(
                        confidence=1.0,
                        transcript="test",
                        words=[
                            cloud_speech_v2.WordInfo(
                                word="test",
                                start_offset=Duration(seconds=0),
                                end_offset=Duration(seconds=1),
                                confidence=1.0,
                            )
                        ],
                    )
                ],
                language_code="te-ST",
            )
        ]
    )
    result = _recognize_response_to_speech_event(resp)
    assert type(result) is SpeechEvent
    assert result.type == SpeechEventType.FINAL_TRANSCRIPT
    assert result.alternatives == [
        SpeechData(
            language=LanguageCode("te-ST"),
            start_time=0.0,
            end_time=1.0,
            text="test",
            confidence=1.0,
            words=[
                TimedString(
                    text="test", start_time=0.0, end_time=1.0, start_time_offset=0.0, confidence=1.0
                )
            ],
        )
    ]

    resp = cloud_speech_v1.RecognizeResponse(
        results=[
            cloud_speech_v1.SpeechRecognitionResult(
                alternatives=[
                    cloud_speech_v1.SpeechRecognitionAlternative(
                        confidence=1.0,
                        transcript="test",
                        words=[
                            cloud_speech_v1.WordInfo(
                                word="test",
                                start_time=Duration(seconds=0),
                                end_time=Duration(seconds=1),
                                confidence=1.0,
                            )
                        ],
                    )
                ],
                language_code="te-ST",
            )
        ]
    )
    result = _recognize_response_to_speech_event(resp)
    assert type(result) is SpeechEvent
    assert result.type == SpeechEventType.FINAL_TRANSCRIPT
    assert result.alternatives == [
        SpeechData(
            language=LanguageCode("te-ST"),
            start_time=0.0,
            end_time=1.0,
            text="test",
            confidence=1.0,
            words=[
                TimedString(
                    text="test", start_time=0.0, end_time=1.0, start_time_offset=0.0, confidence=1.0
                )
            ],
        )
    ]


async def test_voice_activity_timeout_defaults(mock_google_adc):
    """Test voice activity timeouts are not set by default."""
    from livekit.agents.types import NOT_GIVEN
    from livekit.plugins.google import STT

    stt = STT()
    assert stt._config.speech_start_timeout is NOT_GIVEN
    assert stt._config.speech_end_timeout is NOT_GIVEN


async def test_voice_activity_timeout_set(mock_google_adc):
    """Test voice activity timeouts can be set."""
    from livekit.plugins.google import STT

    stt = STT(
        speech_start_timeout=10.0,
        speech_end_timeout=2.5,
    )
    assert stt._config.speech_start_timeout == 10.0
    assert stt._config.speech_end_timeout == 2.5


async def test_voice_activity_timeout_fractional_seconds(mock_google_adc):
    """Test voice activity timeouts handle fractional seconds."""
    from livekit.plugins.google import STT

    stt = STT(
        speech_start_timeout=5.5,
        speech_end_timeout=1.25,
    )
    assert stt._config.speech_start_timeout == 5.5
    assert stt._config.speech_end_timeout == 1.25


async def test_voice_activity_timeout_speech_start_only(mock_google_adc):
    """Test setting only speech_start_timeout."""
    from livekit.agents.types import NOT_GIVEN
    from livekit.plugins.google import STT

    stt = STT(speech_start_timeout=15.0)
    assert stt._config.speech_start_timeout == 15.0
    assert stt._config.speech_end_timeout is NOT_GIVEN


async def test_voice_activity_timeout_speech_end_only(mock_google_adc):
    """Test setting only speech_end_timeout."""
    from livekit.agents.types import NOT_GIVEN
    from livekit.plugins.google import STT

    stt = STT(speech_end_timeout=3.0)
    assert stt._config.speech_end_timeout == 3.0
    assert stt._config.speech_start_timeout is NOT_GIVEN


async def test_voice_activity_timeout_v2_model(mock_google_adc):
    """Test that V2 model detection works correctly."""
    from livekit.plugins.google import STT

    stt_v2 = STT(model="chirp_3")
    assert stt_v2._config.version == 2

    stt_v1 = STT(model="default")
    assert stt_v1._config.version == 1


async def test_voice_activity_timeout_update(mock_google_adc):
    """Test that timeout options can be updated dynamically."""
    from livekit.plugins.google import STT

    stt = STT(
        speech_start_timeout=10.0,
        speech_end_timeout=2.0,
    )
    stt.update_options(
        speech_start_timeout=15.0,
        speech_end_timeout=3.0,
    )
    assert stt._config.speech_start_timeout == 15.0
    assert stt._config.speech_end_timeout == 3.0


async def test_voice_activity_timeout_partial_update(mock_google_adc):
    """Test updating only one timeout at a time."""
    from livekit.plugins.google import STT

    stt = STT(
        speech_start_timeout=10.0,
        speech_end_timeout=2.0,
    )
    stt.update_options(speech_start_timeout=20.0)
    assert stt._config.speech_start_timeout == 20.0
    assert stt._config.speech_end_timeout == 2.0

    stt.update_options(speech_end_timeout=5.0)
    assert stt._config.speech_start_timeout == 20.0
    assert stt._config.speech_end_timeout == 5.0
