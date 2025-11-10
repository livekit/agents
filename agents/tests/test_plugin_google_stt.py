from google.cloud.speech_v2.types import cloud_speech

from livekit.agents.stt import SpeechData
from livekit.plugins.google.stt import (
    _streaming_recognize_response_to_speech_data,  # pyright: ignore[reportPrivateUsage]
)


async def test_streaming_recognize_response_to_speech_data_01():
    srr = cloud_speech.StreamingRecognizeResponse(
        results=[cloud_speech.StreamingRecognitionResult()]
    )
    assert _streaming_recognize_response_to_speech_data(srr, min_confidence_threshold=1.0) is None


async def test_streaming_recognize_response_to_speech_data_02():
    srr = cloud_speech.StreamingRecognizeResponse(
        results=[
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=0.0, transcript="test")
                ],
                is_final=True,
                language_code="te-ST",
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(srr, min_confidence_threshold=0.5)
    assert type(result) is SpeechData
    assert result.text == "test"
    assert result.language == "te-ST"
    assert result.confidence == 0.0


async def test_streaming_recognize_response_to_speech_data_03():
    srr = cloud_speech.StreamingRecognizeResponse(
        results=[
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=0.0, transcript="test")
                ],
                is_final=False,
            )
        ]
    )
    result = _streaming_recognize_response_to_speech_data(srr, min_confidence_threshold=0.5)
    assert result is None


async def test_streaming_recognize_response_to_speech_data_04():
    srr = cloud_speech.StreamingRecognizeResponse(
        results=[
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=1.0, transcript="test01")
                ],
                is_final=False,
                language_code="te-ST",
            ),
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=1.0, transcript="test02")
                ],
                is_final=False,
                language_code="te-ST",
            ),
        ]
    )
    result = _streaming_recognize_response_to_speech_data(srr, min_confidence_threshold=0.5)
    assert type(result) is SpeechData
    assert result.text == "test01test02"
    assert result.language == "te-ST"
    assert result.confidence == 1.0


async def test_streaming_recognize_response_to_speech_data_05():
    srr = cloud_speech.StreamingRecognizeResponse(
        results=[
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=1.0, transcript="test01")
                ],
                is_final=False,
                language_code="te-ST",
            ),
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=1.0, transcript="test02")
                ],
                is_final=False,
                language_code="te-ST",
            ),
            cloud_speech.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(confidence=1.0, transcript="best")
                ],
                is_final=True,
                language_code="te-ST",
            ),
        ]
    )
    result = _streaming_recognize_response_to_speech_data(srr, min_confidence_threshold=0.5)
    assert type(result) is SpeechData
    assert result.text == "best"
    assert result.language == "te-ST"
    assert result.confidence == 1.0
