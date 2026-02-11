from google.cloud.speech_v1.types import cloud_speech as cloud_speech_v1
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_v2
from google.protobuf.duration_pb2 import Duration

from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import TimedString
from livekit.plugins.google.stt import (
    _recognize_response_to_speech_event,  # pyright: ignore[reportPrivateUsage]
    _streaming_recognize_response_to_speech_data,  # pyright: ignore[reportPrivateUsage]
)


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
            language="te-ST",
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
            language="te-ST",
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
