from types import SimpleNamespace

import pytest
from riva.client.proto import riva_asr_pb2 as asr_proto

from livekit.plugins.nvidia import STT

pytestmark = pytest.mark.unit


async def test_word_timestamps_use_the_transcript_seconds_and_offset() -> None:
    provider = STT(api_key="test", use_ssl=False)
    stream = provider.stream(language="en-US")
    stream.start_time_offset = 0.25
    alternative = asr_proto.SpeechRecognitionAlternative(
        transcript="Hello",
        confidence=0.9,
        words=[asr_proto.WordInfo(word="Hello", start_time=1000, end_time=1500)],
    )
    try:
        result = stream._convert_to_speech_data(alternative, is_final=True)
        assert result.start_time == pytest.approx(1.25)
        assert result.end_time == pytest.approx(1.75)
        assert result.words is not None
        assert result.words[0].start_time == pytest.approx(result.start_time)
        assert result.words[0].end_time == pytest.approx(result.end_time)
    finally:
        await stream.aclose()


def test_model_listing_uses_the_riva_protobuf_request() -> None:
    def get_config(request: object) -> asr_proto.RivaSpeechRecognitionConfigResponse:
        return asr_proto.RivaSpeechRecognitionConfigResponse(
            model_config=[
                {
                    "model_name": "streaming-en",
                    "parameters": {"type": "online", "language_code": "en-US"},
                }
            ]
        )

    provider = STT(api_key="test", use_ssl=False)
    service = SimpleNamespace(stub=SimpleNamespace(GetRivaSpeechRecognitionConfig=get_config))
    result = provider.log_asr_models(service)
    assert result == {"en-US": [{"model": ["streaming-en"]}]}
