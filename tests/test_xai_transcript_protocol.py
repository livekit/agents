from __future__ import annotations

import pytest

from livekit.agents import APIConnectionError, stt
from livekit.plugins.xai.stt import _prerecorded_transcription_to_speech_event

pytestmark = pytest.mark.unit


def test_xai_transcript_decoding_preserves_alignment_and_rejects_invalid_words() -> None:
    result = _prerecorded_transcription_to_speech_event(
        {
            "text": "hello",
            "language": "en",
            "words": [{"text": "hello", "start": 0.1, "end": 0.7, "speaker": 0}],
        },
        enable_diarization=True,
    )
    assert result.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    transcript = result.alternatives[0]
    assert transcript.text == "hello"
    assert transcript.speaker_id == "S0"
    assert transcript.words is not None
    assert transcript.words[0].start_time == 0.1
    assert transcript.words[0].end_time == 0.7

    with pytest.raises(APIConnectionError, match="invalid xAI transcript") as error:
        _prerecorded_transcription_to_speech_event(
            {"text": "hello", "words": [{"text": ["bad"]}]}, enable_diarization=False
        )
    assert error.value.__cause__ is not None
