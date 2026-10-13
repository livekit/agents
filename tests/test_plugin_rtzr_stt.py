from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import LanguageCode, stt
from livekit.plugins.rtzr import stt as rtzr_stt

pytestmark = pytest.mark.plugin("rtzr")


@pytest.mark.parametrize("duration", [300, 0, None])
def test_endpoint_preserves_utterance_duration(duration):
    stream = object.__new__(rtzr_stt.SpeechStream)
    stream._start_time_offset = 7.0
    stream._rtzr_stt = SimpleNamespace(_params=SimpleNamespace(language=LanguageCode("ko")))
    message = {"start_at": 100, "final": True, "alternatives": [{"text": "안녕"}]}
    if duration is not None:
        message["duration"] = duration
    events, speaking, started_at = stream._process_transcript_event(message, False, None)
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    assert not speaking
    assert started_at is None
    if duration:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []
