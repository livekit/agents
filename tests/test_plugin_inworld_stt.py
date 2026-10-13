from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.inworld import stt as inworld_stt

pytestmark = pytest.mark.plugin("inworld")


@pytest.mark.parametrize("end_ms", [400, 0, None])
def test_endpoint_requests_and_preserves_word_timing(end_ms):
    stream = object.__new__(inworld_stt.SpeechStream)
    stream._stt = inworld_stt.STT(api_key="test-key")
    stream._language = "en"
    stream._speaking = False
    stream._request_id = "test"
    stream._start_time_offset = 7.0
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    assert stream._build_transcribe_config()["includeWordTimestamps"] is True
    stream._process_stream_event({"result": {"speechStarted": {}}})
    transcript = {"transcript": "hello", "isFinal": True}
    if end_ms is not None:
        transcript["wordTimestamps"] = [{"word": "hello", "startTimeMs": 100, "endTimeMs": end_ms}]
    stream._process_stream_event({"result": {"transcription": transcript}})
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    if end_ms:
        assert events[-1].alternatives[0].end_time == pytest.approx(7.4)
        assert events[-2].alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert events[-1].alternatives == []
