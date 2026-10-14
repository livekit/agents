from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.smallestai import stt as smallestai_stt

pytestmark = pytest.mark.plugin("smallestai")


@pytest.mark.parametrize("endpoint", ["vad", "final", "empty_final", "close"])
@pytest.mark.parametrize("end_time", [0.4, 0.0])
def test_endpoint_preserves_speech_timing(endpoint, end_time):
    stream = object.__new__(smallestai_stt.SpeechStream)
    stream._opts = SimpleNamespace(language="en", diarize=False, vad_events=True)
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._session_id = "test"
    stream._speaking = False
    stream._final_alternatives = []
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    stream._process_stream_event({"type": "speech_started"})
    if endpoint == "vad":
        # VAD can precede the final transcript.
        stream._process_stream_event({"type": "speech_ended", "timestamp": end_time})
    else:
        stream._process_stream_event(
            {
                "transcript": "hello",
                "is_final": True,
                "speech_final": endpoint == "final",
                "words": [{"word": "hello", "start": 0.1, "end": end_time}],
            }
        )
        if endpoint == "empty_final":
            stream._process_stream_event({"is_final": True, "from_finalize": True})
        elif endpoint == "close":
            stream._end_speech()
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert events[-1].alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert events[-1].alternatives == []
    stream._process_stream_event({"type": "speech_started"})
    stream._process_stream_event({"type": "speech_ended"})
    assert events[-1].alternatives == []
