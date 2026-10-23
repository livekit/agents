from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.nvidia import stt as nvidia_stt

pytestmark = pytest.mark.plugin("nvidia")


@pytest.mark.parametrize("end_time", [400, 0, None])
def test_endpoint_preserves_word_timing(end_time):
    stream = object.__new__(nvidia_stt.SpeechStream)
    stream._speaking = False
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._language = "en-US"
    stream._stt = SimpleNamespace(_opts=SimpleNamespace(enable_diarization=False))
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    stream._event_loop = SimpleNamespace(call_soon_threadsafe=lambda fn, *args: fn(*args))
    alternative = SimpleNamespace(
        transcript="hello",
        words=[SimpleNamespace(word="hello", start_time=100, end_time=end_time)]
        if end_time is not None
        else [],
    )
    stream._handle_response(
        SimpleNamespace(results=[SimpleNamespace(is_final=True, alternatives=[alternative])])
    )
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []
