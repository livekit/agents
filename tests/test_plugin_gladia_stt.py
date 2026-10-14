from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.gladia import stt as gladia_stt

pytestmark = pytest.mark.plugin("gladia")


async def test_update_options_stores_region_for_new_streams():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key", region="eu-west")
    stt.update_options(region="us-west")

    assert stt._opts.region == "us-west"


@pytest.mark.parametrize("translate", [False, True])
@pytest.mark.parametrize("end_time", [0.4, 0.0])
def test_endpoint_preserves_source_timing(translate, end_time):
    stream = object.__new__(gladia_stt.SpeechStream)
    stream._opts = gladia_stt.STT(api_key="test-key")._opts
    stream._opts.translation_config.enabled = translate
    stream._speaking = False
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._request_id = "test"
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    utterance = {"text": "hello", "language": "en", "start": 0.1, "end": end_time}
    stream._process_gladia_message(
        {"type": "transcript", "data": {"is_final": True, "utterance": utterance}}
    )
    if translate:
        stream._process_gladia_message(
            {
                "type": "translation",
                "data": {
                    "utterance": utterance,
                    "translated_utterance": {"text": "bonjour", "language": "fr", "end": 9.0},
                },
            }
        )
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []
