from __future__ import annotations

import pytest

from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import stt as elevenlabs_stt

pytestmark = pytest.mark.plugin


class _EventSink:
    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


def _new_stream(*, server_vad=NOT_GIVEN) -> elevenlabs_stt.SpeechStream:
    stream = object.__new__(elevenlabs_stt.SpeechStream)
    stream._opts = elevenlabs_stt.STTOptions(
        model_id="scribe_v2_realtime",
        api_key="test-key",
        base_url=elevenlabs_stt.API_BASE_URL_V1,
        language_code=None,
        tag_audio_events=True,
        include_timestamps=False,
        sample_rate=16000,
        server_vad=server_vad,
        keyterms=NOT_GIVEN,
    )
    stream._language = None
    stream._event_ch = _EventSink()
    stream._speaking = False
    stream._start_time_offset = 0.0
    return stream


def _committed_transcript(text: str) -> dict:
    return {
        "message_type": "committed_transcript",
        "text": text,
        "words": [
            {"text": text, "start": 0.1, "end": 0.4},
        ]
        if text
        else [],
    }


def test_server_vad_commit_emits_end_of_speech() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._event_ch.events[1].alternatives[0].text == "hello"
    assert stream._speaking is False


def test_manual_commit_still_waits_for_empty_commit() -> None:
    stream = _new_stream(server_vad=None)

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert stream._speaking is True

    stream._process_stream_event(_committed_transcript(""))

    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert stream._speaking is False
