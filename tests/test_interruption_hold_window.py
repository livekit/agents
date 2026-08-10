"""Unit tests for the VAD-based adaptive-interruption transcript gate."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import pytest

from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


def _make_recognition(
    *,
    vad_sos: float | None,
    agent_sos: float | None = None,
    end_boundary: float = 1.0,
) -> AudioRecognition:
    recognition = AudioRecognition.__new__(AudioRecognition)
    recognition._agent_speech_started_at = agent_sos
    recognition._active_vad_speech_started_at = vad_sos
    recognition._backchannel_boundary = (0.0, end_boundary)
    recognition._transcript_buffer = deque()
    recognition._transcript_gate_active = True
    recognition._process_stt_event = MagicMock()  # type: ignore[method-assign]
    return recognition


def _event(
    text: str,
    *,
    created_at: float = 0.0,
    start_time: float = 0.0,
    end_time: float = 0.0,
) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(
                language="",
                text=text,
                start_time=start_time,
                end_time=end_time,
            )
        ],
        created_at=created_at,
    )


def test_active_vad_utterance_extends_past_end_boundary() -> None:
    recognition = _make_recognition(vad_sos=5.0, end_boundary=1.0)

    recognition._transcript_buffer.extend(
        [
            _event("before", created_at=4.0),
            _event("utterance start", created_at=5.0),
            _event("utterance end", created_at=9.5),
        ]
    )

    flush_start = recognition._transcript_flush_start(
        now=10.0,
        vad_speech_started_at=5.0,
    )
    recognition._release_transcript_gate(at=10.0, vad_speech_started_at=5.0)

    emitted = [
        call.args[0].alternatives[0].text
        for call in recognition._process_stt_event.call_args_list  # type: ignore[attr-defined]
    ]
    assert flush_start == 5.0
    assert emitted == ["utterance start", "utterance end"]


def test_flush_start_does_not_precede_agent_speech() -> None:
    recognition = _make_recognition(vad_sos=5.0, agent_sos=8.0, end_boundary=1.0)

    assert recognition._transcript_flush_start(now=10.0, vad_speech_started_at=5.0) == 8.0


def test_end_boundary_is_used_without_active_vad_utterance() -> None:
    recognition = _make_recognition(vad_sos=None, end_boundary=1.0)

    recognition._transcript_buffer.extend(
        [_event("old", created_at=8.0), _event("near end", created_at=9.5)]
    )

    recognition._release_transcript_gate(at=10.0, vad_speech_started_at=None)

    emitted = [
        call.args[0].alternatives[0].text
        for call in recognition._process_stt_event.call_args_list  # type: ignore[attr-defined]
    ]
    assert emitted == ["near end"]
    assert not recognition._transcript_buffer


def test_provider_timestamps_do_not_affect_gate() -> None:
    recognition = _make_recognition(vad_sos=None, end_boundary=1.0)
    stale_arrival_with_future_word_time = _event(
        "stale",
        created_at=8.0,
        start_time=10_000.0,
        end_time=20_000.0,
    )
    recent_arrival_without_word_times = _event("recent", created_at=9.5)

    recognition._transcript_buffer.extend(
        [stale_arrival_with_future_word_time, recent_arrival_without_word_times]
    )
    recognition._release_transcript_gate(at=10.0, vad_speech_started_at=None)

    emitted = [
        call.args[0].alternatives[0].text
        for call in recognition._process_stt_event.call_args_list  # type: ignore[attr-defined]
    ]
    assert emitted == ["recent"]


def test_drain_preserves_provider_order() -> None:
    recognition = _make_recognition(vad_sos=5.0)
    events = [
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(language="", text="interim")],
            created_at=5.0,
        ),
        _event("final", created_at=5.1),
        SpeechEvent(type=SpeechEventType.END_OF_SPEECH, created_at=5.2),
    ]
    recognition._transcript_buffer.extend(events)

    recognition._drain_transcript_gate()

    emitted = [call.args[0] for call in recognition._process_stt_event.call_args_list]  # type: ignore[attr-defined]
    assert emitted == events
    assert not recognition._transcript_buffer


def test_disabling_vad_drains_the_transcript_gate() -> None:
    recognition = _make_recognition(vad_sos=5.0)
    event = _event("held", created_at=5.0)
    recognition._transcript_buffer.append(event)
    recognition._interruption_enabled = True
    recognition._interruption_detection = MagicMock()
    recognition._vad = MagicMock()
    recognition._vad_atask = None
    recognition._turn_detector = None

    recognition._update_vad(None)

    recognition._process_stt_event.assert_called_once_with(event)  # type: ignore[attr-defined]
    assert not recognition._transcript_gate_active
    assert not recognition._transcript_buffer
