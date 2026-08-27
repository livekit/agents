"""Unit tests for the VAD-based adaptive-interruption transcript gate."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import ModelSettings
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition, _STTPipeline

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
    recognition._backchannel_boundary_timer = None
    recognition._transcript_buffer = deque()
    recognition._transcript_gate_active = True
    recognition._stt_aligned_transcript = False
    recognition._hooks = MagicMock()
    recognition._hooks.interruption_by_audio_activity_enabled = False
    recognition._process_stt_event = MagicMock()  # type: ignore[method-assign]
    return recognition


def _event(
    text: str,
    *,
    created_at: float = 0.0,
    start_time: float = 0.0,
    end_time: float = 0.0,
    speech_end_time: float | None = None,
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
        speech_end_time=speech_end_time,
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

    recognition._flush_held_transcripts(resolved_at=10.0, vad_speech_started_at=5.0)

    emitted = [
        call.args[0].alternatives[0].text
        for call in recognition._process_stt_event.call_args_list  # type: ignore[attr-defined]
    ]
    assert emitted == ["utterance start", "utterance end"]


def test_trim_does_not_precede_agent_speech() -> None:
    recognition = _make_recognition(vad_sos=5.0, agent_sos=8.0, end_boundary=1.0)
    recognition._transcript_buffer.extend(
        [_event("before agent", created_at=7.5), _event("during agent", created_at=8.0)]
    )

    recognition._flush_held_transcripts(resolved_at=10.0, vad_speech_started_at=5.0)

    emitted = [
        call.args[0].alternatives[0].text
        for call in recognition._process_stt_event.call_args_list  # type: ignore[attr-defined]
    ]
    assert emitted == ["during agent"]


def test_end_boundary_is_used_without_active_vad_utterance() -> None:
    recognition = _make_recognition(vad_sos=None, end_boundary=1.0)

    recognition._transcript_buffer.extend(
        [_event("old", created_at=8.0), _event("near end", created_at=9.5)]
    )

    recognition._flush_held_transcripts(resolved_at=10.0, vad_speech_started_at=None)

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
    recognition._flush_held_transcripts(resolved_at=10.0, vad_speech_started_at=None)

    emitted = [
        call.args[0].alternatives[0].text
        for call in recognition._process_stt_event.call_args_list  # type: ignore[attr-defined]
    ]
    assert emitted == ["recent"]


def test_speech_end_time_preserves_a_delayed_prior_transcript() -> None:
    recognition = _make_recognition(vad_sos=None, agent_sos=8.0, end_boundary=1.0)
    recognition._transcript_buffer.append(
        _event(
            "prior speech",
            created_at=8.5,
            speech_end_time=7.5,
        )
    )

    recognition._flush_held_transcripts(resolved_at=10.0, vad_speech_started_at=None)

    recognition._process_stt_event.assert_called_once()  # type: ignore[attr-defined]


def test_speech_end_time_discards_a_delayed_backchannel() -> None:
    recognition = _make_recognition(vad_sos=None, agent_sos=8.0, end_boundary=1.0)
    recognition._transcript_buffer.append(
        _event(
            "backchannel",
            created_at=9.5,
            speech_end_time=8.5,
        )
    )

    recognition._flush_held_transcripts(resolved_at=10.0, vad_speech_started_at=None)

    recognition._process_stt_event.assert_not_called()  # type: ignore[attr-defined]


def test_flushing_held_transcripts_preserves_provider_order() -> None:
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

    recognition._flush_held_transcripts()

    emitted = [call.args[0] for call in recognition._process_stt_event.call_args_list]  # type: ignore[attr-defined]
    assert emitted == events
    assert not recognition._transcript_buffer
    assert not recognition._transcript_gate_active


def test_new_overlap_rearms_a_released_gate() -> None:
    # a failed interrupt attempt (e.g. below min_words) releases the gate mid-speech;
    # the next overlap in the same agent speech must hold its transcripts again
    recognition = _make_recognition(vad_sos=None, agent_sos=0.0)
    recognition._endpointing = MagicMock()
    recognition._turn_backchannel_over_agent = False
    recognition._overlap_in_current_turn = False
    recognition._overlap_open = False
    recognition._agent_speaking = True
    recognition._interruption_enabled = True
    recognition._interruption_ch = MagicMock(closed=False)
    recognition._transcript_gate_active = False

    recognition._on_start_of_speech(started_at=6.0)

    assert recognition._transcript_gate_active
    recognition._interruption_ch.send_nowait.assert_called_once()


def test_agent_speech_does_not_arm_gate_without_overlap() -> None:
    recognition = _make_recognition(vad_sos=None)
    recognition._agent_speaking = False
    recognition._interruption_enabled = True
    recognition._interruption_ch = MagicMock(closed=False)
    recognition._endpointing = MagicMock()
    recognition._backchannel_boundary = None
    recognition._overlap_in_current_turn = False
    recognition._overlap_open = False
    recognition._turn_backchannel_over_agent = False
    recognition._user_silence_ev = asyncio.Event()
    recognition._user_silence_ev.set()

    recognition._on_start_of_agent_speech(started_at=5.0)

    assert not recognition._transcript_gate_active


async def test_start_boundary_uses_audio_activity_for_agent_speech_interval() -> None:
    recognition = _make_recognition(vad_sos=None)
    recognition._agent_speaking = False
    recognition._interruption_enabled = True
    recognition._interruption_ch = MagicMock(closed=False)
    recognition._endpointing = MagicMock()
    recognition._backchannel_boundary = (1.0, 1.0)
    recognition._overlap_in_current_turn = False
    recognition._overlap_open = False
    recognition._turn_backchannel_over_agent = False
    recognition._user_silence_ev = asyncio.Event()
    recognition._user_silence_ev.set()

    recognition._on_start_of_agent_speech(started_at=5.0)
    recognition._cancel_backchannel_boundary()
    recognition._on_start_of_speech(started_at=5.5)

    assert recognition._hooks.interruption_by_audio_activity_enabled
    assert not recognition._transcript_gate_active
    assert not recognition._overlap_open
    recognition._interruption_ch.send_nowait.assert_called_once()

    recognition._speaking = True
    recognition._on_end_of_speech(ended_at=5.75)
    recognition._speaking = False
    recognition._on_start_of_speech(started_at=6.5)

    assert recognition._hooks.interruption_by_audio_activity_enabled
    assert not recognition._transcript_gate_active
    assert not recognition._overlap_open
    recognition._interruption_ch.send_nowait.assert_called_once()


async def test_ungated_final_during_agent_speech_uses_audio_activity() -> None:
    recognition = _make_recognition(vad_sos=None)
    recognition._agent_speaking = True
    recognition._transcript_gate_active = False
    recognition._turn_detection_mode = "vad"
    recognition._user_turn_committed = False
    recognition._stt_request_ids = []
    recognition._mark_turn_transcribed = MagicMock()  # type: ignore[method-assign]

    event = _event("late final", created_at=5.2)
    await recognition._on_stt_event(event)

    assert recognition._hooks.interruption_by_audio_activity_enabled
    recognition._process_stt_event.assert_called_once_with(event)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("aligned_transcript", "speech_end_time", "end_time", "expected_speech_end_time"),
    [
        (True, None, 9.0, 10.0),
        (False, None, 9.0, None),
        (True, 8.0, 9.0, 8.0),
        (True, None, 11.0, None),
    ],
)
async def test_custom_stt_node_normalizes_aligned_speech_end_time(
    aligned_transcript: bool,
    speech_end_time: float | None,
    end_time: float,
    expected_speech_end_time: float | None,
) -> None:
    event = _event(
        "custom",
        created_at=11.0,
        end_time=end_time,
        speech_end_time=speech_end_time,
    )

    async def custom_stt_node(
        _audio: AsyncIterable[rtc.AudioFrame], _model_settings: ModelSettings
    ) -> AsyncIterator[SpeechEvent]:
        yield event

    pipeline = _STTPipeline(custom_stt_node)
    pipeline.input_started_at = 1.0
    recognition = _make_recognition(vad_sos=None)
    recognition._stt_pipeline = pipeline
    recognition._stt_aligned_transcript = aligned_transcript
    recognition._agent_speaking = True
    recognition._turn_detection_mode = "vad"
    recognition._user_turn_committed = False
    recognition._stt_request_ids = []
    recognition._mark_turn_transcribed = MagicMock()  # type: ignore[method-assign]

    try:
        received = await asyncio.wait_for(pipeline.event_ch.recv(), timeout=1.0)
        await recognition._on_stt_event(received)
    finally:
        await pipeline.aclose()

    assert recognition._transcript_buffer[0].speech_end_time == expected_speech_end_time


def test_disabling_vad_flushes_held_transcripts() -> None:
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
