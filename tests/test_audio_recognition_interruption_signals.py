from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import MagicMock

import pytest

from livekit.agents import NOT_GIVEN
from livekit.agents.inference.interruption import (
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechEndedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


async def test_agent_segments_emit_complete_interruption_boundaries() -> None:
    recognition = AudioRecognition.__new__(AudioRecognition)
    recognition._agent_speaking = False
    recognition._agent_speech_started_at = None
    recognition._endpointing = MagicMock()
    recognition._backchannel_boundary = None
    recognition._backchannel_boundary_timer = None
    recognition._backchannel_boundary_callback = None
    recognition._interruption_enabled = True
    recognition._interruption_ch = MagicMock()
    recognition._interruption_ch.closed = False
    recognition._ignore_user_transcript_until = NOT_GIVEN
    recognition._transcript_buffer = deque()
    recognition._tasks = set()
    recognition._overlap_in_current_turn = False
    recognition._turn_backchannel_over_agent = False
    recognition._user_silence_ev = asyncio.Event()
    recognition._user_silence_ev.set()
    recognition._session = MagicMock()

    recognition._on_start_of_agent_speech(started_at=1.0)
    recognition._speaking = True
    recognition._on_start_of_speech(started_at=2.0)
    recognition._on_end_of_agent_speech(ignore_user_transcript_until=3.0)
    recognition._on_start_of_agent_speech(started_at=4.0)

    frames = [call.args[0] for call in recognition._interruption_ch.send_nowait.call_args_list]
    assert [type(frame) for frame in frames] == [
        _AgentSpeechStartedSentinel,
        _OverlapSpeechStartedSentinel,
        _OverlapSpeechEndedSentinel,
        _AgentSpeechEndedSentinel,
        _AgentSpeechStartedSentinel,
        _OverlapSpeechStartedSentinel,
    ]
    assert frames[-1]._speech_duration == 0.0
    assert frames[-1]._started_at == 4.0

    await asyncio.gather(*recognition._tasks)
