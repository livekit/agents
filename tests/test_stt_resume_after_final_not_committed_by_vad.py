import asyncio

import pytest

from livekit.agents.stt import SpeechEventType
from livekit.agents.utils import aio

from .test_agent_session import (
    _close_test_session,
    _make_stt_turn_recognition,
    _send_stt_event,
    _send_vad_end,
)

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def test_resume_after_final_and_end_is_not_committed_by_vad_before_its_final() -> None:
    """Deepgram Flux order: EndOfTurn sends FINAL_TRANSCRIPT then END_OF_SPEECH."""
    recognition, committed = _make_stt_turn_recognition(min_delay=0.3)
    try:
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I live in Cantu.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        # the caller resumes within min_delay (Flux StartOfTurn, then Update)
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.INTERIM_TRANSCRIPT, "in the province")
        # a short pause: VAD end of speech before Flux ends the resumed turn
        await _send_vad_end(recognition)
        await asyncio.sleep(0.4)
        # Flux EndOfTurn for the resumed speech
        await _send_stt_event(
            recognition, SpeechEventType.FINAL_TRANSCRIPT, "in the province of Como."
        )
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await asyncio.sleep(0.4)

        assert committed == ["I live in Cantu. in the province of Como."]
    finally:
        if recognition._end_of_turn_task is not None:
            await aio.cancel_and_wait(recognition._end_of_turn_task)
        await _close_test_session(recognition._session)


async def test_final_after_resume_rearms_the_vad_end_commit() -> None:
    """A provider final for the resumed speech lets VAD silence end the turn again."""
    recognition, committed = _make_stt_turn_recognition(min_delay=0.3)
    try:
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        # the caller resumes within min_delay
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        # the provider finalizes the continued speech without a new end of speech
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "the OTP")
        # VAD silence now ends the turn with the complete transcript
        await _send_vad_end(recognition)
        await asyncio.sleep(0.4)

        assert committed == ["I need the OTP"]
    finally:
        if recognition._end_of_turn_task is not None:
            await aio.cancel_and_wait(recognition._end_of_turn_task)
        await _close_test_session(recognition._session)
