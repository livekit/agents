"""
Tests to validate that AudioRecognition._speech_start_time reflects the
turn-level start of speech and persists across multiple VAD bursts within
the same logical user turn.

Within a single user turn the VAD can produce several
START_OF_SPEECH/END_OF_SPEECH cycles separated by short silences (e.g. the
user says "Hello." then pauses briefly before continuing with the rest of
their utterance). End-of-turn detection is decoupled from VAD: a turn is
only considered ended once the EOT logic in `_bounce_eou_task` runs and
clears the per-turn state.

Before the fix, `_vad_speech_started` was reset to False in the VAD
END_OF_SPEECH branch. The next START_OF_SPEECH would therefore overwrite
`_speech_start_time` with the *latest* burst's start, losing the original
turn-start timestamp. After the fix, `_vad_speech_started` is only cleared
by the EOT cleanup, so `_speech_start_time` correctly anchors to the first
burst within the turn.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.audio_recognition import AudioRecognition


class TestSpeechStartTimePersistence:
    """Test cases for `AudioRecognition._speech_start_time` lifecycle."""

    def _create_audio_recognition(self) -> AudioRecognition:
        """Create an AudioRecognition instance with mocked dependencies."""
        with patch.object(AudioRecognition, "__init__", lambda self, *args, **kwargs: None):
            audio_recognition = AudioRecognition.__new__(AudioRecognition)

        # state read/written by _on_vad_event SOS/EOS branches
        audio_recognition._speech_start_time = None
        audio_recognition._vad_speech_started = False
        audio_recognition._speaking = False
        audio_recognition._end_of_turn_task = None
        audio_recognition._user_turn_span = None
        audio_recognition._user_turn_committed = False
        # disable EOU detection from EOS branch — we're testing VAD state, not EOT
        audio_recognition._vad_base_turn_detection = False
        audio_recognition._turn_detection_mode = None
        audio_recognition._stt = None
        audio_recognition._stt_model = None
        audio_recognition._stt_provider = None
        audio_recognition._audio_transcript = ""
        audio_recognition._last_speaking_time = None

        # collaborators
        audio_recognition._hooks = MagicMock()
        audio_recognition._session = MagicMock()
        audio_recognition._session.amd = None
        audio_recognition._session._room_io = None

        return audio_recognition

    @staticmethod
    def _vad_event(
        type_: VADEventType,
        *,
        speech_duration: float = 0.0,
        silence_duration: float = 0.0,
        inference_duration: float = 0.0,
    ) -> VADEvent:
        return VADEvent(
            type=type_,
            samples_index=0,
            timestamp=time.time(),
            speech_duration=speech_duration,
            silence_duration=silence_duration,
            inference_duration=inference_duration,
        )

    @pytest.mark.asyncio
    async def test_first_sos_sets_speech_start_time(self):
        """A single START_OF_SPEECH event sets _speech_start_time to the
        back-calculated burst start (time.time() - speech_duration - inference_duration).
        """
        audio_recognition = self._create_audio_recognition()

        before = time.time()
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        after = time.time()

        assert audio_recognition._speech_start_time is not None
        assert before - 5.0 - 0.5 <= audio_recognition._speech_start_time <= after - 5.0 + 0.5
        assert audio_recognition._vad_speech_started is True

    @pytest.mark.asyncio
    async def test_speech_start_time_persists_across_intra_turn_bursts(self):
        """
        Within a single turn, VAD may fire multiple START_OF_SPEECH/END_OF_SPEECH
        cycles before EOT detection commits the turn. `_speech_start_time` must
        reflect the *first* burst's start and persist across subsequent bursts —
        it is only cleared by the EOT cleanup in `_bounce_eou_task`.

        Sequence:
            SOS (burst 1, speech_duration=5.0)  →  _speech_start_time = T1
            EOS (burst 1)
            SOS (burst 2, speech_duration=0.0)  →  _speech_start_time should remain T1

        Without the fix, the second SOS overwrites _speech_start_time with the
        burst-2 start time, losing T1.
        """
        audio_recognition = self._create_audio_recognition()

        # Burst 1 — speech started ~5s before this event fired
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        first_burst_start = audio_recognition._speech_start_time
        assert first_burst_start is not None

        # End of burst 1
        await audio_recognition._on_vad_event(
            self._vad_event(
                VADEventType.END_OF_SPEECH, speech_duration=5.0, silence_duration=0.6
            )
        )

        # Brief silence between bursts — same logical turn (no EOT yet)
        await asyncio.sleep(0.05)

        # Burst 2 — speech started "right now" (speech_duration=0)
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.0)
        )

        assert audio_recognition._speech_start_time == pytest.approx(
            first_burst_start, abs=0.01
        ), (
            "_speech_start_time was overwritten by the second SOS within the same turn. "
            f"Expected {first_burst_start:.3f}, got {audio_recognition._speech_start_time:.3f}. "
            "It should only be cleared by the EOT cleanup in _bounce_eou_task."
        )

    @pytest.mark.asyncio
    async def test_eos_does_not_clear_vad_speech_started(self):
        """
        END_OF_SPEECH should not flip `_vad_speech_started` to False on its own —
        otherwise the next START_OF_SPEECH within the same turn would overwrite
        `_speech_start_time`. The flag is owned by the EOT cleanup.
        """
        audio_recognition = self._create_audio_recognition()

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=1.0)
        )
        assert audio_recognition._vad_speech_started is True

        await audio_recognition._on_vad_event(
            self._vad_event(
                VADEventType.END_OF_SPEECH, speech_duration=1.0, silence_duration=0.6
            )
        )

        assert audio_recognition._vad_speech_started is True, (
            "_vad_speech_started was reset by END_OF_SPEECH. "
            "It should only be cleared by the EOT cleanup in _bounce_eou_task."
        )
