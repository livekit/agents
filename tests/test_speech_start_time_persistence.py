"""
Tests to validate that AudioRecognition._user_turn_start reflects the
turn-level start of speech and persists across multiple VAD bursts within
the same logical user turn.

Within a single user turn the VAD can produce several
START_OF_SPEECH/END_OF_SPEECH cycles separated by short silences (e.g. the
user says "Hello." then pauses briefly before continuing with the rest of
their utterance). End-of-turn detection is decoupled from VAD: a turn is
only considered ended once the EOT logic in `_bounce_eou_task` runs and
clears the per-turn state.

`_speech_start_time` reflects the *latest* VAD burst start (it is
overwritten by every new SOS) and is used as the start of the per-burst
`user_speaking` OTEL spans. The new `_user_turn_start` is set alongside
the `_user_turn_span` on the first SOS of a turn and cleared together with
the span on EOT cleanup. It is the value passed into `_bounce_eou_task`
and ultimately ends up as `started_speaking_at` on the EOT metrics report.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.audio_recognition import AudioRecognition


class TestUserTurnStartPersistence:
    """Test cases for `AudioRecognition._user_turn_start` lifecycle."""

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
        audio_recognition._user_turn_start = None
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
    async def test_first_sos_sets_user_turn_start(self):
        """A single START_OF_SPEECH event sets _user_turn_start to the
        back-calculated burst start (time.time() - speech_duration - inference_duration).
        """
        audio_recognition = self._create_audio_recognition()

        before = time.time()
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        after = time.time()

        assert audio_recognition._user_turn_start is not None
        assert before - 5.0 - 0.5 <= audio_recognition._user_turn_start <= after - 5.0 + 0.5

    @pytest.mark.asyncio
    async def test_user_turn_start_persists_across_intra_turn_bursts(self):
        """
        Within a single turn, VAD may fire multiple START_OF_SPEECH/END_OF_SPEECH
        cycles before EOT detection commits the turn. `_user_turn_start` must
        reflect the *first* burst's start and persist across subsequent bursts —
        it is only cleared by the EOT cleanup in `_bounce_eou_task`, alongside
        the `_user_turn_span` it travels with.

        Sequence:
            SOS (burst 1, speech_duration=5.0)  →  _user_turn_start = T1
            EOS (burst 1)
            SOS (burst 2, speech_duration=0.0)  →  _user_turn_start should remain T1

        `_speech_start_time` (per-burst, used for OTEL spans) is allowed to be
        overwritten by the second SOS — that's a separate concern.
        """
        audio_recognition = self._create_audio_recognition()

        # Burst 1 — speech started ~5s before this event fired
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        first_burst_start = audio_recognition._user_turn_start
        assert first_burst_start is not None

        # End of burst 1
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=5.0, silence_duration=0.6)
        )

        # Brief silence between bursts — same logical turn (no EOT yet)
        await asyncio.sleep(0.05)

        # Burst 2 — speech started "right now" (speech_duration=0)
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.0)
        )

        assert audio_recognition._user_turn_start == pytest.approx(first_burst_start, abs=0.01), (
            "_user_turn_start was overwritten by the second SOS within the same turn. "
            f"Expected {first_burst_start:.3f}, got {audio_recognition._user_turn_start:.3f}. "
            "It should only be cleared by the EOT cleanup in _bounce_eou_task."
        )

    @pytest.mark.asyncio
    async def test_speech_start_time_updates_per_burst(self):
        """
        `_speech_start_time` is per-burst by design (used as the start of OTEL
        `user_speaking` spans), so it *should* update when a new SOS fires
        after an EOS. This test pins down that behaviour so we don't regress it.
        """
        audio_recognition = self._create_audio_recognition()

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        first_burst_speech_start = audio_recognition._speech_start_time
        assert first_burst_speech_start is not None

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=5.0, silence_duration=0.6)
        )

        await asyncio.sleep(0.05)

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.0)
        )

        # _speech_start_time should now reflect the second burst's start, not the first
        assert audio_recognition._speech_start_time is not None
        assert audio_recognition._speech_start_time > first_burst_speech_start
