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
from livekit.agents.voice.audio_recognition import (
    _SPEECH_DURATION_STALE_AFTER,
    AudioRecognition,
)

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class TestUserTurnStartPersistence:
    """Test cases for `AudioRecognition._user_turn_start` lifecycle."""

    def _create_audio_recognition(self) -> AudioRecognition:
        """Create an AudioRecognition instance with mocked dependencies."""
        with patch.object(AudioRecognition, "__init__", lambda self, *args, **kwargs: None):
            audio_recognition = AudioRecognition.__new__(AudioRecognition)

        # state read/written by _on_vad_event SOS/EOS branches
        audio_recognition._speech_start_time = None
        audio_recognition._vad_speech_started = False
        # _speaking is a property backed by this event (set == silent/not speaking)
        audio_recognition._user_silence_ev = asyncio.Event()
        audio_recognition._speaking = False
        audio_recognition._agent_speaking = False
        audio_recognition._turn_detector_stream = None
        audio_recognition._end_of_turn_task = None
        audio_recognition._user_turn_span = None
        audio_recognition._user_turn_start = None
        audio_recognition._user_turn_committed = False
        # disable EOU detection from EOS branch — we're testing VAD state, not EOT
        audio_recognition._vad_base_turn_detection = False
        audio_recognition._turn_detection_mode = None
        audio_recognition._stt = None
        audio_recognition._stt_pipeline = None
        audio_recognition._stt_model = None
        audio_recognition._stt_provider = None
        audio_recognition._audio_transcript = ""
        audio_recognition._audio_interim_transcript = ""
        audio_recognition._last_speaking_time = None
        audio_recognition._transcription_timeout_handle = None
        audio_recognition._turn_speech_duration = 0.0
        audio_recognition._vad_speech_duration = None
        audio_recognition._turn_detector_prediction_fut = None
        audio_recognition._turn_detector_flushed = False

        # collaborators
        audio_recognition._hooks = MagicMock()
        audio_recognition._session = MagicMock()
        audio_recognition._session.amd = None
        audio_recognition._session._room_io = None
        audio_recognition._session.options.transcription_timeout = None

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

    @pytest.mark.asyncio
    async def test_vad_speech_duration_survives_post_eos_zero_inference(self):
        """Silero zeros pub_speech_duration after EOS; keep the segment final.

        Late STT finals use ``current_speech_duration`` for
        ``interruption.min_duration``. If a post-EOS INFERENCE_DONE with
        ``speech_duration=0`` overwrites the EOS value, a real interrupt is
        blocked as "too short".
        """
        audio_recognition = self._create_audio_recognition()

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.1)
        )
        await audio_recognition._on_vad_event(
            self._vad_event(
                VADEventType.INFERENCE_DONE,
                speech_duration=0.55,
            )
        )
        assert audio_recognition.current_speech_duration == pytest.approx(0.55)

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=0.6, silence_duration=0.5)
        )
        assert audio_recognition.current_speech_duration == pytest.approx(0.6)

        # Silero-style post-EOS inference with zeroed duration
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.INFERENCE_DONE, speech_duration=0.0)
        )
        assert audio_recognition.current_speech_duration == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_vad_speech_duration_cleared_when_turn_commits(self):
        """EOT cleanup must drop ``_vad_speech_duration`` with other speech anchors.

        Otherwise the next turn's STT-failsafe path (no fresh VAD events) reuses
        the previous segment length for ``interruption.min_duration``.
        """
        audio_recognition = self._create_audio_recognition()
        audio_recognition._closing = asyncio.Event()
        audio_recognition._endpointing = MagicMock()
        audio_recognition._endpointing.min_delay = 0.0
        audio_recognition._endpointing.max_delay = 0.0
        audio_recognition._turn_detector = None
        audio_recognition._hooks.on_end_of_turn.return_value = True
        audio_recognition._turn_backchannel_over_agent = False
        audio_recognition._overlap_in_current_turn = False
        audio_recognition._stt_request_ids = []
        audio_recognition._last_final_transcript_time = None
        audio_recognition._final_transcript_confidence = []
        audio_recognition._reset_transcription_timeout = MagicMock()
        audio_recognition._audio_transcript = "hello"

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.1)
        )
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=0.8, silence_duration=0.1)
        )
        assert audio_recognition.current_speech_duration == pytest.approx(0.8)

        chat_ctx = MagicMock()
        chat_ctx.copy.return_value = chat_ctx
        chat_ctx.add_message = MagicMock()
        chat_ctx.items = []

        audio_recognition._run_eou_detection(chat_ctx, trigger="vad")
        task = audio_recognition._end_of_turn_task
        assert task is not None
        await task

        assert audio_recognition.current_speech_duration is None
        assert audio_recognition._vad_speech_duration is None

    @pytest.mark.asyncio
    async def test_inference_done_hook_sees_current_vad_duration(self):
        """``on_vad_inference_done`` must observe this frame's duration, not the last.

        ``_interrupt_by_audio_activity`` gates on ``current_speech_duration``. If
        that is updated after the hook, barge-in waits one extra VAD window.
        """
        audio_recognition = self._create_audio_recognition()
        seen: list[float | None] = []

        def _capture(_ev: object) -> None:
            seen.append(audio_recognition.current_speech_duration)

        audio_recognition._hooks.on_vad_inference_done.side_effect = _capture

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.1)
        )
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.INFERENCE_DONE, speech_duration=0.48)
        )
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.INFERENCE_DONE, speech_duration=0.51)
        )

        assert seen == [pytest.approx(0.48), pytest.approx(0.51)]

    @pytest.mark.asyncio
    async def test_stale_blip_duration_does_not_gate_vad_missed_barge_in(self):
        """A VAD blip that never produced a transcript must stop gating.

        A short noise blip fires SOS/EOS with no transcript, so the turn never
        commits and the blip's 0.2s measurement is never cleared. If VAD then
        misses the user's real barge-in, the ``on_final_transcript`` failsafe
        gates on ``current_speech_duration`` — past the late-final window it
        must see unknown (None), not the blip's 0.2s, or the agent keeps
        talking over the user.
        """
        audio_recognition = self._create_audio_recognition()

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.05)
        )
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=0.2, silence_duration=0.4)
        )

        # within the late-final window the blip's measurement still applies
        assert audio_recognition.current_speech_duration == pytest.approx(0.2)

        await asyncio.sleep(_SPEECH_DURATION_STALE_AFTER + 0.5)

        assert audio_recognition.current_speech_duration is None

    @pytest.mark.asyncio
    async def test_stt_speaking_zero_vad_inference_keeps_duration_unknown(self):
        """STT can mark speaking before VAD has a voiced duration.

        ``turn_detection="stt"`` sets ``_speaking`` on START_OF_SPEECH while
        Silero still emits ``speech_duration=0`` until its own onset. Writing
        that zero into ``_vad_speech_duration`` turns "unknown" into 0.0, so
        ``interruption.min_duration`` blocks the STT-failsafe barge-in.
        """
        audio_recognition = self._create_audio_recognition()
        audio_recognition._turn_detection_mode = "stt"
        audio_recognition._speaking = True
        assert audio_recognition._vad_speech_duration is None
        assert audio_recognition.current_speech_duration is None

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.INFERENCE_DONE, speech_duration=0.0)
        )

        assert audio_recognition._vad_speech_duration is None
        assert audio_recognition.current_speech_duration is None

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.INFERENCE_DONE, speech_duration=0.42)
        )
        assert audio_recognition.current_speech_duration == pytest.approx(0.42)
