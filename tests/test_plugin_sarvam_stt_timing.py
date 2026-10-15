"""Sarvam streaming STT -> voice pipeline: the ``transcription_delay`` anchor.

Drives the real Sarvam ``SpeechStream`` handlers and hands every END_OF_SPEECH and
FINAL_TRANSCRIPT they emit to ``AudioRecognition`` as they happen, in STT turn
detection with no local VAD (Sarvam's VAD owns the turn). ``transcription_delay``
must measure Sarvam's speech end -> final transcript received, whichever of the two
events the plugin emits first.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.agents.voice.audio_recognition import AudioRecognition, _compute_end_of_turn_metrics
from livekit.plugins.sarvam.stt import SpeechStream

from .test_transcription_delay_anchor import _make_recognition

pytestmark = pytest.mark.unit

FINAL_LATENCY = 0.2  # END_SPEECH -> final transcript


def _make_sarvam_stream(*, eos_fallback_timeout: float) -> tuple[SpeechStream, list[Any]]:
    """Wire the attributes the Sarvam message handlers touch."""
    stream = SpeechStream.__new__(SpeechStream)
    captured: list[Any] = []
    stream._event_ch = SimpleNamespace(send_nowait=captured.append)  # type: ignore[assignment]
    stream._logger = MagicMock()
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._server_request_id = "req-session"
    stream._opts = SimpleNamespace(language="hi-IN", sample_rate=16000)  # type: ignore[assignment]
    stream._speaking = False
    stream._should_flush = False
    stream._start_time_offset = 0.0
    stream._utterance_speech_start_wall = None
    stream._utterance_speech_end_wall = None
    stream._audio_sent_duration = 0.0
    stream._utterance_speech_end_audio_pos = None
    stream._pending_final_data = None
    stream._pending_eos = False
    stream._eos_fallback_task = None
    stream._eos_fallback_timeout = eos_fallback_timeout
    stream._final_received_for_utterance = False
    stream._eos_emitted_for_utterance = False
    return stream, captured


def _signal(signal_type: str) -> dict[str, Any]:
    return {"type": "events", "data": {"signal_type": signal_type}}


def _transcript() -> dict[str, Any]:
    # what the socket sends today: no word timestamps, null speech_start/speech_end
    return {
        "type": "data",
        "data": {
            "transcript": "नमस्ते, मुझे अपने खाते का बैलेंस जानना है",
            "language_code": "hi-IN",
            "speech_start": None,
            "speech_end": None,
            "metrics": {"audio_duration": 2.7},
            "request_id": "req-test",
        },
    }


async def _deliver(ar: AudioRecognition, captured: list[Any], seen: int) -> int:
    """Hand the newly emitted turn events to the pipeline, as the STT node would."""
    for ev in captured[seen:]:
        if ev.type in (stt.SpeechEventType.END_OF_SPEECH, stt.SpeechEventType.FINAL_TRANSCRIPT):
            await ar._on_stt_event(ev)
    return len(captured)


@pytest.mark.parametrize(
    "final_after_eos_fallback",
    [False, True],
    ids=["final_within_grace_period", "final_after_eos_fallback"],
)
async def test_transcription_delay_measures_sarvam_speech_end_to_final(
    final_after_eos_fallback: bool,
) -> None:
    input_started_at = time.time() - 10.0
    ar = _make_recognition(vad=None, input_started_at=input_started_at, mode="stt")
    # the grace period either outlasts the final's latency or expires well before it
    grace = FINAL_LATENCY / 4 if final_after_eos_fallback else FINAL_LATENCY * 4
    stream, captured = _make_sarvam_stream(eos_fallback_timeout=grace)
    seen = 0

    await stream._handle_message(_signal("START_SPEECH"))
    # audio streamed in real time: the audio sent so far matches the wall clock
    stream._audio_sent_duration = time.time() - input_started_at
    await stream._handle_message(_signal("END_SPEECH"))
    speech_end = time.time()
    seen = await _deliver(ar, captured, seen)

    # audio keeps flowing while Sarvam finalizes; the EOS fallback may fire meanwhile
    await asyncio.sleep(FINAL_LATENCY)
    seen = await _deliver(ar, captured, seen)
    stream._audio_sent_duration = time.time() - input_started_at
    await stream._handle_message(_transcript())
    seen = await _deliver(ar, captured, seen)

    order = [
        ev.type
        for ev in captured
        if ev.type in (stt.SpeechEventType.END_OF_SPEECH, stt.SpeechEventType.FINAL_TRANSCRIPT)
    ]
    if final_after_eos_fallback:
        assert order == [stt.SpeechEventType.END_OF_SPEECH, stt.SpeechEventType.FINAL_TRANSCRIPT]
    else:
        assert order == [stt.SpeechEventType.FINAL_TRANSCRIPT, stt.SpeechEventType.END_OF_SPEECH]

    assert ar._user_turn_committed is True
    assert ar._last_speaking_time == pytest.approx(speech_end, abs=0.02)

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=speech_end - 2.7,
        last_speaking_time=ar._last_speaking_time,
        last_final_transcript_time=ar._last_final_transcript_time,
        now=time.time(),
    )
    assert metrics.transcription_delay == pytest.approx(FINAL_LATENCY, abs=0.05)
