"""Sarvam streaming STT -> voice pipeline: the ``transcription_delay`` anchor.

Drives the real Sarvam ``SpeechStream`` handlers and hands every END_OF_SPEECH and
FINAL_TRANSCRIPT they emit to ``AudioRecognition`` as they happen, in STT turn
detection with no local VAD (Sarvam's VAD owns the turn). ``transcription_delay``
must measure Sarvam's speech end -> final transcript received.

A final that misses the EOS grace period reaches the pipeline after END_OF_SPEECH
without an ``end_time`` (Sarvam's socket sends no ``speech_end`` today), so it
re-anchors the turn at its own arrival; that case needs provider audio positions.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import stt
from livekit.agents.voice.audio_recognition import AudioRecognition, _compute_end_of_turn_metrics
from livekit.plugins import sarvam
from livekit.plugins.sarvam.stt import SpeechStream

from .test_transcription_delay_anchor import _make_recognition

pytestmark = pytest.mark.unit

FINAL_LATENCY = 0.2  # END_SPEECH -> final transcript


@pytest.fixture
async def sarvam_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[SpeechStream, list[Any]]]:
    """A real Sarvam stream that never connects; the events it emits are captured."""

    async def _no_connection(self: SpeechStream) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(SpeechStream, "_run", _no_connection)
    stream = sarvam.STT(api_key="sk_test", language="hi-IN").stream()
    captured: list[Any] = []
    send = stream._event_ch.send_nowait

    def _capture(ev: stt.SpeechEvent) -> None:
        captured.append(ev)
        send(ev)

    stream._event_ch.send_nowait = _capture  # type: ignore[method-assign]
    yield stream, captured
    await stream.aclose()


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


async def test_transcription_delay_measures_sarvam_speech_end_to_final(
    sarvam_stream: tuple[SpeechStream, list[Any]],
) -> None:
    ar = _make_recognition(vad=None, input_started_at=time.time() - 10.0, mode="stt")
    stream, captured = sarvam_stream
    # the final lands well within the grace period, as it does in practice
    stream._eos_fallback_timeout = FINAL_LATENCY * 4
    seen = 0

    await stream._handle_message(_signal("START_SPEECH"))
    await stream._handle_message(_signal("END_SPEECH"))
    speech_end = time.time()
    seen = await _deliver(ar, captured, seen)

    await asyncio.sleep(FINAL_LATENCY)
    await stream._handle_message(_transcript())
    seen = await _deliver(ar, captured, seen)

    order = [
        ev.type
        for ev in captured
        if ev.type in (stt.SpeechEventType.END_OF_SPEECH, stt.SpeechEventType.FINAL_TRANSCRIPT)
    ]
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
