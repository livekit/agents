from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt, vad
from livekit.plugins.mistralai import stt as mistral_stt

pytestmark = pytest.mark.plugin("mistralai")


async def test_endpoint_preserves_vad_time_before_flush(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(mistral_stt.time, "time", lambda: now)
    stream = object.__new__(mistral_stt.SpeechStream)
    stream._speaking = True
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)

    async def vad_events():
        yield vad.VADEvent(
            type=vad.VADEventType.END_OF_SPEECH,
            samples_index=0,
            timestamp=1.0,
            speech_duration=0.5,
            silence_duration=0.3,
            inference_duration=0.05,
        )

    async def flush():
        nonlocal now
        now += 2.0

    await stream._vad_task(vad_events(), SimpleNamespace(flush_audio=flush))
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert events[-1].speech_end_time == pytest.approx(999.65)
