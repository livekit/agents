"""Live Gnani TTS checks against the real Vachana API. Skipped without ``GNANI_API_KEY``.

Short texts, a handful of requests: these are billed, so they only run when asked for.
"""

from __future__ import annotations

import array
import os

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APIStatusError

pytestmark = [
    pytest.mark.plugin("gnani"),
    pytest.mark.skipif(not os.environ.get("GNANI_API_KEY"), reason="GNANI_API_KEY not set"),
]

SAMPLE_RATE = 16000
OPTS = APIConnectOptions(max_retry=0, timeout=30)
TEXT = "Namaste, aap kaise hain? Main aapki madad ke liye yahan hoon."


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


def _tts(session, **kwargs):
    from livekit.plugins.gnani import TTS

    tts = TTS(sample_rate=SAMPLE_RATE, **kwargs)
    tts._session = session
    return tts


async def _synth(tts, text: str = TEXT) -> bytes:
    audio = bytearray()
    async with tts.synthesize(text, conn_options=OPTS) as stream:
        async for ev in stream:
            audio += bytes(ev.frame.data)
    return bytes(audio)


def _seconds(audio: bytes) -> float:
    return len(audio) / 2 / SAMPLE_RATE


def _peak(audio: bytes) -> int:
    return max(abs(s) for s in array.array("h", audio))


@pytest.mark.parametrize("method", ["rest", "sse", "websocket"])
@pytest.mark.parametrize("model", ["timbre-v2.5", "timbre-v3.0"])
async def test_live_synthesize_returns_speech(session, model, method):
    audio = await _synth(_tts(session, model=model, synthesize_method=method))
    assert 1.0 < _seconds(audio) < 20.0
    assert _peak(audio) > 1000  # actual speech, not silence


async def test_live_v3_speed_changes_duration(session):
    slow = await _synth(_tts(session, model="timbre-v3.0", speed=0.85))
    fast = await _synth(_tts(session, model="timbre-v3.0", speed=1.15))
    assert _seconds(fast) < _seconds(slow)


async def test_live_v3_language_and_audio_tags(session):
    tts = _tts(session, model="timbre-v3.0", voice="Shivani", language="hi-IN")
    audio = await _synth(tts, "Haha <laugh> sach mein? <sigh> theek hai.")
    assert _seconds(audio) > 1.0


async def test_live_stream_uses_v3(session):
    tts = _tts(session, model="timbre-v3.0")
    audio = bytearray()
    async with tts.stream(conn_options=OPTS) as stream:
        stream.push_text("Namaste, ")
        stream.push_text("aap kaise hain?")
        stream.end_input()
        async for ev in stream:
            audio += bytes(ev.frame.data)
    assert _seconds(bytes(audio)) > 0.8
    assert _peak(bytes(audio)) > 1000


async def test_live_retired_model_is_rejected_before_any_request():
    from livekit.plugins.gnani import TTS

    with pytest.raises(ValueError, match="retired"):
        TTS(model="timbre-v2.0")


async def test_live_api_rejects_unknown_voice_with_400(session):
    """The API stays the source of truth for voices the plugin does not know about."""
    tts = _tts(session, model="timbre-v3.0", voice="NoSuchVoice")
    with pytest.raises(APIStatusError) as exc:
        await _synth(tts, "hi")
    assert exc.value.status_code == 400
