"""livekit-plugins-sherpa TTS behaviour against a scripted engine (no models, no network).

The LiveKit side (ChunkedStream, AudioEmitter, frames) is the real SDK.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import get_args

import numpy as np
import numpy.typing as npt
import pytest

from livekit.plugins import sherpa
from livekit.plugins.sherpa.tts import ChunkCallback

KOKORO = get_args(sherpa.KokoroVoice)

pytestmark = pytest.mark.unit

RATE = 24000


@dataclass
class _Audio:
    samples: npt.NDArray[np.float32]
    sample_rate: int = RATE


@dataclass
class ScriptedEngine:
    """Produces `chunks` through the callback, one per call, like Kokoro does per sentence."""

    chunks: list[npt.NDArray[np.float32]]
    calls: list[tuple[str, int, float]] = field(default_factory=list)
    gate: threading.Semaphore | None = None
    """When set, each chunk waits for one release, so the test decides when audio is made."""
    stopped_early: bool = False

    @property
    def sample_rate(self) -> int:
        return RATE

    def generate(
        self, text: str, sid: int = 0, speed: float = 1.0, callback: ChunkCallback | None = None
    ) -> _Audio:
        self.calls.append((text, sid, speed))
        for chunk in self.chunks:
            if self.gate is not None:
                self.gate.acquire(timeout=5)
            if callback is not None and callback(chunk, 0.0) != 0:
                self.stopped_early = True
                break
        return _Audio(np.concatenate(self.chunks))


def _tone(seconds: float, value: float) -> npt.NDArray[np.float32]:
    return np.full(int(RATE * seconds), value, dtype=np.float32)


async def test_audio_arrives_in_order_as_one_utterance() -> None:
    engine = ScriptedEngine([_tone(0.3, 0.25), _tone(0.2, -0.5)])
    frame = await sherpa.TTS(engine=engine).synthesize("Hello there. Bye.").collect()

    expected = np.concatenate([_tone(0.3, 0.25), _tone(0.2, -0.5)]) * 32767
    pcm = np.frombuffer(frame.data, dtype=np.int16)
    assert frame.sample_rate == RATE
    # The emitter may close the segment with a 10 ms silent marker frame.
    assert np.array_equal(pcm[: len(expected)], expected.astype(np.int16))
    assert not pcm[len(expected) :].any()


async def test_voice_and_speed_reach_the_engine() -> None:
    engine = ScriptedEngine([_tone(0.1, 0.0)])
    voice = sherpa.TTS(engine=engine, speakers=KOKORO, voice="bm_george", speed=1.2)

    await voice.synthesize("Hi.").collect()
    voice.update_options(voice="af_heart", speed=0.9)
    await voice.synthesize("Hi again.").collect()

    # Speaker ids from the Kokoro v1.0 voice table.
    assert engine.calls == [("Hi.", 26, 1.2), ("Hi again.", 3, 0.9)]


async def test_blank_text_produces_no_audio_and_skips_the_engine() -> None:
    engine = ScriptedEngine([_tone(0.1, 0.1)])
    frames = [ev async for ev in sherpa.TTS(engine=engine).synthesize("   ")]

    assert frames == []
    assert engine.calls == []


async def test_interruption_stops_the_engine() -> None:
    gate = threading.Semaphore(0)
    engine = ScriptedEngine([_tone(0.25, 0.1)] * 50, gate=gate)
    stream = sherpa.TTS(engine=engine).synthesize("A long answer the user talks over.")

    gate.release()
    await anext(stream)  # first audio is out, then the user interrupts
    await stream.aclose()
    gate.release()  # the engine reaches its next chunk and must be told to stop

    for _ in range(200):
        if engine.stopped_early:
            break
        await asyncio.sleep(0.01)
    assert engine.stopped_early
    assert len(engine.calls) == 1


async def test_a_speaker_id_works_without_a_speaker_table() -> None:
    # Piper, Matcha or any custom engine: no names known, ids still select the voice.
    engine = ScriptedEngine([_tone(0.1, 0.0)])
    await sherpa.TTS(engine=engine, voice=5).synthesize("Hi.").collect()

    assert engine.calls == [("Hi.", 5, 1.0)]


def test_unknown_voice_names_the_choices() -> None:
    with pytest.raises(ValueError, match="expr-voice-2-m"):
        sherpa.TTS(engine=ScriptedEngine([]), speakers=["expr-voice-2-m"], voice="af_heart")
    with pytest.raises(ValueError, match="speaker id"):
        sherpa.TTS(engine=ScriptedEngine([]), voice="af_heart")
