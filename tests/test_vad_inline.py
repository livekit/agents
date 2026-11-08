import asyncio
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents.inference import VAD

pytestmark = pytest.mark.unit


def samples():
    with wave.open(str(Path(__file__).parent / "test_realtime/weather_question.wav")) as w:
        raw = w.readframes(w.getnframes())
    # Resample once: independent Sox instances dither differently.
    rs = rtc.AudioResampler(input_rate=24000, output_rate=16000)
    fs = rs.push(rtc.AudioFrame(raw, 24000, 1, len(raw) // 2)) + rs.flush()
    return b"".join(f.data.tobytes() for f in fs) + bytes(16000 * 2)


async def collect(data, chunk, mode):
    s = VAD(execution_mode=mode).stream()
    try:
        for i in range(0, len(data), chunk * 2):
            d = data[i : i + chunk * 2]
            s.push_frame(rtc.AudioFrame(d, 16000, 1, len(d) // 2))
        s.end_input()
        result = []
        async for ev in s:
            result.append(
                (
                    ev.type,
                    ev.probability,
                    ev.timestamp,
                    ev.samples_index,
                    ev.speaking,
                    ev.speech_duration,
                    ev.silence_duration,
                    b"".join(f.data.tobytes() for f in ev.frames),
                )
            )
        return result
    finally:
        await s.aclose()


@pytest.mark.parametrize("chunk", [160, 512, 800, 1168])
async def test_inline_preserves_events_and_pcm(chunk):
    data = samples()
    a = await collect(data, chunk, "thread")
    b = await collect(data, chunk, "inline")
    assert a == b
    assert any(str(e[0]).endswith("START_OF_SPEECH") for e in a)


async def test_default_executor_and_inline_teardown():
    default = VAD().stream()
    executor = default._executor
    assert isinstance(executor, ThreadPoolExecutor)
    await default.aclose()
    assert executor._shutdown
    for _ in range(20):
        inline = VAD(execution_mode="inline").stream()
        assert inline._executor is None
        await inline.aclose()


async def test_inline_large_frame_yields_and_cancels():
    class Slow:
        calls = 0

        def predict(self, data):
            self.calls += 1
            time.sleep(0.001)
            return 0.0

    stream = VAD(execution_mode="inline").stream()
    slow = Slow()
    stream._native_vad = slow
    stream.push_frame(rtc.AudioFrame(bytes(5 * 16000 * 2), 16000, 1, 5 * 16000))
    await asyncio.sleep(0.005)
    await stream.aclose()
    assert 0 < slow.calls < 156


def test_bad_mode_and_executor_conflict():
    with pytest.raises(ValueError):
        VAD(execution_mode="wrong")
    with ThreadPoolExecutor(1) as ex:
        with pytest.raises(ValueError):
            VAD(execution_mode="inline", executor=ex)
