from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins import sarvam

pytestmark = pytest.mark.plugin("sarvam")


async def _send_speech(stream: stt.SpeechStream, name: str, *, silence_ms: int) -> None:
    path = Path(__file__).parent / "test_realtime" / f"{name}.wav"
    with wave.open(str(path)) as wav:
        sample_rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    pcm += bytes(sample_rate * silence_ms // 1000 * 2)
    chunk_bytes = sample_rate // 50 * 2
    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        stream.push_frame(rtc.AudioFrame(chunk, sample_rate, 1, len(chunk) // 2))
        await asyncio.sleep(len(chunk) / (sample_rate * 2))


@pytest.mark.parametrize("model_name", ["saaras:v3", "saaras:v4"])
@pytest.mark.parametrize("client_flush", [True, False])
async def test_live_sarvam_transcribes_two_utterances_on_one_connection(
    model_name: str, client_flush: bool
) -> None:
    async with sarvam.STT(
        model=model_name,
        language="en-IN",
        flush_signal=True,
    ) as model:
        stream = model.stream()
        finals: asyncio.Queue[str] = asyncio.Queue()

        async def collect() -> None:
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    finals.put_nowait(event.alternatives[0].text)

        collector = asyncio.create_task(collect())
        connection = None
        try:
            for name, expected in [
                ("hello_world", ["hello", "world"]),
                ("weather_question", ["weather", "paris"]),
            ]:
                await _send_speech(stream, name, silence_ms=100 if client_flush else 2000)
                if client_flush:
                    stream.flush()
                transcript = await asyncio.wait_for(finals.get(), timeout=20)
                assert all(word in transcript.lower() for word in expected), transcript
                assert not collector.done()
                assert stream._ws is not None
                assert not stream._ws.closed
                if connection is None:
                    connection = stream._ws
                else:
                    assert stream._ws is connection
        finally:
            await stream.aclose()
            await collector
