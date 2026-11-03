"""Live ElevenLabs transcription with manual and automatic turn commits."""

from __future__ import annotations

import asyncio
import os
import time
import wave
from pathlib import Path

import aiohttp
import pytest
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import elevenlabs, silero

pytestmark = pytest.mark.plugin("elevenlabs")


class _FixedTurnDetector:
    model = "fixed-test-prediction"
    provider = "test"

    def __init__(self, probability: float) -> None:
        self.probability = probability

    async def supports_language(self, language):
        return True

    async def unlikely_threshold(self, language):
        return 0.5

    async def predict_end_of_turn(self, chat_ctx, *, timeout=None):
        return self.probability


@pytest.mark.parametrize("turn_detection", ["vad", "manual", 0.1, 0.9])
async def test_live_flush_preserves_each_turn(turn_detection) -> None:
    load_dotenv(Path(__file__).parents[1] / ".env")
    if not os.environ.get("ELEVEN_API_KEY"):
        pytest.skip("ELEVEN_API_KEY is required for live ElevenLabs validation")

    with wave.open(str(Path(__file__).parent / "test_realtime/weather_question.wav")) as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        audio = wav.readframes(wav.getnframes())

    async with aiohttp.ClientSession() as http:
        stt = elevenlabs.STT(model="scribe_v2_realtime", language_code="en", http_session=http)
        session = AgentSession(
            stt=stt,
            vad=silero.VAD.load(),
            turn_handling={
                "turn_detection": _FixedTurnDetector(turn_detection)
                if isinstance(turn_detection, float)
                else turn_detection,
            },
            session_close_transcript_timeout=0,
        )
        finals: asyncio.Queue[str] = asyncio.Queue()

        @session.on("user_input_transcribed")
        def on_transcript(event):
            if event.is_final:
                finals.put_nowait(event.transcript)

        async def feed(data: bytes) -> None:
            deadline = time.monotonic()
            chunk_bytes = sample_rate // 50 * channels * 2
            for offset in range(0, len(data), chunk_bytes):
                chunk = data[offset : offset + chunk_bytes]
                assert session._activity is not None
                session._activity.push_audio(
                    rtc.AudioFrame(chunk, sample_rate, channels, len(chunk) // (2 * channels))
                )
                deadline += len(chunk) / (2 * channels * sample_rate)
                await asyncio.sleep(max(0, deadline - time.monotonic()))

        try:
            await session.start(Agent(instructions="Transcription test."))
            silence = bytes(sample_rate * channels * 2 * 2)
            await feed(silence)
            assert finals.empty()
            for _ in range(2):
                await feed(audio)
                if turn_detection == "manual":
                    committed = await session.commit_user_turn(
                        transcript_timeout=10, skip_reply=True
                    )
                    assert "weather" in committed.lower()
                else:
                    await feed(silence)
                transcript = await asyncio.wait_for(finals.get(), timeout=20)
                assert "weather" in transcript.lower()
                assert "paris" in transcript.lower()
                assert finals.empty()
        finally:
            await session.aclose()
            await stt.aclose()
