"""Live ElevenLabs transcription with local VAD."""

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


async def test_live_vad_flush_preserves_each_turn() -> None:
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
            turn_handling={"turn_detection": "vad"},
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
                await feed(silence)
                transcript = await asyncio.wait_for(finals.get(), timeout=20)
                assert "weather" in transcript.lower()
                assert "paris" in transcript.lower()
                assert finals.empty()
        finally:
            await session.aclose()
            await stt.aclose()
