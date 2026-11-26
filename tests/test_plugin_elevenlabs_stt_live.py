"""Live ElevenLabs transcription with manual and automatic turn commits."""

from __future__ import annotations

import asyncio
import os
import time
import wave
from pathlib import Path

import aiohttp
import av
import pytest
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import elevenlabs, silero

from .fake_llm import FakeLLM
from .fake_stt import FakeUserSpeech
from .fake_vad import FakeVAD

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


@pytest.mark.parametrize("turn_detection", ["vad", "manual", 0.1, 0.9, "stt"])
async def test_live_flush_preserves_each_turn(turn_detection) -> None:
    load_dotenv(Path(__file__).parents[1] / ".env")
    if not os.environ.get("ELEVEN_API_KEY"):
        pytest.skip("ELEVEN_API_KEY is required for live ElevenLabs validation")

    with wave.open(str(Path(__file__).parent / "test_realtime/weather_question.wav")) as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        audio = wav.readframes(wav.getnframes())

    async with aiohttp.ClientSession() as http:
        stt = elevenlabs.STT(
            model="scribe_v3_live" if turn_detection == "stt" else "scribe_v2_realtime",
            commit_strategy="turn_prediction" if turn_detection == "stt" else "manual",
            language_code="en",
            http_session=http,
        )
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


@pytest.mark.no_concurrent
@pytest.mark.parametrize("speech_seconds,stuck_vad", [(63.0, False), (34.5, True)])
async def test_live_forced_commit_preserves_turn_and_eventually_ends(
    monkeypatch: pytest.MonkeyPatch, speech_seconds: float, stuck_vad: bool
) -> None:
    load_dotenv(Path(__file__).parents[1] / ".env")
    if not os.environ.get("ELEVEN_API_KEY"):
        pytest.skip("ELEVEN_API_KEY is required for live ElevenLabs validation")

    rate = 16000
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    parts: list[bytes] = []
    with av.open(str(Path(__file__).parent / "long.mp3")) as container:
        for frame in container.decode(audio=0):
            parts.extend(f.to_ndarray().tobytes() for f in resampler.resample(frame))
    parts.extend(f.to_ndarray().tobytes() for f in resampler.resample(None))
    original = b"".join(parts)
    speech_size = int(speech_seconds * rate * 2)
    audio = (original * (speech_size // len(original) + 1))[:speech_size]

    # Keep VAD positive through the silence to test background audio's effect on
    # the fallback. The long-turn case uses actual Silero inference instead.
    detector = (
        FakeVAD(
            fake_user_speeches=[
                FakeUserSpeech(start_time=0, end_time=100, transcript="", stt_delay=0)
            ]
        )
        if stuck_vad
        else silero.VAD.load()
    )
    turns: list[str] = []

    class ProbeAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx, new_message):
            turns.append(new_message.text_content)

    async with aiohttp.ClientSession() as http:
        provider = elevenlabs.STT(
            model="scribe_v3_live",
            commit_strategy="turn_prediction",
            language_code="en",
            include_language_detection=False,
            server_vad={"vad_silence_threshold_secs": 3.0},
            vad=detector,
            http_session=http,
        )
        suppressed = 0
        original_process = elevenlabs.SpeechStream._process_stream_event

        def observe(stream, data):
            nonlocal suppressed
            original_process(stream, data)
            if (
                stream._opts is provider._opts
                and data.get("message_type") == "committed_transcript"
                and stream._turn_detection._eos_timeout_handle is not None
            ):
                suppressed += 1

        monkeypatch.setattr(elevenlabs.SpeechStream, "_process_stream_event", observe)
        session = AgentSession(
            stt=provider,
            vad=None,
            llm=FakeLLM(),
            tts=None,
            turn_handling={
                "turn_detection": "stt",
                "endpointing": {"min_delay": 0},
                "preemptive_generation": {"enabled": False},
                "interruption": {"enabled": False},
            },
            session_close_transcript_timeout=0,
        )

        async def feed(data: bytes) -> None:
            deadline = time.monotonic()
            chunk_bytes = rate * 2 // 50
            for offset in range(0, len(data), chunk_bytes):
                chunk = data[offset : offset + chunk_bytes]
                assert session._activity is not None
                session._activity.push_audio(rtc.AudioFrame(chunk, rate, 1, len(chunk) // 2))
                deadline += len(chunk) / (rate * 2)
                await asyncio.sleep(max(0, deadline - time.monotonic()))

        try:
            await session.start(ProbeAgent(instructions="Transcription test."))
            await feed(audio)
            assert not turns, "the user turn ended before speech finished"
            await feed(bytes(7 * rate * 2))
            assert len(turns) == 1, "the turn must end once, including when no more finals arrive"
            assert "hands" in turns[0].lower()
            assert suppressed, "the live run must exercise the forced-commit guard"
        finally:
            await session.aclose()
            await provider.aclose()
