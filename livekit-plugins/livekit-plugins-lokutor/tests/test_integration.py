"""Live integration test against the real Lokutor API.

Skipped unless LOKUTOR_API_KEY is set, so the default `pytest` run stays
offline and hermetic. To run it:

    LOKUTOR_API_KEY="sk_your_key" pytest tests/test_integration.py -v

It exercises the full path — WebSocket connect, a request carrying the
canonical `language` field, binary audio frames, and the `EOS` end signal —
so it will catch a wire-protocol regression that the unit tests (which only
build the request dict) cannot.

Written as sync functions driving ``asyncio.run`` so the suite needs no
``pytest-asyncio`` plugin or event-loop configuration.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LOKUTOR_API_KEY"),
    reason="LOKUTOR_API_KEY not set; skipping live Lokutor API test",
)


async def _stream_synthesis() -> tuple[int, int]:
    import aiohttp

    from livekit.plugins import lokutor

    async with aiohttp.ClientSession() as session:
        tts = lokutor.TTS(voice="F1", language="en", steps=8, http_session=session)
        async with tts:
            stream = tts.stream()
            stream.push_text("Testing the Lokutor LiveKit integration end to end.")
            stream.end_input()

            total_bytes = 0
            frames = 0
            async for audio in stream:
                total_bytes += len(audio.frame.data)
                frames += 1
    return frames, total_bytes


async def _chunked_synthesis() -> int:
    import aiohttp

    from livekit.plugins import lokutor

    async with aiohttp.ClientSession() as session:
        tts = lokutor.TTS(voice="M1", language="en", steps=8, http_session=session)
        async with tts:
            total_bytes = 0
            async for audio in tts.synthesize("A short one-shot synthesis."):
                total_bytes += len(audio.frame.data)
    return total_bytes


def test_live_streaming_synthesis_returns_audio():
    frames, total_bytes = asyncio.run(_stream_synthesis())
    assert frames > 0, "no audio frames were emitted"
    assert total_bytes > 0, "audio frames were empty"


def test_live_chunked_synthesis_returns_audio():
    total_bytes = asyncio.run(_chunked_synthesis())
    assert total_bytes > 0, "chunked synthesis produced no audio"


async def _synthesize_then_transcribe() -> tuple[str, str]:
    """Lokutor speaks a sentence and Lokutor recognises it, both ways: one-shot and streamed."""
    import aiohttp

    from livekit import rtc
    from livekit.agents import stt as agents_stt
    from livekit.plugins import lokutor

    sentence = "Please book a table for four people on Friday evening."
    async with aiohttp.ClientSession() as session:
        tts = lokutor.TTS(voice="F1", language="en", http_session=session)
        async with tts:
            frames = [audio.frame async for audio in tts.synthesize(sentence)]
        speech = rtc.combine_audio_frames(frames)
        # Trailing silence so the stream sees the utterance end.
        silence = rtc.AudioFrame(
            b"\x00\x00" * speech.sample_rate,
            sample_rate=speech.sample_rate,
            num_channels=1,
            samples_per_channel=speech.sample_rate,
        )

        stt = lokutor.STT(language="en", http_session=session)
        one_shot = await stt.recognize([speech])

        stream = stt.stream()
        stream.push_frame(speech)
        stream.push_frame(silence)
        stream.end_input()
        finals = [
            ev.alternatives[0].text
            async for ev in stream
            if ev.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        await stream.aclose()
    return one_shot.alternatives[0].text, " ".join(finals)


def test_live_speech_to_text_recognises_lokutor_speech():
    one_shot, streamed = asyncio.run(_synthesize_then_transcribe())
    for text in (one_shot, streamed):
        words = text.lower()
        assert "table" in words and "friday" in words, f"unexpected transcript: {text!r}"
