"""Synthesize speech through LiveKit Inference, then check it for the provider's signal.

Usage (from examples/inference):
    uv run watermark/tts_roundtrip.py

Shows the end-to-end path: inference.TTS -> WAV bytes -> WatermarkDetector.
Cartesia watermarks Sonic 3.6 and newer only, and needs at least 5 seconds of
audio; Inworld scores inworld-tts-2 and inworld-tts-2-flash.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import APIError, inference, utils

load_dotenv(Path(__file__).parents[2] / ".env")

# Long enough for Cartesia's 5-second minimum at a normal speaking rate.
TEXT = (
    "Thanks for calling. I can help you reschedule your appointment, update your "
    "contact details, or answer questions about your bill. Which of those would "
    "you like to start with today?"
)

# The TTS model to synthesize with, and the provider whose detector checks it.
PAIRS = [
    ("cartesia/sonic-3.6", "cartesia"),
    ("inworld/inworld-tts-2", "inworld"),
]


async def synthesize_wav(model: str) -> bytes:
    tts = inference.TTS(model)
    try:
        async with tts.synthesize(TEXT) as stream:
            frame = await stream.collect()
    finally:
        await tts.aclose()
    # The detector needs a container header; raw PCM frames are refused.
    return frame.to_wav_bytes()


async def roundtrip(detector: inference.WatermarkDetector, model: str, provider: str) -> bool:
    audio = await synthesize_wav(model)
    try:
        check = await detector.check(audio, provider=provider)
    except APIError as e:
        print(f"{model}: check failed: {e}", file=sys.stderr)
        return False
    for r in check.results:
        verdict = r.outcome if r.outcome is not None else r.probability
        print(f"{model} -> {provider}: {r.type} = {verdict}")
    return True


async def main() -> int:
    async with utils.http_context.open():
        detector = inference.WatermarkDetector()
        ok = [await roundtrip(detector, model, provider) for model, provider in PAIRS]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
