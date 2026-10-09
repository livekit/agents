"""
Standalone TTS example using Lokutor outside of the agent pipeline.

Run:
  LOKUTOR_API_KEY="your-api-key" uv run python examples/standalone-tts.py
"""

import asyncio
import os

import aiohttp

from livekit import rtc
from livekit.plugins import lokutor


async def main():
    async with aiohttp.ClientSession() as session:
        tts = lokutor.TTS(
            api_key=os.environ["LOKUTOR_API_KEY"],
            voice="F1",
            language="en",
            speed=1.05,
            steps=5,
            http_session=session,
        )

        async with tts:
            stream = tts.stream()
            stream.push_text(
                "Hello! This is a test of the Lokutor TTS integration with LiveKit Agents."
            )
            stream.end_input()

            frames = []
            async for audio in stream:
                frames.append(audio.frame)

            if frames:
                combined = rtc.combine_audio_frames(frames)
                import wave

                with wave.open("output.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(combined.sample_rate)
                    wf.writeframes(combined.data.tobytes())
                print(f"Saved output.wav ({combined.sample_rate} Hz, {combined.duration:.2f}s)")
            else:
                print("No audio generated")


if __name__ == "__main__":
    asyncio.run(main())
