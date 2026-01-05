#!/usr/bin/env python3
"""
Simplest possible Camb.ai TTS demo.

Usage:
    export CAMB_API_KEY=your_key_here
    cd livekit-plugins/livekit-plugins-camb
    uv run python examples/simple_demo.py
"""

import asyncio
import os
import sys
import wave

# Load .env file from examples directory
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not required, can use export instead

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from livekit.plugins.camb import TTS


async def main():
    if not os.getenv("CAMB_API_KEY"):
        print("Error: CAMB_API_KEY environment variable not set")
        return

    # Create TTS instance with Attic voice and MARS-8-Flash model
    tts = TTS(
        api_key=os.getenv("CAMB_API_KEY"),
        voice_id=2681,  # Attic voice
        model="mars-8-flash",  # Faster inference
    )

    # Synthesize
    print("Synthesizing with Attic voice using MARS-8-Flash model...")
    stream = tts.synthesize("Hello from Camb.ai!")
    audio = await stream.collect()

    # Save as proper WAV file with headers
    output_path = "examples/output.wav"
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(audio.num_channels)
        wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.data)

    print(f"✅ Generated {audio.duration:.2f}s of audio")
    print(f"   Sample rate: {audio.sample_rate} Hz")
    print(f"   Channels: {audio.num_channels}")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
