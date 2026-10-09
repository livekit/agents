"""Transcribe a PCM16 WAV locally: python orukeet_transcribe.py recording.wav.

Install livekit-plugins-orukeet first. Add --offline after the model is cached.
"""

import argparse
import asyncio
import wave

from livekit import rtc
from livekit.plugins import orukeet


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_file")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    with wave.open(args.wav_file, "rb") as source:
        if source.getsampwidth() != 2:
            parser.error("the input must be a 16-bit PCM WAV")
        frame = rtc.AudioFrame(
            source.readframes(source.getnframes()),
            source.getframerate(),
            source.getnchannels(),
            source.getnframes(),
        )
    recognizer = orukeet.STT(local_files_only=args.offline)
    try:
        event = await recognizer.recognize(frame)
        print(event.alternatives[0].text)
    finally:
        await recognizer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
