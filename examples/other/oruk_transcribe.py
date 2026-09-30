"""Transcribe a PCM16 WAV file using ORUK_API_KEY, without a LiveKit room."""

import argparse
import asyncio
import wave

from livekit import rtc
from livekit.plugins import oruk


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="PCM16 WAV, 45 ms to 60 seconds")
    parser.add_argument("--model", default="oruk-spectra-2")
    args = parser.parse_args()
    with wave.open(args.audio) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("The example requires PCM16 WAV")
        frame = rtc.AudioFrame(
            wav.readframes(wav.getnframes()),
            wav.getframerate(),
            wav.getnchannels(),
            wav.getnframes(),
        )
    recognizer = oruk.STT(model=args.model)
    try:
        event = await recognizer.recognize(frame)
        print(event.alternatives[0].text)
    finally:
        await recognizer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
