"""
- The caller's voice for the e2e test, as a WAV Chromium plays into the fake microphone
  (`--use-file-for-fake-audio-capture`): a few seconds of silence while the agent greets,
  one line synthesized with the LiveKit inference TTS, then silence so the realtime model's turn detection hears the line end and the
  expert has time to answer.
- Run with the environment of examples/README.md (the inference credentials):
  `python console/e2e/caller-wav.py /tmp/lk-console/caller.wav [line]`.
"""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import wave

from livekit.agents import inference, utils

SAMPLE_RATE = 24000
LEAD = float(os.environ.get("E2E_CALLER_LEAD", "6"))
TAIL = float(os.environ.get("E2E_CALLER_TAIL", "60"))
LINE = "I would like to book a visit with Doctor Lin on Friday, please."


async def main(path: str, line: str) -> None:
    pcm = bytearray()
    async with utils.http_context.open():
        tts = inference.TTS(os.environ.get("V0_CALLER_TTS", "cartesia/sonic-3"), sample_rate=SAMPLE_RATE)
        async for ev in tts.synthesize(line):
            pcm += bytes(ev.frame.data)
        await tts.aclose()
    silence = lambda seconds: struct.pack("<h", 0) * int(SAMPLE_RATE * seconds)  # noqa: E731
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(silence(LEAD) + bytes(pcm) + silence(TAIL))
    print(f"{path}: {LEAD:.0f}s silence, {len(pcm) / 2 / SAMPLE_RATE:.1f}s of speech, {TAIL:.0f}s silence")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], " ".join(sys.argv[2:]) or LINE))
