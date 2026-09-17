"""Experiment matrix for the empty-STS-transcript issue.

Tries longer audio, LINEAR16 encoding, and trailing silence before stop,
through the plugin's SpeechStream, and reports which variants return text.
"""

import asyncio
import wave

from livekit.agents import stt
from livekit.plugins._60db import STT
from livekit.rtc import AudioFrame

CHUNK_MS = 30


async def run_variant(
    name: str, wav_path: str, *, encoding: str, sample_rate: int, trailing_silence_s: float
) -> str:
    model = STT(encoding=encoding, sample_rate=sample_rate)
    wf = wave.open(wav_path, "rb")
    rate, ch = wf.getframerate(), wf.getnchannels()

    final = ""
    async with model.stream() as stream:
        while True:
            data = wf.readframes(int(rate * CHUNK_MS / 1000))
            if not data:
                break
            frame = AudioFrame(
                data=bytes(data),
                sample_rate=rate,
                num_channels=ch,
                samples_per_channel=len(data) // (2 * ch),
            )
            stream.push_frame(frame)
            await asyncio.sleep(CHUNK_MS / 1000)

        if trailing_silence_s > 0:
            silence = b"\x00" * int(rate * 2 * trailing_silence_s)  # 16-bit mono
            sent = 0
            chunk = b"\x00" * int(rate * 2 * CHUNK_MS / 1000)
            while sent < len(silence):
                frame = AudioFrame(
                    data=chunk,
                    sample_rate=rate,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
                stream.push_frame(frame)
                await asyncio.sleep(CHUNK_MS / 1000)
                sent += len(chunk)

        stream.end_input()
        async for ev in stream:
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                final = ev.alternatives[0].text if ev.alternatives else ""

    print(f"{name:38} -> {final!r}")
    return final


async def main() -> None:
    variants = [
        (
            "A: mulaw/8k, longer audio (5.9s)",
            "tests/change-sophie.wav",
            {"encoding": "mulaw", "sample_rate": 8000, "trailing_silence_s": 0},
        ),
        (
            "B: mulaw/8k, short audio + 1s silence",
            "tts_output.wav",
            {"encoding": "mulaw", "sample_rate": 8000, "trailing_silence_s": 1.0},
        ),
        (
            "C: LINEAR16/16k, longer audio",
            "tests/change-sophie.wav",
            {"encoding": "LINEAR16", "sample_rate": 16000, "trailing_silence_s": 0},
        ),
    ]
    results = {}
    for name, path, kwargs in variants:
        try:
            results[name] = await run_variant(name, path, **kwargs)
        except Exception as exc:
            print(f"{name:38} -> ERROR: {exc}")
            results[name] = ""

    print("\nsummary:")
    for name, txt in results.items():
        status = "TEXT RETURNED" if txt.strip() else "empty"
        print(f"  [{status:13}] {name}")


asyncio.run(main())
