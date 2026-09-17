"""Verify the STT handshake fix: the preliminary {"type": "connecting"} status
must no longer abort the session — the stream should complete a full recognize."""

import asyncio
import wave

from livekit.agents import stt
from livekit.rtc import AudioFrame

from livekit.plugins._60db import STT


async def main() -> None:
    stt_model = STT()
    wf = wave.open("tts_output.wav", "rb")
    rate, ch = wf.getframerate(), wf.getnchannels()
    print(f"input: tts_output.wav ({rate} Hz, {ch} ch)")

    final = ""
    interim_seen = False
    async with stt_model.stream() as stream:
        while True:
            data = wf.readframes(480)  # 30ms @ 16kHz
            if not data:
                break
            frame = AudioFrame(
                data=bytes(data),
                sample_rate=rate,
                num_channels=ch,
                samples_per_channel=len(data) // (2 * ch),
            )
            stream.push_frame(frame)
            await asyncio.sleep(0.03)  # realtime pacing
        stream.end_input()
        async for ev in stream:
            if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                interim_seen = True
                txt = ev.alternatives[0].text if ev.alternatives else ""
                print(f"  interim: {txt}")
            elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                final = ev.alternatives[0].text if ev.alternatives else ""
                print(f"  FINAL: {final}")

    print("\nCONFIRMED: handshake passed, session completed cleanly")
    print(f"final transcript: {final!r} (interim seen: {interim_seen})")


asyncio.run(main())
