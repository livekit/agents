"""Cartesia STT transcribe on flush use-case: manual turn detection with Silero VAD.

This example showcases advanced usage.
If you're building your first voice agent, try examples/other/cartesia.py

When configured to "transcribe_on_flush", Cartesia STT only emits a
:attr:`~livekit.agents.stt.SpeechEventType.FINAL_TRANSCRIPT` when *you* call
:meth:`~livekit.agents.stt.RecognizeStream.flush`.

It never emits ``START_OF_SPEECH`` / ``END_OF_SPEECH``.

This example shows how you can decide turn boundaries yourself with your own VAD:

1. The same audio is pushed to a ``silero.VAD`` stream and the STT stream
2. Every time the VAD reports :attr:`~livekit.agents.vad.VADEventType.END_OF_SPEECH`,
    we call ``stt_stream.flush()`` to close out the turn and get its transcript.

Note: this is *not* how ``AgentSession`` works. The session never calls ``flush()``, so
``transcribe_on_flush`` is meant for code that drives a ``RecognizeStream`` directly, like this.

The audio is generated with ``cartesia.TTS`` so the example is self-contained. Several short
utterances are concatenated with silence gaps so the VAD sees distinct turns.

Run with ``CARTESIA_API_KEY`` set:

    uv run examples/other/cartesia_transcribe_on_flush_vad.py
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import stt, utils, vad
from livekit.plugins import cartesia, silero

UTTERANCES = [
    "What is the weather like today?",
    "Set a timer for ten minutes.",
    "Thanks, that is all for now.",
]

SAMPLE_RATE = 16000
FRAME_MS = 20
# Silence between utterances; must exceed the VAD's min_silence_duration so it reports
# END_OF_SPEECH (and we flush) before the next utterance begins.
SILENCE_GAP_S = 1.0
MIN_SILENCE_DURATION = 0.55


def _resample(frame: rtc.AudioFrame, sample_rate: int) -> rtc.AudioFrame:
    if frame.sample_rate == sample_rate:
        return frame
    resampler = rtc.AudioResampler(
        input_rate=frame.sample_rate,
        output_rate=sample_rate,
        num_channels=frame.num_channels,
    )
    frames = resampler.push(frame)
    frames.extend(resampler.flush())
    return rtc.combine_audio_frames(frames)


async def build_timeline(tts: cartesia.TTS) -> list[rtc.AudioFrame]:
    """Synthesize each utterance, resample to 16 kHz, and append a silence gap.

    Returns the whole timeline chunked into ``FRAME_MS`` frames, ready to stream.
    """
    samples_per_frame = SAMPLE_RATE * FRAME_MS // 1000
    bstream = utils.audio.AudioByteStream(
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples_per_frame,
    )
    silence = b"\x00\x00" * int(SAMPLE_RATE * SILENCE_GAP_S)

    frames: list[rtc.AudioFrame] = []
    for text in UTTERANCES:
        combined = _resample(await tts.synthesize(text).collect(), SAMPLE_RATE)
        frames.extend(bstream.write(combined.data.tobytes()))
        frames.extend(bstream.write(silence))
    frames.extend(bstream.flush())
    return frames


async def main() -> None:
    load_dotenv()

    logger = logging.getLogger("cartesia-transcribe-on-flush-vad")

    logging.basicConfig(level=logging.INFO)

    vad_model = silero.VAD.load(min_silence_duration=MIN_SILENCE_DURATION)

    # A standalone script has no agent http context, so share one explicit session.
    async with aiohttp.ClientSession() as http_session:
        tts = cartesia.TTS(model="sonic-3.5", http_session=http_session)
        speech_to_text = cartesia.STT(
            model="ink-2",
            behavior="transcribe_on_flush",
            http_session=http_session,
        )

        frames = await build_timeline(tts)
        vad_stream = vad_model.stream()
        stt_stream = speech_to_text.stream()

        async def push_audio() -> None:
            for frame in frames:
                # Push to STT first so a flush triggered by VAD always sees at least the
                # audio the VAD has already consumed.
                stt_stream.push_frame(frame)
                vad_stream.push_frame(frame)
                # Pace in real time so END_OF_SPEECH for one turn fires during its silence
                # gap, before the next turn's audio reaches the STT.
                await asyncio.sleep(frame.duration)
            vad_stream.end_input()

        async def drive_flush_from_vad() -> None:
            async for ev in vad_stream:
                if ev.type == vad.VADEventType.START_OF_SPEECH:
                    logger.info("user started speaking")
                elif ev.type == vad.VADEventType.END_OF_SPEECH:
                    logger.info("user stopped speaking -> stt_stream.flush()")
                    stt_stream.flush()
            # No more turns; close the STT input once the VAD is done.
            stt_stream.end_input()

        async def print_finals() -> None:
            turn = 0
            async for ev in stt_stream:
                if ev.type != stt.SpeechEventType.FINAL_TRANSCRIPT:
                    continue
                if not ev.alternatives:
                    continue
                turn += 1
                logger.info("turn %d transcript: %s", turn, ev.alternatives[0].text)

        try:
            await asyncio.gather(push_audio(), drive_flush_from_vad(), print_finals())
        finally:
            await vad_stream.aclose()
            await stt_stream.aclose()


if __name__ == "__main__":
    asyncio.run(main())
