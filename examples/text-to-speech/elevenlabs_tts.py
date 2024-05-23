import asyncio
import logging
from typing import Optional

from livekit import rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    tts,
)
from livekit.plugins import elevenlabs


def _text_to_chunks(text: str) -> list[str]:
    """Split the text into chunks of 2, 3, and 4 words"""
    sizes = [2, 3, 4]
    chunks, i = [], 0

    for size in sizes:
        while i + size <= len(text):
            chunks.append(text[i : i + size])
            i += size

    chunks.append(text[i:])  # remaining
    return chunks


async def _playout_task(
    playout_q: asyncio.Queue, audio_source: rtc.AudioSource
) -> None:
    """Playout audio frames from the queue to the audio source"""
    while True:
        frame = await playout_q.get()
        if frame is None:
            break

        await audio_source.capture_frame(frame)


async def entrypoint(job: JobContext):
    # use another voice for this demo
    # you can get a list of the voices using 'await tts_11labs.list_voices()'
    voice = elevenlabs.Voice(
        id="ODq5zmih8GrVes37Dizd",
        name="Patrick",
        category="premade",
    )

    tts_11labs = elevenlabs.TTS(
        model_id="eleven_multilingual_v2",
        voice=voice,
    )

    source = rtc.AudioSource(tts_11labs.sample_rate, tts_11labs.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await job.room.local_participant.publish_track(track, options)

    await asyncio.sleep(1)
    logging.info('Saying "Bonjour, comment allez-vous?"')
    async for output in tts_11labs.synthesize("Bonjour, comment allez-vous?"):
        await source.capture_frame(output.data)

    await asyncio.sleep(1)
    logging.info('Saying "Au revoir."')
    async for output in tts_11labs.synthesize("Au revoir."):
        await source.capture_frame(output.data)

    await asyncio.sleep(1)
    streamed_text = (
        "Bonjour, ceci est un autre example avec la méthode utilisant un websocket."
    )
    logging.info('Streaming text "%s"', streamed_text)
    stream = tts_11labs.stream()
    for chunk in _text_to_chunks(
        streamed_text
    ):  # split into chunk just for the demonstration
        stream.push_text(chunk)

    stream.mark_segment_end()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async def _synth_task():
        async for ev in stream:
            if ev.type != tts.SynthesisEventType.AUDIO:
                continue
            assert ev.audio is not None

            playout_q.put_nowait(ev.audio.data)

        playout_q.put_nowait(None)

    synth_task = asyncio.create_task(_synth_task())
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    await stream.aclose(wait=True)
    await asyncio.gather(synth_task, playout_task)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
