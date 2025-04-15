import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.plugins import elevenlabs

logger = logging.getLogger("elevenlabs-tts-demo")
logger.setLevel(logging.INFO)

load_dotenv()


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


async def _playout_task(playout_q: asyncio.Queue, audio_source: rtc.AudioSource) -> None:
    """Playout audio frames from the queue to the audio source"""
    while True:
        frame = await playout_q.get()
        if frame is None:
            break

        await audio_source.capture_frame(frame)


async def entrypoint(job: JobContext):
    # use another voice for this demo
    # you can get a list of the voices using 'await tts_11labs.list_voices()'
    tts_11labs = elevenlabs.TTS(voice_id="ODq5zmih8GrVes37Dizd", model="eleven_multilingual_v2")

    source = rtc.AudioSource(tts_11labs.sample_rate, tts_11labs.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect()
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    logger.info('Saying "Bonjour, comment allez-vous?"')
    async for output in tts_11labs.synthesize("Bonjour, comment allez-vous?"):
        await source.capture_frame(output.frame)

    await asyncio.sleep(1)
    logger.info('Saying "Au revoir."')
    async for output in tts_11labs.synthesize("Au revoir."):
        await source.capture_frame(output.frame)

    await asyncio.sleep(1)
    streamed_text = "Bonjour, ceci est un autre example avec la méthode utilisant un websocket."
    logger.info('Streaming text "%s"', streamed_text)
    stream = tts_11labs.stream()
    for chunk in _text_to_chunks(streamed_text):  # split into chunk just for the demonstration
        stream.push_text(chunk)

    stream.flush()
    stream.end_input()

    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()

    async def _synth_task():
        async for ev in stream:
            playout_q.put_nowait(ev.frame)

        playout_q.put_nowait(None)

    synth_task = asyncio.create_task(_synth_task())
    playout_task = asyncio.create_task(_playout_task(playout_q, source))

    await asyncio.gather(synth_task, playout_task)
    await stream.aclose()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
