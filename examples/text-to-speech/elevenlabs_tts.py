import asyncio
import logging

from livekit import rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.plugins import elevenlabs


async def entrypoint(job: JobContext):
    tts = elevenlabs.TTS(
        model_id="eleven_multilingual_v2",
    )

    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await job.room.local_participant.publish_track(track, options)

    await asyncio.sleep(1)
    logging.info('Saying "Bonjour, comment allez-vous?"')
    async for output in tts.synthesize("Bonjour, comment allez-vous?"):
        await source.capture_frame(output.data)

    await asyncio.sleep(1)
    logging.info('Saying "Au revoir."')
    async for output in tts.synthesize("Au revoir."):
        await source.capture_frame(output.data)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
