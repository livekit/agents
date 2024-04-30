import asyncio
import logging
import os

from livekit import rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.plugins.openai import TTS


async def entrypoint(job: JobContext):
    logging.info("starting tts example agent")
    # OpenAI TTS Outputs 24kHz mono audio
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await job.room.local_participant.publish_track(track, options)

    tts = TTS(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="tts-1",
        voice="nova",
    )
    await asyncio.sleep(1)
    logging.info("Speaking Hello!")
    async for output in tts.synthesize("Hello!"):
        await source.capture_frame(output.data)

    await asyncio.sleep(1)
    logging.info("Speaking Goodbye.")
    async for output in tts.synthesize("Goodbye."):
        await source.capture_frame(output.data)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
