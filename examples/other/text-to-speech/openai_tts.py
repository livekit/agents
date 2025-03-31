import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import openai

load_dotenv()

logger = logging.getLogger("openai-tts-demo")
logger.setLevel(logging.INFO)


async def entrypoint(job: JobContext):
    logger.info("starting tts example agent")

    tts = openai.TTS(model="tts-1", voice="nova")

    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    logger.info('Saying "Hello!"')
    async for output in tts.synthesize("Hello!"):
        await source.capture_frame(output.frame)

    await asyncio.sleep(1)
    logger.info('Saying "Goodbye."')
    async for output in tts.synthesize("Goodbye."):
        await source.capture_frame(output.frame)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
