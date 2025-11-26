import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.plugins import inworld

load_dotenv()

logger = logging.getLogger("inworld-tts-demo")
logger.setLevel(logging.INFO)

server = AgentServer()


@server.rtc_session()
async def entrypoint(job: JobContext):
    logger.info("starting tts example agent")

    tts = inworld.TTS(
        # voice="Ashley",  # default voice
        # speaking_rate=1.0,
        # pitch=0.0,
    )

    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    text = "Hello from Inworld. I hope you are having a great day."

    # Inworld TTS does not support streaming, so we use synthesize()
    logger.info(f'synthesizing: "{text}"')
    async for audio in tts.synthesize(text):
        await source.capture_frame(audio.frame)

    logger.info("synthesis complete")


if __name__ == "__main__":
    cli.run_app(server)
