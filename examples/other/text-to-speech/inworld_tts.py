import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

# For local development, use direct import from the plugin source:
import sys
sys.path.insert(0, "livekit-plugins/livekit-plugins-inworld")
from livekit.plugins import inworld
# from livekit.plugins import inworld

load_dotenv()

logger = logging.getLogger("inworld-tts-demo")
logger.setLevel(logging.INFO)

server = AgentServer()


@server.rtc_session()
async def entrypoint(job: JobContext):
    logger.info("starting tts example agent")

    tts = inworld.TTS(
        voice="Alex", # Voice ID (or custom cloned voice ID)
        timestamp_type="TIMESTAMP_TYPE_UNSPECIFIED", # CHARACTER or WORD
        text_normalization="ON", # ON or OFF
    )

    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    text = "Hello from Inworld. I hope you are having a spectacular day."

    logger.info(f'synthesizing: "{text}"')
    async for audio in tts.synthesize(text):
        # Print timestamp information if available
        timed_strings = audio.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
        for ts in timed_strings:
            start = f"{ts.start_time:.3f}s" if hasattr(ts, "start_time") and ts.start_time else "N/A"
            end = f"{ts.end_time:.3f}s" if hasattr(ts, "end_time") and ts.end_time else "N/A"
            logger.info(f"  [{start} - {end}] {ts}")

        await source.capture_frame(audio.frame)

    logger.info("synthesis complete")


if __name__ == "__main__":
    cli.run_app(server)
