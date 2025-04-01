import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import neuphonic

load_dotenv()

logger = logging.getLogger("neuphonic-tts-demo")
logger.setLevel(logging.INFO)


async def entrypoint(job: JobContext):
    logger.info("starting tts example agent")

    SAMPLE_RATE = 22050
    NUM_CHANNELS = 1

    tts = neuphonic.TTS(
        # voice_id=<uuid>,
        sample_rate=SAMPLE_RATE  # defaults to 22050
    )

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    stream = tts.stream()

    async def _playback_task():
        async for audio in stream:
            await source.capture_frame(audio.frame)

    task = asyncio.create_task(_playback_task())

    text = "Hello from Neuphonic. You have just successfully run the example!"

    # split into two word chunks to simulate LLM streaming
    words = text.split()
    for i in range(0, len(words), 2):
        chunk = " ".join(words[i : i + 2])
        if chunk:
            logger.info(f'pushing chunk: "{chunk} "')
            stream.push_text(chunk + " ")

    # Mark end of input segment
    stream.flush()
    stream.end_input()
    await asyncio.gather(task)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
