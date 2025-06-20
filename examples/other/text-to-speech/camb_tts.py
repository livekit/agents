import logging

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import camb

logger = logging.getLogger("camb-tts-demo")
logger.setLevel(logging.INFO)


async def entrypoint(job: JobContext):
    await job.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create TTS with CAMB
    tts = camb.TTS(
        voice_id=20298,  # Daniel voice
        language=1,
        gender=camb.Gender.MALE,
        age=30,
    )

    # Create audio source and track
    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)

    # Publish the track
    await job.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    # Synthesize and play
    async for audio in tts.synthesize("Hello! I'm using CAMB AI for text-to-speech."):
        await source.capture_frame(audio.frame)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
