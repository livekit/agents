import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    ATTRIBUTE_AGENT_STATE,
    AgentState,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.vad import VADEventType
from livekit.plugins import silero

load_dotenv()
logger = logging.getLogger("echo-agent")


# An example agent that echos each utterance from the user back to them
# the example uses a queue to buffer incoming streams, and uses VAD to detect
# when the user is done speaking.
async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant: rtc.Participant = await ctx.wait_for_participant()
    stream = rtc.AudioStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
    )
    vad = silero.VAD.load(
        min_speech_duration=0.2,
        min_silence_duration=0.6,
    )
    vad_stream = vad.stream()

    source = rtc.AudioSource(sample_rate=48000, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("echo", source)
    await ctx.room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    # speech queue holds AudioFrames
    queue = asyncio.Queue(maxsize=1000)  # 10 seconds of audio (1000 frames * 10ms)
    is_speaking = False
    is_echoing = False

    async def _set_state(state: AgentState):
        await ctx.room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: state})

    await _set_state("listening")

    async def _process_input():
        async for audio_event in stream:
            if is_echoing:  # Skip processing while echoing
                continue
            vad_stream.push_frame(audio_event.frame)
            try:
                queue.put_nowait(audio_event.frame)
            except asyncio.QueueFull:
                # Remove oldest frame when queue is full
                queue.get_nowait()
                queue.put_nowait(audio_event.frame)

    async def _process_vad():
        nonlocal is_speaking, is_echoing
        async for vad_event in vad_stream:
            if is_echoing:  # Skip VAD processing while echoing
                continue
            if vad_event.type == VADEventType.START_OF_SPEECH:
                is_speaking = True
                frames_to_keep = 100
                frames = []
                while not queue.empty():
                    frames.append(queue.get_nowait())
                for frame in frames[-frames_to_keep:]:
                    queue.put_nowait(frame)
            elif vad_event.type == VADEventType.END_OF_SPEECH:
                is_speaking = False
                is_echoing = True
                logger.info("end of speech, playing back")
                await _set_state("speaking")
                try:
                    while not queue.empty():
                        frame = queue.get_nowait()
                        await source.capture_frame(frame)
                except asyncio.QueueEmpty:
                    pass
                finally:
                    is_echoing = False  # Reset echoing flag after playback
                    await _set_state("listening")

    await asyncio.gather(
        _process_input(),
        _process_vad(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
