import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli, io
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioOutput

sys.path.insert(0, str(Path(__file__).parent))
from avatar_runner import AudioWaveGenerator

logger = logging.getLogger("audio-echo-example")
logger.setLevel(logging.INFO)

load_dotenv()


class RoomAudioForwarder:
    def __init__(self, room: rtc.Room, *, sample_rate: int = 24000, num_channels: int = 1):
        self.room = room
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        # there can be multiple input streams for multiple participants
        self._input_stream: rtc.AudioStream | None = None
        self._participant_linked = asyncio.Future[str]()
        self._forward_atask: asyncio.Task | None = None

    def start(self, audio_output: io.AudioOutput):
        self._forward_atask = asyncio.create_task(self._forward_audio(audio_output))

        self.room.on("participant_connected", self._on_participant_connected)
        for participant in self.room.remote_participants.values():
            self._on_participant_connected(participant)

    async def _forward_audio(self, audio_output: io.AudioOutput):
        await self._participant_linked
        assert self._input_stream is not None
        async for ev in self._input_stream:
            await audio_output.capture_frame(ev.frame)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info("participant connected", extra={"participant": participant.identity})

        # link the first participant joined the room
        # in this example only one participant is linked at a time
        # it can be extended to support multiple participants by adding more input streams
        if not self._participant_linked.done():
            self._input_stream = rtc.AudioStream.from_participant(
                participant=participant,
                track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
            self._participant_linked.set_result(participant.identity)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    options = AvatarOptions(
        video_width=1280,
        video_height=720,
        video_fps=30,
        audio_sample_rate=24000,
        audio_channels=1,
    )
    video_gen = AudioWaveGenerator(options)

    # audio buffer between the input and the avatar runner
    # QueueAudioOutput is a subclass of both AudioOutput and AudioReceiver
    audio_queue = QueueAudioOutput(sample_rate=options.audio_sample_rate)

    # forward audio from the first linked participant to the buffer
    audio_forwarder = RoomAudioForwarder(ctx.room)
    audio_forwarder.start(audio_queue)

    # avatar runner
    # path: participant audio -> audio buffer -> avatar runner -> video generator -> room
    avatar_runner = AvatarRunner(
        ctx.room, audio_recv=audio_queue, video_gen=video_gen, options=options
    )
    await avatar_runner.start()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM,
        )
    )
