import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli, io, utils
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioOutput
from livekit.agents.voice.room_io._input import _ParticipantAudioInputStream

sys.path.insert(0, str(Path(__file__).parent))
from avatar_runner import AudioWaveGenerator

logger = logging.getLogger("audio-echo-example")
logger.setLevel(logging.INFO)

load_dotenv()


class RoomAudioForwarder:
    def __init__(self, room: rtc.Room, *, sample_rate: int = 24000, num_channels: int = 1):
        self.room = room
        # input stream can be reused by linking to different participants
        # or can have multiple input streams for multiple participants
        self._input_stream = _ParticipantAudioInputStream(
            self.room,
            sample_rate=sample_rate,
            num_channels=num_channels,
            noise_cancellation=None,
        )
        self._participant_linked = asyncio.Future[str]()
        self._forward_atask: asyncio.Task | None = None

    def start(self, audio_output: io.AudioOutput):
        self._forward_atask = asyncio.create_task(self._forward_audio(audio_output))

        # listen for participant connections and disconnections
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        for participant in self.room.remote_participants.values():
            self._on_participant_connected(participant)

    async def _forward_audio(self, audio_output: io.AudioOutput):
        async for frame in self._input_stream:
            await audio_output.capture_frame(frame)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info("participant connected", extra={"participant": participant.identity})

        # link the first participant joined the room
        # in this example only one participant is linked at a time
        # it can be extended to support multiple participants
        # by adding more input streams and setting them when a new participant connects
        if not self._participant_linked.done():
            self._input_stream.set_participant(participant)
            self._participant_linked.set_result(participant.identity)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info("participant disconnected", extra={"participant": participant.identity})
        if (
            not self._participant_linked.done()
            or self._participant_linked.result() != participant.identity
        ):
            return

        self._input_stream.set_participant(None)
        self._participant_linked = asyncio.Future[str]()
        for participant in self.room.remote_participants.values():
            # link the next participant
            self._on_participant_connected(participant)
            break

    async def aclose(self):
        await self._input_stream.aclose()
        if self._forward_atask:
            await utils.aio.cancel_and_wait(self._forward_atask)


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
