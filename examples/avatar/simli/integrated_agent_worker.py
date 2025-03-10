import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask, CallContext, VoiceAgent
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioSink
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.plugins import cartesia, deepgram, openai

sys.path.insert(0, str(Path(__file__).parent))
from simli_avatar_runner import SimliVideoGenerator

load_dotenv()

logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.INFO)
logging.getLogger("aioice.ice").setLevel(logging.INFO)

logger = logging.getLogger("avatar-example")


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo.",
            # llm=openai.realtime.RealtimeModel(voice="echo"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
        )

    @ai_function
    async def talk_to_alloy(self, context: CallContext):
        return AlloyTask(), "Transferring you to Alloy."


class AlloyTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    @ai_function
    async def talk_to_echo(self, context: CallContext):
        return EchoTask(), "Transferring you to Echo."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # create agent
    agent = VoiceAgent(
        task=AlloyTask(),
    )
    agent.output.audio = QueueAudioSink()

    # create video generator
    avatar_options = AvatarOptions(
        video_width=512,
        video_height=512,
        video_fps=30,
        audio_sample_rate=48000,
        audio_channels=2,
    )
    video_gen = SimliVideoGenerator(avatar_options)
    await video_gen.start(os.getenv("SIMLI_API_KEY"), os.getenv("SIMLI_FACE_ID"))
    await video_gen.wait_for_first_frame()

    # create avatar runner
    avatar_runner = AvatarRunner(
        room=ctx.room,
        video_gen=video_gen,
        audio_recv=agent.output.audio,
        options=avatar_options,
    )
    await avatar_runner.start()

    # start agent
    await agent.start(
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            audio_enabled=False, audio_sample_rate=avatar_options.audio_sample_rate
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, job_memory_warn_mb=1500))
