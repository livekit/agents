import logging
import os
import pathlib

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.utils.audio import audio_frames_from_file
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import CloseEvent, ErrorEvent
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.rtc import ParticipantKind

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

os.environ["OPENAI_API_KEY"] = "BAD_KEY"
load_dotenv()


class MyTask(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant.",
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Create session
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    custom_error_audio = os.path.join(pathlib.Path(__file__).parent.absolute(), "error_message.ogg")

    @session.on("error")
    def on_error(ev: ErrorEvent):
        if ev.error.recoverable:
            return
        logger.info(f"Session is closing due to error in {ev.source.__class__.__name__}")
        logger.info(f"Playing error audio file from: {custom_error_audio}")
        session.say(
            "I'm having trouble connecting right now. Let me transfer your call.",
            # If you define a custom audio file, it will play out even if the TTS provider is down.
            audio=audio_frames_from_file(custom_error_audio),
            allow_interruptions=False,
        )

    @session.on("close")
    def on_close(_: CloseEvent):
        logger.info("Session is closing")
        participant = list(ctx.room.remote_participants.values())[0]
        if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            ctx.transfer_sip_participant(participant, "tel:+18003310500")
        ctx.delete_room()

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(agent=MyTask(), room=ctx.room)


if __name__ == "__main__":
    # Set agent_name to enable explicit dispatch
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="inbound-agent"))
