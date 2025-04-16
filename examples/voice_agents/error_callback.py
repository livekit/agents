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

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

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

        logger.info(f"session is closing due to unrecoverable error {ev.error}")

        # To bypass the TTS service in case it's unavailable, we use a custom audio file instead
        session.say(
            "I'm having trouble connecting right now. Let me transfer your call.",
            audio=audio_frames_from_file(custom_error_audio),
            allow_interruptions=False,
        )

    @session.on("close")
    def on_close(_: CloseEvent):
        logger.info("Session is closing")

        # Assume there is only one caller in the room
        participant = ctx.get_sip_participants()[0]

        # See https://docs.livekit.io/sip/ on how to set up SIP participants
        if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            ctx.transfer_sip_participant(participant, "tel:+18003310500")
        ctx.delete_room()

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    # Set agent_name to enable explicit dispatch
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="inbound-agent"))
