import asyncio
import logging
import pathlib

from dotenv import load_dotenv

from livekit.agents import AgentServer, JobContext, cli, inference
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import CloseEvent, ErrorEvent
from livekit.rtc import ParticipantKind

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example demonstrates how to handle errors from STT, TTS, and LLM
# and how to continue the conversation after an error if the error is recoverable

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        # play a pre-recorded file (or any AudioSource) before the session closes on an
        # unrecoverable error; bypasses TTS, so it's still heard when TTS is the failed
        # resource. A non-file str is synthesized through TTS instead.
        unrecoverable_error_message=str(pathlib.Path(__file__).parent / "error_message.ogg"),
    )

    # Advanced path: for full control (e.g. continuing the conversation on recoverable
    # errors, or triggering a SIP transfer) handle the "error" event yourself. This runs
    # in addition to the built-in unrecoverable_error_message above.
    @session.on("error")
    def on_error(ev: ErrorEvent):
        if ev.error.recoverable:
            return

        logger.info(f"session is closing due to unrecoverable error {ev.error}")

        # If want to continue the conversation, we can set the recoverable to True

        # TTS and LLM errors can be marked as recoverable
        # since these components are recreated for each response

        # if isinstance(ev.source, (tts.TTS, llm.LLM)):
        #     ev.error.recoverable = True
        #     return

        # STT stream persists for the entire agent lifetime
        # we can reset the agent if we want to continue the conversation
        # if isinstance(ev.source, stt.STT):
        #     session.update_agent(session.current_agent)
        #     ev.error.recoverable = True
        #     return

    @session.on("close")
    def on_close(_: CloseEvent):
        logger.info("Session is closing")

        # Assume there is only one caller in the room
        participant = [
            p
            for p in ctx.room.remote_participants.values()
            if p.kind == ParticipantKind.PARTICIPANT_KIND_SIP
        ][0]

        def on_sip_transfer_done(f: asyncio.Future):
            if f.exception():
                logger.error(f"Error transferring SIP participant: {f.exception()}")
            else:
                logger.info("SIP participant transferred")
            ctx.delete_room()

        # See https://docs.livekit.io/sip/ on how to set up SIP participants
        if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            ctx.transfer_sip_participant(participant, "tel:+18003310500").add_done_callback(
                on_sip_transfer_done
            )

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
