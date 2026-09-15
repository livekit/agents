import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    AMD,
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    AMDMenuObservedEvent,
    AMDPredictionEvent,
    JobContext,
    cli,
    inference,
)

logger = logging.getLogger("amd-example")
load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Alex from Acme Dental. You are calling Sam to confirm "
                "a dental appointment tomorrow at 10 AM. Keep replies brief. "
                "If asked to leave a message, give the appointment details and "
                "ask Sam to call the office to confirm. Do not invent a phone number."
            ),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
    )

    async def hangup():
        await ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=ctx.room.name,
            )
        )

    ctx.add_shutdown_callback(hangup)

    phone_number = os.getenv("SIP_PHONE_NUMBER")
    participant_identity = os.getenv("SIP_PARTICIPANT_IDENTITY") or NOT_GIVEN
    outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

    if not session.room_io:
        raise RuntimeError(
            "session room_io is unavailable. Make sure you use dev or start commands"
        )

    detector = AMD(
        session,
        participant_identity=participant_identity,
    )

    @detector.on("amd_prediction")
    def on_prediction(event: AMDPredictionEvent) -> None:
        logger.info("amd prediction", extra={"lk.pii.amd.prediction": event.model_dump_json()})

    @detector.on("amd_menu_observed")
    def on_menu(event: AMDMenuObservedEvent) -> None:
        logger.info("amd menu (informational)", extra={"lk.pii.amd.menu": event.model_dump_json()})

    async with detector:
        # Start AMD before creating the SIP participant.
        if ctx.room.isconnected() and phone_number and outbound_trunk_id and participant_identity:
            try:
                await ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=ctx.room.name,
                        sip_trunk_id=outbound_trunk_id,
                        sip_call_to=phone_number,
                        participant_identity=participant_identity,
                        wait_until_answered=True,
                    ),
                    timeout=45,
                )
            except (api.SipCallError, asyncio.TimeoutError):
                logger.info("sip call was not answered")
                ctx.shutdown("call not answered")
                return
            # The call may end just before wait_until_answered returns.
            if participant_identity not in ctx.room.remote_participants:
                logger.info("sip participant missing, ending")
                ctx.shutdown("participant missing")
                return

        result = await detector.execute()
        logger.info("amd completed", extra={"lk.pii.amd.result": result.model_dump_json()})
        # The application decides whether to continue or end the call.


if __name__ == "__main__":
    cli.run_app(server)
