"""Experimental client-side, multi-turn AMD. See amd.md for the design and API.

Run this agent with console for local audio, or dev for a room/SIP call.
Set SIP_PHONE_NUMBER, SIP_PARTICIPANT_IDENTITY, and SIP_OUTBOUND_TRUNK_ID
only when you want the dev worker to place an outbound call.

AMD reuses this Agent's Gemma model and Ink 2 transcript. Pass an explicit
llm or stt to AMD to use other models. Realtime models are not supported yet.
"""

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
from livekit.plugins import silero

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
    session = AgentSession(
        stt=inference.STT("cartesia/ink-2", language="en"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=silero.VAD.load(),
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
            "preemptive_generation": {"enabled": True},
        },
    )
    await session.start(agent=MyAgent(), room=ctx.room)

    phone_number = os.getenv("SIP_PHONE_NUMBER")
    participant_identity = os.getenv("SIP_PARTICIPANT_IDENTITY")
    outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

    detector = AMD(
        session,
        participant_identity=participant_identity or NOT_GIVEN,
    )

    @detector.on("amd_prediction")
    def on_prediction(event: AMDPredictionEvent) -> None:
        logger.info("AMD prediction: %s", event.model_dump_json())

    @detector.on("amd_menu_observed")
    def on_menu(event: AMDMenuObservedEvent) -> None:
        logger.info("AMD menu (informational): %s", event.model_dump_json())

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
                logger.info("SIP call was not answered")
                ctx.shutdown("call not answered")
                return

        result = await detector.execute()
        logger.info("AMD completed: %s", result.model_dump_json())
        # The application decides whether to continue or end the call.


if __name__ == "__main__":
    cli.run_app(server)
