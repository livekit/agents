import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from livekit import api
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference

logger = logging.getLogger("vobiz-agent")

load_dotenv()

AGENT_NAME = "vobiz-agent"
PARTICIPANT_IDENTITY = "phone-user"


class PhoneAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant on a phone call. Keep your replies short and "
                "conversational, and do not use lists, markdown or emojis."
            ),
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # the dispatch metadata carries a phone number only for outbound calls
    metadata = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    phone_number = metadata.get("phone_number")

    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    async def hangup() -> None:
        # deleting the room ends the phone call for the SIP participant
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))

    ctx.add_shutdown_callback(hangup)

    # start the session first so that none of the callee's audio is lost after they answer
    await session.start(agent=PhoneAgent(), room=ctx.room)

    if phone_number:
        if not session.room_io:
            raise RuntimeError("room_io is unavailable, use the dev or start command")
        session.room_io.set_participant(PARTICIPANT_IDENTITY)

        try:
            # the API timeout must outlast the ring window
            await asyncio.wait_for(
                ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=ctx.room.name,
                        sip_trunk_id=os.environ["SIP_OUTBOUND_TRUNK_ID"],
                        sip_call_to=phone_number,
                        participant_identity=PARTICIPANT_IDENTITY,
                        wait_until_answered=True,
                    )
                ),
                timeout=45,
            )
        except (api.SipCallError, asyncio.TimeoutError) as e:
            logger.info(f"call was not answered: {e}")
            ctx.shutdown("call not answered")
            return

    await session.generate_reply(instructions="Greet the person and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
