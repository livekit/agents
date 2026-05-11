import logging
import os

from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import (
    AMD,
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are reaching out to a customer with a phone call. "
                "You are calling to see if they are home. "
                "You might encounter an answering machine with a DTMF menu or IVR system. "
                "If you do, you will try to leave a message to ask them to call back."
            ),
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
    )

    phone_number = os.getenv("SIP_PHONE_NUMBER")
    participant_identity = os.getenv("SIP_PARTICIPANT_IDENTITY")
    outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

    # focus the session on the callee before AMD starts so audio recognition
    # doesn't push frames from any pre-existing participant into AMD's pipeline
    if not session.room_io:
        raise RuntimeError(
            "session room_io is unavailable. Make sure you use dev or start commands"
        )
    if participant_identity:
        session.room_io.set_participant(participant_identity)

    async with AMD(
        session,
        participant_identity=participant_identity or NOT_GIVEN,
    ) as detector:
        # start running amd before the SIP participant joins to avoid audio loss
        if phone_number and outbound_trunk_id and participant_identity:
            logger.info(f"creating SIP participant for {participant_identity}")
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=outbound_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=participant_identity,
                    wait_until_answered=True,
                )
            )
            participant = await ctx.wait_for_participant(identity=participant_identity)
            logger.info(
                "participant joined",
                extra={
                    "actual_identity": participant.identity,
                    "expected_identity": participant_identity,
                    "kind": participant.kind,
                    "audio_tracks_subscribed": [
                        pub.sid
                        for pub in participant.track_publications.values()
                        if pub.subscribed and pub.kind == rtc.TrackKind.KIND_AUDIO
                    ],
                },
            )

        result = await detector.execute()
        logger.info(f"AMD result: {result}")
        if result.category == "human" or result.category == "uncertain":
            logger.info(
                "human answered the call or amd is uncertain, proceeding with normal conversation",
                extra={"transcript": result.transcript},
            )

        elif result.category == "machine-ivr":
            logger.info(
                "ivr menu detected, starting navigation",
                extra={"transcript": result.transcript},
            )

        elif result.category == "machine-vm":
            logger.info(
                "voicemail detected, leaving a message",
                extra={"transcript": result.transcript},
            )

            speech_handle = session.generate_reply(
                instructions=(
                    "You've reached voicemail. Leave a brief message asking "
                    "the customer to call back."
                ),
            )
            await speech_handle.wait_for_playout()
            ctx.shutdown("voicemail detected")

        elif result.category == "machine-unavailable":
            logger.info("mailbox unavailable, ending call", extra={"transcript": result.transcript})

            ctx.shutdown("mailbox unavailable")

    async def hangup():
        await ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=ctx.room.name,
            )
        )

    ctx.add_shutdown_callback(hangup)


if __name__ == "__main__":
    cli.run_app(server)
