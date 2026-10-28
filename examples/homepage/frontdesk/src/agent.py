import asyncio
import json
import logging
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import ai_coustics

from clinic import Clinic
from phone import outbound_line, ring
from receptionist import (
    BACKEND_MODEL,
    GREETING,
    VOICE_MODEL,
    CallState,
    ConfirmationCall,
    Receptionist,
    caller_script,
    outbound_caller_script,
    outbound_greeting,
)
from simulation import on_simulation_end
from suggested_replies import SUGGESTIONS_MODEL, SuggestedReplies

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = "homepage-front-desk"
MODELS = {"voice": f"openai/{VOICE_MODEL}", "backend": f"openai/{BACKEND_MODEL}"}

server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME, on_simulation_end=on_simulation_end)
async def front_desk(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    outbound = meta.get("direction") == "outbound"
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "direction": "outbound" if outbound else "inbound",
    }
    await ctx.connect()
    if dial := meta.get("dial"):
        caller = await ring(ctx, dial, *outbound_line(meta))
        if caller is None:
            ctx.delete_room()
            return
    else:
        caller = await ctx.wait_for_participant()
    on_phone = caller.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP

    async def publish(key: str, value: Any) -> None:
        await ctx.room.local_participant.set_attributes(
            {f"clinic.{key}": json.dumps(value)}
        )

    clinic = Clinic.for_now(publish=publish)
    await publish(
        "call",
        {
            "channel": "phone" if on_phone else "browser",
            "direction": "outbound" if outbound else "inbound",
            "caller": _masked(caller.attributes.get("sip.phoneNumber", "")),
            "models": MODELS,
        },
    )

    session = AgentSession[CallState](
        vad=inference.VAD(),
        userdata=CallState(clinic=clinic),
        tools=[EndCallTool(end_instructions=None)],
    )
    confirming = clinic.appointments[0]
    SuggestedReplies(
        session,
        ctx.room,
        llm=inference.LLM(SUGGESTIONS_MODEL),
        persona="Riley",
        script=lambda: (
            outbound_caller_script(clinic, session.userdata, confirming)
            if outbound
            else caller_script(clinic, session.userdata)
        ),
    ).attach()

    await session.start(
        agent=ConfirmationCall(clinic, confirming)
        if outbound
        else Receptionist(clinic),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=caller.identity,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    if on_phone and not await _answered(ctx.room, caller):
        ctx.shutdown(reason="the call was not answered")
        return
    greeting = outbound_greeting(confirming) if outbound else GREETING
    session.generate_reply(
        instructions=f'Greet the caller with exactly these words: "{greeting}"'
    )


async def _answered(room: rtc.Room, caller: rtc.RemoteParticipant) -> bool:
    answered: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

    def check(*_: object) -> None:
        if not answered.done() and caller.attributes.get("sip.callStatus") == "active":
            answered.set_result(True)

    def left(participant: rtc.RemoteParticipant, *_: object) -> None:
        if not answered.done() and participant.identity == caller.identity:
            answered.set_result(False)

    room.on("participant_attributes_changed", check)
    room.on("participant_disconnected", left)
    check()
    try:
        return await answered
    finally:
        room.off("participant_attributes_changed", check)
        room.off("participant_disconnected", left)


def _masked(number: str) -> str | None:
    digits = "".join(c for c in number if c.isdigit())
    return f"•••-{digits[-4:]}" if len(digits) >= 4 else None


if __name__ == "__main__":
    cli.run_app(server)
