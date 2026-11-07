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
    STTContextOptions,
    TurnHandlingOptions,
    UserStateChangedEvent,
    cli,
    inference,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.utils.participant import wait_for_participant_attribute
from livekit.plugins import ai_coustics

from bank import BANK_NAME, Bank
from card_services import (
    GREETING,
    CardServices,
    FraudAlert,
    caller_script,
    outbound_caller_script,
    outbound_greeting,
)
from phone import dial_phone, outbound_line
from simulation import on_simulation_end
from suggested_replies import SUGGESTIONS_MODEL, SuggestedReplies

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = "card-verification"
STT_MODEL = "assemblyai/universal-3-6-pro"
LLM_MODEL = "google/gemma-4-31b-it"
TTS_MODEL = "rime/coda"
VOICE = "lintel"
KEYTERMS = [BANK_NAME, "Avery"]

STILL_THERE = "Are you still there?"
SILENT_GOODBYE = "I'll let you go. Give us a call back anytime."

server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME, on_simulation_end=on_simulation_end)
async def card_verification(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    if sim := ctx.simulation_context():
        meta |= sim.userdata()
    outbound = meta.get("direction") == "outbound"
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "direction": "outbound" if outbound else "inbound",
    }
    await ctx.connect()
    if dial := meta.get("dial"):
        caller = await dial_phone(ctx, dial, *outbound_line(meta))
        if caller is None:
            ctx.delete_room()
            return
    else:
        caller = await ctx.wait_for_participant()
    on_phone = caller.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP

    async def publish(key: str, value: Any) -> None:
        await ctx.room.local_participant.set_attributes(
            {f"auth.{key}": json.dumps(value)}
        )

    bank = Bank(publish=publish)
    await publish("status", "unauthenticated")
    await publish(
        "call",
        {
            "channel": "phone" if on_phone else "browser",
            "direction": "outbound" if outbound else "inbound",
            "models": {"stt": STT_MODEL, "llm": LLM_MODEL, "tts": TTS_MODEL},
        },
    )

    session = AgentSession[Bank](
        userdata=bank,
        stt=inference.STT(model=STT_MODEL, language="en"),
        stt_context_options=STTContextOptions(
            keyterms=KEYTERMS, keyterm_detection={"enabled": True}
        ),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model=TTS_MODEL, voice=VOICE, language="en"),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
        tools=[
            EndCallTool(
                end_instructions="Say what any tool called with this one returned, then a "
                "short goodbye if you haven't said one."
            )
        ],
    )
    if outbound:
        await bank.open_alert()
    session.on("close", lambda _: ctx.delete_room())
    if not ctx.is_fake_job():
        # a live line: the console's text turns have no silence to check on
        check_in_when_user_away(session)
    SuggestedReplies(
        session,
        ctx.room,
        llm=inference.LLM(SUGGESTIONS_MODEL),
        persona="Avery",
        script=lambda: (
            outbound_caller_script(bank) if outbound else caller_script(bank)
        ),
    ).attach()

    await session.start(
        agent=FraudAlert(bank) if outbound else CardServices(bank),
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

    if on_phone:
        try:
            await wait_for_participant_attribute(
                ctx.room,
                identity=caller.identity,
                attribute="sip.callStatus",
                value="active",
            )
        except RuntimeError:
            ctx.shutdown(reason="the call was not answered")
            return
    session.say(outbound_greeting() if outbound else GREETING)


def check_in_when_user_away(session: AgentSession) -> None:
    """Asks once whether the caller's still there when the line goes quiet, and says goodbye
    if it stays quiet. Fixed lines, so silence never agrees to anything."""
    check_in: asyncio.Task[None] | None = None

    async def ask_then_hang_up() -> None:
        await session.say(STILL_THERE)
        await asyncio.sleep(session.options.user_away_timeout)
        await session.say(SILENT_GOODBYE, allow_interruptions=False)
        session.shutdown()

    @session.on("user_state_changed")
    def on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal check_in
        if check_in:
            check_in.cancel()
        if ev.new_state == "away":
            check_in = asyncio.create_task(ask_then_hang_up())


if __name__ == "__main__":
    cli.run_app(server)
