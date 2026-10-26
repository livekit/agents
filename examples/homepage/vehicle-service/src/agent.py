import json
import logging
import os
from datetime import date
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
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
from livekit.plugins import ai_coustics

from advisor import RepairCallback, ServiceAdvisor, caller_script
from phone import PHONE, outbound_line, ring
from service_desk import ServiceDesk, today_in
from simulation import on_simulation_end
from suggested_replies import SUGGESTIONS_MODEL, SuggestedReplies

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = "vehicle-service"
STT_MODEL = "assemblyai/universal-3-6-pro"
LLM_MODEL = "google/gemma-4-31b-it"
TTS_MODEL = "rime/coda"
VOICE = "thread"
KEYTERMS = ["Bayside", "BMW", "X3", "330i", "xDrive", "recall", "Nick"]

STILL_THERE = "Are you still there?"
SILENT_GOODBYE = "I'll let you go. Give us a call back anytime."

server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME, on_simulation_end=on_simulation_end)
async def vehicle_service(ctx: JobContext) -> None:
    # which call it is, and optionally the day pinned: from the dispatch, from a scenario, or,
    # in the console, from CALL_DIRECTION and SERVICE_DESK_TODAY
    meta = {
        "direction": os.getenv("CALL_DIRECTION"),
        "today": os.getenv("SERVICE_DESK_TODAY"),
        **json.loads(ctx.job.metadata or "{}"),
    }
    if sim := ctx.simulation_context():
        meta |= sim.userdata()
    today = (
        date.fromisoformat(meta["today"])
        if meta.get("today")
        else today_in(meta.get("timezone"))
    )
    callback = meta.get("direction") == "outbound"
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "direction": "outbound" if callback else "inbound",
    }
    await ctx.connect()

    dial = meta.get("dial")
    if dial and await ring(ctx, dial, *outbound_line(meta)) is None:
        ctx.delete_room()
        return
    on_phone = any(
        p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        for p in ctx.room.remote_participants.values()
    )

    async def publish(key: str, value: Any) -> None:
        await ctx.room.local_participant.set_attributes(
            {f"service.{key}": json.dumps(value)}
        )

    desk = (
        ServiceDesk.for_callback(today, publish)
        if callback
        else ServiceDesk(today, publish)
    )
    await publish(
        "call",
        {
            "channel": "phone" if on_phone else "browser",
            "direction": "outbound" if callback else "inbound",
            "models": {"stt": STT_MODEL, "llm": LLM_MODEL, "tts": TTS_MODEL},
        },
    )
    await desk.publish_all()

    session = AgentSession[ServiceDesk](
        userdata=desk,
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
    )
    session.on("close", lambda _: ctx.delete_room())
    if not ctx.is_fake_job():
        # a live line: the console's text turns have no silence to check on
        check_in_when_user_away(session)
    SuggestedReplies(
        session,
        ctx.room,
        llm=inference.LLM(SUGGESTIONS_MODEL),
        persona="Nick",
        script=lambda: caller_script(desk),
    ).attach()

    await session.start(
        agent=RepairCallback(desk) if callback else ServiceAdvisor(desk),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=PHONE if dial else NOT_GIVEN,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )


def check_in_when_user_away(session: AgentSession) -> None:
    """Asks once whether the caller's still there when the line goes quiet, and says goodbye
    if it stays quiet. Fixed lines, so silence never agrees to anything."""
    quiet_spells = 0

    def on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal quiet_spells
        if ev.new_state == "speaking":
            quiet_spells = 0
        elif ev.new_state == "away":
            quiet_spells += 1
            if quiet_spells == 1:
                session.say(STILL_THERE)
                session.reset_away_timer()
            elif quiet_spells == 2:
                session.say(
                    SILENT_GOODBYE, allow_interruptions=False
                ).add_done_callback(lambda _: session.shutdown())

    session.on("user_state_changed", on_user_state)


if __name__ == "__main__":
    cli.run_app(server)
