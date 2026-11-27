import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    APIError,
    ConversationItemAddedEvent,
    JobContext,
    UserStateChangedEvent,
    cli,
    inference,
    llm,
    room_io,
)
from livekit.agents.utils.participant import wait_for_participant_attribute
from livekit.plugins import ai_coustics

from call_summary import SUMMARY_MODEL, CallSummary
from interviewer import BACKEND_INSTRUCTIONS, MODEL, VOICE, Interviewer, docs_tools
from playout_hold import hold_playout

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = "homepage-feedback-interview"
DECISION_MODEL = "openai/gpt-6-luna-decisions"

LAST_MESSAGE = llm.Choice(
    "What is the assistant's last message doing?",
    {
        "closing": "It is the final farewell that ends the call, with no question for the user.",
        "continuing": "It asks the user something or otherwise keeps the conversation going.",
    },
)

STILL_THERE = (
    "The line has gone quiet. Ask, in a few words, whether they're still there."
)
SILENT_GOODBYE = (
    "The line is still quiet. Say you'll let them go, thank them, and say goodbye."
)

server = AgentServer()
hold_playout()


@server.rtc_session(agent_name=AGENT_NAME)
async def feedback_interview(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    await ctx.connect()
    caller = await ctx.wait_for_participant()
    on_phone = caller.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP

    session = AgentSession(
        llm=inference.DuplexModel(
            model=MODEL,
            voice=VOICE,
            extra_kwargs={"responses_options": {"instructions": BACKEND_INSTRUCTIONS}},
        ),
        vad=inference.VAD(),
        tools=[docs_tools()],
    )
    session.on("close", lambda _: ctx.delete_room())
    summary = CallSummary(session, ctx.room, llm=inference.LLM(SUMMARY_MODEL)).attach()
    hang_up_after_farewell(session, summary)
    if not ctx.is_fake_job():
        # a live line: the console's text turns have no silence to check on
        check_in_when_user_away(session, summary)

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

    await session.start(
        agent=Interviewer(),
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


def hang_up_after_farewell(session: AgentSession, summary: CallSummary) -> None:
    """Ends the call once Morgan has said its farewell. GPT-Live rarely delegates a hang-up,
    so a decision model reads each of Morgan's turns instead."""
    judge = inference.DecisionModel(DECISION_MODEL)
    checks: set[asyncio.Task[None]] = set()

    async def check() -> None:
        try:
            response = await judge.evaluate(
                input=session.history, decisions={"last_message": LAST_MESSAGE}
            )
        except APIError as e:
            logger.warning("couldn't judge whether the call is over: %s", e)
            return
        verdict = response.results.get("last_message")
        if verdict and verdict.value == "closing":
            await summary.finish()
            session.shutdown()

    @session.on("conversation_item_added")
    def on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            task = asyncio.create_task(check())
            checks.add(task)
            task.add_done_callback(checks.discard)


def check_in_when_user_away(session: AgentSession, summary: CallSummary) -> None:
    """Asks once whether the caller's still there when the line goes quiet, and says goodbye
    if it stays quiet."""
    check_in: asyncio.Task[None] | None = None

    async def ask_then_hang_up() -> None:
        session.generate_reply(instructions=STILL_THERE)
        await asyncio.sleep(session.options.user_away_timeout)
        await session.generate_reply(instructions=SILENT_GOODBYE)
        await summary.finish()
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
