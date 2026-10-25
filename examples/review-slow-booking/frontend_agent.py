import logging
import os
import time
from typing import Literal, cast

from review_log import describe, rows

from livekit.agents import (
    Agent,
    AgentHandler,
    ConversationItemAddedEvent,
    InputRequired,
    JobContext,
    LiveSession,
    RunContext,
    ToolExecutionUpdatedEvent,
    ToolResult,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.plugins import openai

log = logging.getLogger("review.frontend")

# the travel desk's text door inbox; a run sets another to try the call taking the lane per turn
FRONT_INBOX = cast(
    Literal["live", "batch", "interrupt", "queue"], os.environ.get("REVIEW_FRONT_INBOX", "live")
)

# the back office on the project's own host, which the worker connects to
BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/booking/default/backend"

PERSONA = (
    "You are the travel desk of Skyline Travel. Warm and brief. Booking a flight or changing "
    "its day goes to the booking office, stated in full with what the traveller said; tell the "
    "traveller what it answered. A correction such as another day goes to it too."
)


class LoggingDelegate(A2ADelegate):
    """The booking office as the travel desk's tool, with each request and answer logged."""

    async def _ask(self, ctx: RunContext, request: str) -> str | InputRequired | ToolResult:
        """The tool: the request to the expert with what the run knows, the answer back.

        Args:
            request: the request, in full
        """
        call_id = ctx.function_call.call_id
        log.info(f"delegate ask {call_id}: request={request!r}")
        try:
            result = await super()._ask(ctx, request)
        except Exception as e:
            log.info(f"delegate ask {call_id}: raised {type(e).__name__}: {e}")
            raise
        if isinstance(result, InputRequired):
            log.info(f"delegate ask {call_id}: input_required={result.question!r}")
        else:
            log.info(f"delegate ask {call_id}: answer={result!r}")
        return result


# the booking office as the travel desk's one tool, described from its card
back_office = LoggingDelegate(BACKEND)

chat = Agent(
    instructions=PERSONA + " By text: one or two plain sentences.",
    llm="google/gemma-4-31b-it",
    tools=[back_office],
)

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time. While the booking office "
    "works, tell the traveller in one line that it is on it, then wait.",
    tools=[back_office],
)


front_desk = AgentHandler()  # the worker's own lane: the phone and the text door share it


@front_desk.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)

    @session.on("tool_execution_updated")
    def _tool(ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            call = update.function_call
            log.info(f"call tool started {call.call_id}: {call.name}({call.arguments})")
        elif update.type == "tool_call_updated":
            log.info(f"call tool updated {update.call_id}: {update.message!r}")
        elif update.type == "tool_call_ended":
            log.info(
                f"call tool ended {update.call_id}: status={update.status} "
                f"message={update.message!r}"
            )

    @session.on("conversation_item_added")
    def _item(ev: ConversationItemAddedEvent) -> None:
        log.info(f"call item added: {describe(ev.item)}")

    await session.start(agent=voice, room=ctx.room)
    log.info(f"call started: thread={ctx.thread.id if ctx.thread else None}")
    if not session.history.items:
        await session.generate_reply(instructions="Greet the traveller in one sentence.")


@front_desk.a2a_session(endpoint="frontend", description="Skyline Travel desk.", inbox=FRONT_INBOX)
async def text(ctx: A2ASessionContext) -> str:
    turn = ctx.task_id[-6:]
    wait = f"{time.time() - ctx.messages[-1].received_at:.1f}s" if ctx.messages else "none"
    log.info(
        f"frontend turn {turn}: thread={ctx.thread.id} lease_wait={wait} "
        f"message={ctx.message.text_content!r} rows={rows(ctx.tasks)}"
    )
    result = await chat.run(ctx.message, message_history=ctx.message_history)
    for item in result.new_messages():
        log.info(f"frontend turn {turn}: added {describe(item)}")
    log.info(
        f"frontend turn {turn}: input_required={result.input_required!r} "
        f"interrupted={result.interrupted}"
    )
    return result.input_required or str(result.output)
