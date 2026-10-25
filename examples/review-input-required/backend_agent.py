from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel
from review_log import describe, rows

from livekit.agents import (
    Agent,
    AgentHandler,
    AgentTask,
    Completed,
    InputRequired,
    RunContext,
    function_tool,
)
from livekit.agents.a2a import A2ASessionContext

log = logging.getLogger("review.backend")

FRONT_DESK = "inputreq"  # the front desk's lane: the worker's agent name


class Patient(BaseModel):
    """What the back office keeps about the patient, on its lane."""

    email: str | None = None


class Signup(BaseModel):
    """The newsletter flow's own state."""

    email: str | None = None


@dataclass
class CollectEmail(AgentTask[str, Signup]):
    """Asks for the email for the newsletter; completes with it."""

    def __post_init__(self) -> None:
        super().__init__(
            instructions=(
                "Ask the patient, in one sentence, which email the newsletter should go to. When "
                "they give one, call record_email with it."
            ),
            llm="google/gemma-4-31b-it",
        )

    @function_tool
    async def record_email(self, ctx: RunContext[None, Patient], email: str) -> Completed[str]:
        """Finish with the email the patient gave.

        Args:
            email: the email, as the patient gave it
        """
        log.info(f"backend CollectEmail.record_email {ctx.function_call.call_id}: email={email!r}")
        self.state.email = email
        return Completed(f"signed up {email} for the newsletter")


backend = Agent(
    instructions=(
        "You are the back office of Pearl Dental. Answer in one sentence. To book a visit, call "
        "book_visit with the day and the patient's email if they gave one. When a tool asks the "
        "patient something, the patient's answer continues your request as its next message: an "
        "email means calling book_visit again with it; a question means "
        "answering it in one sentence and asking for the email again. For the newsletter, call "
        "newsletter_signup."
    ),
    llm="google/gemma-4-31b-it",
    state_type=Patient,
)


@backend.tool
async def book_visit(
    ctx: RunContext[None, Patient], day: str, email: str | None = None
) -> str | InputRequired:
    """Book a visit; asks for the email when it is missing.

    Args:
        day: the day the patient named, as they said it
        email: where the confirmation goes, as the patient gave it
    """
    log.info(f"backend book_visit {ctx.function_call.call_id}: day={day!r} email={email!r}")
    if email is None or "@" not in email:
        return InputRequired("Which email address should the confirmation go to?")
    ctx.state.email = email
    return f"booked {day}; confirmation to {email}"


@backend.tool
async def newsletter_signup(ctx: RunContext[None, Patient]) -> str:
    """Sign the patient up for the newsletter; a short flow asks for the email."""
    call_id = ctx.function_call.call_id
    log.info(f"backend newsletter_signup {call_id}: awaiting CollectEmail")
    result = await CollectEmail()
    log.info(f"backend newsletter_signup {call_id}: code after the await ran, result={result!r}")
    return result


back_office = AgentHandler(lane="backend")


@back_office.a2a_session(
    endpoint="backend",
    description="Pearl Dental's back office: books visits and signs patients up.",
    public=False,
    state=Patient,
)
async def text(ctx: A2ASessionContext[Patient]) -> str:
    turn = ctx.task_id[-6:]
    channel = ctx.messages[-1].channel if ctx.messages else "none"
    caller = "none" if ctx.caller_history is None else f"{len(ctx.caller_history.items)} items"
    # the caller names its lane on a delegation in the shared thread
    front = (await ctx.thread.chat_history(ctx.caller_lane or FRONT_DESK)).items
    log.info(
        f"backend turn {turn}: thread={ctx.thread.id} channel={channel} "
        f"message={ctx.message.text_content!r}"
    )
    log.info(
        f"backend turn {turn}: task={ctx.task_id} own history={len(ctx.message_history.items)} "
        f"items caller_history={caller} caller_lane={ctx.caller_lane}, last 2: "
        + (" | ".join(describe(item) for item in ctx.message_history.items[-2:]) or "empty")
    )
    # the front desk's lane as the store has it right now: one item per line, the whole lane,
    # so what is missing (the turn that led to this delegation) is visible
    lines = [f"  {i:>3}. {describe(item)}" for i, item in enumerate(front, 1)] or ["  (empty)"]
    log.info(
        f"backend turn {turn}: {ctx.caller_lane or FRONT_DESK} lane as stored, {len(front)} "
        "items:\n" + "\n".join(lines)
    )
    log.info(f"backend turn {turn}: own rows: {rows(ctx.tasks)}")
    log.info(f"backend turn {turn}: front desk rows: {rows(await ctx.thread.tasks(FRONT_DESK))}")

    result = await backend.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    log.info(
        f"backend turn {turn}: output={result.output!r} input_required={result.input_required!r}"
    )
    return result.input_required or str(result.output)
