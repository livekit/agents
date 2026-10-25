from __future__ import annotations

import asyncio
import logging
import os
import time
from contextvars import ContextVar
from typing import Literal, cast

from pydantic import BaseModel
from review_log import rows

from livekit.agents import Agent, AgentHandler, RunContext
from livekit.agents.a2a import A2ASessionContext

log = logging.getLogger("review.backend")

Policy = Literal["live", "batch", "interrupt", "queue", "concurrent", "reject"]

# the inbox policy under test, and how long each of the booking's nine steps takes
INBOX = cast(Policy, os.environ.get("REVIEW_INBOX", "live"))
STEP_SECONDS = float(os.environ.get("REVIEW_STEP_SECONDS", "5"))


class Trip(BaseModel):
    """What the booking office keeps about the trip, on its lane."""

    booked_day: str | None = None


# the turn a tool runs in, so a step can read its own task row as it is now
current_turn: ContextVar[A2ASessionContext[Trip]] = ContextVar("current_turn")

backend = Agent(
    instructions=(
        "You are the booking office of Skyline Travel. Answer in one sentence. To book a "
        "flight, call book_flight with the day; to change the day of a booking, call "
        "change_flight with the new day."
    ),
    llm="google/gemma-4-31b-it",
    state_type=Trip,
)


@backend.tool
async def book_flight(ctx: RunContext[None, Trip], day: str) -> str:
    """Book a flight on the day; it takes a while.

    Args:
        day: the day the traveller named, as they said it
    """
    turn = current_turn.get()
    task_id = turn.task_id
    tag = f"backend turn {task_id[-6:]}: book_flight"
    try:
        for i in range(1, 10):
            await ctx.update(f"booking step {i}/9 for {day}")
            row = await turn.task(task_id)
            log.info(f"{tag} step {i}/9 day={day} row={row.status if row else 'missing'}")
            await asyncio.sleep(STEP_SECONDS)
        # the seat is taken in the airline's system: a cancel waits for this to finish
        with ctx.uninterruptible():
            log.info(f"{tag} taking the seat")
            await asyncio.sleep(STEP_SECONDS)
            ctx.state.booked_day = day
    except asyncio.CancelledError:
        log.info(f"{tag} stopped by a cancel, booked_day={ctx.state.booked_day!r}")
        raise
    log.info(f"{tag} booked {day}")
    return f"booked {day}"


@backend.tool
async def change_flight(ctx: RunContext[None, Trip], day: str) -> str:
    """Change the booked flight to another day.

    Args:
        day: the new day, as the traveller said it
    """
    log.info(
        f"backend change_flight {ctx.function_call.call_id}: "
        f"from={ctx.state.booked_day!r} to={day!r}"
    )
    ctx.state.booked_day = day
    return f"changed to {day}"


back_office = AgentHandler(lane="backend")


@back_office.a2a_session(
    endpoint="backend",
    description="Skyline Travel's booking office: books flights and changes their day.",
    public=False,
    inbox=INBOX,
    state=Trip,
)
async def text(ctx: A2ASessionContext[Trip]) -> str:
    current_turn.set(ctx)
    started = time.time()
    turn = ctx.task_id[-6:]
    wait = f"{started - ctx.messages[-1].received_at:.1f}s" if ctx.messages else "none"
    log.info(
        f"backend turn {turn}: thread={ctx.thread.id} task={ctx.task_id} policy={INBOX} "
        f"lease_wait={wait} batch={len(ctx.messages)} message={ctx.message.text_content!r}"
    )
    log.info(f"backend turn {turn}: own rows: {rows(ctx.tasks)}")

    result = await backend.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    log.info(
        f"backend turn {turn}: done in {time.time() - started:.1f}s output={result.output!r} "
        f"interrupted={result.interrupted} booked_day={ctx.state.booked_day!r}"
    )
    return result.input_required or str(result.output)
