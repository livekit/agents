from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentTask,
    Completed,
    ModelRetry,
    RunContext,
    function_tool,
)

log = logging.getLogger("review.frontend")

PERSONA = "You are a support desk for an online shop. One sentence answers. Use the tools as asked."


def now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def on_call(ctx: RunContext) -> bool:
    """Whether the tool runs on the call, where a task takes the session's realtime model."""
    try:
        _ = ctx.session
    except RuntimeError:
        return False
    return True


class Attempts(BaseModel):
    count: int = 0


@dataclass
class CollectCode(AgentTask[str, Attempts]):
    """Asks for a four-digit code; completes with it."""

    live: bool = False  # on a call: the session's model, else a text model

    def __post_init__(self) -> None:
        super().__init__(
            instructions=(
                "Ask the customer, in one sentence, for the four-digit code from their "
                "confirmation email. When they give four digits, call record_code with them."
            ),
            llm=NOT_GIVEN if self.live else "google/gemma-4-31b-it",
        )

    @function_tool
    async def record_code(self, ctx: RunContext, code: str) -> Completed[str]:
        """Finish with the code the customer gave.

        Args:
            code: the four digits, as given
        """
        self.state.count += 1
        log.info(f"CollectCode.record_code {ctx.function_call.call_id}: code={code!r}")
        return Completed(code)


@function_tool
async def lookup_order(ctx: RunContext, order_id: str) -> str:
    """Look up one order's shipping status; call it once per order.

    Args:
        order_id: the order number, as the customer gave it
    """
    call_id = ctx.function_call.call_id
    log.info(f"lookup_order {call_id} {order_id}: start at {now()}")
    for update in ("looking up " + order_id, "almost there"):
        await ctx.update(update)
        log.info(f"lookup_order {call_id} {order_id}: update {update!r} returned at {now()}")
        await asyncio.sleep(4)
    log.info(f"lookup_order {call_id} {order_id}: end at {now()}")
    return f"order {order_id}: shipped yesterday"


@function_tool
async def verify_identity(ctx: RunContext) -> str:
    """Verify the customer with the code from their confirmation email."""
    code = await CollectCode(live=on_call(ctx))
    log.info(f"verify_identity {ctx.function_call.call_id}: after the await: code={code!r}")
    return f"verified with {code}"


@function_tool
async def verify_identity_safely(ctx: RunContext) -> str:
    """Verify the customer with the code, failing softly; use it when asked to verify safely."""
    try:
        code = await CollectCode(live=on_call(ctx))
        log.info(f"verify_identity_safely {ctx.function_call.call_id}: after the await: {code=}")
        return f"verified with {code}"
    except Exception as e:
        log.info(f"verify_identity_safely {ctx.function_call.call_id}: caught {e}")
        return "verification failed"


@function_tool
async def flaky_lookup(ctx: RunContext) -> str:
    """A lookup that needs more from the customer; use it only when asked for the flaky lookup."""
    log.info(f"flaky_lookup {ctx.function_call.call_id}: raising ModelRetry")
    raise ModelRetry("ask for the order number first")


TOOLS = [lookup_order, verify_identity, verify_identity_safely, flaky_lookup]

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time.",
    tools=TOOLS,
)

chat = Agent(
    instructions=PERSONA + " By text: plain sentences, no lists.",
    llm="google/gemma-4-31b-it",
    tools=TOOLS,
)
