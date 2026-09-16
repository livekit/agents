"""The fare desk: an expert served over A2A, with no voice of its own.

Run it alongside `voice.py`, which talks to it:

    python expert.py dev

It serves one endpoint on the agent server's HTTP app:

    curl localhost:8321/fare-desk/.well-known/agent-card.json

Anything that speaks A2A can use it, ours or not. To drive it by hand:

    curl -X POST localhost:8321/fare-desk/v1/message:stream -N \
         -H 'content-type: application/json' \
         -d '{"message": {"messageId": "m1", "role": "ROLE_USER",
              "parts": [{"text": "what does it cost to move my Monday flight to Tuesday?"}]}}'

The expert never speaks to the caller. It returns facts, and the voice agent phrases them.
`ctx.update()` inside a tool reports progress while the work is still running, and that
report is relayed as the tool wrote it rather than being handed to a model to restate.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, RunContext, cli, function_tool
from livekit.agents.a2a import TextSessionContext
from livekit.agents.llm import ToolFlag

logger = logging.getLogger("fare-desk")

load_dotenv()

server = AgentServer()


@dataclass
class Booking:
    code: str
    day: str


@dataclass
class Caller:
    """One caller's file, as a store would hold it."""

    bookings: list[Booking] = field(default_factory=list)
    holds: dict[str, str] = field(default_factory=dict)


CHANGE_FEE_USD = 75
_TODAY = date.today()


def _next(weekday: str) -> str:
    wanted = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(
        weekday.strip().lower()
    )
    ahead = (wanted - _TODAY.weekday()) % 7 or 7
    return (_TODAY + timedelta(days=ahead)).isoformat()


class FareDesk(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the fare desk of Northwind Air. You answer the agent talking to the "
                "caller, not the caller: give it the facts it needs and nothing else. Never "
                "invent a fare, a fee or a booking — call a tool. If a tool has not confirmed "
                "something, say so. Keep every answer to one or two sentences. "
                f"Today is {_TODAY.isoformat()}."
            )
        )

    @function_tool
    async def change_fee(self, ctx: RunContext, from_day: str, to_day: str) -> str:
        """The fee to move a booking between two days.

        Args:
            from_day: The day the booking is on, as a weekday name.
            to_day: The day the caller wants instead, as a weekday name.
        """
        return f"moving {from_day} to {to_day} costs {CHANGE_FEE_USD} USD"

    @function_tool
    async def check_availability(self, ctx: RunContext, day: str) -> str:
        """Whether seats are left on a day, which takes a moment to look up.

        Args:
            day: The day to check, as a weekday name.
        """
        # reports while the work is still running, and releases the turn: the voice agent
        # hears this as it happens rather than waiting for the answer
        await ctx.update(f"checking {day}")
        return f"{day} ({_next(day)}) has seats in economy"

    @function_tool(flags={ToolFlag.CANCELLABLE})
    async def hold_seat(self, ctx: RunContext, day: str) -> str:
        """Hold a seat on a day, which takes a while and can be cancelled.

        Args:
            day: The day to hold, as a weekday name.
        """
        await ctx.update(f"holding a seat on {day}")
        caller: Caller = ctx.userdata
        code = f"NW{len(caller.bookings) + len(caller.holds) + 1:03d}"
        caller.holds[code] = day
        return f"held {code} on {day}, unconfirmed"

    @function_tool
    async def confirm(self, ctx: RunContext, code: str) -> str:
        """Confirm a held seat.

        Args:
            code: The hold code to confirm.
        """
        caller: Caller = ctx.userdata
        day = caller.holds.pop(code, None)
        if day is None:
            return f"{code} is not a hold I have"
        caller.bookings.append(Booking(code=code, day=day))
        return f"{code} is confirmed on {day}"

    @function_tool
    async def end_of_call(self, ctx: RunContext) -> str:
        """Called when the caller has everything they came for and is done."""
        # advice to whoever asked, acted on after the answer is said. In an ordinary session
        # nobody is waiting on an answer, so there is nothing to advise and the session ends.
        if (request := ctx.session.request) is not None:
            request.set_directive("end_session", reason="caller_done")
        return "nothing outstanding"


@server.text_session(
    endpoint="fare-desk",
    description="Answers fare, baggage and change-fee questions for Northwind Air.",
)
async def fare_desk(ctx: TextSessionContext) -> None:
    session = AgentSession[Caller](llm="openai/gpt-4.1", userdata=Caller())
    await session.start(agent=FareDesk())
    # TODO(v1): runs in the server process; the same handler moves to a job process with #4337
    ctx.attach(session)


if __name__ == "__main__":
    cli.run_app(server)
