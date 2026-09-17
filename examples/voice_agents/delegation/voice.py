"""The voice half: a realtime model talks, the fare desk in the other process reasons.

Start the desk first, in another terminal:

    python expert.py dev

then this, and call in:

    python voice.py console

Two processes on one machine, talking A2A over localhost. The phone agent has no tools of
its own beyond the one delegation gives it and the one that has to talk, and it knows
nothing about fares, seats or rules. Ask it something real — "my flight to Tokyo tomorrow
is delayed, what else can you put me on?" — and you will hear it acknowledge, report what
the desk is doing while it does it, then answer.

Dana Whitfield <dana@example.com> is a Gold member whose Tokyo flight tomorrow is delayed
245 minutes, which is the interesting case: the delay is our fault, so the fee is waived and
the seat moves for nothing. Miguel Ortiz <ortiz@example.com> is on a BASIC fare, which
cannot be changed or refunded at all. Priya Raman <raman@example.com> holds travel credit.
"""

import json
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    DirectiveReceivedEvent,
    JobContext,
    RunContext,
    ToolExecutionUpdatedEvent,
    cli,
)
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.delegation import DELEGATE_TOOL_NAME, A2ADelegate
from livekit.agents.llm import function_tool
from livekit.plugins import openai

logger = logging.getLogger("voice")

load_dotenv()

FARE_DESK_URL = "http://localhost:8321/fare-desk"

server = AgentServer()


def _short(text: str | None, limit: int = 90) -> str:
    """One clipped line. Tool payloads are long and it is the shape of the call that reads."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(
            # purely conversational, and deliberately not told who does the rest. Calling the
            # other half "the fare desk" here would invite it to decide a weather question is
            # out of scope and answer from memory instead of delegating
            instructions=(
                "You are the voice of Northwind Air's support line. Keep every reply to one "
                "or two short sentences. You are on the phone, so no emojis, asterisks or "
                "markdown, and never read a booking reference out loud. "
                "You do not look anything up or work anything out yourself. Every request "
                "goes to the delegate tool — flights, prices, rules, bookings, dates, "
                "baggage, the weather, whatever the caller asks — and you say what comes "
                "back in your own words. Nothing is off-topic for it, so never decide by "
                "yourself that something is out of scope or cannot be done. "
                "Say dates the way a person would, 'Friday the 31st', but never convert one "
                "yourself — pass on the caller's own words, because the other half is the "
                "one that knows today's date. If what comes back lists more than one "
                "booking, ask which flight they mean by route and date. "
                "The one thing you do yourself is take an email address, with collect_email, "
                "and only when what came back asks for one — never to open with. "
                "Wait for the caller to speak before you use any tool: until they have asked "
                "for something there is nothing to delegate and nobody to look up."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="greet the caller as Northwind Air and ask how you can help"
        )

    @function_tool
    async def collect_email(self, ctx: RunContext, change: bool = False) -> str:
        """Ask the caller for their email address, reading it back to confirm it.

        This one talks, which is why it lives here and not on the other side: spelling an
        address out and confirming it is a conversation, and the other half is not on the
        phone. Reach for it only once an answer has asked for an address.

        Args:
            change: only when the caller wants a different address from the one already
                confirmed in this call.
        """
        async with ctx.foreground():
            result = await GetEmailTask(chat_ctx=self.chat_ctx)

        email = result.email_address.strip().lower()
        logger.info(f"caller identified as {email}")
        # said back into the conversation, so the next delegation carries it to the desk
        return f"confirmed with the caller: {email}"


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        # one delegate per conversation: the session closes it when the call ends, which is
        # what tells the desk it can drop this conversation rather than wait for it to idle
        delegate={"delegate": A2ADelegate(FARE_DESK_URL), "announce": False},
        llm=openai.realtime.RealtimeModel(model="gpt-realtime"),
        # llm=inference.LLM("openai/gpt-4.1-mini"),
        # stt=inference.STT("deepgram/nova-3", language="multi"),
        # llm=inference.LLM("google/gemma-4-31b-it"),
        # tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    @session.on("directive_received")
    def _on_directive(ev: DirectiveReceivedEvent) -> None:
        # advice, acted on after the answer: shutdown drains, so whatever is queued plays
        # out before the call ends. what to do about a directive is yours
        logger.info(f"── directive: {ev.kind} ({ev.reason})")
        if ev.kind == "end_session":
            session.shutdown()

    # this side's half of the trace: what was asked of the desk, what it relayed back while
    # it worked, and what it answered. The desk's own tool calls are in the other terminal.
    #
    #   ▶ delegated: caller's flight to Tokyo is delayed, find them something tomorrow
    #   … relayed: holding a seat on NW812
    #   ◀ answered: moved to NW812, 302.40 charged, the delay waived the fee
    @session.on("tool_execution_updated")
    def _on_tool_execution_updated(ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            call = update.function_call
            if call.name != DELEGATE_TOOL_NAME:
                logger.info(f"     → {call.name}({_short(call.arguments)})")
                return
            try:
                task = json.loads(call.arguments or "{}").get("task", "")
            except ValueError:
                task = call.arguments
            logger.info(f"▶ delegated: {_short(task, 200)}")
        elif update.type == "tool_call_updated":
            # the dispatch note is recorded for the model and never spoken, and is not worth
            # a line; everything else here is the desk reporting as it works
            if not update.silent:
                logger.info(f"… relayed: {_short(update.message)}")
        elif update.type == "tool_call_ended":
            logger.info(f"◀ answered: {_short(update.message, 200)}")

    await session.start(agent=Receptionist(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
