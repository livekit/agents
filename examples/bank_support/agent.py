"""A bank card and transfer support agent whose flow rules live at the tool boundary.

The customer's flow document is a table of rules in ``flows/bank.py``; ``contract.py``
turns the table into a check before every tool call and a sentence after every reply.
This file is the part that does not change between businesses: the session settings,
the parent agent that verifies and routes, the task that owns each write, the filler
speech during tool calls, and the caller-turn annotation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import simulation
from bank_db import FakeBank
from contract import CallState, digits_heard
from dotenv import load_dotenv
from flows.bank import CONTRACT, FLOWS, TOOLS, Flow

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    RunContext,
    SimulationContext,
    cli,
    function_tool,
    inference,
)
from livekit.agents.llm import ChatContext, ChatMessage, FunctionTool

load_dotenv()

logger = logging.getLogger("bank-support")

# How long the session stays open after the caller hangs up, so the last turn's
# events are delivered before the room closes.
SESSION_DRAIN_S = 2.0

# The business's own prompt. Nothing about the rules below lives here: a rule that
# must hold is enforced at the tool boundary, not asked for in prose.
INSTRUCTIONS = (
    "You are the phone assistant of Harbor Bank. You help account holders block a lost or stolen "
    "card and order a replacement, dispute transactions they do not recognise, move money between "
    "their own accounts, and add travel notices. Your words are spoken aloud: keep each turn short, "
    "no lists, no symbols, and say numbers digit by digit when you read them back. Verify the caller "
    "first with the phone number and date of birth on file. Before any change, say exactly what you "
    "are about to do and wait for the caller's yes. When a tool replies with a note that starts with "
    "NOT DONE, ALREADY DONE or NOT ENDED, follow that note. Before ending, ask whether there is "
    "anything else you can help with. If the caller asks for something you have no tool for, or asks "
    "for a person, use transfer_to_human."
)

# Deepgram hears digits better with these; the keyterms are the words a caller uses
# when reading a card or phone number.
STT_EXTRA_KWARGS = {"dictation": True, "smart_format": True, "numerals": True}
DIGIT_KEYTERMS = [
    "zero",
    "oh",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "double",
    "triple",
]


@dataclass
class Userdata:
    bank: FakeBank
    state: CallState = field(default_factory=CallState)


def annotate_user_turn(state: CallState, message: ChatMessage) -> None:
    """Append the digit strings the caller spoke beneath their words, and record the turn.

    The caller's words stay as they were: the model may have asked for exactly that
    reading. The appended form is what the lookup needs.
    """
    text = (message.text_content or "").strip()
    if not text:
        return
    state.user_turns.append(text)
    heard = digits_heard(text)
    if heard:
        message.content = [f"{text}\n[heard: {', '.join(heard)}]"]


def _guarded_tools(names: list[str]) -> list[FunctionTool]:
    return [function_tool(CONTRACT.guarded(TOOLS[name])) for name in names]


class WriteTask(AgentTask[dict[str, Any]]):
    """One unit of work: its own instructions, only its tools, the call's records, and a
    completion the model cannot talk past. It completes on the flow's write, or when the
    caller declines through ``end_request``."""

    def __init__(self, flow: Flow, state: CallState, chat_ctx: ChatContext | None = None) -> None:
        self._flow = flow
        self._state = state
        super().__init__(
            instructions=flow.instructions,
            tools=[*_guarded_tools(flow.tools), self._end_request_tool()],
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self._state.completions[self._flow.completes_on] = self._on_write
        self.session.generate_reply()

    async def on_exit(self) -> None:
        self._state.completions.pop(self._flow.completes_on, None)

    def _on_write(self, name: str, arguments: dict[str, Any], result: str) -> None:
        if not self.done():
            self.complete({"flow": self._flow.name, "done": True, "tool": name, "result": result})

    def _end_request_tool(self) -> FunctionTool:
        async def end_request(reason: str) -> str:
            """End this request without the change, when the caller declines it or changes their mind.

            Args:
                reason: Why, in the caller's words.
            """
            self._state.declined = True
            if not self.done():
                self.complete({"flow": self._flow.name, "done": False, "reason": reason})
            return "Request closed without a change."

        return function_tool(end_request)


class BankAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS, tools=self._parent_tools())

    def _parent_tools(self) -> list[FunctionTool]:
        tools = _guarded_tools(["verify_identity", "get_accounts", "get_card", "get_transactions"])
        for flow in FLOWS.values():
            tools.append(self._start_tool(flow))
        tools.append(function_tool(self.transfer_to_human))
        tools.append(function_tool(CONTRACT.guarded(self._end_call_impl)))
        return tools

    def _start_tool(self, flow: Flow) -> FunctionTool:
        async def start(ctx: RunContext[Userdata]) -> str:
            result = await WriteTask(flow, ctx.userdata.state, chat_ctx=self.chat_ctx)
            if result.get("done"):
                return f"The {flow.name} request is complete: {result.get('result', '')}"
            return (
                f"The {flow.name} request was closed without a change: {result.get('reason', '')}"
            )

        start.__name__ = f"start_{flow.name}"
        start.__doc__ = (
            f"Start the {flow.name} request once the caller is verified. {flow.description}"
        )
        return function_tool(start)

    async def transfer_to_human(self, ctx: RunContext[Userdata], summary: str) -> str:
        """Hand the call to a person, for a request no tool here covers or when the caller asks for one.

        Args:
            summary: What the caller needs, in one or two sentences.
        """
        logger.info("transfer to human: %s", summary)
        return "A colleague will take the call from here. Tell the caller, then end the call."

    async def _end_call_impl(self, ctx: RunContext[Userdata], closing_message: str) -> str:
        """End the call after the caller has said there is nothing else.

        Args:
            closing_message: The goodbye to say first.
        """
        await ctx.session.say(closing_message, allow_interruptions=False)
        session = ctx.session

        async def close_after_drain() -> None:
            await asyncio.sleep(SESSION_DRAIN_S)
            await session.aclose()

        self._close_task = asyncio.create_task(close_after_drain())
        return "Ending the call."

    _end_call_impl.__name__ = "end_call"

    async def on_enter(self) -> None:
        await self.session.say(
            "Thanks for calling Harbor Bank. To get started, may I have the phone number on your account?"
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        annotate_user_turn(self.session.userdata.state, new_message)


def build_session(userdata: Userdata) -> AgentSession[Userdata]:
    """The session settings for a tool-using voice agent: speak during tool waits, never
    reason out loud, close a turn within 3 s, yield to a real interruption and not to a
    backchannel, keep tool calls sequential."""
    return AgentSession[Userdata](
        userdata=userdata,
        stt=inference.STT(
            "deepgram/nova-3", extra_kwargs={**STT_EXTRA_KWARGS, "keyterm": DIGIT_KEYTERMS}
        ),
        llm=inference.LLM("google/gemma-4-31b-it", extra_kwargs={"parallel_tool_calls": False}),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_handling={
            "endpointing": {"min_delay": 1.2, "max_delay": 3.0},
            "interruption": {"mode": "adaptive"},
            # or, to stop faster on a real barge-in at the cost of some backchannels:
            # "interruption": {"mode": "vad", "min_duration": 0.9, "min_words": 2},
        },
        max_tool_steps=8,
    )


server = AgentServer()


async def on_simulation_end(ctx: SimulationContext) -> None:
    userdata = ctx.userdata()
    expected = userdata.get("expected")
    if not expected:
        return  # graded on the conversation only
    bank: FakeBank = ctx.job_context.primary_session.userdata.bank
    if (why := simulation.grade(bank, expected)) is not None:
        ctx.fail(reason=why)


@server.rtc_session(on_simulation_end=on_simulation_end)
async def bank_support(ctx: JobContext) -> None:
    await ctx.connect()
    if sim := ctx.simulation_context():
        bank = simulation.bank_from_userdata(sim.userdata())
    else:
        bank = FakeBank.seeded()
    userdata = Userdata(bank=bank)
    session = build_session(userdata)

    async def drain_and_close() -> None:
        await asyncio.sleep(SESSION_DRAIN_S)
        await session.aclose()
        ctx.shutdown(reason="caller left")

    drain_task: asyncio.Task[None] | None = None

    @ctx.room.on("participant_disconnected")
    def _caller_left(_participant: Any) -> None:
        nonlocal drain_task
        if drain_task is None:
            drain_task = asyncio.create_task(drain_and_close())

    await session.start(agent=BankAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
