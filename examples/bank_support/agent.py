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
    "no lists, no symbols. Read phone and card numbers back digit by digit, and say money as you would aloud, for example two hundred twenty-three dollars and seventy-four cents. Verify the caller "
    "first with the phone number and date of birth on file. Do not ask for permission before calling a "
    "change tool: the tool asks you to confirm, and then you say exactly what will change and wait "
    "for a yes. If the caller's answer is unclear, ask again before calling it. After end_call or "
    "transfer_to_human, say nothing more. When a tool replies with a note that starts with "
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
    customer_id: str | None = None  # set by verify_identity


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


def _with_undo_tools(names: list[str]) -> list[str]:
    """The flow's tools plus the undo tool of each write, so a caller who changes their
    mind right after a write is served in the same task."""
    out = list(names)
    for name in names:
        undo = CONTRACT.rules[name].undo if name in CONTRACT.rules else None
        if undo and undo not in out:
            out.append(undo)
    return out


class WriteTask(AgentTask[dict[str, Any]]):
    """One unit of work: its own instructions, only its tools, and the call's records.

    The task stays open after a write, so the caller can add to the same request (a
    replacement after a block). It finishes through ``finish_request`` once one of its
    writes succeeded, or through ``end_request`` when the caller drops the request."""

    def __init__(self, flow: Flow, state: CallState, chat_ctx: ChatContext | None = None) -> None:
        self._flow = flow
        self._state = state
        self._first_record = len(state.records)
        super().__init__(
            instructions=flow.instructions
            + " When the caller has nothing more for this request, call finish_request.",
            tools=[
                *_guarded_tools(_with_undo_tools(flow.tools)),
                self._finish_request_tool(),
                self._end_request_tool(),
            ],
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self._first_record = len(self._state.records)
        self.session.generate_reply()

    async def finish_request_reply(self, summary: str) -> str | None:
        """None when the task finished; otherwise the reply the model gets instead."""
        done = [
            r.name
            for r in self._state.records[self._first_record :]
            if r.ok and r.name in self._flow.writes
        ]
        if not done:
            return (
                f"NOT DONE: none of {', '.join(self._flow.writes)} has succeeded for this request. "
                "Finish the change after the caller's yes, or call end_request if they dropped it."
            )
        if not self.done():
            self.complete(
                {"flow": self._flow.name, "done": True, "writes": done, "summary": summary}
            )
        return None

    def _finish_request_tool(self) -> FunctionTool:
        async def finish_request(summary: str) -> str:
            """Finish this request once the change is made and the caller has nothing more for it.

            Args:
                summary: What was done, in one sentence.
            """
            return await self.finish_request_reply(summary) or (
                "Request finished. Do not say goodbye: the main assistant continues the call."
            )

        return function_tool(finish_request)

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
        tools.append(function_tool(CONTRACT.guarded(self._transfer_to_human_impl, fillers=False)))
        tools.append(function_tool(CONTRACT.guarded(self._end_call_impl, fillers=False)))
        return tools

    def _start_tool(self, flow: Flow) -> FunctionTool:
        async def start(ctx: RunContext[Userdata]) -> str:
            result = await WriteTask(flow, ctx.userdata.state, chat_ctx=self.chat_ctx)
            if result.get("done"):
                return (
                    f"The {flow.name} request is complete: {result.get('summary', '')}. If the caller "
                    "already said there is nothing else, call end_call with their words now; do not "
                    "ask again."
                )
            return (
                f"The {flow.name} request was closed without a change: {result.get('reason', '')}"
            )

        start.__name__ = f"start_{flow.name}"
        start.__doc__ = (
            f"Start the {flow.name} request once the caller is verified. {flow.description}"
        )
        return function_tool(start)

    async def _transfer_to_human_impl(
        self, ctx: RunContext[Userdata], summary: str, closing_message: str
    ) -> str:
        """Hand the call to a person, for a request no tool here covers or when the caller asks
        for one. This ends your part of the call: no closing question is needed.

        Args:
            summary: What the caller needs, in one or two sentences, for the colleague.
            closing_message: What to tell the caller before the hand-off.
        """
        logger.info("transfer to human: %s", summary)
        # A real deployment transfers the SIP call here; the example ends the session.
        await self._say_and_close(ctx, closing_message)
        return "Transferred to a colleague."

    _transfer_to_human_impl.__name__ = "transfer_to_human"

    async def _end_call_impl(
        self, ctx: RunContext[Userdata], caller_reply: str, closing_message: str
    ) -> str:
        """End the call after you asked whether there is anything else and the caller said no.

        Args:
            caller_reply: The caller's last words, exactly as they said them.
            closing_message: The goodbye to say first.
        """
        await self._say_and_close(ctx, closing_message)
        return "Ending the call."

    async def _say_and_close(self, ctx: RunContext[Userdata], message: str) -> None:
        await ctx.session.say(message, allow_interruptions=False)
        session = ctx.session

        async def close_after_drain() -> None:
            await asyncio.sleep(SESSION_DRAIN_S)
            await session.aclose()

        self._close_task = asyncio.create_task(close_after_drain())

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
