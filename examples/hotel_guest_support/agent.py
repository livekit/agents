from __future__ import annotations

import logging
import os
import sys
from datetime import date, time
from typing import Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# scenarios.yaml dates are literals against this date. hotel_db.TODAY freezes at
# import time, so the pin has to precede every import that reaches hotel_db.
if "--simulation" in sys.argv:
    os.environ.setdefault("HOTEL_TODAY", "2026-06-08")

from benchmark import build_expected, diff_databases
from common import Userdata, _speak_code, speech_only
from dotenv import load_dotenv
from hotel_db import (
    TODAY,
    FollowupKind,
    HotelDB,
    NotFound,
    RoomBooking,
    Unavailable,
    speak_time,
    speak_usd,
)
from instructions import INSTRUCTIONS
from policies import build_lookup_policy_tool
from seed import build_seed_bytes
from verify_booking import VerifyBookingTask

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    SimulationContext,
    ToolError,
    cli,
    function_tool,
    inference,
)
from livekit.agents.evals import (
    JudgeGroup,
    accuracy_judge,
    coherence_judge,
    conciseness_judge,
    handoff_judge,
    relevancy_judge,
    safety_judge,
    task_completion_judge,
    tool_use_judge,
)

load_dotenv(".env.local")

logger = logging.getLogger("hotel-guest-support")


@function_tool
async def record_followup(
    ctx: RunContext[Userdata],
    kind: FollowupKind,
    caller_name: str,
    caller_phone: str,
    summary: str,
) -> str:
    """Capture something for a human to follow up on - housekeeping requests, callback requests, verification-failed callers, in-house early-checkout requests, lost-and-found reports, and any other request you can't handle on this line. ALWAYS use this instead of saying "someone will follow up" with no record; otherwise the request vanishes.

    Args:
        kind: One of housekeeping, callback, verification_help, early_checkout, lost_and_found, other.
        caller_name: Caller's name (ask if you don't already have it).
        caller_phone: Caller's callback number - for an in-house guest, the room number works.
        summary: One sentence describing what they want, with enough detail for a human to act on it.
    """
    code = await ctx.userdata.db.record_followup(
        kind=kind, caller_name=caller_name, caller_phone=caller_phone, summary=summary
    )
    return (
        f"recorded; reference {_speak_code(code)} | read it back so the caller knows it's "
        f"actually on the list: who it's for ({caller_name}, {caller_phone}) and what's noted "
        f'("{summary}"). Don\'t just say "logged", and don\'t promise anyone will follow up or '
        "call back unless that's what was actually recorded."
    )


class GuestSupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS, tools=[build_lookup_policy_tool()])

    async def on_enter(self) -> None:
        # The caller may have already said what they want before we speak -
        # pick up from there instead of re-asking "how can I help?".
        await self.session.generate_reply(
            instructions=(
                "Greet the caller in one short sentence. If they've already named a need "
                "(towels, a wake-up call, a message for a guest...), move straight into helping; "
                "otherwise ask how you can help."
            )
        )

    @function_tool
    async def dispatch_emergency(
        self,
        ctx: RunContext[Userdata],
        room: str,
        kind: Literal["medical", "fire", "security"],
        situation: str,
    ) -> str:
        """EMERGENCY ONLY - a real, in-progress danger. Use it the MOMENT you have the room number and what's happening: no verification, no other questions first. It alerts the duty manager and sends hotel staff/security to the room - that dispatch is the PRIMARY action and shows the hotel owns it; outside help (911 / fire brigade / police) is a secondary direction you give the caller, never a substitute for sending the hotel's own people. Classify the kind:
          - "medical" - someone hurt, collapsed, unresponsive, not breathing, a health crisis.
          - "fire" - fire, smoke, or a fire alarm going off.
          - "security" - a safety/security threat: an intruder or someone forcing a door, assault or violence, a theft.
        NOT for nuisances - a noisy neighbour with nobody in danger is record_followup (kind="other"), not this.

        Args:
            room: The room number (e.g. "206"). Get this first if you don't have it.
            kind: medical, fire, or security - classify what's happening.
            situation: One short sentence: what's happening to whom.
        """
        try:
            code = await ctx.userdata.db.dispatch_emergency(
                room=room, kind=kind, situation=situation
            )
        except NotFound:
            raise ToolError(
                f"no room {room} exists - re-confirm the room number, calmly, right now"
            ) from None
        head = (
            f"DISPATCHED (ref {code}): duty manager alerted, staff heading to room {room} now | "
            "tell the caller, short and calm, that our people are on their way up right now"
        )
        if kind == "medical":
            tail = (
                " - then have them hang up and dial 9-1-1; the dispatcher stays on the line and "
                "tells them exactly what to do until the ambulance arrives. Don't give medical "
                "instructions yourself - the 911 dispatcher is the right person for that."
            )
        elif kind == "fire":
            tail = (
                " - tell them to get out now via the stairs or fire escapes, NOT the elevator, "
                "stay low if there's smoke, and once safe call the fire brigade on 9-1-1. Don't "
                "tell them to fight the fire or go investigate it."
            )
        else:  # security
            tail = (
                " - if they're in any immediate danger tell them to call 9-1-1 (police) now and "
                "stay somewhere safe with the door locked; otherwise our security and duty manager "
                "will be right there to help and take care of what's needed (a police report, and "
                "for a lost passport the consulate can help). Don't tell them to confront anyone."
            )
        return head + tail

    @function_tool
    async def take_guest_message(
        self,
        ctx: RunContext[Userdata],
        recipient: str,
        caller_name: str,
        caller_phone: str,
        message: str,
    ) -> str:
        """Take a message for someone the caller says is staying at the hotel. It gets delivered only if that person is in fact a guest - the result never tells you whether they are, and you must never tell the caller either: no confirming or denying anyone's presence, no room numbers, no connecting calls (see lookup_policy topic "guest_privacy"). Read the caller's name, number, and message back before calling this.

        Args:
            recipient: Full name of the person the message is for - first AND last. If the caller only gave a first name, ask for the last name before calling.
            caller_name: The caller's own name.
            caller_phone: The caller's callback number.
            message: The message, in the caller's words.
        """
        if len(recipient.split()) < 2:
            raise ToolError(
                f"'{recipient}' is only one name - a message needs the recipient's full name "
                "to reach the right person. Ask the caller for the last name, then call again."
            )
        code = await ctx.userdata.db.take_guest_message(
            recipient=recipient,
            caller_name=caller_name,
            caller_phone=caller_phone,
            message=message,
        )
        return (
            f"message recorded; reference {_speak_code(code)} | tell the caller it's logged and "
            "give the reference. You don't know whether the recipient is staying here and never "
            "say either way - but the general policy IS shareable: messages for in-house guests "
            "reach the room within about thirty minutes (message light, slip under the door). "
            "Promise delivery timing only, never that the person will read or act on it."
        )

    @function_tool
    async def schedule_wakeup_call(
        self,
        ctx: RunContext[Userdata],
        room: str,
        guest_name: str,
        call_date: date,
        call_time: time,
    ) -> str:
        """Schedule a wake-up call to a guest's room. This actually sets the call - never log a wake-up request as a followup note instead. Collect the room, the name, and the exact date and time from the caller, read them back, and call this once they've agreed. No booking verification needed.

        Args:
            room: The room number as the caller gave it (e.g. "304").
            guest_name: The guest's name.
            call_date: The date of the wake-up call in ISO YYYY-MM-DD format ("tomorrow morning" = tomorrow's date).
            call_time: The wake-up time in 24-hour HH:MM format (4:45 a.m. = "04:45").
        """
        try:
            code = await ctx.userdata.db.schedule_wakeup_call(
                room=room, guest_name=guest_name, call_date=call_date, call_time=call_time
            )
        except NotFound:
            raise ToolError(
                f"no room {room} exists - re-confirm the room number with the caller"
            ) from None
        except Unavailable as e:
            raise ToolError(f"can't schedule that: {e} - re-confirm the date") from None
        return (
            f"wake-up call set for room {room}, {call_date.strftime('%A, %B %-d')} at "
            f"{speak_time(call_time)}; reference {_speak_code(code)} | confirm it's set. If the "
            "caller worries about sleeping through: a second call comes about five minutes later "
            "if there's no answer, and no response to that sends staff up for an in-person room "
            "check - they will be woken."
        )

    @function_tool
    async def set_do_not_disturb(self, ctx: RunContext[Userdata], room: str) -> str:
        """Place a Do-Not-Disturb hold on an in-house guest's room when they ask not to be disturbed / to hold their calls and messages. It's a standing hold (until lifted), not a one-off like a single message or a wake-up call. Take the room number. Always tell the guest that a genuine emergency or hotel safety matter still overrides DND.

        Args:
            room: The guest's room number.
        """
        try:
            code = await ctx.userdata.db.set_do_not_disturb(room=room)
        except NotFound:
            raise ToolError(f"no room {room} exists - re-confirm the room number") from None
        return (
            f"Do-Not-Disturb set on room {room}; reference {_speak_code(code)} | confirm it holds "
            "their calls and messages until they ask to lift it, and that a genuine emergency "
            "still gets through."
        )

    async def _verified_booking(self, ctx: RunContext[Userdata]) -> RoomBooking:
        """Verify the caller once per call. Tools that mutate the booking
        (modify, cancel) update or clear the cache themselves."""
        if ctx.userdata.verified_booking is None:
            verify = await VerifyBookingTask(
                db=ctx.userdata.db, chat_ctx=speech_only(self.chat_ctx)
            )
            ctx.userdata.verified_booking = verify.booking
        return ctx.userdata.verified_booking

    @function_tool
    async def lookup_booking(self, ctx: RunContext[Userdata]) -> str:
        """Read-only lookup of a confirmed room booking. Use this when the caller wants to
        check or recall their booking details (dates, room type, what they're paying, who
        it's under, check-in time) without changing anything. Verifies the caller first."""
        b = await self._verified_booking(ctx)
        nights = b.nights
        extras = ", ".join(b.extras) if b.extras else "no extras"
        smoking = "smoking-permitted" if b.smoking else "non-smoking"
        info = (
            f"Booking for {b.first_name} {b.last_name}, {b.room_type.replace('_', ' ')} ({smoking}), "
            f"checking in {b.check_in.strftime('%A %B %-d')} and out {b.check_out.strftime('%A %B %-d')} "
            f"({nights} night{'s' if nights != 1 else ''}, {b.guests} guest{'s' if b.guests != 1 else ''}), "
            f"extras: {extras}. Total {speak_usd(b.total)} on card ending in {b.card_last4}."
        )
        if conflict := await ctx.userdata.db.room_conflict(booking_code=b.code):
            info += (
                f" | WARNING: the room is double-booked {conflict[0].strftime('%B %-d')} to "
                f"{conflict[1].strftime('%B %-d')} - no room is assigned to this booking for that "
                "period. Break the news with ownership and an apology, then run "
                'resolve_room_conflict to fix it (procedure: lookup_policy topic "guest_walks"). '
                "Don't pretend the booking is fine."
            )
        return info

    @function_tool
    async def resolve_room_conflict(self, ctx: RunContext[Userdata]) -> str:
        """Fix a double-booked / no-room situation on the caller's verified booking - run this when lookup_booking warned the room is double-booked. It applies the house procedure in fixed order: move the guest to a free room of the same or better category (an upgrade is free), and only if nothing in the house fits, arrange the walk (partner hotel tonight on us, covered taxi, their room back from the return date). Returns the concrete facts; deliver them following the guest_walks policy (own the overbooking, explain plainly why it happened, "at no extra cost to you"). Full procedure: lookup_policy topic "guest_walks"."""
        booking = await self._verified_booking(ctx)
        try:
            r = await ctx.userdata.db.resolve_room_conflict(booking_code=booking.code)
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        if r.moved_to:
            what = "an upgrade, free of charge" if r.upgraded else "same category, no charge"
            return (
                f"resolved: moved to {r.moved_to} - a {r.moved_to_view}-view "
                f"{r.moved_to_type.replace('_', ' ')} ({what}), same dates, total unchanged | "
                "deliver this per the guest_walks policy: own the overbooking and explain plainly "
                "why it happened, then confirm they still have a place for the whole stay, at the "
                "same total, at no extra cost. No further tool call is needed."
            )
        assert r.walk_return_date is not None
        return (
            f"no room in the house fits (every room was checked) - walk arranged at "
            f"{r.walk_partner} (two blocks away, room and taxi both on us), guest's room back "
            f"here {r.walk_return_date.strftime('%A, %B %-d')} | deliver this per the guest_walks "
            "policy: own the overbooking and explain plainly why it happened, then the plan above, "
            "all at no extra cost to them. The guest is angry and will interrupt - give it in short "
            "pieces and make sure every piece lands before the call ends, resuming any that got "
            "talked over. If still upset after the full plan, record a manager callback "
            '(record_followup, kind="callback") before wrapping up.'
        )

    @function_tool
    async def transfer_call(
        self,
        ctx: RunContext[Userdata],
        destination: Literal["restaurant", "duty_manager", "housekeeping"],
        summary: str,
    ) -> str:
        """Transfer the caller to a hotel DEPARTMENT - the restaurant, the duty manager, or housekeeping. NOT a guest's room (never connect a caller to a guest). Before calling this you must have told the caller you're putting them on hold to connect them to that department AND gotten their okay; only then transfer. Pass a one-line summary of what the caller needs so the department is briefed.

        Args:
            destination: The department to transfer to.
            summary: A one-line summary of what the caller needs.
        """
        # A transfer happens exactly once. If the agent re-calls this (the caller reacts
        # and it "re-confirms", or it retries after thinking the first failed), don't write
        # a second transfer row - the deterministic grader counts rows, and a duplicate
        # fails the run. Just reassure the caller they're being connected.
        if destination in ctx.userdata.transferred_to:
            return (
                f"already transferred to the {destination.replace('_', ' ')} on this call - do NOT "
                "transfer again. Just briefly reassure the caller they're being connected."
            )
        try:
            await ctx.userdata.db.transfer_call(destination=destination, summary=summary)
        except NotFound as e:
            raise ToolError(str(e)) from None
        ctx.userdata.transferred_to.add(destination)
        # Don't disconnect the session here: the caller may have a last reaction, and going
        # silent mid-call reads as a hang (the conversation ends when the caller is done, not
        # when we drop off). Close out briefly instead so the call can wrap up naturally.
        return (
            f"Transferred to the {destination.replace('_', ' ')} - your part of the call is done. "
            'Give ONE short closing hand-off ("You\'re all set - connecting you now"), NOT '
            '"anything else?", so the call can wrap up. Do NOT transfer again or take the request '
            "down as a followup; if the caller reacts, keep it to a brief acknowledgement and "
            "don't reopen the conversation."
        )


server = AgentServer()

_SEED_DB_BYTES = build_seed_bytes(TODAY)


def _expected_state_statements(userdata: dict[str, object]) -> list[str] | None:
    """No key means the DB isn't checked; null or [] means it must come back unchanged."""
    if "expected_state" not in userdata:
        return None
    statements = userdata["expected_state"]
    if statements is None:
        return []
    if not isinstance(statements, list) or not all(isinstance(s, str) for s in statements):
        raise TypeError("expected_state must be a list of SQL statements")
    return statements


async def on_simulation_end(ctx: SimulationContext) -> None:
    tagger = ctx.job_context.tagger
    failure: str | None = None
    graded = False
    try:
        expected_state = _expected_state_statements(ctx.userdata())
        graded = expected_state is not None
        if expected_state is not None:
            # Grade the run on final DB state: build the scenario's `expected_state` on a
            # fresh seed, then diff it against the agent's DB. The diff compares
            # agent-decided facts only (room type, dates, extras, status), so minted
            # codes / order / which-king don't matter and the agent need not reproduce the
            # statements — while collateral damage still surfaces.
            session = ctx.job_context.primary_session
            expected = await build_expected(_SEED_DB_BYTES, expected_state)
            try:
                diffs = diff_databases(expected.connection, session.userdata.db.connection)
            finally:
                await expected.aclose()
            if diffs:
                failure = "final DB diverges from expected: " + " | ".join(diffs[:8])
    except Exception as exc:
        # Grading that can't run is not a pass. The prefix tells a broken scenario
        # apart from a failing agent.
        logger.exception("expected-state grading failed")
        failure = f"expected-state grading failed: {exc}"

    # Most scenarios skip the DB check on purpose, so the run has to record
    # whether it ran at all.
    tagger.add("state:graded" if graded else "state:ungraded")

    # A run passes only if both the conversation and the DB check pass. A scenario
    # with no DB check is graded on the conversation alone.
    if failure:
        ctx.fail(reason=failure)
        tagger.fail(reason=failure)
    elif ctx.simulator_verdict.success:
        tagger.success(reason=ctx.simulator_verdict.reason)
    else:
        tagger.fail(reason=ctx.simulator_verdict.reason)


async def on_session_end(ctx: JobContext) -> None:
    try:
        report = ctx.make_session_report()
    except RuntimeError:
        return

    chat = report.chat_history.copy(exclude_function_call=True, exclude_instructions=True)
    if len(chat.items) < 3:
        return

    judges = JudgeGroup(
        llm="openai/gpt-4.1-mini",
        judges=[
            task_completion_judge(),
            accuracy_judge(),
            tool_use_judge(),
            handoff_judge(),
            safety_judge(),
            relevancy_judge(),
            coherence_judge(),
            conciseness_judge(),
        ],
    )
    await judges.evaluate(report.chat_history)

    userdata = ctx.primary_session.userdata

    logger.info("session tags: %s", ctx.tagger.tags)

    try:
        await userdata.db.aclose()
    except Exception:
        logger.exception("error closing hotel DB")


@server.rtc_session(on_session_end=on_session_end, on_simulation_end=on_simulation_end)
async def hotel_guest_support_agent(ctx: JobContext) -> None:
    await ctx.connect()

    userdata = Userdata(db=HotelDB.from_bytes(_SEED_DB_BYTES))
    session = AgentSession[Userdata](
        userdata=userdata,
        # Session-scoped, so every agent and every AgentTask in the call can reach
        # it - a caller can drop off mid-flow, and the followup has to be recordable there.
        tools=[record_followup],
        # An explicit VAD is required (not the bundled default): without it the
        # speaking anchor falls back to the STT stream clock, which drifts into the
        # future across a long call / nested-task switch and makes the turn-commit
        # logic sleep for that offset (~the elapsed call time) before replying.
        vad=inference.VAD(model="silero"),
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("inworld/inworld-tts-2"),
        max_tool_steps=5,
    )

    await session.start(agent=GuestSupportAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
