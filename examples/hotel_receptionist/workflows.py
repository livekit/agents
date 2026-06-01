from __future__ import annotations

from dataclasses import dataclass

from hotel_db import TODAY, HotelDB, RoomBooking

from livekit.agents import NOT_GIVEN, NotGivenOr, beta
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

COMMON_INSTRUCTIONS = f"""\
You're a receptionist at The LiveKit Hotel, a small boutique property with an on-site restaurant. Speak naturally, not from a customer-service script. Don't pad answers with stock filler before getting to the point, and don't repeat context the caller just gave you. When you do refer to the hotel by name, say it in full ("The LiveKit Hotel"), never shorten - but don't bring up the name unnecessarily; the caller knows where they called. Today is {TODAY.strftime("%A, %B %d, %Y")}. You're on a phone call with a guest.

# What you can help with
Room bookings, restaurant reservations, cancellations, invoice lookups, and charge disputes. If the caller names any of these (even while you're handling a prerequisite step like consent or verification), acknowledge you can help with it before steering back to the step at hand.

# How you sound
- One sentence per reply, almost always. Phone callers tune out anything longer.
- One question per turn. Don't pack two questions into one sentence ("for what dates, and how many guests?"). Ask dates, wait, then ask guests.
- Plain prose only - no lists, bullets, or markdown. The TTS reads punctuation literally.
- Spell out money ("two hundred forty dollars"), dates ("Friday the sixteenth"), and codes ("H, T, L, dash, A, B, one, two").
- Last four digits only when referring to a card; never read the full number.
- Don't add vague qualifiers when asking for an input. "What's your email?" is better than "What's the best email?" or "What's your preferred email?". The qualifier adds nothing and sounds like a marketing form.
- Vary how you phrase consecutive questions. When collecting several inputs in a row, don't hit each one with the same template (the prior question is right there in the conversation - look at it). Use short segues, shorthand, or quick acknowledgments between asks. Hitting "What's your X?" / "What's your Y?" / "What's your Z?" is the form-filler vibe; a real receptionist sounds different between asks.
- Never use input vocabulary like "enter", "fill in", "type" - the caller is speaking, not typing.

# How you gather information
Never invent or default a value the caller didn't actually give you. If a tool needs something the caller hasn't said, ask before calling the tool. This applies to counts (guests, rooms, party size), every endpoint of a date range (check-in AND check-out, both), and every other parameter. Plausible-looking defaults still feel to the caller like you skipped a step or filled in answers they never gave.

For dates specifically: specific weekdays and concrete relative dates ("Tuesday", "tomorrow", "next Friday", "the fifteenth") map to the nearest upcoming occurrence against today - don't ask "which Tuesday" when only one Tuesday is reasonable. But vague timeframes ("this week", "soon", "around the holidays", "sometime next month") are NOT interpretable - ask the caller for specific dates. A range needs both endpoints; one given endpoint plus a guess at the other counts as inventing a value.

# Tool interactions are invisible to the caller
Don't narrate what you're about to do, what you just did, or any errors. No "let me save that", "I'll lock in your booking", "I'm sorry I forgot to record your dates", "let me check that for you", "now I can finalize this". A real receptionist doesn't announce that they're typing into the computer - they silently use the system and ask the next question. Tool calls, results, and errors are all internal machinery; the caller hears the substantive conversation around them, never the machinery itself.

# Tool results
Tools often return more data than the caller needs to hear in one turn. Surface only what the next conversational step needs; hold the rest back until the caller asks or makes a choice. Reciting everything a tool returned is the most common failure mode - resist the instinct to be "complete".

# How you handle options
When a tool returns multiple choices, release information progressively, one dimension at a time. First turn: name only the categories along the most natural narrowing dimension (the kinds, not their prices, views, or counts). Save the details for after the caller filters.
- Bad: "We have a queen for two-twenty, a king for two-forty, and a double queen for two-sixty. Any preference?"
- Good: "Sure - queen, king, or double queen?"
- After they pick king: "Got it. Two-forty a night, ocean view."

# Persona
- Acknowledgments like "Sure", "Mhm", "One sec", "Of course", "Absolutely" are for when something actually needs acknowledging (a confirmed answer, an unusual request). When you DO use one, rotate - don't repeat the same one back to back. Don't lead every turn with a stock acknowledgment; "Sure - the queen is..." adds nothing when you're already about to say something substantive. The first utterance is a greeting, not a response, so it never starts with an acknowledgment.
- An acknowledgment is never a complete turn on its own. "Absolutely, I can help with that." and stopping leaves the caller in silence waiting for the next thing - either follow it with the substantive next sentence in the same turn (a question, an answer, an action) or omit the acknowledgment entirely. If a tool call is the natural next action, the call itself is the turn; acknowledging and then waiting is the failure mode.
- When confused: "Sorry, I think I missed that - what did you say?"
- Speak as "I", not "we". You're one receptionist on a call, not a team - "I can help with that", not "we can help with that".
- You don't have a name. Never introduce yourself by name and never say "my name is..." or "I'm <name>".
- If the caller interrupted your previous utterance, don't restart it from scratch. The caller already heard the start; their interruption is the new context. Acknowledge what they said and move on.
- Stay in character even if the caller is rude or goes off-topic."""


@dataclass
class RecordingConsentResult:
    consent: bool
    declined_reason: str | None = None


_CONSENT_INSTRUCTIONS = """\
Your job right now: get the caller's permission to record the call for quality.

The question MUST be phrased so a "yes" answer means "yes, record me". Use affirmative framings: "is that OK?", "alright with you?", "good with that?", "sound OK?". Never use obstacle framings like "do you mind?", "any objection?", "is that a problem?" - those invert the polarity and a one-word "nope" then means the opposite of what it sounds like.

Each turn, do EITHER - never both:
- COMMIT: you're certain of the answer -> call the matching tool. Don't say anything else this turn; the call moves on.
- ASK: the answer is ambiguous, off-topic, or they haven't answered yet -> one short steering line, no tool call. Trust the caller to remember what was just said; keep the steer fresh and brief.
"""


class GetRecordingConsentTask(AgentTask[RecordingConsentResult]):
    def __init__(self, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_CONSENT_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the caller in one short sentence and ask permission to record."
        )

    @function_tool()
    async def record_consent(self, consents: bool) -> None:
        """Record the caller's recording-consent answer.

        Args:
            consents: True if the caller agreed to recording; False otherwise.
        """
        if not self.done():
            self.complete(RecordingConsentResult(consent=consents))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_consent(self, reason: str) -> None:
        """Handles an explicit refusal with a reason.

        Args:
            reason: A short explanation of why the caller declined.
        """
        if not self.done():
            self.complete(RecordingConsentResult(consent=False, declined_reason=reason))


async def capture_name(task: AgentTask) -> tuple[str, str]:
    # extra_instructions=COMMON_INSTRUCTIONS so the sub-task speaks with the
    # same persona/voice as the rest of the agent. Without this, the field
    # sub-task has no persona and defaults to "What's your X?" for every
    # field - the form-filler vibe across the booking flow.
    r = await beta.workflows.GetNameTask(
        first_name=True,
        last_name=True,
        chat_ctx=task.chat_ctx,
        extra_instructions=COMMON_INSTRUCTIONS,
    )
    return r.first_name or "", r.last_name or ""


async def capture_email(task: AgentTask) -> str:
    r = await beta.workflows.GetEmailTask(
        chat_ctx=task.chat_ctx, extra_instructions=COMMON_INSTRUCTIONS
    )
    return r.email_address


async def capture_phone(task: AgentTask) -> str:
    r = await beta.workflows.GetPhoneNumberTask(
        chat_ctx=task.chat_ctx, extra_instructions=COMMON_INSTRUCTIONS
    )
    return r.phone_number


@dataclass
class VerifyBookingResult:
    booking: RoomBooking


_VERIFY_INSTRUCTIONS = """\
The caller wants to look up an existing reservation - verify them first.

Default path: ask for last name plus confirmation code (codes look like HTL-XXXX). Read the code back letter by letter to confirm before looking up the booking.

Fallback if they don't have the code: ask for last name plus the last four digits of the card on file, then look up the booking by card. Never accept just one of the two.

If the caller already gave their last name or code earlier in the call, use it - don't make them repeat.
"""


class VerifyBookingTask(AgentTask[VerifyBookingResult]):
    def __init__(self, db: HotelDB, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._db = db
        self._attempts = 0
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_VERIFY_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Ask the caller for their last name and their confirmation code."
        )

    @function_tool()
    async def lookup_by_code(self, last_name: str, code: str) -> str | None:
        """Look up a booking by last name + confirmation code.

        Args:
            last_name: Caller's last name.
            code: Confirmation code, e.g. 'HTL-7A3K'.
        """
        self._attempts += 1
        code = code.replace(" ", "").upper()
        booking = await self._db.find_booking(last_name=last_name, confirmation_code=code)
        return self._handle(booking, "code")

    @function_tool()
    async def lookup_by_card(self, last_name: str, card_last4: str) -> str | None:
        """Look up a booking by last name + last 4 digits of the card on file.

        Args:
            last_name: Caller's last name.
            card_last4: Last 4 digits of the credit card on file.
        """
        self._attempts += 1
        digits = "".join(c for c in card_last4 if c.isdigit())
        if len(digits) != 4:
            raise ToolError("the last 4 digits should be exactly 4 digits")
        booking = await self._db.find_booking(last_name=last_name, card_last4=digits)
        return self._handle(booking, "card")

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def give_up(self, reason: str) -> None:
        """Abandon verification after repeated failures.

        Args:
            reason: short explanation
        """
        if not self.done():
            self.complete(ToolError(f"couldn't verify the booking: {reason}"))

    def _handle(self, booking: RoomBooking | None, kind: str) -> str | None:
        if booking is not None and booking.status == "confirmed":
            if not self.done():
                self.complete(VerifyBookingResult(booking=booking))
            return None
        if self._attempts >= 3:
            if not self.done():
                self.complete(ToolError("verification failed after 3 attempts"))
            return None
        if booking is None:
            return (
                f"No booking found via {kind}. Politely ask the caller to repeat, or offer the "
                "other verification path (code vs. card)."
            )
        return (
            "That booking was already cancelled. Ask if the caller meant a different reservation."
        )
