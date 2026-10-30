from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import ToolError

from bank import (
    BANK_NAME,
    DECLINED_CARD,
    VERIFIED_CARD,
    Alert,
    Bank,
    BankError,
    Identity,
    birth_date,
    card_digits,
)
from conversation import caller_spoke_last, heard

BANK_TZ = ZoneInfo("America/New_York")

# What an inbound caller asked for, which the identity check takes care of once they're verified.
Request = Literal["charge", "lock", "travel", "other"]

DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)


def spoken(digits: str) -> str:
    """Card digits the way Avery says them: "1111" is "one-one-one-one", not "eleven eleven"."""
    return "-".join(DIGIT_WORDS[int(d)] for d in digits)


GREETING = (
    "Hi! I'm Avery, and this is a demo for financial verification. Can I get your name?"
)


def outbound_greeting() -> str:
    return (
        f"Hi, it's Avery, calling for your LiveKit demo. I'll play the fraud team at "
        f"{BANK_NAME}, calling about a purchase on your card. Nothing here is real, so "
        f"just play along. Okay: this is Avery from {BANK_NAME} fraud prevention, about "
        f"recent activity on your card ending in {spoken(VERIFIED_CARD)}. "
        "Who am I speaking with?"
    )


# What the suggested replies give as the caller's details. Any name and birthday work.
SAMPLE_NAME = "Jordan Lee"
SAMPLE_BIRTH_DATE = date(1990, 4, 22)

DEMO_NOTE = """
Play the bank agent. Step out of the role only briefly, starting with something like "Quick demo note", at these moments:
- {moments}
- when they ask about the demo itself, like whether this is real or how it works: answer in a sentence or two, mention that the page shows each step as it happens, and go back to the call.
Never claim to see a real account or anyone's real details.
""".strip()

VOICE_RULES = """
You're on a phone call, so talk like a person, not a document:
- Brisk and warm. One or two short sentences at a time, and one question at a time. No lists, markdown, or symbols.
- Never say you've done something, or that you're about to, without calling its tool in that same reply.
- You know nothing about the caller, the card, or the bank except what a tool returned on this call. Never say whether they're verified, describe a charge, give a case number, or say the card is locked or a notice is set until a tool's result says so. Use its exact merchant, place, amount, case number, and date; never make one up.
- When a tool refuses, fix what it says with your tools. Tell the caller only what changes for them.
""".strip()

GOODBYE = """
When they're all set, ask if there's anything else. Once they say that's all, or answer with a no or a thank-you, say a short goodbye that thanks them for trying the demo, and call end_call in that same reply. Never say goodbye before they've had a chance to say they're done.
""".strip()


class _CardDesk(Agent):
    """What both calls can do on the card once the caller is verified."""

    def __init__(self, bank: Bank, *, instructions: str) -> None:
        self._bank = bank
        self._must_hear: dict[str, str] = {}
        super().__init__(instructions=instructions)

    def _must_tell(self, ctx: RunContext, what: str) -> None:
        self._must_hear[what] = ctx.function_call.call_id

    @function_tool
    async def end_call(self, ctx: RunContext) -> None:
        """Hang up. Only once the caller has nothing else, and after a short goodbye in this
        same reply."""
        await ctx.wait_for_playout()
        items = ctx.session.history.items
        if unheard := [
            w for w, token in self._must_hear.items() if not heard(items, token)
        ]:
            self._must_hear[unheard[0]] = ctx.function_call.call_id
            raise ToolError(f"Don't hang up yet: first tell them {unheard[0]}.")
        if caller_spoke_last(items):
            raise ToolError(
                "Not hung up yet. Answer what they just said with a short goodbye, and call "
                "end_call again in that same reply."
            )
        ctx.speech_handle.add_done_callback(lambda _: ctx.session.shutdown())

    @function_tool
    async def report_purchase(self, ctx: RunContext, made_by_cardholder: bool) -> str:
        """Record whether the cardholder made the flagged purchase. If they didn't, the card is locked and the case goes to loss prevention.

        Args:
            made_by_cardholder: True if the cardholder says they made the purchase, False if they didn't.
        """
        with _as_tool_error():
            card = await self._bank.record_answer(made_by_cardholder)
        if made_by_cardholder:
            return (
                "Marked as the cardholder's purchase. Ask whether they're traveling, "
                "and if so until when."
            )
        self._must_tell(ctx, "that the card is locked and the case number")
        return json.dumps(
            {
                "card": "locked",
                "loss_prevention_case": card.case_id,
                "next": "Tell them the card is locked and the case is with loss "
                "prevention, who will follow up.",
            }
        )

    @function_tool
    async def set_travel_notice(
        self, ctx: RunContext, destination: str, back_on: str
    ) -> str:
        """Note that the cardholder is traveling, so purchases there don't trigger a call.

        Args:
            destination: Where they're traveling, like "Italy".
            back_on: The day they're back home, YYYY-MM-DD.
        """
        try:
            until = date.fromisoformat(back_on)
        except ValueError as e:
            raise ToolError(f"Use YYYY-MM-DD, got {back_on!r}.") from e
        today = _today()
        with _as_tool_error():
            card = await self._bank.set_travel_notice(destination, until, today)
        assert card.travel is not None
        self._must_tell(ctx, "the travel notice")
        return (
            f"Travel notice set for {card.travel.destination} through "
            f"{until:%A, %B} {until.day}, {(until - today).days} days from today. "
            "No calls about purchases there until then."
        )


class CardServices(_CardDesk):
    """Avery taking a call to card services, playing the bank agent for the demo."""

    def __init__(self, bank: Bank) -> None:
        today = _today()
        demo_note = DEMO_NOTE.format(
            moments="after a check that didn't go through, as described below;"
        )
        super().__init__(
            bank,
            instructions=f"""
You are Avery, the voice agent in a LiveKit demo. You play a card services agent at {BANK_NAME}, a made-up bank, to show how a voice agent verifies a caller before it touches their account.
Nothing on this call is real. The caller can give any name and birthday. The card decides the identity check: a card ending in {spoken(VERIFIED_CARD)} gets verified, and any other card, like {spoken(DECLINED_CARD)}, doesn't.
Today is {today:%A, %B} {today.day}, {today.year}.
You've already explained the demo and asked for their name.

{demo_note}

{VOICE_RULES}

Verify them first, and only then ask what they need. You can help with a charge they don't recognize, locking their card, or a travel notice. Anything else, even a balance or an address you can't see, goes to a card specialist once they're verified: never turn it down. If they ask for something before they're verified, say you can help, and that first you'll need to verify them.

To verify them, ask for their name, their date of birth, and the last four digits of their card, one at a time, skipping anything they've already given. Ask with these words:
- Date of birth: "Hi [their name], usually I'd check your details against a database to make sure they match, but today we're just role-playing. Can I get a date of birth from you? It doesn't need to be a real one."
- Card: "Okay, great. And what are the last four digits of your card? We can use four ones for a success, or four zeros for a failure."
As soon as you have all three, call verify_identity with them and what they asked for, if they've said. Call it every time you have all three, even when you think you know how it will come out, like with {spoken(DECLINED_CARD)}: never decide the result yourself.

If the check doesn't go through, say only that you weren't able to verify them with those details. Then add a quick demo note: a real bank wouldn't say which detail was off, and you don't know either, since the bank only answers yes or no. For the demo, a card ending in {spoken(VERIFIED_CARD)} always gets through. Offer to try again. If they want to try again with the same details, call verify_identity again with them right away. If they'd rather give new ones, ask for all three again in one go, say them back in one sentence, and call verify_identity again.
If the check says they're locked out, say you can't verify them on this call and that they can call the number on the back of their card. Then say goodbye and call end_call.

Once they're verified, if they haven't said what they need yet, say exactly: "Great, you're all verified. I can look into a charge you don't recognize, lock your card, or set up a travel notice. What can I help you with?" Otherwise say "Great, you're all verified." Then help with what they asked:
- A charge they don't recognize: verify_identity returns it when that's what they asked for; otherwise call look_up_recent_charges. Describe it right away: where, which city and country, and the amount. Ask whether they made it, and call report_purchase as soon as they say. If it wasn't them, tell them the card is locked now, so nothing else can go through on it, and the case is with the loss prevention team, who will follow up. Give the case number from the result: the letters, then each digit on its own, without saying the dash.
- Locking their card: verify_identity locks it when that's what they asked for; otherwise call lock_card. Once it's locked, tell them nothing new can go through on it.
- A travel notice: ask where they're going and how long they'll be there, until they're back. Call set_travel_notice with the day they're back, counting a length like "two weeks" from today, then tell them they won't get a call about purchases there until then. If they add to or change the dates, call it again.
- Anything else: say you've passed it to a card specialist, who will follow up.

{GOODBYE}
""".strip(),
        )

    async def on_enter(self) -> None:
        self.session.say(GREETING)

    @function_tool
    async def verify_identity(
        self,
        ctx: RunContext,
        full_name: str,
        date_of_birth: str,
        card_last_four: str,
        request: Request,
    ) -> str:
        """Check the caller's details with the bank. Returns only whether they're verified, and whether too many failed checks have locked them out. Once they're verified, it also takes care of what they asked for: it returns the charge they don't recognize, or locks the card.

        Args:
            full_name: The caller's full name.
            date_of_birth: The caller's date of birth, YYYY-MM-DD.
            card_last_four: The last four digits of the card, like 1111.
            request: What they called about: "charge" for a charge they don't recognize, "lock" to lock their card, "travel" for a travel notice, or "other".
        """
        result: dict[str, Any] = {}
        with _as_tool_error():
            if not full_name.strip():
                raise BankError("Ask for their full name first.")
            claimed = Identity(
                name=full_name.strip(),
                date_of_birth=birth_date(date_of_birth, _today()),
                last_four=card_digits(card_last_four),
            )
            result["verified"] = await self._bank.authenticate(claimed)
            if result["verified"] and request == "charge":
                result["recent_charge"] = _charge(await self._bank.recent_charges())
            elif result["verified"] and request == "lock":
                card = await self._bank.lock_card()
                self._must_tell(ctx, "that the card is locked")
                result["card"] = (
                    f"locked: nothing new can go through on the card ending in {card.last_four}"
                )
        if self._bank.locked_out:
            result["locked_out"] = True
        return json.dumps(result)

    @function_tool
    async def look_up_recent_charges(self) -> str:
        """Look up the recent charge on the verified caller's card, when they're asking about one they don't recognize."""
        with _as_tool_error():
            alert = await self._bank.recent_charges()
        return json.dumps({"recent_charge": _charge(alert)})

    @function_tool
    async def lock_card(self, ctx: RunContext) -> str:
        """Lock the verified caller's card, so nothing new can go through on it."""
        with _as_tool_error():
            card = await self._bank.lock_card()
        self._must_tell(ctx, "that the card is locked")
        return f"The card ending in {card.last_four} is locked. Nothing new can go through on it."


class FraudAlert(_CardDesk):
    """Avery calling about a flagged purchase abroad, playing the bank's fraud team for the demo."""

    def __init__(self, bank: Bank) -> None:
        today = _today()
        demo_note = DEMO_NOTE.format(moments="if they seem unsure what this call is;")
        super().__init__(
            bank,
            instructions=f"""
You are Avery, the voice agent in a LiveKit demo. You placed this call to show how a voice agent handles a fraud alert: you play {BANK_NAME}'s fraud prevention team, calling about a purchase the bank flagged on their card ending in {spoken(VERIFIED_CARD)}.
Nothing on this call is real. Any name and birthday they give is fine.
Today is {today:%A, %B} {today.day}, {today.year}.
You've already explained the demo and asked who you're speaking with.

{demo_note}

{VOICE_RULES}

If it's not a good time, say you'll try again later, then say goodbye and call end_call.
Once they've said their name, ask for their date of birth to confirm it's them. Say it back, and call verify_identity with their name and date of birth in that same reply. Never say anything about the purchase until it comes back verified. If it says the date can't be right, ask for it once more.
Once verified, describe the flagged purchase exactly as it came back: where, which city and country, and the amount. Ask whether they made it, and call report_purchase as soon as they answer.
- If they didn't make it: say the card is locked now, so nothing else can go through on it, and that you're forwarding their case to the loss prevention team, who will follow up with them. Give the case number from the result: the letters, then each digit on its own, without saying the dash.
- If they did: ask whether they're traveling, and if so, how long they'll be there, until they're back. Call set_travel_notice with where they are and the day they're back, counting a length like "two weeks" from today, then tell them they won't get another call about purchases there until then. If they aren't traveling, just thank them for confirming.

{GOODBYE}
""".strip(),
        )

    async def on_enter(self) -> None:
        self.session.say(outbound_greeting())

    @function_tool
    async def verify_identity(self, full_name: str, date_of_birth: str) -> str:
        """Confirm the person on the phone is the cardholder. Returns the flagged purchase once they're confirmed.

        Args:
            full_name: The name they gave.
            date_of_birth: The date of birth they gave, YYYY-MM-DD.
        """
        with _as_tool_error():
            if not full_name.strip():
                raise BankError("Ask who you're speaking with first.")
            await self._bank.confirm_cardholder(
                full_name.strip(), birth_date(date_of_birth, _today())
            )
        assert self._bank.alert is not None
        return json.dumps(
            {"verified": True, "flagged_purchase": _charge(self._bank.alert)}
        )


def _charge(alert: Alert) -> dict[str, str]:
    """A flagged purchase the way Avery describes it."""
    purchase = alert.purchase
    return {
        "merchant": purchase.merchant,
        "where": purchase.place,
        "amount": f"${purchase.amount}",
    }


@contextmanager
def _as_tool_error() -> Iterator[None]:
    try:
        yield
    except BankError as e:
        raise ToolError(str(e)) from e


def _today() -> date:
    return datetime.now(BANK_TZ).date()


def _sample_caller(bank_line: str) -> str:
    born = f"{SAMPLE_BIRTH_DATE:%B} {SAMPLE_BIRTH_DATE.day}, {SAMPLE_BIRTH_DATE.year}"
    return f"""
{bank_line} Nothing on the call is real. The caller's details, the only ones to use:
- Full name: {SAMPLE_NAME}
- Date of birth: {born}
- Card ending in: {spoken(VERIFIED_CARD).replace("-", " ")}
""".strip()


def outbound_caller_script(bank: Bank) -> str | None:
    """What the person Avery called knows and wants right now, for the suggested replies."""
    alert = bank.alert
    if alert is None or bank.state == "authenticating":
        return None
    back = _today() + timedelta(days=14)
    on_file = _sample_caller(
        f"Avery called the caller for a demo of {BANK_NAME}'s fraud alerts."
    )
    trip = (
        f"They are on vacation in {alert.purchase.country} for two weeks, back on "
        f"{back:%A, %B} {back.day}, and the flagged purchase at "
        f"{alert.purchase.merchant} in {alert.purchase.place} was theirs."
    )
    if bank.card.travel is not None or bank.card.state == "locked":
        stage = (
            "Everything's sorted. When Avery asks if there's anything else, "
            "say that's all, thanks, and goodbye."
        )
    elif alert.made_by_cardholder:
        stage = (
            "When Avery asks if they're traveling, say they're on vacation in "
            f"{alert.purchase.country} for two weeks."
        )
    elif bank.state == "authenticated":
        stage = "When Avery describes the purchase, say yes, that was them."
    else:
        stage = (
            "They haven't been verified yet. Give their name when Avery asks who she's "
            "speaking with, and the date of birth when she asks for it."
        )
    return f"{on_file}\n{trip}\n{stage}"


def caller_script(bank: Bank) -> str | None:
    """What the caller knows and wants right now, for the suggested replies; nothing while a check runs."""
    if bank.state == "authenticating":
        return None
    on_file = _sample_caller(
        f"The caller is trying out Avery's demo of {BANK_NAME} card services."
    )
    alert = bank.alert
    if bank.locked_out:
        stage = "Avery can't verify them on this call. Say okay, thanks, and goodbye."
    elif bank.card.state == "locked" or bank.card.travel is not None:
        stage = (
            "That's sorted. When Avery asks if there's anything else, say that's all, "
            "thanks, and goodbye."
        )
    elif alert is not None and alert.made_by_cardholder is None:
        stage = "When Avery describes the charge, say that's the one, and they didn't make it."
    elif bank.state == "authenticated":
        stage = (
            "They're verified. Answer what Avery asks. When she asks if there's anything "
            "else, say that's all, thanks, and goodbye."
        )
    elif any(a["authenticated"] is False for a in bank.attempts):
        stage = (
            "The last check didn't go through. If Avery offers to try again, say yes. "
            "When she asks for the details again, give all three in one line, with the "
            f"card ending in {spoken(VERIFIED_CARD).replace('-', ' ')}."
        )
    else:
        stage = (
            "They're calling because there's a charge on their card they don't "
            "recognize: say so when Avery asks how she can help. Then give each detail "
            "as Avery asks for it."
        )
    return f"{on_file}\n{stage}"
