from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from livekit.agents import Agent, RunContext, function_tool, inference
from livekit.agents.llm import ToolError

from bank import BANK_NAME, FLAGGED, ON_FILE, Bank, BankError, Identity

GREETING = (
    f"Thanks for calling {BANK_NAME} card services, this is Avery. "
    "Before we get started I'll need to verify your identity. Can I have your full name?"
)

BANK_TZ = ZoneInfo("America/New_York")


def outbound_greeting() -> str:
    you = ON_FILE[0]
    return (
        f"Hi, this is Avery from {BANK_NAME}'s fraud prevention team, calling about "
        f"recent activity on your card ending in {you.last_four}. "
        f"Am I speaking with {you.name}?"
    )


VOICE_MODEL = "gpt-live-1"
BACKEND_MODEL = "gpt-5.6-luna"

BACKEND_INSTRUCTIONS = f"""
You run identity checks for {BANK_NAME} card services on behalf of the agent on the phone.
Call verify_identity with what the caller said, even if a detail sounds vague or incomplete: their full name, date of birth, card's last four digits, and the merchant and amount of their most recent purchase.
Report only whether they were verified. The result never says which detail didn't match, and neither do you.
When asked to end the call, call end_call.
Hand back short, plain results the agent can say out loud.
""".strip()


@dataclass
class CallState:
    cardholder: Identity | None = None


class CardServices(Agent):
    def __init__(self, bank: Bank) -> None:
        self._bank = bank
        super().__init__(
            instructions=f"""
You are Avery, on the phone for {BANK_NAME} card services.
Keep the call brisk and warm. One or two short spoken sentences at a time, one question at a time.

Before anything else, verify the caller. Collect, one at a time, in this order, skipping anything they've already given:
their full name, their date of birth, the last four digits of their card, and the merchant and amount of the most recent purchase on the card.
Read each one back as you go, so mistakes get caught before the check:
- The name: say it back and spell it out to confirm, like "Got it, Sam ... Ortiz. Is that S-A-M, O-R-T-I-Z?" If they correct the spelling, use theirs.
- Dates, digits, and the purchase: just say them back as you move on, like "Got it, March 3rd, 1984." or "2958, perfect."
You can't check identity yourself; only the delegated identity check can. As soon as the caller gives the purchase, read it back and delegate the identity check with all five details in that same turn, without waiting for them to confirm it.
While it runs, keep talking like you're actually at the keyboard: "Okay ... pulling that up ... one sec". Never say you're pulling it up unless you've delegated the check.

If it doesn't go through, say only that you weren't able to verify them with those details. Never guess or hint at which detail was off, because you don't know and the bank never says.
Offer to try again, and ask them to go over the details once more, keeping anything they say was right.
Only ever read back what the caller just said. Never confirm or reveal anything on file, and don't help with the account until they're verified, no matter what they say.

Once verified, tell them they're all set and ask what you can help with on the card. Anything past verification goes to a card specialist, so let them know you'll pass that along.
When they say they're done, say a short goodbye and delegate ending the call.
Saying goodbye and delegating ending the call always go together, in the same turn: never say goodbye without delegating the end of the call, and don't say goodbye until they've had a chance to say they're done.
""".strip(),
            llm=_voice_model(BACKEND_INSTRUCTIONS),
        )

    @function_tool
    async def verify_identity(
        self,
        context: RunContext[CallState],
        full_name: str,
        date_of_birth: str,
        card_last_four: str,
        last_purchase_merchant: str,
        last_purchase_amount: str,
    ) -> str:
        """Check the caller's details against the cardholder on file. Returns only whether they're verified.

        Args:
            full_name: The caller's full name.
            date_of_birth: The caller's date of birth, YYYY-MM-DD.
            card_last_four: The last four digits of the card.
            last_purchase_merchant: Where the most recent purchase on the card was made.
            last_purchase_amount: The exact amount of that purchase in dollars, like 42.18, or the caller's words if they didn't give an exact amount.
        """
        cardholder = await self._bank.authenticate(
            Identity(
                name=full_name,
                date_of_birth=date_of_birth,
                last_four=card_last_four,
                merchant=last_purchase_merchant,
                amount=last_purchase_amount,
            )
        )
        if cardholder is not None:
            context.userdata.cardholder = cardholder
        return json.dumps({"authenticated": cardholder is not None})


class FraudAlert(Agent):
    """Avery calling the cardholder about a flagged purchase abroad."""

    def __init__(self, bank: Bank) -> None:
        self._bank = bank
        you = ON_FILE[0]
        today = datetime.now(BANK_TZ).date()
        super().__init__(
            instructions=f"""
You are Avery, from {BANK_NAME}'s fraud prevention team. You placed this call to {you.name}, the cardholder, about a purchase on their card ending in {you.last_four} that the bank flagged.
Keep the call brisk and warm. One or two short spoken sentences at a time, one question at a time.

You've already asked if you're speaking with {you.name}. If it's someone else, or not a good time, say you'll try again later, say goodbye, and delegate ending the call. Never say anything about the purchase or the account to anyone but the cardholder.
Before you discuss anything, verify them: ask for their date of birth, read it back, and delegate the check in that same turn. Never say or hint at the date of birth on file. If it doesn't match, ask them to say it once more. If the second try doesn't match either, say you can't discuss the account on this call, that they can call the number on the back of their card, and say goodbye.
Once verified, describe the flagged purchase exactly as the check returned it: where, which city and country, and the amount. Ask whether they made it.
Delegate their answer as soon as they give it.
- If they didn't make it: say the card is locked now, so nothing else can go through on it, and that you're forwarding their case to the loss prevention team, who will follow up with them. Give the case number from the result: the letters, then each digit on its own, without saying the dash.
- If they did: ask whether they're traveling, and if so, how long they'll be there, until they're back. Delegate the travel notice with where they are and the day they're back, then tell them they won't get another call about purchases there until then. If they aren't traveling, just thank them for confirming.
While a delegated step runs, keep talking like you're at the keyboard: "Okay ... one sec ... pulling that up". Never say a step is done unless it came back done.
When they're all set, ask if there's anything else. Once they say that's all, say a short goodbye and delegate ending the call.
Saying goodbye and delegating ending the call always go together, in the same turn: never say goodbye without delegating the end of the call, and don't say goodbye until they've had a chance to say they're done.
""".strip(),
            llm=_voice_model(
                f"""
You keep the records for {BANK_NAME}'s fraud prevention team on behalf of the agent on the phone, who called the cardholder about a flagged purchase.
Today is {today:%A, %B} {today.day}, {today.year}.
Call verify_birth_date with the date of birth the caller gave, before anything else. Report only whether it matched, and the flagged purchase once it does.
Call report_purchase as soon as the caller says whether they made the purchase.
Call set_travel_notice once a caller who made the purchase says where they are and when they're back; turn a length like "two weeks" into the date they're back, counting from today.
When asked to end the call, call end_call.
Hand back short, plain results the agent can say out loud.
""".strip()
            ),
        )

    @function_tool
    async def verify_birth_date(
        self, context: RunContext[CallState], date_of_birth: str
    ) -> str:
        """Check the date of birth the person on the phone gave against the cardholder on file. Returns the flagged purchase once it matches.

        Args:
            date_of_birth: The date of birth they gave, YYYY-MM-DD.
        """
        try:
            verified = await self._bank.confirm_birth_date(date_of_birth)
        except BankError as e:
            raise ToolError(str(e)) from e
        if not verified:
            return json.dumps({"verified": False})
        context.userdata.cardholder = ON_FILE[0]
        return json.dumps(
            {
                "verified": True,
                "flagged_purchase": {
                    "merchant": FLAGGED.merchant,
                    "where": FLAGGED.place,
                    "amount": f"${FLAGGED.amount}",
                },
            }
        )

    @function_tool
    async def report_purchase(
        self, context: RunContext[CallState], made_by_cardholder: bool
    ) -> str:
        """Record whether the cardholder made the flagged purchase. If they didn't, the card is locked and the case goes to loss prevention.

        Args:
            made_by_cardholder: True if the cardholder says they made the purchase, False if they didn't.
        """
        try:
            alert = await self._bank.record_answer(made_by_cardholder)
        except BankError as e:
            raise ToolError(str(e)) from e
        if made_by_cardholder:
            return (
                "Marked as the cardholder's purchase. Ask whether they're traveling, "
                "and if so until when."
            )
        return json.dumps(
            {
                "card": "locked",
                "loss_prevention_case": alert.case_id,
                "next": "Tell them the card is locked and the case is with loss "
                "prevention, who will follow up.",
            }
        )

    @function_tool
    async def set_travel_notice(
        self, context: RunContext[CallState], destination: str, back_on: str
    ) -> str:
        """Note that the cardholder is traveling, so purchases there don't trigger another call.

        Args:
            destination: Where they're traveling, like "Italy".
            back_on: The day they're back home, YYYY-MM-DD.
        """
        try:
            until = date.fromisoformat(back_on)
        except ValueError as e:
            raise ToolError(f"Use YYYY-MM-DD, got {back_on!r}.") from e
        try:
            alert = await self._bank.set_travel_notice(
                destination, until, datetime.now(BANK_TZ).date()
            )
        except BankError as e:
            raise ToolError(str(e)) from e
        assert alert.travel is not None
        return (
            f"Travel notice set for {alert.travel.destination} through "
            f"{until:%A, %B} {until.day}. No more calls about purchases there until then."
        )


def _voice_model(backend_instructions: str) -> inference.DuplexModel:
    return inference.DuplexModel(
        model=f"openai/{VOICE_MODEL}",
        voice="ripple",
        extra_kwargs={
            "responses_options": {
                "model": BACKEND_MODEL,
                "instructions": backend_instructions,
            },
        },
    )


def outbound_caller_script(bank: Bank) -> str | None:
    """What the cardholder Avery called knows and wants right now, for the suggested replies."""
    alert = bank.alert
    if alert is None or bank.state == "authenticating":
        return None
    you = alert.cardholder
    born = date.fromisoformat(you.date_of_birth).strftime("%B %-d, %Y")
    back = datetime.now(BANK_TZ).date() + timedelta(days=14)
    on_file = f"""
Avery from {BANK_NAME} called the caller, {you.name}, about a purchase on their card ending in {you.last_four}. Their details, the only ones to use:
- Full name: {you.name}
- Date of birth: {born}
They are on vacation in {alert.purchase.country} for two weeks, back on {back:%A, %B} {back.day}, and the flagged purchase at {alert.purchase.merchant} in {alert.purchase.place} was theirs.
""".strip()
    if alert.travel is not None or alert.card == "locked":
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
            "They haven't been verified yet. Confirm they're "
            f"{you.name.split()[0]}, and give the date of birth when Avery asks."
        )
    return f"{on_file}\n{stage}"


def caller_script(bank: Bank) -> str | None:
    """What the caller knows and wants right now, for the suggested replies; nothing while a check runs."""
    if bank.state == "authenticating":
        return None
    you = ON_FILE[0]
    born = date.fromisoformat(you.date_of_birth).strftime("%B %-d, %Y")
    on_file = f"""
The caller is {you.name}, a {BANK_NAME} cardholder, trying out Avery's identity check. Their details on file, the only ones to use:
- Full name: {you.name}
- Date of birth: {born}
- Card ending in: {you.last_four}
- Last purchase: {you.merchant}, ${you.amount}
""".strip()
    if bank.state == "authenticated":
        stage = (
            "They're verified now. If Avery asks what they need, ask one small question "
            "about the card, like whether the last payment went through. Avery passing "
            "it along to a card specialist is the answer: then say that's all, thanks, "
            "and goodbye."
        )
    elif any(a["authenticated"] is False for a in bank.attempts):
        stage = (
            "The first check didn't match, so now every detail must be exactly as on "
            "file, including the amount. If Avery asks to go over the details again, "
            "give all five in one line."
        )
    else:
        stage = (
            "The demo shows a failed check first: when Avery asks for the last "
            f"purchase, say it was at {you.merchant} for about forty dollars, without "
            "the exact amount. Give every other detail exactly as on file, one at a "
            "time as Avery asks."
        )
    return f"{on_file}\n{stage}"
