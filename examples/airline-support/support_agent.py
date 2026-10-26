from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field
from skyline import (
    BAG_DAILY_ALLOWANCE_EUR,
    CHANGE_FEE_EUR,
    LOST_BAG_CAP_EUR,
    POLICY,
    Booking,
    Skyline,
)

from livekit.agents import (
    Agent,
    AgentServer,
    InputRequired,
    JobContext,
    LiveSession,
    RunContext,
    RunResult,
    ToolError,
    cli,
    persisted,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

AUTO_PAY_LIMIT_EUR = 400  # above it a person signs the payment off
MAX_ROUNDS = 2  # drafts the reviewer sees before a person takes over
MAX_ASKS = 3  # questions to the customer before a person takes over

Kind = Literal["delay_or_cancellation", "baggage", "refund", "change", "spam"]
Team = Literal["claims", "baggage", "fares"]
TEAM: dict[Kind, Team] = {
    "delay_or_cancellation": "claims",
    "baggage": "baggage",
    "refund": "fares",
    "change": "fares",
}
BOOKING_REF = re.compile(r"\b(?=[A-Z]*\d)[A-Z0-9]{6}\b")
PIR_REF = re.compile(r"\b[A-Z]{5}\d{5}\b")


class Email(BaseModel):
    sender: str = Field(description="the customer's email address")
    subject: str
    body: str = Field(description="what the customer wrote, word for word")


class Classification(BaseModel):
    kind: Kind
    booking_ref: str | None = None
    pir_ref: str | None = None


class Verdict(BaseModel):
    approved: bool
    objection: str = Field(default="", description="what is wrong, empty when approved")


class TriageResult(BaseModel):
    action: Literal["resolved", "to_human", "close"]
    team: Team | None
    priority: Literal["low", "normal", "high"]
    paid_eur: int
    reply_draft: str | None
    reason: str


@dataclass
class Ticket:
    """The customer's request, saved between separate agent invocations."""

    email: str = ""
    sender: str = ""
    kind: Kind | None = None
    booking_ref: str | None = None
    booking: Booking | None = None
    pir_ref: str | None = None
    asked: int = 0
    amount_eur: int = 0
    facts: list[str] = field(default_factory=list)
    draft: str | None = None
    review_notes: list[str] = field(default_factory=list)
    rounds: int = 0

    def end(
        self,
        action: Literal["resolved", "to_human", "close"],
        reason: str,
        *,
        priority: Literal["low", "normal", "high"] = "normal",
        paid_eur: int = 0,
    ) -> TriageResult:
        return TriageResult(
            action=action,
            team=TEAM.get(self.kind) if self.kind else None,
            priority=priority,
            paid_eur=paid_eur,
            reply_draft=self.draft,
            reason=reason,
        )

    def brief(self) -> str:
        lines = [f"Customer email:\n{self.email}"]
        if self.booking is not None:
            b = self.booking
            lines.append(
                f"Booking {b.ref}: {b.passenger}, flights {', '.join(b.flights)}, "
                f"{b.fare_class} fare, EUR {b.price_eur}"
            )
        lines += [f"Fact: {fact}" for fact in self.facts]
        lines.append(f"Amount due to the customer: EUR {self.amount_eur}")
        return "\n".join(lines)


LLM = "google/gemma-4-31b-it"
T = TypeVar("T")


async def call(agent: Agent[Any, T], prompt: str) -> T:
    """An agent called as a function: a fresh conversation, a typed output."""
    output = (await agent.run(prompt)).output
    if output is None or isinstance(output, InputRequired):
        raise ToolError("The specialist needs more information; ask the customer before retrying.")
    return output


classifier = Agent(
    instructions=(
        "You triage customer emails for Skyline Air. kind: delay_or_cancellation for a late or "
        "cancelled flight, baggage for a late, damaged or lost bag, refund for money back on a "
        "ticket, change for another date or flight, spam for marketing, phishing, automated "
        "mail, a duplicate, or a message with no request. booking_ref: the six-character "
        "booking reference, like SKY4F2; pir_ref: the baggage report reference, like "
        "AMSSK10422. Copy references exactly as written; null when absent, never a guess."
    ),
    llm=LLM,
    output_type=Classification,
)
writer = Agent(
    instructions=(
        "You write Skyline Air's email replies to customers: plain, warm, under 120 words, "
        "signed Skyline Air Customer Care, no subject line. Use only the facts given and state "
        "the amount exactly as given; an amount due is being paid to the original payment "
        f"method. Answer with the reply's text only.\n\n{POLICY}"
    ),
    llm=LLM,
    output_type=str,
)
reviewer = Agent(
    instructions=(
        "You review a Skyline Air reply before it is sent. Approve only if every fact and "
        "amount in it matches the facts given, it follows the policy, it promises nothing "
        "more, and it answers what the customer asked. Otherwise give the one objection that "
        f"matters most, in one sentence.\n\n{POLICY}"
    ),
    llm=LLM,
    output_type=Verdict,
)
skyline = Skyline.seeded()


@dataclass
class SupportContext:
    ticket: Ticket = persisted(default_factory=Ticket)


def owed(ticket: Ticket, booking: Booking, due: int) -> int:
    """Remaining entitlement after payments already made for the booking."""
    paid = skyline.paid_eur(booking.ref)
    if paid:
        ticket.facts.append(f"EUR {paid} was already paid on this booking")
    return max(due - paid, 0)


def eu261_band(distance_km: int) -> int:
    return 250 if distance_km <= 1500 else 400 if distance_km <= 3500 else 600


def ask(ticket: Ticket, question: str) -> InputRequired | TriageResult:
    if ticket.asked >= MAX_ASKS:
        return ticket.end("to_human", "the booking could not be identified")
    ticket.asked += 1
    return InputRequired(question)


async def resolve_ticket(ticket: Ticket) -> TriageResult | InputRequired:
    if ticket.kind == "spam":
        return ticket.end("close", "spam, or nothing to act on", priority="low")
    booking = skyline.find(ticket.booking_ref, ticket.sender) if ticket.booking_ref else None
    if booking is None:
        return ask(ticket, "What is your six-character booking reference, such as SKY4F2?")
    ticket.booking = booking
    if question := calculate_due(ticket, booking):
        return ask(ticket, question.question)
    for _ in range(MAX_ROUNDS):
        prompt = ticket.brief()
        if ticket.draft and ticket.review_notes:
            prompt += f"\nYour last draft: {ticket.draft}\nReviewer: {ticket.review_notes[-1]}"
        ticket.draft = await call(writer, prompt)
        ticket.rounds += 1
        verdict = await call(reviewer, f"{ticket.brief()}\nDraft reply: {ticket.draft}")
        if not verdict.approved:
            ticket.review_notes.append(verdict.objection)
            continue
        if ticket.amount_eur == 0:
            return ticket.end("resolved", "nothing to pay")
        if ticket.amount_eur > AUTO_PAY_LIMIT_EUR:
            return ticket.end(
                "to_human", f"EUR {ticket.amount_eur} requires human approval", priority="high"
            )
        reason = "; ".join(ticket.facts)
        skyline.pay(booking.ref, ticket.amount_eur, reason)
        return ticket.end("resolved", reason, paid_eur=ticket.amount_eur)
    return ticket.end("to_human", "the reply failed review twice")


def calculate_due(ticket: Ticket, booking: Booking) -> InputRequired | None:
    ticket.facts.clear()
    ticket.amount_eur = 0
    if ticket.kind == "delay_or_cancellation":
        due = 0
        for number in booking.flights:
            f = skyline.flights[number]
            route = f"{f.number} {f.origin}-{f.destination} ({f.distance_km} km)"
            if f.actual_arrival is None:
                notice = f.cancellation_notice_days
                disrupted = notice is None or notice < 14
                ticket.facts.append(f"{route} was cancelled {notice} days ahead: {f.cause}")
            else:
                disrupted = f.delay >= timedelta(hours=3)
                minutes = int(f.delay.total_seconds()) // 60
                ticket.facts.append(
                    f"{route} arrived {minutes // 60}h{minutes % 60:02d} late: {f.cause or 'on time'}"
                )
            if disrupted and f.extraordinary:
                ticket.facts.append(f"{f.cause} is an extraordinary circumstance: nothing due")
            elif disrupted:
                due = max(due, eu261_band(f.distance_km))
        ticket.amount_eur = owed(ticket, booking, due)
    elif ticket.kind == "baggage":
        bag = skyline.bags.get((ticket.pir_ref or "").strip().upper())
        if bag is None or bag.booking_ref != booking.ref:
            return InputRequired("What is the baggage report reference, such as AMSSK10422?")
        b = bag
        if b.status == "lost":
            due = min(b.declared_value_eur, LOST_BAG_CAP_EUR)
            ticket.facts.append(f"bag {b.pir} is lost, declared value EUR {b.declared_value_eur}")
        else:
            due = min(b.receipts_eur, b.days_late * BAG_DAILY_ALLOWANCE_EUR)
            ticket.facts.append(
                f"bag {b.pir} is {b.status.replace('_', ' ')} after {b.days_late} days, receipts EUR {b.receipts_eur}"
            )
        ticket.amount_eur = owed(ticket, booking, due)
    else:
        b = booking
        if ticket.kind == "refund":
            ticket.facts.append(
                f"the {b.fare_class} fare is {('' if b.refundable else 'not ')}refundable"
            )
            ticket.amount_eur = owed(ticket, b, b.price_eur if b.refundable else 0)
        else:
            ticket.facts.append(
                f"changing a {b.fare_class} fare costs EUR {CHANGE_FEE_EUR[b.fare_class]} plus any fare difference; the customer changes it in Manage booking"
            )
    return None


support = Agent[SupportContext](
    instructions=(
        "You are Skyline Air's customer care, by text: one or two plain sentences. A late or "
        "cancelled flight, a bag, a refund or a change goes to triage with what the customer "
        "wrote, word for word, and the address they write from; ask for that address first if "
        "they have not given it. When triage answers resolved, send its reply_draft as it is. "
        "On to_human the draft is a colleague's to send, never yours: say a colleague will "
        "answer by email, and to any question about money, that the colleague will confirm "
        "it. On close, say you can only help with Skyline Air trips. A question about an "
        "answer already given is answered from it; only a new request goes to triage again. "
        "Use continue_request with newly supplied booking or baggage references after a question. "
        "Never decide, compute or promise anything yourself."
    ),
    llm=LLM,
)


@support.tool
async def triage(ctx: RunContext[SupportContext], email: Email) -> TriageResult | InputRequired:
    """Start investigating a new customer request, with the original email and sender."""
    ticket = Ticket(
        sender=email.sender, email=f"From: {email.sender}\nSubject: {email.subject}\n\n{email.body}"
    )
    classification = await call(classifier, ticket.email)
    ticket.kind = classification.kind
    ticket.booking_ref = classification.booking_ref
    ticket.pir_ref = classification.pir_ref
    ctx.context.ticket = ticket
    outcome = await resolve_ticket(ticket)
    await ctx.store.save_context(ctx.context)
    return outcome


@support.tool
async def continue_request(
    ctx: RunContext[SupportContext],
    booking_ref: str | None = None,
    pir_ref: str | None = None,
) -> TriageResult | InputRequired:
    """Continue the current request using references supplied by the customer."""
    ticket = ctx.context.ticket
    if not ticket.email:
        raise ToolError("Start triage with the customer email first")
    if booking_ref is not None:
        ticket.booking_ref = booking_ref.strip().upper()
    if pir_ref is not None:
        ticket.pir_ref = pir_ref.strip().upper()
    outcome = await resolve_ticket(ticket)
    await ctx.store.save_context(ctx.context)
    return outcome


server = AgentServer(storage=AgentDBStorage(), agent_name="skyline-support")


@server.a2a_session(endpoint="triage", description="Skyline Air customer care.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    app = SupportContext()
    await ctx.store.load_context(app)
    result = await support.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result


@server.rtc_session()
async def chat(ctx: JobContext) -> None:
    session = LiveSession(llm=LLM, context=SupportContext(), thread=ctx.thread)
    await session.start(agent=support, room=ctx.room)
    await session.store.load_context(session.context)
    if not session.history.items:
        await session.generate_reply(instructions="Greet the customer in one sentence.")


if __name__ == "__main__":
    cli.run_app(server)
