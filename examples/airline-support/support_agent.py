from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, Field
from skyline import (
    BAG_DAILY_ALLOWANCE_EUR,
    CHANGE_FEE_EUR,
    LOST_BAG_CAP_EUR,
    POLICY,
    BaggageFile,
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
    cli,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.graph import BaseNode, Edge, End, Graph, GraphRunContext
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
    """The graph's state, saved while it waits on the customer."""

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
    ) -> End[TriageResult]:
        return End(
            TriageResult(
                action=action,
                team=TEAM.get(self.kind) if self.kind else None,
                priority=priority,
                paid_eur=paid_eur,
                reply_draft=self.draft,
                reason=reason,
            )
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


async def call(agent: Agent[Any, Any, T], prompt: str) -> T:
    """An agent called as a function: a fresh conversation, a typed output."""
    output = (await agent.run(prompt)).output
    assert output is not None  # these agents have no tools, so nothing asks for input
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
class Classify(BaseNode[Ticket, None, TriageResult]):
    email: Email

    async def run(
        self, ctx: GraphRunContext[Ticket]
    ) -> LookUpBooking | Annotated[End[TriageResult], Edge(label="close")]:
        ticket = ctx.state
        ticket.sender = self.email.sender
        ticket.email = (
            f"From: {self.email.sender}\nSubject: {self.email.subject}\n\n{self.email.body}"
        )
        triage = await call(classifier, ticket.email)
        ticket.kind = triage.kind
        ticket.booking_ref, ticket.pir_ref = triage.booking_ref, triage.pir_ref
        if triage.kind == "spam":
            return ticket.end("close", "spam, or nothing to act on", priority="low")
        return LookUpBooking()


@dataclass
class LookUpBooking(BaseNode[Ticket, None, TriageResult]):
    async def run(
        self, ctx: GraphRunContext[Ticket]
    ) -> (
        EU261
        | Baggage
        | FareRules
        | InputRequired
        | Annotated[End[TriageResult], Edge(label="to_human")]
    ):
        ticket = ctx.state
        if ctx.reply:  # the answer to this node's question
            said = ctx.reply.upper()
            if found := BOOKING_REF.search(said):
                ticket.booking_ref = found.group()
            if found := PIR_REF.search(said):
                ticket.pir_ref = found.group()

        ref = ticket.booking_ref
        booking = skyline.find(ref, ticket.sender) if ref else None
        if booking is None:
            if ref is None:
                return ask(
                    ticket, "What is your booking reference? It has six characters, like SKY4F2."
                )
            return ask(
                ticket,
                f"I could not find booking {ref} under {ticket.sender}. Could you "
                "check the booking reference?",
            )

        ticket.booking = booking
        if ticket.kind == "delay_or_cancellation":
            return EU261(booking)
        if ticket.kind == "baggage":
            bag = skyline.bags.get((ticket.pir_ref or "").strip().upper())
            if bag is None or bag.booking_ref != booking.ref:
                return ask(
                    ticket,
                    "What is the reference on the report the baggage desk gave you? It looks "
                    "like AMSSK10422.",
                )
            return Baggage(booking, bag)
        return FareRules(booking, refund=ticket.kind == "refund")


def ask(ticket: Ticket, question: str) -> InputRequired | End[TriageResult]:
    """The customer is asked, up to MAX_ASKS times; then a person takes over."""
    if ticket.asked >= MAX_ASKS:
        return ticket.end("to_human", "the booking could not be identified")
    ticket.asked += 1
    return InputRequired(question)


def owed(ticket: Ticket, booking: Booking, due: int) -> int:
    """What is still due once earlier payments on the booking are taken off."""
    if paid := skyline.paid_eur(booking.ref):
        ticket.facts.append(f"EUR {paid} was already paid on this booking")
    return max(due - paid, 0)


def eu261_band(distance_km: int) -> int:
    return 250 if distance_km <= 1500 else 400 if distance_km <= 3500 else 600


@dataclass
class EU261(BaseNode[Ticket]):
    booking: Booking

    async def run(self, ctx: GraphRunContext[Ticket]) -> DraftReply:
        due = 0
        for number in self.booking.flights:
            f = skyline.flights[number]
            route = f"{f.number} {f.origin}-{f.destination} ({f.distance_km} km)"
            if f.actual_arrival is None:
                notice = f.cancellation_notice_days
                disrupted = notice is None or notice < 14
                ctx.state.facts.append(f"{route} was cancelled {notice} days ahead: {f.cause}")
            else:
                disrupted = f.delay >= timedelta(hours=3)
                minutes = int(f.delay.total_seconds()) // 60
                ctx.state.facts.append(
                    f"{route} arrived {minutes // 60}h{minutes % 60:02d} late: {f.cause or 'on time'}"
                )
            if disrupted and f.extraordinary:
                ctx.state.facts.append(f"{f.cause} is an extraordinary circumstance: nothing due")
            elif disrupted:
                due = max(due, eu261_band(f.distance_km))
        ctx.state.amount_eur = owed(ctx.state, self.booking, due)
        return DraftReply()


@dataclass
class Baggage(BaseNode[Ticket]):
    booking: Booking
    bag: BaggageFile

    async def run(self, ctx: GraphRunContext[Ticket]) -> DraftReply:
        b = self.bag
        if b.status == "lost":
            due = min(b.declared_value_eur, LOST_BAG_CAP_EUR)
            ctx.state.facts.append(
                f"bag {b.pir} is lost, declared value EUR {b.declared_value_eur}"
            )
        else:
            due = min(b.receipts_eur, b.days_late * BAG_DAILY_ALLOWANCE_EUR)
            ctx.state.facts.append(
                f"bag {b.pir} is {b.status.replace('_', ' ')} after {b.days_late} days, "
                f"receipts EUR {b.receipts_eur}"
            )
        ctx.state.amount_eur = owed(ctx.state, self.booking, due)
        return DraftReply()


@dataclass
class FareRules(BaseNode[Ticket]):
    booking: Booking
    refund: bool

    async def run(self, ctx: GraphRunContext[Ticket]) -> DraftReply:
        b = self.booking
        if self.refund:
            ctx.state.facts.append(
                f"the {b.fare_class} fare is {'' if b.refundable else 'not '}refundable"
            )
            ctx.state.amount_eur = owed(ctx.state, b, b.price_eur if b.refundable else 0)
        else:
            ctx.state.facts.append(
                f"changing a {b.fare_class} fare costs EUR {CHANGE_FEE_EUR[b.fare_class]} plus any "
                "fare difference; the customer changes it in Manage booking"
            )
        return DraftReply()


@dataclass
class DraftReply(BaseNode[Ticket]):
    objection: str | None = None

    async def run(self, ctx: GraphRunContext[Ticket]) -> Review:
        prompt = ctx.state.brief()
        if self.objection and ctx.state.draft:
            prompt += f"\n\nYour last draft:\n{ctx.state.draft}\n\nThe reviewer: {self.objection}"
        draft = await call(writer, prompt)
        ctx.state.draft = draft
        ctx.state.rounds += 1
        return Review(draft)


@dataclass
class Review(BaseNode[Ticket, None, TriageResult]):
    draft: str

    async def run(
        self, ctx: GraphRunContext[Ticket]
    ) -> (
        Annotated[DraftReply, Edge(label="rejected")]
        | Annotated[Pay, Edge(label="approved, EUR 400 at most")]
        | Annotated[End[TriageResult], Edge(label="resolved, to_human")]
    ):
        ticket = ctx.state
        verdict = await call(reviewer, f"{ticket.brief()}\n\nDraft reply:\n{self.draft}")
        if not verdict.approved:
            ticket.review_notes.append(verdict.objection)
            if ticket.rounds >= MAX_ROUNDS:
                return ticket.end("to_human", f"the reply failed review twice: {verdict.objection}")
            return DraftReply(objection=verdict.objection)
        if ticket.amount_eur == 0 or ticket.booking is None:
            return ticket.end("resolved", "nothing to pay")
        if ticket.amount_eur > AUTO_PAY_LIMIT_EUR:
            return ticket.end(
                "to_human",
                f"EUR {ticket.amount_eur} is above the EUR {AUTO_PAY_LIMIT_EUR} paid without a person",
                priority="high",
            )
        return Pay(ticket.booking.ref, ticket.amount_eur)


@dataclass
class Pay(BaseNode[Ticket, None, TriageResult]):
    booking_ref: str
    eur: int

    async def run(
        self, ctx: GraphRunContext[Ticket]
    ) -> Annotated[End[TriageResult], Edge(label="resolved")]:
        reason = "; ".join(ctx.state.facts)
        skyline.pay(self.booking_ref, self.eur, reason)
        return ctx.state.end("resolved", f"paid EUR {self.eur}: {reason}", paid_eur=self.eur)


class Triage(Graph[TriageResult, Ticket]):
    nodes = [Classify, LookUpBooking, EU261, Baggage, FareRules, DraftReply, Review, Pay]


support = Agent(
    instructions=(
        "You are Skyline Air's customer care, by text: one or two plain sentences. A late or "
        "cancelled flight, a bag, a refund or a change goes to triage with what the customer "
        "wrote, word for word, and the address they write from; ask for that address first if "
        "they have not given it. When triage answers resolved, send its reply_draft as it is. "
        "On to_human the draft is a colleague's to send, never yours: say a colleague will "
        "answer by email, and to any question about money, that the colleague will confirm "
        "it. On close, say you can only help with Skyline Air trips. A question about an "
        "answer already given is answered from it; only a new request goes to triage again. "
        "Never decide, compute or promise anything yourself."
    ),
    llm=LLM,
)


@support.tool
async def triage(ctx: RunContext, email: Email) -> TriageResult:
    """Look into the customer's request and answer it: the booking, what is due, the reply.

    Args:
        email: the customer's request
    """
    return await Triage(Classify(email), state=Ticket())


server = AgentServer(storage=AgentDBStorage(), agent_name="skyline-support")


@server.a2a_session(endpoint="triage", description="Skyline Air customer care.")
async def text(ctx: A2ASessionContext[None]) -> RunResult[str]:
    return await support.run(ctx.message, message_history=ctx.message_history)


@server.rtc_session()
async def chat(ctx: JobContext) -> None:
    session = LiveSession(llm=LLM, thread=ctx.thread)
    await session.start(agent=support, room=ctx.room)
    if not session.history.items:
        await session.generate_reply(instructions="Greet the customer in one sentence.")


if __name__ == "__main__":
    cli.run_app(server)
