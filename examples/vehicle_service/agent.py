"""Bayside BMW's service desk on the phone. Inbound, Leo talks a caller through a problem with
their car and books the drop-off; outbound, he calls back once the technician has found the
cause. The hard rules live in backend.py.

Inbound is the default. Dispatch with the job metadata `{"direction": "outbound"}` (or set
CALL_DIRECTION=outbound for `console`) for the follow-up, where the person in the room plays the
customer; add `"dial": "+1..."` with an outbound line (SIP_OUTBOUND_TRUNK_ID, or the metadata's
`trunk` or `from`) to ring a real phone instead.

The demo website watches calls through page.py; without it, the page hooks here do nothing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from backend import (
    DECISIONS,
    ENDED_EARLY,
    LOANER_MIN_HOURS,
    MENU,
    SHUTTLE_MILES,
    EndedEarly,
    Finding,
    NoteKind,
    RepairDecision,
    ServiceDesk,
    ServiceError,
    ServiceId,
    demo_desk,
    first_name,
    local_today,
    spoken_span,
)
from call import (
    VOICE_RULES,
    end_after_reply,
    hang_up,
    pause_between_sentences,
    room_options,
    spoken_number,
    transfer,
)
from dotenv import load_dotenv
from grade import grade
from page import Page, visitor_name

from livekit import api
from livekit.agents import (
    AMD,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    SimulationContext,
    ToolError,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)

logger = logging.getLogger("vehicle-service")

load_dotenv()

PERSONA = "Leo"
BUSINESS = "Bayside BMW"
VOICE = "228fca29-3a0a-435c-8728-5cb483251068"  # Cartesia "Kiefer"
# the number a customer calls back on, said on the follow-up and on voicemail
SERVICE_DESK_NUMBER = os.getenv("SERVICE_DESK_NUMBER", "")
# the participant identity of a phone the agent rings
PHONE = "phone"


@dataclass
class Userdata:
    desk: ServiceDesk
    # the demo website's view of the call: inert unless a page is watching (page.py)
    page: Page = field(init=False)

    def __post_init__(self) -> None:
        self.page = Page(PERSONA, self.desk)


Ctx = RunContext[Userdata]
# what a good advisor asks about a problem, in order (backend.NoteKind)
_QUESTIONS = ("sound", "where", "when", "since", "warning_lights", "recent")


@contextlib.contextmanager
def _desk_rules(ctx: Ctx) -> Iterator[ServiceDesk]:
    """Run a desk call; a refusal shows on the page and goes back to the agent to relay."""
    try:
        yield ctx.userdata.desk
    except ServiceError as e:
        logger.info("desk refused %s: %s", ctx.function_call.name, e)
        ctx.userdata.page.record(ctx.function_call.name, f"Refused: {str(e).split('. ')[0]}")
        raise ToolError(str(e)) from e


def _with_notes(result: str, notes: list[str]) -> str:
    # what a correction knocked out (a loaner, a warranty repair, a slot), for Leo to say
    return " ".join([result, *notes]) if notes else result


class LeoAgent(Agent):
    def __init__(self, desk: ServiceDesk) -> None:
        super().__init__(instructions=_instructions(desk), tools=[transfer_to_advisor])
        self._replies_at_booking: int | None = None

    async def on_enter(self) -> None:
        self.session.userdata.page.push()
        self.session.generate_reply(
            instructions=(
                f'Answer the phone: "Thanks for calling {BUSINESS} service, this is {PERSONA}. '
                'How can I help you today?"'
            )
        )

    # ---- who and what ----

    @function_tool
    async def lookup_customer(self, ctx: Ctx, phone: str, name: str = "") -> str:
        """Find the customer's file by the phone number they give. Do this before anything
        else about their car.

        Args:
            phone: The phone number as digits, e.g. "4155550142": exactly the digits they
                said. Never fill in, pad, or guess a digit; if it's short, ask again.
            name: The caller's name, if they've said it.
        """
        with _desk_rules(ctx) as desk:
            c = desk.lookup_customer(phone, name)
        ctx.userdata.page.record(
            "lookup_customer", f"{c.name} · {len(c.vehicles)} vehicles on file"
        )
        cars = "; ".join(
            f"{v.label}, VIN ending {v.vin[-6:]}, {v.miles:,} miles on file" for v in c.vehicles
        )
        return (
            f"Found {c.name}. Vehicles on file: {cars}. If they already said which car and "
            "what it needs, call select_vehicle and set_services now; otherwise ask."
        )

    @function_tool
    async def select_vehicle(self, ctx: Ctx, vehicle: str) -> str:
        """Set which of their cars the visit is for. Call it again if they switch cars.

        Args:
            vehicle: The VIN, or the year and model as the caller said it, e.g. "the X3".
        """
        with _desk_rules(ctx) as desk:
            notes = desk.select_vehicle(vehicle)
            v = desk.require_vehicle()
        ctx.userdata.page.record("select_vehicle", v.label)
        recall_hint = "" if v.vin in desk.recalls_checked else " Check its recalls."
        return _with_notes(
            f"Visit is for the {v.label} (VIN {v.vin}). {v.warranty_reason(desk.today)} "
            "(mention warranty only if it matters for this visit)." + recall_hint,
            notes,
        )

    @function_tool
    async def update_mileage(self, ctx: Ctx, miles: int) -> str:
        """Record the current mileage the caller reads off the odometer. Call it again with
        the new number if they correct themselves.

        Args:
            miles: The mileage as a whole number, e.g. 31000.
        """
        with _desk_rules(ctx) as desk:
            notes = desk.set_mileage(miles)
            v = desk.require_vehicle()
        ctx.userdata.page.record("update_mileage", f"{miles:,} miles")
        return _with_notes(
            f"Mileage is {miles:,}. {v.warranty_reason(desk.today)} (mention warranty only if "
            "it matters for this visit).",
            notes,
        )

    # ---- what's wrong ----

    @function_tool
    async def add_tech_note(self, ctx: Ctx, kind: NoteKind, detail: str) -> str:
        """Log one thing the caller said about the problem, for the technician, as soon as
        they say it. One note per answer; a correction is the same kind again.

        Args:
            kind: sound (what it sounds like), where (front or back, left or right), when
                (braking, turning, over bumps, at speed, idling, a cold start), since (how
                long it's been going on), warning_lights (which, or none), recent (recent
                work, new tires, an accident), or other.
            detail: What they said, in a few words, e.g. "grinding, like metal on metal".
        """
        with _desk_rules(ctx) as desk:
            note = desk.add_note(kind, detail)
        ctx.userdata.page.record("add_tech_note", f"{note.label}: {note.text}")
        out = f"Noted for the technician: {note.label.lower()}, {note.text}."
        asked = {n.kind for n in desk.draft.notes}
        left = [k.replace("_", " ") for k in _QUESTIONS if k not in asked]
        if cause := desk.likely_cause():
            out += (
                f" What they've described often points to {cause.cause}. If you haven't said "
                "so yet, say it as a possibility the technician will confirm, never a "
                f"diagnosis. {cause.advice}"
            )
        if left:
            out += f" Not asked yet, if it helps narrow it down: {', '.join(left)}."
        return out

    @function_tool
    async def set_concern(self, ctx: Ctx, summary: str) -> str:
        """Sum up the problem for the technician once you've asked about it. It goes on the
        visit and is read back. Call it again if the picture changes.

        Args:
            summary: The problem in a few words, e.g. "grinding from the front when braking".
        """
        with _desk_rules(ctx) as desk:
            desk.set_concern(summary)
        ctx.userdata.page.record("set_concern", desk.draft.concern)
        return f"Concern: {desk.draft.concern}. Now work out the service (set_services)."

    # ---- the service ----

    @function_tool
    async def quote_price(self, ctx: Ctx, service: str) -> str:
        """Look up the price of anything before saying a price: a service, the recall, the
        shuttle or a loaner, or the visit's total. Only menu services have prices; for
        anything else this returns no price, and you must not estimate one.

        Args:
            service: What the caller asked about, in their words, e.g. "brake job".
        """
        with _desk_rules(ctx) as desk:
            q = desk.quote(service)
        shown = "no price" if q.price is None else f"${q.price:.0f}"
        ctx.userdata.page.record("quote_price", f"{service} → {shown}")
        return q.answer

    @function_tool
    async def set_services(self, ctx: Ctx, services: list[ServiceId], concern: str = "") -> str:
        """Set the services on the visit, replacing what's there. To add or drop one, pass
        the whole new list.

        Args:
            services: oil_change, tire_rotation, diagnostic (any problem to look at, like a
                noise), or warranty_repair (a defect on a car that's in warranty; never wear
                items like brakes or tires).
            concern: Leave out to use the summary from set_concern.
        """
        with _desk_rules(ctx) as desk:
            notes = desk.set_services(list(services), concern)
        labels = ", ".join(MENU[s].label for s in desk.draft.services)
        detail = labels + (f" · {desk.draft.concern}" if concern else "")
        ctx.userdata.page.record("set_services", detail)
        cost = f"estimate ${desk.estimate():.0f}" if desk.estimate() else "no charge"
        return _with_notes(
            f"Services: {labels}. Visit is {spoken_span(desk.visit_hours())}, {cost}.", notes
        )

    @function_tool
    async def check_recalls(self, ctx: Ctx, vin: str) -> str:
        """Check open recalls for the car. Required on every call before booking.

        Args:
            vin: The car's full VIN.
        """
        with _desk_rules(ctx) as desk:
            recalls = desk.check_recalls(vin)
        if not recalls:
            ctx.userdata.page.record("check_recalls", f"{vin[-6:]}: no open recalls")
            return "No open recalls on this car. Tell the caller."
        r = recalls[0]
        ctx.userdata.page.record("check_recalls", f"{vin[-6:]}: {r.id} {r.title}")
        stock = "Parts are in stock." if r.parts_in_stock else "Parts are on order."
        return (
            f"Open recall: {r.title}, about {spoken_span(r.hours)}, no charge. {stock} (Its "
            f"number is {r.id}; say it only if they ask.) Offer to add it to this visit, then "
            "call decide_recall."
        )

    @function_tool
    async def decide_recall(self, ctx: Ctx, add_to_visit: bool) -> str:
        """Record whether the caller wants the open recall done on this visit.

        Args:
            add_to_visit: True if they accept, False if they decline.
        """
        with _desk_rules(ctx) as desk:
            desk.decide_recall(add_to_visit)
        ctx.userdata.page.record("decide_recall", "Added to visit" if add_to_visit else "Declined")
        return f"Recall {desk.draft.recall}. Visit is {spoken_span(desk.visit_hours())}."

    # ---- when, and how they get around ----

    @function_tool
    async def get_slots(self, ctx: Ctx, day: str | None = None) -> str:
        """Open advisor slots that fit this visit, over the next five business days.

        Args:
            day: The day to narrow to, as the caller said it, e.g. "wednesday", "tomorrow",
                "Oct 8". Leave out for all days.
        """
        with _desk_rules(ctx) as desk:
            slots = desk.open_slots(day)
        ctx.userdata.page.record("get_slots", f"{day or 'all days'}: {len(slots)} open")
        if not slots:
            return f"Nothing open that fits that day. Open days are {desk.open_days()}."
        return (
            "Open: "
            + "; ".join(f"{s.id} = {s.spoken()} with {s.advisor}" for s in slots)
            + ". Offer two or three times that suit what they asked for, not the whole list. Say"
            " the times only, without the advisors' names: the caller doesn't know who they are"
            ' ("I have tomorrow at seven thirty, eight, or ten thirty."). Name the advisor once'
            " a time is picked, saying who they are."
        )

    @function_tool
    async def choose_slot(self, ctx: Ctx, slot_id: str) -> str:
        """Put a slot on the draft. Call it again right away if they change the day or time.

        Args:
            slot_id: An id from get_slots, like "wed-0730".
        """
        with _desk_rules(ctx) as desk:
            desk.choose_slot(slot_id)
        slot = next(s for s in desk.slots if s.id == desk.draft.slot_id)
        ctx.userdata.page.record("choose_slot", f"{slot.spoken()} · {slot.advisor}")
        return (
            f"Held {slot.spoken()} with {slot.advisor}. Tell them who that is the first time you"
            f' name them: "{slot.advisor}, one of our service advisors, will meet you at the'
            ' service drive."'
        )

    @function_tool
    async def choose_transport(self, ctx: Ctx, option: Literal["loaner", "shuttle", "wait"]) -> str:
        """Set how the caller gets around during the visit.

        Args:
            option: loaner (only warranty or recall visits over three hours), shuttle (free,
                anyone), or wait (in the lounge).
        """
        with _desk_rules(ctx) as desk:
            desk.choose_transport(option)
        ctx.userdata.page.record("choose_transport", option)
        return f"Transportation: {option}."

    # ---- confirm ----

    @function_tool
    async def read_back(self, ctx: Ctx) -> str:
        """Get the appointment to read back. Say all of it to the caller, then ask if it's
        right. If they change anything, update that one field and call this again."""
        with _desk_rules(ctx) as desk:
            summary = desk.read_back()
        ctx.userdata.page.record("read_back", f"Revision {desk.draft.revision}")
        return (
            f"Read this back, in plain speech: {summary} It isn't booked yet, so say something "
            'like "here\'s what I have", not "booked", and ask if it\'s right.'
        )

    @function_tool
    async def book_service(self, ctx: Ctx) -> str:
        """Book the appointment. Only after the read-back, once the caller says it's right."""
        with _desk_rules(ctx) as desk:
            appt = desk.book()
        ctx.userdata.page.record("book_service", f"{appt.confirmation} · {appt.when}")
        self._replies_at_booking = _agent_replies(ctx)
        return (
            f"Booked, confirmation {appt.confirmation} (give it only if they ask). Tell them "
            f"they're all set and that {appt.advisor}, one of our service advisors, will meet "
            "them at the service drive, even if they've already said goodbye, and ask if "
            "there's anything else. When there isn't (or they've already said goodbye), say "
            "goodbye and call end_call in that same turn."
        )

    @function_tool
    async def end_call(self, ctx: Ctx) -> None:
        """Hang up. Only call this after you've said goodbye and the caller has nothing else."""
        # a goodbye in the same breath as "yes, book it" can't skip telling them they're set
        await ctx.wait_for_playout()
        if self._replies_at_booking is not None and _agent_replies(ctx) <= self._replies_at_booking:
            raise ToolError(
                "First tell them they're all set and which advisor will meet them, then say "
                "goodbye and call end_call."
            )
        await end_after_reply(ctx)


@function_tool
async def transfer_to_advisor(ctx: Ctx, reason: str) -> str:
    """Hand the call to a service advisor: for a price you don't have, a question you can't
    answer, a change to a booked visit, a tow, or when the caller asks for a person. Tell them
    you're connecting them first.

    Args:
        reason: A few words on why, for the advisor.
    """
    ctx.userdata.page.transfer("service_advisor", reason)
    return await transfer(ctx, "a service advisor", reason)


def _agent_replies(ctx: Ctx) -> int:
    return sum(1 for m in ctx.session.history.messages() if m.role == "assistant")


def _instructions(desk: ServiceDesk) -> str:
    today = desk.today
    return f"""\
{VOICE_RULES}

You're {PERSONA} at the {BUSINESS} service desk, answering the phone. Today is \
{today:%A, %B} {today.day}. Most callers have something wrong with their car: you find out
what's going on the way a good service advisor would, so the technician knows what to look
for, then book the drop-off with an advisor. You know nothing about the caller until you look
them up. Bring some energy: upbeat and genuinely friendly, like someone who likes their job and
is glad they called, without being salesy or over the top.

The flow (a price question can be answered at any point, even before the lookup):
1. Ask for their name, then (once they've answered) the phone number on their account,
   one at a time and skipping whatever they've already given, and call lookup_customer.
   Then find out which car (select_vehicle). If they mention mileage, call update_mileage.
   Don't ask for anything else about them.
2. If something's wrong (a noise, a vibration, a warning light, a leak), narrow it down, one
   question at a time, skipping whatever they've already told you:
   - what it sounds like: grinding, squealing, a clunk, a rattle, a hum;
   - where it seems to come from: front or back, left or right;
   - when it happens: braking, turning, over bumps, at speed or idling, on a cold start;
   - how long it's been going on;
   - any warning lights on the dash;
   - anything recent: new tires, brake work, an accident, a pothole.
   Log each answer the moment you hear it with add_tech_note, one note per answer, in their
   words. If they correct one, call add_tech_note again with the same kind.
3. add_tech_note says what the notes so far often point to. Say it once, cautiously, as a
   possibility ("that often points to worn brake pads, but the technician will confirm"),
   never as a diagnosis. If it says it may not be safe to drive, say that plainly, once.
4. Sum the problem up with set_concern, e.g. "grinding from the front when braking". Don't
   ask them to describe it again.
5. Work out what it needs (set_services). A problem to look at is a diagnostic ($159). But
   when the car is in warranty and a part has failed (a liftgate, a window, electronics, a
   leak), it's a warranty_repair, not a diagnostic: no charge, and the warranty covers
   finding the fault. Brakes, tires, and wipers are wear items and never warranty.
6. Always call check_recalls for that car's VIN. If there's an open recall, offer to add it
   and call decide_recall with their answer.
7. Offer times (get_slots, then choose_slot) and transportation (choose_transport). Offer
   two or three times, not the whole calendar; if they haven't said a day, ask what works.
   Offer the times alone, not the advisors' names (the caller doesn't know them); once they
   pick one, say who they'll meet: "Dana, one of our service advisors, will meet you there."
   If it may not be safe to drive, offer the soonest drop-off first.
8. Call read_back and say what it returns, the concern included. Never sum up the
   appointment from memory instead: the read-back the caller agrees to must come from
   read_back. When they say it's right, call book_service.

Rules:
- Whenever the caller asks what anything costs, call quote_price first, every time, even
  when you expect there's no price, and answer only from what it returns. Never say a
  price you didn't get from it. For a repair with no set price, like a brake job, say the
  technician will quote it after the diagnostic ($159), and that you'll call them first.
- Loaners are only for warranty or recall visits over {LOANER_MIN_HOURS:g} hours. Otherwise
  offer the free shuttle or waiting in the lounge. Don't promise a loaner the tool refused.
- When the caller corrects one thing (the day, the mileage, which car, a symptom), call only
  the tool for that field, straight away, then carry on. If you'd already read back, call
  read_back again and read the new version before booking.
- Say times like "seven thirty a.m." and dates like "Wednesday, September thirtieth". Call
  cars what people call them ("your X3"), not the full badge every time. Don't read slot
  ids, recall numbers, or VINs aloud unless they ask.
- Record what they've told you as soon as a tool can take it: if they said "a grinding
  noise on the X3" in their first sentence, that's a note and the car; don't ask again.
- If a tool refuses over a rule that affects the caller (no loaner, out of warranty, a
  time that's taken), tell them why in plain words and offer what the tool suggests. If
  it's a step you skipped, just do that step; don't mention it to the caller.
- After book_service, always say the confirmation number before hanging up.
- If the time they asked for isn't open, say so and offer the nearest ones before holding
  one. If they haven't said how they'll get around, ask before calling choose_transport."""


class RepairCallAgent(Agent):
    """The follow-up call: the car is in the shop for the problem the customer called about,
    and the technician has found the cause."""

    def __init__(self, desk: ServiceDesk) -> None:
        super().__init__(instructions=_outbound_instructions(desk), tools=[transfer_to_advisor])
        self._wrapped_at: float | None = None  # the pickup summary, or ending without it

    async def on_enter(self) -> None:
        ud: Userdata = self.session.userdata
        desk = ud.desk
        ud.page.push(stage="reach")
        # a fixed opener: who's calling, and for whom; nothing about the car until it's them
        who = first_name(desk.customer.name if desk.customer else "")
        self.session.say(
            f"Hi, this is {PERSONA} calling from {BUSINESS} service. May I speak with {who}?"
        )

    @function_tool
    async def confirm_customer(self, ctx: Ctx) -> str:
        """Whoever picked up said they're the customer. Returns what the technician found.
        Call it before saying anything about the car."""
        with _desk_rules(ctx) as desk:
            desk.confirm_customer()
        who = desk.customer.name if desk.customer else ""
        ctx.userdata.page.record("confirm_customer", f"Speaking with {who}")
        assert desk.visit is not None
        now = [f for f in desk.visit.findings if f.urgency == "now"]
        return (
            f"It's them. {_findings(desk)} Start with what explains the noise: what the "
            "technician found, the price, how long it takes, and when the car would be "
            f"ready ({_ready_choices(desk, now)}). Then ask whether they'd like it done, or "
            "would rather not. Bring up anything that can wait after that, honestly, as "
            "something for the next visit."
        )

    @function_tool
    async def repair_details(self, ctx: Ctx, question: str) -> str:
        """Look up the technician's findings to answer a question about a repair: what it is,
        why it's needed, the parts and labor, how long, warranty, whether it can wait, or a
        cheaper option. Answer only from what it returns.

        Args:
            question: What they asked, in a few words, e.g. "why the rotors too".
        """
        with _desk_rules(ctx) as desk:
            findings = desk.repair_details(question)
        ctx.userdata.page.record("repair_details", question or "Findings")
        details = " ".join(
            f"{f.title} ({f.id}): {f.why} {f.covered} Cheaper option: {f.cheaper}" for f in findings
        )
        return f"{_findings(desk)} {details} Answer what they asked in a sentence or two."

    @function_tool
    async def approve_repair(self, ctx: Ctx, repair: str) -> str:
        """The customer wants this repair done. Call it the moment they say yes.

        Args:
            repair: The finding's id, e.g. "front_brakes", or its name, e.g. "rear pads".
        """
        with _desk_rules(ctx) as desk:
            f, notes = desk.decide_repair(repair, "approved")
        ctx.userdata.page.record("approve_repair", f"{f.title} · ${f.price:.0f}")
        return _decided(desk, f"Approved: {f.title}, ${f.price:.0f}.", notes)

    @function_tool
    async def decline_repair(self, ctx: Ctx, repair: str, think_it_over: bool = False) -> str:
        """The customer would rather not have this repair done today, or wants to think it
        over. No pressure: record it and move on.

        Args:
            repair: The finding's id, e.g. "front_brakes", or its name, e.g. "rear pads".
            think_it_over: True if they want to think about it rather than a plain no.
        """
        decision: RepairDecision = "deferred" if think_it_over else "declined"
        with _desk_rules(ctx) as desk:
            f, notes = desk.decide_repair(repair, decision)
        said = "thinking it over" if think_it_over else "declined"
        ctx.userdata.page.record("decline_repair", f"{f.title} · {said}")
        later = (
            " It's noted on their file; they can call to approve it whenever they're ready."
            if think_it_over
            else ""
        )
        warn = (
            f" The technician says: {f.soon} Say that once, plainly, without pushing."
            if f.urgency == "now"
            else ""
        )
        said = DECISIONS[decision].lower()
        return _decided(desk, f"Recorded: {f.title}, {said}.{later}{warn}", notes)

    @function_tool
    async def note_for_next_visit(self, ctx: Ctx, note: str) -> str:
        """Put something on the customer's file for the next visit, like a repair that can
        wait. Only once they're fine with it being noted.

        Args:
            note: What to do next time, in a few words, e.g. "rear brake pads".
        """
        with _desk_rules(ctx) as desk:
            note = desk.note_for_next_visit(note)
        ctx.userdata.page.record("note_for_next_visit", note)
        return f"Noted for the next visit: {note}."

    @function_tool
    async def arrange_pickup(
        self, ctx: Ctx, ride: Literal["shuttle", "self"], where: str = ""
    ) -> str:
        """How the customer gets back to the dealership for the car.

        Args:
            ride: shuttle (the free courtesy shuttle picks them up) or self (they'll come in
                on their own).
            where: For the shuttle, where to pick them up, as they said it, e.g. "home".
        """
        with _desk_rules(ctx) as desk:
            v = desk.arrange_pickup(ride, where)
        if ride == "shuttle":
            detail = f"Shuttle from {v.pickup_from}, {desk.shuttle_when()}"
            ctx.userdata.page.record("arrange_pickup", detail)
            return f"The shuttle picks them up from {v.pickup_from} {desk.shuttle_when()}."
        ctx.userdata.page.record("arrange_pickup", "Coming in themselves")
        return "They're coming in on their own."

    @function_tool
    async def pickup_summary(self, ctx: Ctx) -> str:
        """Get when the car is ready, the total due, and the ride, to tell the customer. Call
        it once every repair that needs doing now has an answer and the ride is set, and again
        after any change."""
        with _desk_rules(ctx) as desk:
            summary = desk.pickup_summary()
        detail = f"Ready {desk.ready_when()} · ${desk.ticket_total():.0f}"
        ctx.userdata.page.record("pickup_summary", detail)
        self._wrapped_at = time.time()
        return (
            f"Tell them, in plain speech, in two or three sentences: {summary} Then ask if "
            "there's anything else, say goodbye, and call end_call."
        )

    @function_tool
    async def end_without_update(self, ctx: Ctx, reason: EndedEarly) -> str:
        """End the call without going through the findings: whoever answered isn't the
        customer, or it's a bad time. The car stays as it is.

        Args:
            reason: wrong_person (a wrong number, or someone else picked up) or
                call_back_later (driving, busy, "call me later").
        """
        with _desk_rules(ctx) as desk:
            desk.end_without_update(reason)
        ctx.userdata.page.record("end_without_update", ENDED_EARLY[reason])
        self._wrapped_at = time.time()
        who = first_name(desk.customer.name if desk.customer else "")
        if reason == "wrong_person":
            return (
                f"Recorded. Ask that {who} call back at {_callback()}, and say a polite goodbye "
                f"in a sentence: nothing about the car, the visit, or why you're calling, even "
                "if they ask. Then call end_call."
            )
        return (
            f"Recorded. Don't push: say they can call back at {_callback()} whenever suits, "
            "say goodbye in a sentence, and call end_call."
        )

    @function_tool
    async def end_call(self, ctx: Ctx) -> None:
        """Hang up, once you've told them when the car's ready and the total (or called
        end_without_update) and said goodbye."""
        await ctx.wait_for_playout()
        ud = ctx.userdata
        desk = ud.desk
        spoken = [
            m.created_at
            for m in ctx.session.history.messages()
            if m.role == "assistant" and m.text_content
        ]
        # the call isn't done until they've heard when the car's ready and what's due, unless
        # it ended without the update
        if not desk.call_done or not any(t > (self._wrapped_at or 0) for t in spoken):
            ud.page.record("end_call", "Refused: ready time and total not given yet")
            raise ToolError(
                "Get an answer on each repair that needs doing now, set the ride, call "
                "pickup_summary, and tell them what it returns before ending the call. If it's "
                "the wrong person, or a bad time, call end_without_update instead."
            )
        v = desk.visit
        reason = ENDED_EARLY[v.ended_early] if v and v.ended_early else "Call complete"
        await end_after_reply(ctx, reason)


def _callback() -> str:
    """The number the customer calls back on, as it's said."""
    return (
        spoken_number(SERVICE_DESK_NUMBER) if SERVICE_DESK_NUMBER else "the service desk's number"
    )


def _findings(desk: ServiceDesk) -> str:
    v = desk.visit
    assert v is not None
    return " ".join(
        f"{f.title} ({f.id}): {f.found} ${f.price:.0f}: parts ${f.parts:.0f} ({f.parts_detail}), "
        f"labor {spoken_span(f.labor_hours)} at ${f.labor:.0f}. "
        f"{'Needs doing now' if f.urgency == 'now' else 'Can wait'}: {f.soon}"
        for f in v.findings
    )


def _ready_choices(desk: ServiceDesk, now: list[Finding]) -> str:
    v = desk.visit
    assert v is not None
    out = [f"with nothing done, ready {desk.ready_when([])}"]
    out.append(
        f"with the {' and '.join(f.title.lower() for f in now)}, ready {desk.ready_when(now)}"
    )
    if later := [f for f in v.findings if f not in now]:
        out.append(f"with everything, ready {desk.ready_when(now + later)}")
    return "; ".join(out)


def _decided(desk: ServiceDesk, result: str, notes: list[str]) -> str:
    out = [
        result,
        *notes,
        f"Ticket total now ${desk.ticket_total():.0f}, ready {desk.ready_when()}.",
    ]
    if left := desk.undecided():
        out.append(f"Still to ask about: {', '.join(f.title.lower() for f in left)}.")
    else:
        out.append(
            "Every repair that needs doing now has an answer: next, how they'll get back for "
            f"the car (the free courtesy shuttle, within {SHUTTLE_MILES} miles, or on their own)."
        )
    return " ".join(out)


def _outbound_instructions(desk: ServiceDesk) -> str:
    v = desk.visit
    assert v is not None and desk.customer is not None
    name = desk.customer.name
    who = first_name(name)
    car = desk.vehicle(v.appointment.vin).name.split()[0]
    today = desk.today
    return f"""\
{VOICE_RULES}

You're {PERSONA} at the {BUSINESS} service desk, making an outbound call to {name}. Today is \
{today:%A, %B} {today.day}. Their {car} came in at 7:30 this morning for a diagnostic: they'd
called about "{v.appointment.concern.lower()}". The open recall was added too, and both are
done. The technician has found the cause, and you're calling to go over it. Be warm, upbeat,
plain, and brief: an advisor who's on their side and glad to help, not a salesperson.

# Who picked up
You've already said who you are and asked for {who}.
- Once they say it's them, call confirm_customer: it returns what the technician found.
  Until it succeeds, say nothing about the car, the visit, or why you're calling.
- If it's someone else, or {who} can't come to the phone: say only that it's {PERSONA} from
  {BUSINESS} calling for {who}, and ask that {who} call back at {_callback()}. If they ask what
  it's about, say you can only discuss it with {who}. Call end_without_update with
  wrong_person, say goodbye, and call end_call.
- If it's {who} but a bad time (driving, at work, "call me later"): don't push. Call
  end_without_update with call_back_later, say they can call back at {_callback()} whenever
  suits, say goodbye, and call end_call.

# What the technician found
Go over it a sentence or two at a time, not all at once: what they found that explains the
noise, the price, how long it takes, and when the car would be ready. Break the price into
parts and labor only if they ask. Then ask plainly whether they'd like it done, or would
rather not.
- Any question about a repair (what it is, why, the cost, how long, warranty, whether it can
  wait, a cheaper option): call repair_details and answer only from what it returns. Never
  guess a price, a part, or a time. Offer a cheaper option only if it says there is one.
- Never pressure them. Be honest about urgency, once: if the technician says it shouldn't be
  driven like this, say so plainly; if something can wait, say that just as plainly.
- Yes: call approve_repair. No: call decline_repair. They want to think about it: call
  decline_repair with think_it_over true; it's noted on their file. If they change their
  mind, call the other tool.
- Something that can wait (like pads that are low but fine for now): say it's not urgent
  and fine to do at the next visit, and offer to note it (note_for_next_visit). If they'd
  rather do it today too, approve it.

# Your car is ready
When every repair that needs doing now has an answer, ask how they'll get back for the car:
the free courtesy shuttle can pick them up (arrange_pickup with shuttle and where from), or
they can come in on their own (arrange_pickup with self). Then call pickup_summary and say
what it returns in two or three sentences: when it's ready, the total due, and the ride.
Ask if there's anything else, say goodbye, and call end_call."""


def _voicemail() -> str:
    # who's calling, for whom, and a number to call back: never why
    return (
        f"Hi, this is {PERSONA} from {BUSINESS} service, calling for Alex Morgan. Please call "
        f"us back at {_callback()}. Thanks, goodbye."
    )


server = AgentServer()


async def on_simulation_end(sim: SimulationContext) -> None:
    """Fails a run whose final service desk state isn't what the scenario's `expect` says."""
    ud: Userdata = sim.job_context.primary_session.userdata
    if (expect := sim.userdata().get("expect")) and (problems := grade(ud.desk, expect)):
        sim.fail(reason="; ".join(problems))


@server.rtc_session(on_simulation_end=on_simulation_end)
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    sim = ctx.simulation_context()
    if sim:
        # a simulated caller: the scenario's userdata says which direction
        meta = {**meta, **sim.userdata()}
    # console has no job metadata: CALL_DIRECTION=outbound picks the follow-up there
    outbound = (meta.get("direction") or os.getenv("CALL_DIRECTION")) == "outbound"
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "direction": "outbound" if outbound else "inbound",
    }

    await ctx.connect()
    dial = None if sim else meta.get("dial")
    trunk, from_number = _outbound_line(meta)
    ring_phone = bool(outbound and dial and (trunk or from_number))
    if outbound and dial and not ring_phone:
        logger.warning(
            "no outbound line (SIP_OUTBOUND_TRUNK_ID); the person in the room plays the callee"
        )
    caller = None
    if not sim and not ring_phone:
        # wait for them, so an outbound opener isn't spoken to an empty room
        caller = await ctx.wait_for_participant()

    # "today" is the website visitor's (the metadata's `timezone`), else the dealership's; the
    # file is under the name they gave, and the phone number is what finds it
    today = local_today(meta.get("timezone"))
    name = visitor_name(caller) or "Alex Morgan"
    agent: Agent
    if outbound:
        # the visit is already booked, and the car is in the shop
        desk = demo_desk(name, today, direction="outbound")
        agent = RepairCallAgent(desk)
    else:
        desk = demo_desk(name, today)
        agent = LeoAgent(desk)

    tts = inference.TTS(
        "cartesia/sonic-3.5",
        voice=VOICE,
        language="en",
        extra_kwargs={"emotion": "enthusiastic"},
    )
    session = AgentSession[Userdata](
        userdata=Userdata(desk=desk),
        tts_text_transforms=["filter_markdown", "filter_emoji", pause_between_sentences],
        stt=inference.STT("deepgram/nova-3", language="en"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1"),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
    )

    if sim:
        await session.start(agent=agent, room=ctx.room, room_options=room_options())
        return

    # the caller hung up, or the agent did: close the room, and the phone line with it
    session.on("close", lambda _: ctx.delete_room())
    # the demo website, if it's watching: the page that asked to ring a phone, the browser
    # caller, or viewers of a call to the service desk's number
    session.userdata.page.watch(ctx.room, session, page=meta.get("page"), caller=caller)
    if dial and ring_phone:
        await ring(ctx, session, agent, dial, trunk, from_number, tts)
        return
    assert caller is not None
    await session.start(agent=agent, room=ctx.room, room_options=room_options(caller.identity))


def _outbound_line(meta: dict[str, str]) -> tuple[str, str]:
    """The line a call to a real phone goes out on, as (trunk, from): the metadata's (a SIP
    trunk, or a number of the project's own to dial from), else SIP_OUTBOUND_TRUNK_ID."""
    if meta.get("trunk") or meta.get("from"):
        return meta.get("trunk") or "", meta.get("from") or ""
    return os.getenv("SIP_OUTBOUND_TRUNK_ID", ""), ""


async def ring(
    ctx: JobContext,
    session: AgentSession[Userdata],
    agent: Agent,
    dial: str,
    trunk: str,
    from_number: str,
    tts: inference.TTS,
) -> None:
    """Rings a real phone; Leo joins only once a person has picked up."""
    page = session.userdata.page
    # a listener of its own answers first, so nothing reaches the line until it's known who,
    # or what, picked up
    probe = AgentSession[None](tts=tts)
    await probe.start(agent=Agent(instructions=""), room=ctx.room, room_options=room_options(PHONE))
    async with AMD(probe, participant_identity=PHONE, ivr_detection=False) as detector:
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    # a carrier's trunk reaches any country; without one, LiveKit dials out
                    # from one of the project's own numbers, US only
                    sip_trunk_id=trunk,
                    sip_number=from_number,
                    sip_call_to=dial,
                    participant_identity=PHONE,
                    wait_until_answered=True,
                )
            )
        except api.TwirpError as e:
            sip = e.metadata or {}
            logger.warning(
                "outbound call to %s failed: %s %s",
                dial,
                sip.get("sip_status_code", ""),
                sip.get("sip_status", e.message),
            )
            await page.dial_failed(e, dial)
            ctx.delete_room()
            return
        result = await detector.execute()
    logger.info("answered by %s", result.category, extra={"transcript": result.transcript})
    if result.category == "machine-vm":
        await probe.say(_voicemail(), allow_interruptions=False).wait_for_playout()
        await probe.aclose()
        await hang_up("Reached voicemail; left a message", page)
        return
    await probe.aclose()
    if result.category in ("machine-unavailable", "machine-ivr"):
        await hang_up("No one answered", page)
        return
    await session.start(agent=agent, room=ctx.room, room_options=room_options(PHONE))


if __name__ == "__main__":
    cli.run_app(server)
