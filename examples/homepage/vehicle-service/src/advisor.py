"""Nick at Bayside BMW's service desk: answering the phone, and calling a customer back once the
technician has found the cause of the problem they called about.

Tools return facts and the next step; how to say things lives in the instructions, once. Anything
that books, moves, cancels, or approves work is read back first and done only after the caller has
heard the read-back in full and answered it (conversation.py)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import ToolError

from conversation import answered, caller_spoke_last, heard
from service_desk import (
    DEALERSHIP,
    LOANER_MIN_HOURS,
    BmwModel,
    EndedEarly,
    Finding,
    Moment,
    PartOfDay,
    PickupRide,
    Priceable,
    RepairDecision,
    RepairId,
    Ride,
    Service,
    ServiceDesk,
    ServiceDeskError,
    Sound,
    UnsafeReason,
    WarrantyPart,
    likely_cause,
    service_label,
    spoken,
)

PERSONA = "Nick"
GREETING = f"Thanks for calling {DEALERSHIP} service, this is {PERSONA}. How can I help you today?"

VOICE_RULES = """\
You're on a phone call, so talk like a person, not a document:
- One or two short sentences at a time, and one question at a time. No lists, markdown, or symbols.
- Say numbers, prices, dates, and times as words, the way people say them ("seventy-nine dollars", "seven thirty Tuesday morning"). Never read out ids or codes. Call cars by model: "your X3", "your 3 Series".
- Never say you've done something, or that you're about to, without calling its tool in that same reply. "Let me check" means calling the tool now, in this reply.
- Only state a price, time, recall, cause, or booking that a tool gave you.
- When a tool refuses, fix what it says with your tools. Tell the caller only what changes for them, as the rule itself: never "the system".
- The caller can't see anything: never "does that look right".
- You handle everything yourself. There's nobody to transfer to, and no callbacks to promise or messages to pass on.
- Payment is in person at the service desk at pickup, cash or card. Nothing is paid over the phone, and there are no discount codes.
- When they're done, say a short goodbye and call end_call in that same reply."""


@dataclass(frozen=True)
class Review:
    """Something read back to the caller: a booking, a move, a cancellation, or a repair to approve."""

    action: str
    target: str
    revision: int
    token: str


@contextmanager
def _desk_rules() -> Iterator[None]:
    try:
        yield
    except ServiceDeskError as e:
        raise ToolError(str(e)) from e


def _facts(**fields: Any) -> str:
    return json.dumps(fields)


class Nick(Agent):
    """What both of Nick's calls share: the desk, read-backs, and hanging up."""

    def __init__(self, desk: ServiceDesk, *, instructions: str) -> None:
        self._desk = desk
        self._review: Review | None = None
        # what the caller must hear before the call can end, by what it is
        self._must_hear: dict[str, str] = {}
        super().__init__(instructions=instructions)

    def _answered_review(self, ctx: RunContext, action: str, target: str) -> bool:
        r = self._review
        return (
            r is not None
            and (r.action, r.target, r.revision)
            == (action, target, self._desk.revision)
            and answered(ctx.session.history.items, r.token)
        )

    def _read_back(self, ctx: RunContext, action: str, target: str) -> None:
        self._review = Review(
            action, target, self._desk.revision, ctx.function_call.call_id
        )

    def _must_tell(self, ctx: RunContext, what: str) -> None:
        self._must_hear[what] = ctx.function_call.call_id

    @function_tool
    async def end_call(self, ctx: RunContext) -> None:
        """Hang up. Only once the caller has nothing else, and after a short goodbye in this
        same reply."""
        # what this same reply says, a goodbye with the booking in it, lands once it has played
        await ctx.wait_for_playout()
        items = ctx.session.history.items
        if unheard := [
            w for w, token in self._must_hear.items() if not heard(items, token)
        ]:
            raise ToolError(f"Don't hang up yet: first tell them {unheard[0]}.")
        if missing := self._unfinished():
            raise ToolError(f"Don't hang up yet: {missing}")
        if caller_spoke_last(items):
            raise ToolError(
                "Answer what they just said with a short goodbye, then end_call."
            )
        ctx.speech_handle.add_done_callback(lambda _: ctx.session.shutdown())

    def _unfinished(self) -> str | None:
        return None


class ServiceAdvisor(Nick):
    """Nick answering the phone: finds the caller's file and the car, takes down the problem,
    books the visit, and moves or cancels what's already booked."""

    def __init__(self, desk: ServiceDesk) -> None:
        super().__init__(desk, instructions=_inbound_instructions(desk))

    async def on_enter(self) -> None:
        self.session.say(GREETING)

    @function_tool
    async def find_customer(self, ctx: RunContext, full_name: str, phone: str) -> str:
        """Look up the caller's file by their name and phone number. Ask for the name, then
        the number, then call this.

        Args:
            full_name: Their first and last name, as they gave it.
            phone: The phone number's digits, exactly as they said them.
        """
        with _desk_rules():
            customer = await self._desk.find_customer(full_name, phone)
        if customer is None:
            return _facts(
                found=False,
                next="If the name may have been misheard, ask them to spell their last name "
                "and look again. Otherwise call open_new_file.",
            )
        return _facts(
            found=True,
            file=self._desk.file_json(),
            next="Record what they've already told you now: choose_vehicle for the car they "
            "named, describe_problem for the problem. If they've asked to move or cancel an "
            "appointment, call that tool now: it gives you what to read back.",
        )

    @function_tool
    async def open_new_file(self, ctx: RunContext, full_name: str, phone: str) -> str:
        """Set up a new file for a caller who doesn't have one.

        Args:
            full_name: Their first and last name.
            phone: The phone number's digits.
        """
        with _desk_rules():
            await self._desk.open_new_file(full_name, phone)
        return _facts(
            file=self._desk.file_json(),
            next="Record what they've already told you about the problem with describe_problem, "
            "then ask what car it is.",
        )

    @function_tool
    async def choose_vehicle(
        self, ctx: RunContext, vehicle_id: str, current_mileage: int | None = None
    ) -> str:
        """Put one of the cars on their file on this visit. Call again if they switch cars, or
        to record the mileage they read off the odometer.

        Args:
            vehicle_id: The car's vehicleId from their file.
            current_mileage: Only if they told you the current mileage.
        """
        with _desk_rules():
            changed = await self._desk.choose_vehicle(vehicle_id, current_mileage)
        car = self._desk.visit.vehicle
        assert car is not None
        return _facts(car=car.to_json(self._desk.today), changed=changed)

    @function_tool
    async def add_vehicle(
        self, ctx: RunContext, year: int, model: BmwModel, mileage: int
    ) -> str:
        """Add a car that isn't on their file, and put it on this visit. Ask the year and model,
        then the mileage. This desk services BMWs only.

        Args:
            year: The model year.
            model: The BMW model, e.g. "330i" is the 3 Series.
            mileage: Their best guess at the odometer reading.
        """
        with _desk_rules():
            car = await self._desk.add_vehicle(year, model, mileage)
        return _facts(car=car.to_json(self._desk.today))

    @function_tool
    async def describe_problem(
        self,
        ctx: RunContext,
        summary: str | None = None,
        sound: Sound | None = None,
        where: str | None = None,
        when: Moment | None = None,
        since: str | None = None,
        warning_lights: str | None = None,
        recent_work: str | None = None,
    ) -> str:
        """Record what the caller tells you about the problem, as soon as they say it. Pass only
        what they just told you; leave out what they haven't said, never "unknown". Anything you
        leave out stays as it was. A correction is the same field again.

        Args:
            summary: The problem in a few words, for the technician and the read-back.
            sound: What it sounds like.
            where: Where it seems to come from, in their words.
            when: When it happens.
            since: How long it's been going on, in their words.
            warning_lights: Which lights are on, or "none" when they said there are none.
            recent_work: Recent work, new tires, an accident, or "none" when they said there's been none.
        """
        with _desk_rules():
            problem = await self._desk.describe_problem(
                summary=summary,
                sound=sound,
                where=where,
                when=when,
                since=since,
                warning_lights=warning_lights,
                recent_work=recent_work,
            )
        cause = likely_cause(problem.sound, problem.when)
        return _facts(
            problem=problem.to_json(),
            oftenPointsTo=None
            if cause is None
            else {"cause": cause.cause, "advice": cause.advice},
        )

    @function_tool
    async def flag_unsafe_to_drive(self, ctx: RunContext, reason: UnsafeReason) -> str:
        """The car isn't safe to drive at all: the brakes don't stop it, smoke or burning,
        steering that fails, a red or flashing warning light, or overheating. Call it the moment
        they describe one. A noise on its own isn't one.

        Args:
            reason: What they described.
        """
        with _desk_rules():
            await self._desk.flag_unsafe_to_drive(reason)
        return _facts(
            unsafeToDrive=reason,
            next="Tell them now, plainly, not to drive it and to have it towed in; the desk "
            "can't arrange the tow. Then offer the earliest time.",
        )

    @function_tool
    async def quote_price(self, ctx: RunContext, item: Priceable) -> str:
        """Look up a price before saying one.

        Args:
            item: What they asked about. Anything that isn't a menu service, like a brake job,
                is something_else. this_visit is the total for what's on the visit.
        """
        return _facts(**self._desk.quote(item))

    @function_tool
    async def add_service(
        self,
        ctx: RunContext,
        service: Service,
        failed_part: WarrantyPart | None = None,
    ) -> str:
        """Add a service to the visit. A problem to look at, with the cause unknown, is a
        diagnostic. A warranty_repair is only for a specific part the caller says has stopped
        working, on a car in warranty: pass that part as failed_part.
        recall adds the car's open recall, at no charge.

        Args:
            service: The service to add.
            failed_part: For a warranty_repair only: what failed.
        """
        with _desk_rules():
            await self._desk.add_service(service, failed_part)
        return self._visit_facts()

    @function_tool
    async def remove_service(self, ctx: RunContext, service: Service) -> str:
        """Take a service off the visit.

        Args:
            service: The service to remove.
        """
        with _desk_rules():
            await self._desk.remove_service(service)
        return self._visit_facts()

    @function_tool
    async def find_times(
        self,
        ctx: RunContext,
        day: str | None = None,
        part_of_day: PartOfDay = "any",
        appointment_id: str | None = None,
    ) -> str:
        """Find open times that fit the visit, or an appointment that's being moved.

        Args:
            day: The day, YYYY-MM-DD, from the bookable days. Leave out for any day.
            part_of_day: morning, afternoon, or any.
            appointment_id: Only when moving an appointment that's already booked.
        """
        with _desk_rules():
            slots = self._desk.openings(day, part_of_day, appointment_id)
        await self._desk.publish_openings(slots)
        return _facts(
            openings=[
                {"slotId": s.id, "when": spoken(s.starts_at), "advisor": s.advisor}
                for s in slots[:3]
            ],
            more=max(0, len(slots) - 3),
        )

    @function_tool
    async def choose_time(self, ctx: RunContext, slot_id: str) -> str:
        """Put a time on the visit.

        Args:
            slot_id: A slotId from find_times.
        """
        with _desk_rules():
            await self._desk.choose_time(slot_id)
        return self._visit_facts()

    @function_tool
    async def choose_ride(self, ctx: RunContext, ride: Ride) -> str:
        """How they'll get around during the visit: wait in the lounge, the free shuttle, or a
        loaner. Only once they've picked one.

        Args:
            ride: wait, shuttle, or loaner.
        """
        with _desk_rules():
            await self._desk.choose_ride(ride)
        return self._visit_facts()

    @function_tool
    async def set_approval_limit(self, ctx: RunContext, dollars: int) -> str:
        """The caller wants a call before any work over an amount.

        Args:
            dollars: The amount they gave.
        """
        with _desk_rules():
            await self._desk.set_approval_limit(dollars)
        return self._visit_facts()

    @function_tool
    async def book_visit(self, ctx: RunContext) -> str:
        """Book the visit. The first call checks it and returns what to read back to the caller.
        Call it again once they've said it's right. After any change, it's read back again; a
        visit already booked on this call is updated under the same confirmation."""
        if self._answered_review(ctx, "book", ""):
            with _desk_rules():
                appointment = await self._desk.book()
            self._must_tell(ctx, "their booking")
            return _facts(
                booked=appointment.to_json(),
                next="Tell them they're booked: the day, time, advisor, and ride. Then ask if "
                "there's anything else.",
            )
        with _desk_rules():
            visit = self._desk.read_back()
        self._read_back(ctx, "book", "")
        return _facts(
            booked=False,
            readBack=visit,
            next="It isn't booked yet. Read this back as a question, the concern included, "
            "and ask if it's right. Call book_visit again once they agree.",
        )

    @function_tool
    async def reschedule_appointment(
        self, ctx: RunContext, appointment_id: str, slot_id: str
    ) -> str:
        """Move an appointment that's already booked to another time. The first call returns
        the move to read back; call it again once the caller agrees.

        Args:
            appointment_id: The appointmentId from their file.
            slot_id: A slotId from find_times for that appointment.
        """
        target = f"{appointment_id}@{slot_id}"
        if self._answered_review(ctx, "move", target):
            with _desk_rules():
                appointment = await self._desk.move(appointment_id, slot_id)
            self._must_tell(ctx, f"the move of {appointment_id}")
            return _facts(moved=appointment.to_json())
        with _desk_rules():
            move = self._desk.move_read_back(appointment_id, slot_id)
        self._read_back(ctx, "move", target)
        return _facts(
            moved=False,
            readBack=move,
            next="It isn't moved yet. Read the move back and ask if that's right. Call reschedule_appointment "
            "again once they agree.",
        )

    @function_tool
    async def cancel_appointment(self, ctx: RunContext, appointment_id: str) -> str:
        """Cancel an appointment that's already booked. The first call returns the cancellation
        to read back; call it again once the caller agrees.

        Args:
            appointment_id: The appointmentId from their file.
        """
        if self._answered_review(ctx, "cancel", appointment_id):
            with _desk_rules():
                appointment = await self._desk.cancel(appointment_id)
            self._must_tell(ctx, f"the cancellation of {appointment_id}")
            return _facts(cancelled=appointment.to_json(), due=0)
        with _desk_rules():
            cancellation = self._desk.cancel_read_back(appointment_id)
        self._read_back(ctx, "cancel", appointment_id)
        return _facts(
            cancelled=False,
            readBack=cancellation,
            next="It isn't cancelled yet. Ask if they're sure they want it cancelled. Call cancel_appointment again "
            "once they agree.",
        )

    def _visit_facts(self) -> str:
        desk = self._desk
        visit = desk.visit
        car = visit.vehicle
        recall = car.recall if car else None
        return _facts(
            services=[
                {"service": service_label(s, car), "price": desk.price(s)}
                for s in visit.services
            ],
            openRecallNotAdded=recall.title
            if recall and "recall" not in visit.services
            else None,
            time=spoken(visit.slot.starts_at) if visit.slot else None,
            advisor=visit.slot.advisor if visit.slot else None,
            ride=visit.ride,
            approvalLimit=visit.approval_limit,
            estimate=desk.estimate(),
            loaner="This visit qualifies for a loaner."
            if desk.loaner_eligible()
            else f"No loaner: they're only for warranty or recall visits over "
            f"{LOANER_MIN_HOURS:g} hours.",
            changedSinceBooked=desk.changed_since_booked(),
        )


class RepairCallback(Nick):
    """Nick calling the customer back: what the technician found, what it costs, and how they'll
    get the car back."""

    def __init__(self, desk: ServiceDesk) -> None:
        super().__init__(desk, instructions=_callback_instructions(desk))

    async def on_enter(self) -> None:
        self.session.say(callback_opener(self._desk))

    @function_tool
    async def go_over_findings(self, ctx: RunContext) -> str:
        """Get what the technician found on the repair that needs doing now, to tell the
        customer. Only once they've confirmed they're the customer."""
        desk = self._desk
        with _desk_rules():
            finding = await desk.share_findings()
        ready = desk.ready_at([finding])
        # what they hear next is the read-back their yes approves
        self._read_back(ctx, "approve", finding.id)
        return _facts(
            repairId=finding.id,
            repair=finding.title,
            found=finding.spoken,
            price=finding.price,
            readyIfApproved="now" if ready is None else desk.said(ready),
            next="Tell them what the technician found, the price, and when the car would be "
            "ready, and ask whether to go ahead.",
        )

    @function_tool
    async def decide_repair(
        self, ctx: RunContext, repair_id: RepairId, decision: RepairDecision
    ) -> str:
        """Record what the customer wants done about one repair. Approving one whose price they
        haven't heard returns it to read back first; call again once they agree.

        Args:
            repair_id: The repair.
            decision: approved (do it now), next_visit (note it for next time), declined, or
                thinking_it_over.
        """
        desk = self._desk
        shop = desk.shop
        assert shop is not None
        with _desk_rules():
            finding = shop.finding(repair_id)
        if decision == "approved" and not self._answered_review(
            ctx, "approve", repair_id
        ):
            self._read_back(ctx, "approve", repair_id)
            return _facts(
                approved=False,
                readBack={"repair": finding.title, "price": finding.price},
                next="It isn't approved yet. Give them the price and ask if they'd like it "
                "done. Call decide_repair again once they agree.",
            )
        ready_before = desk.pickup_summary()["readyAt"]
        with _desk_rules():
            await desk.decide(repair_id, decision)
        extra: dict[str, Any] = {}
        if (ready := desk.pickup_summary()["readyAt"]) != ready_before:
            extra["readyAtChanged"] = {"from": ready_before, "to": ready}
        warning = ""
        if decision in ("declined", "thinking_it_over") and finding.urgency == "now":
            warning = (
                f"First say once, plainly and without pushing: {finding.can_wait} "
            )
        return self._pickup_facts(ctx, warning=warning, **extra)

    @function_tool
    async def arrange_pickup(
        self,
        ctx: RunContext,
        ride: PickupRide | None = None,
        shuttle_from: str | None = None,
        collected_by: str | None = None,
    ) -> str:
        """How the car gets back to them. Pass only what they just told you.

        Args:
            ride: shuttle (the free shuttle picks them up) or self (they, or whoever's
                collecting it, come in).
            shuttle_from: For the shuttle: where to pick them up, in their words.
            collected_by: Someone else collecting the car for them.
        """
        with _desk_rules():
            await self._desk.arrange_pickup(ride, shuttle_from, collected_by)
        return self._pickup_facts(ctx)

    @function_tool
    async def note_new_concern(self, ctx: RunContext, concern: str) -> str:
        """Something new they've noticed with the car, noted for the technician to check.

        Args:
            concern: What they noticed, in a few words.
        """
        with _desk_rules():
            await self._desk.note_concern(concern)
        return _facts(
            noted=concern,
            next="Say it's noted for the technician, without guessing the cause or promising "
            "free work. " + self._next_step(),
        )

    @function_tool
    async def end_without_update(self, ctx: RunContext, reason: EndedEarly) -> str:
        """Before anything's decided: the wrong person, or a bad time to talk.

        Args:
            reason: wrong_person or call_back_later.
        """
        with _desk_rules():
            await self._desk.end_early(reason)
        self._must_hear.clear()
        return _facts(
            next="Apologize for the mix-up and say goodbye. Share nothing about the car, and "
            "don't offer to pass anything on."
            if reason == "wrong_person"
            else "Say no problem: they can call the service desk whenever suits. Then say "
            "goodbye.",
        )

    def _pickup_facts(self, ctx: RunContext, warning: str = "", **extra: Any) -> str:
        self._must_tell(ctx, "when the car's ready and the total due")
        if later := self._raising():
            # bringing up a repair that can wait, with its price, is its read-back
            self._read_back(ctx, "approve", later.id)
        return _facts(
            **self._desk.pickup_summary(), **extra, next=warning + self._next_step()
        )

    def _raising(self) -> Finding | None:
        """The repair that can wait, once the urgent one is approved and before it's decided."""
        desk = self._desk
        shop = desk.shop
        assert shop is not None
        approved = any(d == "approved" for d in shop.decisions.values())
        later = desk.undecided_later()
        return later[0] if approved and not desk.undecided() and later else None

    def _next_step(self) -> str:
        desk = self._desk
        shop = desk.shop
        assert shop is not None
        if left := desk.undecided():
            return f"Ask what they'd like to do about the {left[0].title.lower()}."
        if f := self._raising():
            return (
                f"Bring up the {f.title.lower()} ({_dollars(f.price)}): {f.can_wait} Ask "
                "whether to do them now or at the next visit."
            )
        if shop.ride is None:
            return (
                "Ask how they'll get back for the car: the free shuttle (and from where) or "
                "coming in themselves."
            )
        return "Tell them when it's ready and the total due, then ask if there's anything else."

    def _unfinished(self) -> str | None:
        shop = self._desk.shop
        if shop is None or shop.ended_early:
            return None
        if left := self._desk.undecided():
            return f"ask what they'd like to do about the {left[0].title.lower()}."
        if shop.ride is None:
            return "ask how they'll get back for the car."
        return None


def callback_opener(desk: ServiceDesk) -> str:
    shop = desk.shop
    assert shop is not None and desk.customer is not None
    name = desk.customer.name
    return (
        f"Hi, this is {PERSONA} from {DEALERSHIP} service, calling for {name} about the "
        f"{shop.appointment.vehicle.name}. Is this {name.split()[0]}?"
    )


def _dollars(amount: int) -> str:
    return f"${amount:,}"


def _bookable_days(desk: ServiceDesk) -> str:
    return "; ".join(f"{d:%A, %B} {d.day} = {d.isoformat()}" for d in desk.days)


def _inbound_instructions(desk: ServiceDesk) -> str:
    today = desk.today
    return f"""\
You're {PERSONA} at the {DEALERSHIP} service desk, answering the phone. Calm and attentive, like a good service advisor: never a salesperson.
Today is {today:%A, %B} {today.day}, {today.year}. Bookable days: {_bookable_days(desk)}. Nothing today.

{VOICE_RULES}

Take whatever the caller tells you whenever they say it, in any order, and record it right away with its tool; never ask again for something they've said. Answer every question they ask before moving on.

How the call usually goes:
1. Their file: ask their name, then their phone number, then find_customer. With no file, ask them to spell their last name once in case it was misheard, unless they've said they're new; if there's still no file, open_new_file and tell them you've set one up.
2. The car: choose_vehicle from their file, asking which only if they haven't said. For a car that isn't on it, ask the year and model, then the mileage, and add_vehicle. Only BMWs. Don't ask the mileage of a car already on file.
3. What's wrong. For a noise: what it sounds like, where it comes from, when it happens, and how long. For a warning light: which one, and whether it's steady or flashing. For something that's stopped working: what happens, and since when. Then warning lights, then anything recent, as separate questions. One question at a time, skipping what they've said, and record each answer with describe_problem as you hear it. Only when describe_problem says what it often points to, say that once as a possibility the technician will confirm.
   Brakes that don't stop the car, smoke or burning, losing steering, a red or flashing warning light, or overheating: call flag_unsafe_to_drive and tell them right away not to drive it. A noise on its own isn't one of these.
4. What it needs: add_service. A problem to look at is a diagnostic. A warranty_repair is only for a specific part they say has stopped working (the liftgate, a window, the screen) on a car in warranty: no charge, finding the fault included. When the cause is unknown (a noise, smoke, a warning light), it's a diagnostic. Brakes, tires, wipers, and maintenance are never covered by warranty: say so plainly. If the car has an open recall, offer it, and add it only if they say yes. Never suggest anything else.
5. When: find_times for the day they want and offer what it returns. "First thing", "the earliest", or "any time" means the first opening: choose_time with it and tell them the time. Then ask how they'll get around: the lounge or the free shuttle, and a loaner only when the visit qualifies (the loaner fact says).
6. Once the visit has a service, a time, and a ride, call book_visit; don't sum it up or call it set yourself first. It returns the visit, not yet booked: read it back as a question, the car, services, concern, day and time, advisor, ride, and estimate. When they agree, book_visit again, which books it.

Already booked: to move an appointment, find_times with its appointment_id, and call reschedule_appointment the moment they pick a time. To cancel, call cancel_appointment the moment they ask. Each returns what to read back: that read-back is the only confirmation, so don't confirm in your own words first.
A change after booking (another service, the time, the ride) needs book_visit again to update it.
Prices: quote_price first, every time, and only what it returns. A repair is quoted after the diagnostic.
A spending limit ("call me before anything over five hundred"): set_approval_limit."""


def _callback_instructions(desk: ServiceDesk) -> str:
    shop = desk.shop
    assert shop is not None and desk.customer is not None
    a = shop.appointment
    findings = "\n".join(
        f"- {f.id}: {f.title}, {_dollars(f.price)} (parts {_dollars(f.parts)}, labor "
        f"{f.labor_hours:g} hours). Found: {f.found} Why: {f.why} Can it wait? {f.can_wait} "
        f"Warranty: {f.warranty} Cheaper option: {f.cheaper_option}"
        for f in shop.findings
    )
    done = ", ".join(
        f"{service_label(s, a.vehicle).lower()} ({_dollars(desk.price(s)) if desk.price(s) else 'no charge'})"
        for s in a.services
    )
    return f"""\
You're {PERSONA} at the {DEALERSHIP} service desk, calling {desk.customer.name} back about their {a.vehicle.label}. It came in this morning for {a.concern}. Done so far: {done}. Calm and plain, on their side: never a salesperson.
Today is {desk.today:%A, %B} {desk.today.day}.

{VOICE_RULES}

You've just asked whether you're speaking with {desk.customer.name.split()[0]}. Say nothing about the car or the repair until they confirm.
- Someone else, a wrong number, or they can't come to the phone: end_without_update with wrong_person.
- A bad time: end_without_update with call_back_later. You can't schedule a call back: they can call the service desk whenever suits.
- Once they confirm: go_over_findings, then tell them what it returns, the price, and when the car would be ready, and ask whether to go ahead.

What the technician found, for their questions:
{findings}

- Record each decision with decide_repair. Each result says what to bring up next: follow it. Be honest about urgency once, without pushing.
- How they'll get the car back: arrange_pickup.
- Every total and ready time you say comes from the latest tool result; when one changes, say so.
- Something new they've noticed: note_new_concern.
- When nothing's left, tell them when it's ready and the total due, ask if there's anything else, then say goodbye and end_call."""


def caller_script(desk: ServiceDesk) -> str:
    """Who the caller is and what they want right now, for the suggested replies."""
    return _callback_script(desk) if desk.shop else _inbound_script(desk)


def _inbound_script(desk: ServiceDesk) -> str:
    visit = desk.visit
    car = visit.vehicle
    facts = """\
The caller is Alex Morgan, a Bayside BMW customer trying out Nick's phone line. The only details to use:
- Phone: four one five, five five five, zero one four two
- The car: their X3
- The problem: a grinding noise from the front when they brake, for about a week. No warning lights, nothing done to the car recently.
They want to bring it in, add any recall, take the first time Nick offers, and ride the shuttle."""
    if desk.customer is None:
        stage = "Give the name when Nick asks for it, then the phone number."
    elif car is None:
        stage = "It's the X3."
    elif "diagnostic" not in visit.services:
        stage = "Answer only what Nick just asked about the problem, one detail at a time. A diagnostic is fine."
    elif car.recall and "recall" not in visit.services:
        stage = "Yes, add the recall."
    elif visit.slot is None:
        stage = "Take the first time Nick offers."
    elif visit.ride is None:
        stage = "The shuttle, please."
    elif visit.booked is None:
        stage = "When Nick reads it back, yes, that's right."
    else:
        stage = "It's booked. When Nick asks if there's anything else, that's all, thanks, goodbye."
    return f"{facts}\n{stage}"


def _callback_script(desk: ServiceDesk) -> str:
    shop = desk.shop
    assert shop is not None
    facts = """\
Nick from Bayside BMW called the caller, Alex Morgan, about their X3, which is in the shop for a grinding noise. The only details to use:
- They want the front brakes done now.
- The rear brakes can wait until the next visit.
- They'll take the shuttle back to pick it up, from home."""
    if not shop.findings_shared:
        stage = "Yes, this is Alex."
    elif "front_brakes" not in shop.decisions:
        stage = "Yes, go ahead with the front brakes."
    elif "rear_brakes" not in shop.decisions:
        stage = "The rear brakes can wait until next time."
    elif shop.ride is None:
        stage = "The shuttle, please, from home."
    else:
        stage = (
            "When Nick goes over the ready time and total, that's all, thanks, goodbye."
        )
    return f"{facts}\n{stage}"
