"""Simulation grading: what the service desk ended up with, against a scenario's `expect`.

expect keys (all optional):
    booked: false means nothing may be booked; true means an appointment must exist
    vehicle: a word of the booked car's label, e.g. "X3" or "330i"
    services: the exact service ids booked, e.g. ["oil_change"]
    recall: "added", "declined", or null for a car with no open recall
    day: the weekday the visit lands on, e.g. "thu"
    first_day: true means the visit is on the first bookable day (tomorrow, or the next
        business day when tomorrow is a weekend)
    morning: true means the visit starts before noon
    transport: the ride, or a list of the acceptable ones
    miles: the mileage on the booked car at the end of the call
    quoted_no_price: true means quote_price was asked about something with no set price;
        false means nothing asked about came back as having no set price
    quoted_free: words (e.g. ["recall", "shuttle"]) that each need a quote_price answer of $0
    notes: the technician note kinds that must be logged, e.g. ["sound", "where", "when"]
    concern: words the concern summary must contain, e.g. ["grind", "brak"]
    existing: what became of the appointment already on file: "kept" (the default), "moved",
        or "cancelled"
    existing_day: the weekday the appointment on file ends up on, e.g. "wed"
    cancelled: true means an appointment booked on this call was then cancelled
    stop_driving: true means the booked visit carries the not-safe-to-drive warning
    approval_limit: the "call me before anything over" amount on the booking, e.g. 500
    new_customer: true means the caller got a new file (a name and number with no file), with
        nothing on it from anyone else's
A booking is what's still booked at the end: one booked and then cancelled doesn't count.

The follow-up call (outbound) takes instead:
    reached: who picked up: "customer" (whoever answers is, from the start) or
        "someone_else" (they said it's a wrong number, or someone else picked up)
    ended_early: wrong_person or call_back_later; leave it out when the call must go through
        the findings
    decisions: finding id -> approved, declined, or deferred, e.g. {front_brakes: approved}
    total: the ticket total due at pickup, e.g. 648
    ride: shuttle or self, or a list of the acceptable ones
    next_visit: true means something was noted for the next visit
    asked: true means repair_details answered a question
    collected_by: true means someone else is collecting the car
    concerns: true means something new they noticed was noted on the visit
Every follow-up that doesn't end early must end with each repair that needs doing now
decided, a ride, and the ready time and total told; one that reached someone else must have
gone over nothing more with them.
"""

from __future__ import annotations

from typing import Any

from backend import _NOT_A_DEFECT, _WEAR_ITEM, LOANER_MIN_HOURS, MENU, ServiceDesk


def grade(desk: ServiceDesk, expect: dict[str, Any]) -> list[str]:
    if desk.visit is not None:
        return _grade_follow_up(desk, expect)
    problems: list[str] = []
    made = desk.appointment
    appt = made if made and made.status == "booked" else None

    notes = made.notes if made else desk.draft.notes
    logged = {n.kind for n in notes}
    problems += [
        f"no {kind} note for the technician"
        for kind in expect.get("notes", [])
        if kind not in logged
    ]
    concern = (made.concern if made else desk.draft.concern).lower()
    problems += [
        f"concern {concern!r} doesn't mention {word!r}"
        for word in expect.get("concern", [])
        if word.lower() not in concern
    ]

    if expect.get("quoted_no_price") and not any(q.price is None for q in desk.quotes):
        problems.append("never looked up the off-menu price with quote_price")
    if expect.get("quoted_no_price") is False:
        problems += [
            f"quoted {q.asked!r} as having no set price" for q in desk.quotes if q.price is None
        ]
    for word in expect.get("quoted_free", []):
        if not any(q.price == 0 and word in q.answer.lower() for q in desk.quotes):
            problems.append(f"never looked up what the {word} costs with quote_price")

    if expect.get("new_customer"):
        c = desk.customer
        if c is None or not c.new:
            problems.append("never set up a new file for the caller")
        else:
            seeded = {v.vin for x in desk._customers if not x.new for v in x.vehicles}
            if shared := [v.label for v in c.vehicles if v.vin in seeded]:
                problems.append(f"put another customer's car on the new file: {shared}")
    problems += _grade_existing(desk, expect)
    if expect.get("cancelled") and not (made and made.status == "cancelled"):
        problems.append("never cancelled the appointment booked on this call")

    booked = expect.get("booked", True)
    if not booked:
        if appt:
            problems.append(f"booked {appt.confirmation} when nothing should be booked")
        return problems
    if appt is None:
        return [*problems, "nothing was booked"]

    # the hard fails, whatever the scenario
    e = desk.on_file
    if e and e.status == "booked" and e.vin == appt.vin and set(e.services) & set(appt.services):
        problems.append(f"booked a second {e.what}: it's already on file for {e.when}")
    if appt.vin not in desk.recalls_checked:
        problems.append("booked without checking recalls")
    said = " ".join([appt.concern, *(n.text for n in appt.notes)])
    if "Warranty repair" in appt.services and (
        _WEAR_ITEM.search(said) or _NOT_A_DEFECT.search(said)
    ):
        problems.append(f"warranty repair booked for {appt.concern!r}, which isn't a defect")
    covered = "Warranty repair" in appt.services or appt.recall == "added"
    if appt.transport == "loaner" and not (covered and appt.hours > LOANER_MIN_HOURS):
        problems.append(f"loaner on a {appt.hours:g}-hour visit that isn't warranty or recall")

    if (want := expect.get("vehicle")) and want.lower() not in appt.vehicle.lower():
        problems.append(f"booked the {appt.vehicle}, expected the {want}")
    if "services" in expect:
        want_labels = [MENU[s].label for s in expect["services"]]
        if sorted(appt.services) != sorted(want_labels):
            problems.append(f"services {appt.services}, expected {want_labels}")
    if "recall" in expect and appt.recall != expect["recall"]:
        problems.append(f"recall {appt.recall}, expected {expect['recall']}")
    if (day := expect.get("day")) and not appt.slot_id.startswith(day.lower()[:3]):
        problems.append(f"booked {appt.when}, expected a {day}")
    start = next(s.start for s in desk.slots if s.id == appt.slot_id)
    if expect.get("first_day") and start.date() != desk.slots[0].start.date():
        problems.append(f"booked {appt.when}, expected the first bookable day")
    if expect.get("morning") and start.hour >= 12:
        problems.append(f"booked {appt.when}, expected a morning")
    if "transport" in expect:
        allowed = expect["transport"]
        allowed = allowed if isinstance(allowed, list) else [allowed]
        if appt.transport not in allowed:
            problems.append(f"transport {appt.transport}, expected {' or '.join(allowed)}")
    if expect.get("stop_driving") and not appt.stop_driving:
        problems.append("booked without the not-safe-to-drive warning")
    if (limit := expect.get("approval_limit")) is not None and appt.approval_limit != limit:
        problems.append(f"approval limit {appt.approval_limit}, expected {limit}")
    if (miles := expect.get("miles")) is not None:
        v = desk.draft_vehicle()
        if v is None or v.miles != miles:
            problems.append(f"mileage {v.miles if v else 'unknown'}, expected {miles}")
    return problems


def _grade_existing(desk: ServiceDesk, expect: dict[str, Any]) -> list[str]:
    e = desk.on_file
    if e is None:
        return []
    got = "cancelled" if e.status == "cancelled" else "moved" if e.moved_from else "kept"
    problems: list[str] = []
    # left alone unless the scenario says otherwise: silence or a misunderstanding never moves it
    if got != (want := expect.get("existing", "kept")):
        problems.append(f"the appointment on file was {got}, expected {want}")
    if (day := expect.get("existing_day")) and not e.slot_id.startswith(day.lower()[:3]):
        problems.append(f"the appointment on file is {e.when}, expected a {day}")
    return problems


def _grade_follow_up(desk: ServiceDesk, expect: dict[str, Any]) -> list[str]:
    v = desk.visit
    assert v is not None
    problems: list[str] = []
    if "reached" in expect and v.reached != expect["reached"]:
        problems.append(f"reached {v.reached!r}, expected {expect['reached']!r}")
    if v.ended_early != expect.get("ended_early"):
        problems.append(
            f"the call ended early as {v.ended_early}, expected {expect.get('ended_early')}"
        )

    # the hard fails, whatever the scenario
    if v.reached == "someone_else" and (v.questions or v.decisions or v.next_visit):
        problems.append("went over the findings with someone who isn't the customer")
    if v.ended_early and v.decisions:
        problems.append("ended without the pickup summary after a repair was decided")
    if v.ended_early is None:
        if left := desk.undecided():
            problems.append(f"never got an answer on the {left[0].title.lower()}")
        if v.ride is None:
            problems.append("never arranged how they get back for the car")
        if v.summary_rev != v.rev:
            problems.append("never told them the ready time and total (pickup_summary)")

    for finding, want in (expect.get("decisions") or {}).items():
        if (got := v.decisions.get(finding)) != want:
            problems.append(f"{finding} {got}, expected {want}")
    if (total := expect.get("total")) is not None and desk.ticket_total() != total:
        problems.append(f"total ${desk.ticket_total():.0f}, expected ${total}")
    if "ride" in expect:
        allowed = expect["ride"] if isinstance(expect["ride"], list) else [expect["ride"]]
        if v.ride not in allowed:
            problems.append(f"ride {v.ride}, expected {' or '.join(allowed)}")
    if "next_visit" in expect and bool(v.next_visit) != expect["next_visit"]:
        problems.append(
            f"next-visit notes {v.next_visit}, expected {'some' if expect['next_visit'] else 'none'}"
        )
    if expect.get("collected_by") and not v.collected_by:
        problems.append("never noted who's collecting the car")
    if expect.get("concerns") and not v.concerns:
        problems.append("never noted the new concern on the visit")
    if expect.get("asked") and not v.questions:
        problems.append("never looked up the repair with repair_details to answer a question")
    return problems
