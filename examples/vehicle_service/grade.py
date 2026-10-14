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

The follow-up call (outbound) takes instead:
    reached: who picked up, "customer" or "someone_else"
    ended_early: wrong_person or call_back_later; leave it out when the call must go through
        the findings
    decisions: finding id -> approved, declined, or deferred, e.g. {front_brakes: approved}
    total: the ticket total due at pickup, e.g. 648
    ride: shuttle or self, or a list of the acceptable ones
    next_visit: true means something was noted for the next visit
    asked: true means repair_details answered a question
Every follow-up that doesn't end early must end with each repair that needs doing now
decided, a ride, and the ready time and total told; one that reached someone else must have
shared nothing.
"""

from __future__ import annotations

from typing import Any

from backend import LOANER_MIN_HOURS, MENU, ServiceDesk


def grade(desk: ServiceDesk, expect: dict[str, Any]) -> list[str]:
    if desk.visit is not None:
        return _grade_follow_up(desk, expect)
    problems: list[str] = []
    appt = desk.appointment

    notes = appt.notes if appt else desk.draft.notes
    logged = {n.kind for n in notes}
    problems += [
        f"no {kind} note for the technician"
        for kind in expect.get("notes", [])
        if kind not in logged
    ]
    concern = (appt.concern if appt else desk.draft.concern).lower()
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

    booked = expect.get("booked", True)
    if not booked:
        if appt:
            problems.append(f"booked {appt.confirmation} when nothing should be booked")
        return problems
    if appt is None:
        return [*problems, "nothing was booked"]

    # the hard fails, whatever the scenario
    if appt.vin not in desk.recalls_checked:
        problems.append("booked without checking recalls")
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
    if (miles := expect.get("miles")) is not None:
        v = desk.draft_vehicle()
        if v is None or v.miles != miles:
            problems.append(f"mileage {v.miles if v else 'unknown'}, expected {miles}")
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
    if expect.get("asked") and not v.questions:
        problems.append("never looked up the repair with repair_details to answer a question")
    return problems
