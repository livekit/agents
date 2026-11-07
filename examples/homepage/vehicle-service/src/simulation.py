"""Grades a simulated call on what the desk ended up with, against the scenario's `expected`.

Inbound keys, all optional:
    booked: true if the visit must be booked on the call, false if nothing may be
    vehicle: the booked car's model, e.g. "X3" or "3 Series"
    services: the exact services booked, e.g. ["diagnostic", "recall"]
    day: the booked day, YYYY-MM-DD
    ride: the ride, or a list of acceptable ones
    unsafe_to_drive: true if the booking must carry the not-safe-to-drive flag
    approval_limit: the "call me before anything over" amount
    new_file: true if the caller must end up with a new file
    mileage: the mileage on the visit's car at the end
    on_file: what became of the appointment already on file: kept (the default), moved,
        or cancelled
Callback keys:
    decisions: repair id to decision, e.g. {front_brakes: approved}
    ride: shuttle or self
    ended_early: wrong_person or call_back_later; leave it out when the call must go through
        the findings
    concern_noted: true if something new they noticed was noted
"""

from __future__ import annotations

from typing import Any

from livekit.agents import SimulationContext

from service_desk import ServiceDesk

ON_FILE = "RO-48194"


async def on_simulation_end(sim: SimulationContext) -> None:
    expected = sim.userdata().get("expected")
    if not expected:
        return
    desk: ServiceDesk = sim.job_context.primary_session.userdata
    if problems := grade(desk, expected):
        sim.fail(reason="; ".join(problems))


def grade(desk: ServiceDesk, expected: dict[str, Any]) -> list[str]:
    return (
        _grade_callback(desk, expected) if desk.shop else _grade_inbound(desk, expected)
    )


def _grade_inbound(desk: ServiceDesk, expected: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    booked = (
        desk.visit.booked
        if desk.visit.booked and desk.visit.booked.status == "booked"
        else None
    )

    on_file = next(a for a in desk.appointments if a.id == ON_FILE)
    became = (
        "cancelled"
        if on_file.status == "cancelled"
        else "moved"
        if on_file.moved_from
        else "kept"
    )
    if became != (want := expected.get("on_file", "kept")):
        problems.append(f"the appointment on file was {became}, expected {want}")

    if expected.get("new_file") and not (desk.customer and desk.customer.new):
        problems.append("never opened a new file")
    if (mileage := expected.get("mileage")) is not None:
        car = desk.visit.vehicle
        if car is None or car.miles != mileage:
            problems.append(f"mileage {car.miles if car else None}, expected {mileage}")

    if "booked" in expected and bool(booked) != expected["booked"]:
        problems.append("booked a visit" if booked else "never booked the visit")
    if booked is None:
        return problems
    if (model := expected.get("vehicle")) and booked.vehicle.model != model:
        problems.append(f"booked the {booked.vehicle.model}, expected the {model}")
    if "services" in expected and sorted(booked.services) != sorted(
        expected["services"]
    ):
        problems.append(f"services {booked.services}, expected {expected['services']}")
    if (day := expected.get("day")) and booked.slot.starts_at.date().isoformat() != day:
        problems.append(f"booked {booked.slot.id}, expected {day}")
    if "ride" in expected:
        rides = (
            expected["ride"]
            if isinstance(expected["ride"], list)
            else [expected["ride"]]
        )
        if booked.ride not in rides:
            problems.append(f"ride {booked.ride}, expected {' or '.join(rides)}")
    if expected.get("unsafe_to_drive") and not booked.unsafe:
        problems.append("booked without the not-safe-to-drive flag")
    if (
        limit := expected.get("approval_limit")
    ) is not None and booked.approval_limit != limit:
        problems.append(f"approval limit {booked.approval_limit}, expected {limit}")
    return problems


def _grade_callback(desk: ServiceDesk, expected: dict[str, Any]) -> list[str]:
    shop = desk.shop
    assert shop is not None
    problems: list[str] = []
    if shop.ended_early != expected.get("ended_early"):
        problems.append(
            f"ended early as {shop.ended_early}, expected {expected.get('ended_early')}"
        )
    for repair, want in (expected.get("decisions") or {}).items():
        if (got := shop.decisions.get(repair)) != want:
            problems.append(f"{repair} {got}, expected {want}")
    if (ride := expected.get("ride")) and shop.ride != ride:
        problems.append(f"ride {shop.ride}, expected {ride}")
    if expected.get("concern_noted") and not shop.concerns:
        problems.append("never noted the new concern")
    return problems
