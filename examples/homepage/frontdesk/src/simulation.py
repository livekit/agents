"""Grades a simulated call on what the clinic ended up with, against the scenario's `expected`.

Keys, all optional:
    new_patient: the name a new chart must be registered under, or false if none may be
    booked: morning or afternoon if a new appointment must be booked then, or false if none may
        be
    moved: morning or afternoon if an appointment must be moved to then, or false if none may
        move
"""

from __future__ import annotations

from typing import Any

from livekit.agents import SimulationContext

from clinic import PATIENTS_ON_FILE, Appointment, Clinic
from receptionist import CallState


async def on_simulation_end(sim: SimulationContext) -> None:
    expected = sim.userdata().get("expected")
    if not expected:
        return
    state: CallState = sim.job_context.primary_session.userdata
    assert state.clinic is not None
    if problems := grade(state.clinic, expected):
        sim.fail(reason="; ".join(problems))


def grade(clinic: Clinic, expected: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if "new_patient" in expected:
        registered = [name for name in clinic.patients if name not in PATIENTS_ON_FILE]
        want = expected["new_patient"]
        if not want and registered:
            problems.append(f"registered {registered[0]} as a new patient")
        elif want and [n.casefold() for n in registered] != [want.casefold()]:
            problems.append(
                f"registered {', '.join(registered) or 'nobody'}, expected {want}"
            )

    seeded = {a.id: a.starts_at for a in Clinic(today=clinic.today).appointments}
    added = [a for a in clinic.appointments if a.id not in seeded]
    moved = [
        a for a in clinic.appointments if a.id in seeded and a.starts_at != seeded[a.id]
    ]
    for key, appointments in (("booked", added), ("moved", moved)):
        if key in expected:
            problems += _check(key, appointments, expected[key])
    return problems


def _check(key: str, appointments: list[Appointment], want: str | bool) -> list[str]:
    if want is False:
        return [f"{key} an appointment it shouldn't have"] if appointments else []
    if len(appointments) != 1:
        return [f"{key} {len(appointments)} appointments, expected one in the {want}"]
    part = "morning" if appointments[0].starts_at.hour < 12 else "afternoon"
    return [] if part == want else [f"{key} one for the {part}, expected the {want}"]
