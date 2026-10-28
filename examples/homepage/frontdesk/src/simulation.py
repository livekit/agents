"""Grades a simulated call on what the clinic ended up with, against the scenario's `expected`.

Keys, all optional:
    patient: the name the caller must end the call verified or registered as; anything booked
        or moved must be theirs
    born: the date of birth, YYYY-MM-DD, that the patient's chart must have
    new_patient: true if the caller must be registered as a new patient, false if they mustn't
    booked: morning or afternoon if a new appointment must be booked then, or false if none may
        be
    moved: morning or afternoon if an appointment must be moved to then, or false if none may
        move
"""

from __future__ import annotations

from datetime import date
from typing import Any

from livekit.agents import SimulationContext

from clinic import PATIENTS_ON_FILE, Appointment, Clinic
from receptionist import CallState


async def on_simulation_end(sim: SimulationContext) -> None:
    expected = sim.userdata().get("expected")
    if not expected:
        return
    state: CallState = sim.job_context.primary_session.userdata
    if problems := grade(state, expected):
        sim.fail(reason="; ".join(problems))


def grade(state: CallState, expected: dict[str, Any]) -> list[str]:
    clinic = state.clinic
    assert clinic is not None
    problems: list[str] = []
    caller = state.caller
    patient = expected.get("patient")
    if patient and (caller is None or caller.name.casefold() != patient.casefold()):
        problems.append(
            f"verified {caller.name if caller else 'nobody'}, expected {patient}"
        )
    elif caller and (born := expected.get("born")):
        chart = clinic.patients.get(caller.name)
        if chart != date.fromisoformat(str(born)):
            problems.append(
                f"{caller.name}'s chart has birthday {chart}, expected {born}"
            )

    if "new_patient" in expected:
        registered = [name for name in clinic.patients if name not in PATIENTS_ON_FILE]
        if bool(registered) != expected["new_patient"]:
            problems.append(
                f"registered {', '.join(registered)} as a new patient"
                if registered
                else "never registered a new patient"
            )

    seeded = {a.id: a.starts_at for a in Clinic(today=clinic.today).appointments}
    added = [a for a in clinic.appointments if a.id not in seeded]
    moved = [
        a for a in clinic.appointments if a.id in seeded and a.starts_at != seeded[a.id]
    ]
    for key, appointments in (("booked", added), ("moved", moved)):
        if key in expected:
            problems += _check(key, appointments, expected[key], patient)
    return problems


def _check(
    key: str, appointments: list[Appointment], want: str | bool, patient: str | None
) -> list[str]:
    if want is False:
        return [f"{key} an appointment it shouldn't have"] if appointments else []
    if len(appointments) != 1:
        return [f"{key} {len(appointments)} appointments, expected one in the {want}"]
    appointment = appointments[0]
    if patient and appointment.patient.casefold() != patient.casefold():
        return [f"{key} one for {appointment.patient}, expected {patient}"]
    part = "morning" if appointment.starts_at.hour < 12 else "afternoon"
    return [] if part == want else [f"{key} one for the {part}, expected the {want}"]
