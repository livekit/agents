from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

SCHEDULE = json.loads((Path(__file__).parent / "schedule.json").read_text())

CLINIC_NAME = SCHEDULE["clinic"]
DOCTOR = SCHEDULE["doctor"]
# most people trying the demo are on the West Coast
CLINIC_TZ = ZoneInfo("America/Los_Angeles")
CLINIC_TZ_NAME = "Pacific time"
CLINIC_DAYS = len(SCHEDULE["busy"])
SLOT_TIMES = [time.fromisoformat(t) for t in SCHEDULE["times"]]
PATIENTS_ON_FILE = {
    p["name"]: date.fromisoformat(p["dob"]) for p in SCHEDULE["patients"]
}

PartOfDay = Literal["morning", "afternoon", "any"]


class ClinicError(Exception):
    pass


@dataclass
class Appointment:
    id: str
    patient: str
    starts_at: datetime
    reason: str
    confirmed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "patient": self.patient,
            "startsAt": self.starts_at.isoformat(),
            "reason": self.reason,
            "confirmed": self.confirmed,
            "notes": self.notes,
        }


Publish = Callable[[str, Any], Awaitable[None]]


async def _discard(_key: str, _value: Any) -> None:
    return None


@dataclass
class Clinic:
    today: date
    publish: Publish = _discard
    patients: dict[str, date] = field(default_factory=lambda: dict(PATIENTS_ON_FILE))
    appointments: list[Appointment] = field(default_factory=list)
    busy: set[datetime] = field(default_factory=set)

    def __post_init__(self) -> None:
        days = self.clinic_days()
        self.busy = {
            _at(day, time.fromisoformat(t))
            for day, times in zip(days, SCHEDULE["busy"], strict=True)
            for t in times
        }
        for a in SCHEDULE["appointments"]:
            self._add(
                a["patient"],
                _at(days[a["day"]], time.fromisoformat(a["time"])),
                a["reason"],
            )

    @classmethod
    def for_now(cls, publish: Publish = _discard) -> Clinic:
        return cls(today=datetime.now(CLINIC_TZ).date(), publish=publish)

    def clinic_days(self) -> list[date]:
        days: list[date] = []
        day = self.today
        while len(days) < CLINIC_DAYS:
            day += timedelta(days=1)
            if day.weekday() < 5:
                days.append(day)
        return days

    def is_open(self, starts_at: datetime) -> bool:
        return (
            starts_at.date() in self.clinic_days()
            and starts_at.time() in SLOT_TIMES
            and starts_at not in self.busy
            and all(a.starts_at != starts_at for a in self.appointments)
        )

    def upcoming(self, patient: str) -> list[Appointment]:
        return [a for a in self.appointments if a.patient == patient]

    def born_on(self, date_of_birth: date) -> list[str]:
        return [name for name, born in self.patients.items() if born == date_of_birth]

    async def verify(self, name: str, date_of_birth: date) -> bool:
        if self.patients.get(name) != date_of_birth:
            return False
        await self._publish_caller(name)
        return True

    async def register(self, name: str, date_of_birth: date) -> None:
        name = name.strip()
        if not name:
            raise ClinicError("A name is required.")
        if name in self.patients:
            raise ClinicError(
                f"{name} already has a chart. Read the whole date of birth back to "
                "the caller to check it."
            )
        self.patients[name] = date_of_birth
        await self._publish_caller(name)

    async def check_availability(
        self, earliest: date, latest: date, part_of_day: PartOfDay = "any"
    ) -> list[datetime]:
        if latest < earliest:
            raise ClinicError("latest must be on or after earliest.")
        openings = [
            _at(d, t)
            for d in self.clinic_days()
            if earliest <= d <= latest
            for t in SLOT_TIMES
            if _in_part(t, part_of_day) and self.is_open(_at(d, t))
        ]
        await self.publish(
            "search",
            {
                "earliest": earliest.isoformat(),
                "latest": latest.isoformat(),
                "partOfDay": part_of_day,
                "openings": [o.isoformat() for o in openings],
            },
        )
        return openings

    async def book(self, patient: str, starts_at: datetime, reason: str) -> Appointment:
        self._require_open(starts_at)
        appointment = self._add(patient, starts_at, reason)
        await self.publish("search", None)
        await self._publish_appointments()
        return appointment

    async def reschedule(
        self, patient: str, appointment_id: str, starts_at: datetime
    ) -> Appointment:
        appointment = self._owned(patient, appointment_id)
        self._require_open(starts_at)
        appointment.starts_at = starts_at
        await self.publish("search", None)
        await self._publish_appointments()
        return appointment

    async def confirm(self, patient: str, appointment_id: str) -> Appointment:
        appointment = self._owned(patient, appointment_id)
        appointment.confirmed = True
        await self._publish_appointments()
        return appointment

    async def add_note(
        self, patient: str, appointment_id: str, note: str
    ) -> Appointment:
        note = note.strip()
        if not note:
            raise ClinicError("The note is empty. Use the patient's own words.")
        appointment = self._owned(patient, appointment_id)
        appointment.notes.append(note)
        await self._publish_appointments()
        return appointment

    def _owned(self, patient: str, appointment_id: str) -> Appointment:
        appointment = next(
            (a for a in self.upcoming(patient) if a.id == appointment_id), None
        )
        if appointment is None:
            raise ClinicError(f"{patient} has no appointment {appointment_id}.")
        return appointment

    def _add(self, patient: str, starts_at: datetime, reason: str) -> Appointment:
        appointment = Appointment(
            f"A-{101 + len(self.appointments)}", patient, starts_at, reason
        )
        self.appointments.append(appointment)
        return appointment

    async def _publish_caller(self, name: str) -> None:
        await self.publish(
            "caller", {"name": name, "new": name not in PATIENTS_ON_FILE}
        )

    def _require_open(self, starts_at: datetime) -> None:
        if not self.is_open(starts_at):
            raise ClinicError(
                f"{starts_at:%a %b %-d %-I:%M %p} isn't open. Check availability again."
            )

    async def _publish_appointments(self) -> None:
        await self.publish("appointments", [a.to_json() for a in self.appointments])


def _at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, tzinfo=CLINIC_TZ)


def _in_part(t: time, part_of_day: PartOfDay) -> bool:
    if part_of_day == "morning":
        return t.hour < 12
    if part_of_day == "afternoon":
        return t.hour >= 12
    return True
