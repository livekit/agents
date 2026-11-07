"""Bayside BMW's service desk for one call: the customer's file, the visit being put together, the
calendar, and, on the callback, the car that's in the shop. Every rule lives here and refuses with
a ServiceDeskError. Language never does: each choice arrives as a typed value or an id."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

SEED = json.loads((Path(__file__).parent / "bayside.json").read_text())

DEALERSHIP: str = SEED["dealership"]
SHOP_TZ = ZoneInfo(SEED["timezone"])
CLOSES = time.fromisoformat(SEED["closes"])
LOANER_MIN_HOURS: float = SEED["loaner_min_hours"]
SHUTTLE_MILES: int = SEED["shuttle_miles"]

Service = Literal[
    "oil_change", "tire_rotation", "diagnostic", "warranty_repair", "recall"
]
Ride = Literal["wait", "shuttle", "loaner"]
Sound = Literal[
    "grinding", "squealing", "clunking", "humming", "clicking", "rattling", "other"
]
Moment = Literal[
    "braking",
    "turning",
    "over_bumps",
    "at_speed",
    "idling",
    "cold_start",
    "all_the_time",
    "other",
]
UnsafeReason = Literal[
    "brakes_failing",
    "smoke_or_burning",
    "steering_failing",
    "red_or_flashing_warning_light",
    "overheating",
]
WarrantyPart = Literal[
    "electrical",
    "infotainment_or_electronics",
    "window_or_door",
    "liftgate_or_trunk",
    "climate",
    "engine",
    "transmission",
    "fluid_leak",
]
BmwModel = Literal[
    "1 Series",
    "2 Series",
    "3 Series",
    "4 Series",
    "5 Series",
    "7 Series",
    "8 Series",
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "XM",
    "Z4",
    "i4",
    "i5",
    "i7",
    "iX",
    "M2",
    "M3",
    "M4",
    "M5",
    "M8",
]
Priceable = Literal[
    "oil_change",
    "tire_rotation",
    "diagnostic",
    "warranty_repair",
    "recall",
    "shuttle",
    "loaner",
    "waiting_lounge",
    "this_visit",
    "something_else",
]
PartOfDay = Literal["morning", "afternoon", "any"]
RepairId = Literal["front_brakes", "rear_brakes"]
RepairDecision = Literal["approved", "declined", "next_visit", "thinking_it_over"]
PickupRide = Literal["shuttle", "self"]
EndedEarly = Literal["wrong_person", "call_back_later"]

MENU: dict[str, dict[str, Any]] = SEED["menu"]
PRICED_SERVICES = ("oil_change", "tire_rotation", "diagnostic")


class ServiceDeskError(Exception):
    """A rule the desk won't break. The message says what to do instead."""


Publish = Callable[[str, Any], Awaitable[None]]


async def _discard(_key: str, _value: Any) -> None:
    return None


@dataclass(frozen=True)
class Recall:
    id: str
    title: str
    hours: float


def open_recall(model: str, year: int) -> Recall | None:
    for r in SEED["recalls"]:
        if r["model"] == model and r["years"][0] <= year <= r["years"][1]:
            return Recall(r["id"], r["title"], r["hours"])
    return None


@dataclass
class Vehicle:
    id: str
    year: int
    model: str
    miles: int
    in_service: date
    trim: str | None = None
    added_on_call: bool = False

    @property
    def name(self) -> str:
        return self.trim or self.model

    @property
    def label(self) -> str:
        return f"{self.year} BMW {self.name}"

    @property
    def recall(self) -> Recall | None:
        return open_recall(self.model, self.year)

    def in_warranty(self, today: date) -> bool:
        years = (today - self.in_service).days / 365.25
        return (
            years < SEED["warranty"]["years"] and self.miles < SEED["warranty"]["miles"]
        )

    def to_json(self, today: date) -> dict[str, Any]:
        years = (today - self.in_service).days / 365.25
        recall = self.recall
        return {
            "vehicleId": self.id,
            "car": self.label,
            "miles": self.miles,
            "inWarranty": self.in_warranty(today),
            "yearsInService": round(years, 1),
            "openRecall": None
            if recall is None
            else {"recallId": recall.id, "title": recall.title, "hours": recall.hours},
            "addedOnCall": self.added_on_call,
        }


@dataclass
class Customer:
    name: str
    phone: str
    vehicles: list[Vehicle]
    new: bool = False

    def vehicle(self, vehicle_id: str) -> Vehicle:
        for v in self.vehicles:
            if v.id == vehicle_id:
                return v
        if not self.vehicles:
            raise ServiceDeskError(
                "There are no cars on this file yet: use add_vehicle."
            )
        ids = ", ".join(f"{v.id} ({v.label})" for v in self.vehicles)
        raise ServiceDeskError(f"No car {vehicle_id!r} on this file. On file: {ids}.")


@dataclass
class Slot:
    starts_at: datetime
    advisor: str
    taken: bool = False

    @property
    def id(self) -> str:
        return self.starts_at.isoformat(timespec="minutes")

    def fits(self, hours: float) -> bool:
        ends = self.starts_at + timedelta(hours=hours)
        return ends.date() == self.starts_at.date() and ends.time() <= CLOSES


@dataclass
class Appointment:
    id: str
    vehicle: Vehicle
    services: list[Service]
    slot: Slot
    ride: Ride
    concern: str = ""
    approval_limit: int | None = None
    unsafe: UnsafeReason | None = None
    status: Literal["booked", "cancelled"] = "booked"
    moved_from: datetime | None = None
    booked_on_call: bool = False

    @property
    def hours(self) -> float:
        return visit_hours(self.vehicle, self.services)

    @property
    def estimate(self) -> int:
        return sum(MENU[s]["price"] for s in self.services if s in MENU)

    def to_json(self) -> dict[str, Any]:
        return {
            "appointmentId": self.id,
            "car": self.vehicle.label,
            "services": [service_label(s, self.vehicle) for s in self.services],
            "concern": self.concern,
            "when": spoken(self.slot.starts_at),
            "startsAt": self.slot.id,
            "advisor": self.slot.advisor,
            "ride": self.ride,
            "estimate": self.estimate,
            "approvalLimit": self.approval_limit,
            "unsafeToDrive": self.unsafe,
            "status": self.status,
            "movedFrom": spoken(self.moved_from) if self.moved_from else None,
            "bookedOnCall": self.booked_on_call,
        }


@dataclass
class Problem:
    summary: str = ""
    sound: Sound | None = None
    where: str = ""
    when: Moment | None = None
    since: str = ""
    warning_lights: str = ""
    recent_work: str = ""
    unsafe: UnsafeReason | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "sound": self.sound,
            "where": self.where,
            "when": self.when,
            "since": self.since,
            "warningLights": self.warning_lights,
            "recentWork": self.recent_work,
            "unsafeToDrive": self.unsafe,
        }


@dataclass(frozen=True)
class LikelyCause:
    cause: str
    advice: str


def likely_cause(sound: Sound | None, moment: Moment | None) -> LikelyCause | None:
    for c in SEED["likely_causes"]:
        if c["sound"] == sound and c["when"] == moment:
            return LikelyCause(c["cause"], c["advice"])
    return None


@dataclass
class Visit:
    """The visit being put together on this call, and the appointment it was booked as."""

    vehicle: Vehicle | None = None
    problem: Problem = field(default_factory=Problem)
    services: list[Service] = field(default_factory=list)
    failed_part: WarrantyPart | None = None
    slot: Slot | None = None
    ride: Ride | None = None
    approval_limit: int | None = None
    booked: Appointment | None = None


@dataclass(frozen=True)
class Finding:
    id: str
    title: str
    spoken: str
    found: str
    why: str
    parts: int
    labor_hours: float
    urgency: Literal["now", "next_visit"]
    can_wait: str
    warranty: str
    cheaper_option: str
    labor_rate: int

    @property
    def labor(self) -> int:
        return round(self.labor_hours * self.labor_rate)

    @property
    def price(self) -> int:
        return self.parts + self.labor


@dataclass
class ShopVisit:
    """The callback: the car is in the shop, and the technician has found the cause."""

    appointment: Appointment
    findings: list[Finding]
    decisions: dict[str, RepairDecision] = field(default_factory=dict)
    ride: PickupRide | None = None
    shuttle_from: str = ""
    collected_by: str = ""
    concerns: list[str] = field(default_factory=list)
    ended_early: EndedEarly | None = None
    findings_shared: bool = False

    def finding(self, repair_id: str) -> Finding:
        for f in self.findings:
            if f.id == repair_id:
                return f
        ids = ", ".join(f.id for f in self.findings)
        raise ServiceDeskError(f"No repair {repair_id!r}. The findings: {ids}.")


def visit_hours(vehicle: Vehicle | None, services: list[Service]) -> float:
    hours = 0.0
    for s in services:
        if s == "recall":
            recall = vehicle.recall if vehicle else None
            hours += recall.hours if recall else 0.0
        else:
            hours += MENU[s]["hours"]
    return hours


def service_label(service: Service, vehicle: Vehicle | None) -> str:
    if service == "recall":
        recall = vehicle.recall if vehicle else None
        return f"{recall.title} recall" if recall else "Recall"
    return MENU[service]["label"]


def spoken(at: datetime) -> str:
    return f"{at:%A, %B} {at.day} at {at:%-I:%M %p}"


def business_days(after: date, count: int) -> list[date]:
    days: list[date] = []
    day = after
    while len(days) < count:
        day += timedelta(days=1)
        if day.weekday() < 5:
            days.append(day)
    return days


def today_in(timezone: str | None) -> date:
    """Today for the caller: their IANA time zone when the page sent one, else the shop's."""
    try:
        tz = ZoneInfo(timezone) if timezone else SHOP_TZ
    except (ValueError, KeyError):
        tz = SHOP_TZ
    return datetime.now(tz).date()


def _phone_digits(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise ServiceDeskError(
            "A phone number is ten digits with the area code. Ask for the whole number again."
        )
    return digits


def _same_name(a: str, b: str) -> bool:
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


class ServiceDesk:
    def __init__(self, today: date, publish: Publish = _discard) -> None:
        self.today = today
        self.publish = publish
        self.days = business_days(today, SEED["bookable_days"])
        self.slots = self._seed_slots()
        self.customers = [self._seed_customer(c) for c in SEED["customers"]]
        self.appointments: list[Appointment] = []
        self.customer: Customer | None = None
        self.visit = Visit()
        self.shop: ShopVisit | None = None
        self.revision = 0
        self._next_ro = 48231
        self._seed_appointment()

    @classmethod
    def for_callback(cls, today: date, publish: Publish = _discard) -> ServiceDesk:
        """The desk on the day the customer's car is in the shop, with the technician's findings."""
        desk = cls(today, publish)
        desk._seed_shop_visit()
        return desk

    # ---- the customer's file ----

    async def find_customer(self, name: str, phone: str) -> Customer | None:
        digits = _phone_digits(phone)
        match = next(
            (
                c
                for c in self.customers
                if c.phone == digits and _same_name(c.name, name)
            ),
            None,
        )
        if match is not None:
            await self._open(match)
        return match

    async def open_new_file(self, name: str, phone: str) -> Customer:
        digits = _phone_digits(phone)
        name = " ".join(name.split())
        if not name:
            raise ServiceDeskError("Ask for their name first.")
        if any(c.phone == digits and _same_name(c.name, name) for c in self.customers):
            raise ServiceDeskError(
                "There's already a file under that name and number: use find_customer."
            )
        customer = Customer(name=name, phone=digits, vehicles=[], new=True)
        self.customers.append(customer)
        await self._open(customer)
        return customer

    async def _open(self, customer: Customer) -> None:
        if self.customer is customer:
            return
        if self.visit.vehicle is not None:
            raise ServiceDeskError(
                f"This call already has {self.customer.name if self.customer else 'a'} file "
                "open with a car on the visit."
            )
        self.customer = customer
        await self._publish_customer()
        await self._publish_appointments()

    def file_json(self) -> dict[str, Any]:
        return {
            **self._customer_json(),
            "appointments": [a.to_json() for a in self.customer_appointments()],
        }

    def _customer_json(self) -> dict[str, Any]:
        c = self._require_customer()
        return {
            "name": c.name,
            "newFile": c.new,
            "vehicles": [v.to_json(self.today) for v in c.vehicles],
        }

    def customer_appointments(self) -> list[Appointment]:
        c = self.customer
        return [
            a for a in self.appointments if c is not None and a.vehicle in c.vehicles
        ]

    # ---- the car ----

    async def choose_vehicle(
        self, vehicle_id: str, mileage: int | None = None
    ) -> list[str]:
        """Puts a car on the visit. Returns what changed with it, for the agent to say."""
        c = self._require_customer()
        v = c.vehicle(vehicle_id)
        if mileage is not None:
            if mileage < v.miles:
                raise ServiceDeskError(
                    f"The file shows {v.miles:,} miles, and mileage only goes up. Ask them to "
                    "check the odometer."
                )
            if mileage != v.miles:
                v.miles = mileage
                self._changed()
        dropped: list[str] = []
        if self.visit.vehicle is not v:
            if "recall" in self.visit.services:
                self.visit.services.remove("recall")
                dropped.append(
                    "The recall came off the visit: it was for the other car."
                )
            self.visit.vehicle = v
            self._changed()
        await self._publish_customer()
        await self._publish_visit()
        return dropped

    async def add_vehicle(self, year: int, model: BmwModel, mileage: int) -> Vehicle:
        c = self._require_customer()
        if not 1990 <= year <= self.today.year + 1:
            raise ServiceDeskError(
                "That model year doesn't look right. Ask for it again."
            )
        if not 0 <= mileage < 400_000:
            raise ServiceDeskError(
                "That mileage doesn't look right. Ask them to check it."
            )
        for v in c.vehicles:
            if v.year == year and v.model == model:
                raise ServiceDeskError(
                    f"The {v.label} is already on their file as {v.id}: use choose_vehicle."
                )
        v = Vehicle(
            id=f"V{sum(len(x.vehicles) for x in self.customers) + 1}",
            year=year,
            model=model,
            miles=mileage,
            in_service=min(date(year, 7, 1), self.today),
            added_on_call=True,
        )
        c.vehicles.append(v)
        await self.choose_vehicle(v.id)
        return v

    # ---- the problem ----

    async def describe_problem(
        self,
        *,
        summary: str | None = None,
        sound: Sound | None = None,
        where: str | None = None,
        when: Moment | None = None,
        since: str | None = None,
        warning_lights: str | None = None,
        recent_work: str | None = None,
    ) -> Problem:
        """Records what the caller said; anything left out stays as it was."""
        p = self.visit.problem
        before = p.to_json()
        if summary is not None:
            p.summary = " ".join(summary.split())
        if sound is not None:
            p.sound = sound
        if where is not None:
            p.where = where.strip()
        if when is not None:
            p.when = when
        if since is not None:
            p.since = since.strip()
        if warning_lights is not None:
            p.warning_lights = warning_lights.strip()
        if recent_work is not None:
            p.recent_work = recent_work.strip()
        if p.to_json() != before:
            self._changed()
        await self._publish_visit()
        return p

    async def flag_unsafe_to_drive(self, reason: UnsafeReason) -> None:
        if self.visit.problem.unsafe != reason:
            self.visit.problem.unsafe = reason
            self._changed()
        await self._publish_visit()

    # ---- what it needs ----

    async def add_service(
        self, service: Service, failed_part: WarrantyPart | None = None
    ) -> None:
        v = self._require_vehicle()
        if service == "recall" and v.recall is None:
            raise ServiceDeskError(f"The {v.name} has no open recall.")
        if (
            service in ("diagnostic", "warranty_repair")
            and not self.visit.problem.summary
        ):
            raise ServiceDeskError(
                "Record the problem first: describe_problem with a short summary."
            )
        if service == "warranty_repair":
            if failed_part is None:
                raise ServiceDeskError(
                    "A warranty repair needs the part that failed: pass failed_part."
                )
            if not v.in_warranty(self.today):
                raise ServiceDeskError(
                    f"The {v.name} is out of warranty ({SEED['warranty']['years']} years or "
                    f"{SEED['warranty']['miles']:,} miles). Offer a diagnostic instead."
                )
            if self.visit.failed_part != failed_part:
                self.visit.failed_part = failed_part
                self._changed()
        if service not in self.visit.services:
            self.visit.services.append(service)
            self._changed()
        await self._publish_visit()

    async def remove_service(self, service: Service) -> None:
        if service in self.visit.services:
            self.visit.services.remove(service)
            if service == "warranty_repair":
                self.visit.failed_part = None
            self._changed()
        await self._publish_visit()

    def quote(self, item: Priceable) -> dict[str, Any]:
        if item in MENU:
            entry = MENU[item]
            notes = {
                "diagnostic": "The fee to find the cause; any repair is quoted after it.",
                "warranty_repair": (
                    "No charge, for a part that's failed on a car in warranty "
                    f"({SEED['warranty']['years']} years and "
                    f"{SEED['warranty']['miles']:,} miles). Never brakes, tires, wipers, "
                    "or maintenance."
                ),
            }
            return {"item": item, "price": entry["price"], "note": notes.get(item, "")}
        if item == "recall":
            return {"item": item, "price": 0, "note": "Recall work is always free."}
        if item == "shuttle":
            return {
                "item": item,
                "price": 0,
                "note": f"Free, within {SHUTTLE_MILES} miles.",
            }
        if item == "waiting_lounge":
            return {"item": item, "price": 0, "note": "Free."}
        if item == "loaner":
            return {
                "item": item,
                "price": 0,
                "note": (
                    f"Free, but only for warranty or recall visits over "
                    f"{LOANER_MIN_HOURS:g} hours."
                ),
                "thisVisitQualifies": self.loaner_eligible(),
            }
        if item == "this_visit":
            return {
                "item": item,
                "price": self.estimate(),
                "lines": [
                    {
                        "service": service_label(s, self.visit.vehicle),
                        "price": self.price(s),
                    }
                    for s in self.visit.services
                ],
            }
        return {
            "item": item,
            "price": None,
            "note": (
                "No set price. Only the oil change, tire rotation, and diagnostic have menu "
                "prices; any repair is quoted after a diagnostic."
            ),
        }

    # ---- when, and the ride ----

    def openings(
        self,
        day: str | None = None,
        part_of_day: PartOfDay = "any",
        appointment_id: str | None = None,
    ) -> list[Slot]:
        """Open times that fit the visit (or the appointment being moved)."""
        if appointment_id:
            moving = self.appointment(appointment_id)
            hours, own = moving.hours, moving.slot
        else:
            hours, own = visit_hours(self.visit.vehicle, self.visit.services), None
        wanted: date | None = None
        if day:
            try:
                wanted = date.fromisoformat(day)
            except ValueError as e:
                raise ServiceDeskError(
                    f"Pass the day as YYYY-MM-DD, got {day!r}."
                ) from e
            if wanted not in self.days:
                raise ServiceDeskError(
                    "That day isn't bookable. Bookable days: "
                    + ", ".join(d.isoformat() for d in self.days)
                    + "."
                )
        return [
            s
            for s in self.slots
            if not s.taken
            and s is not own
            and s.fits(hours)
            and (wanted is None or s.starts_at.date() == wanted)
            and _in_part(s.starts_at, part_of_day)
        ]

    async def publish_openings(self, slots: list[Slot]) -> None:
        await self.publish("openings", [_slot_json(s) for s in slots])

    async def choose_time(self, slot_id: str) -> Slot:
        slot = self._slot(slot_id)
        held = self.visit.booked.slot if self.visit.booked else None
        if slot.taken and slot is not held:
            raise ServiceDeskError("That time is taken. Offer another.")
        hours = visit_hours(self.visit.vehicle, self.visit.services)
        if not slot.fits(hours):
            raise ServiceDeskError(
                "A visit this long won't finish by closing from that time. Offer an earlier one."
            )
        if self.visit.slot is not slot:
            self.visit.slot = slot
            self._changed()
        await self._publish_visit()
        return slot

    async def choose_ride(self, ride: Ride) -> None:
        if ride == "loaner" and not self.loaner_eligible():
            hours = visit_hours(self.visit.vehicle, self.visit.services)
            raise ServiceDeskError(
                f"Tell them the rule: loaners are only for warranty or recall visits over "
                f"{LOANER_MIN_HOURS:g} hours, and this visit is {hours:g} hours. Then offer the "
                "free shuttle or the lounge."
            )
        if self.visit.ride != ride:
            self.visit.ride = ride
            self._changed()
        await self._publish_visit()

    async def set_approval_limit(self, dollars: int) -> None:
        if dollars <= 0:
            raise ServiceDeskError("Pass the dollar amount they gave.")
        if self.visit.approval_limit != dollars:
            self.visit.approval_limit = dollars
            self._changed()
        await self._publish_visit()

    def loaner_eligible(self) -> bool:
        covered = {"warranty_repair", "recall"} & set(self.visit.services)
        hours = visit_hours(self.visit.vehicle, self.visit.services)
        return bool(covered) and hours > LOANER_MIN_HOURS

    def estimate(self) -> int:
        return sum(self.price(s) for s in self.visit.services)

    def price(self, service: Service) -> int:
        return 0 if service == "recall" else MENU[service]["price"]

    def changed_since_booked(self) -> bool:
        """The visit no longer matches what it was booked as on this call."""
        b, v = self.visit.booked, self.visit
        if b is None:
            return False
        return (
            b.vehicle,
            b.services,
            b.slot,
            b.ride,
            b.concern,
            b.approval_limit,
            b.unsafe,
        ) != (
            v.vehicle,
            v.services,
            v.slot,
            v.ride,
            v.problem.summary,
            v.approval_limit,
            v.problem.unsafe,
        )

    # ---- booking ----

    def read_back(self) -> dict[str, Any]:
        """The whole visit, checked against every rule, as it would be booked."""
        v = self._require_vehicle()
        missing = [
            what
            for what, have in (
                ("a service", self.visit.services),
                ("a time", self.visit.slot),
                ("how they'll get around", self.visit.ride),
            )
            if not have
        ]
        if missing:
            raise ServiceDeskError(
                f"Not ready to book. Still need: {', '.join(missing)}."
            )
        slot = self.visit.slot
        assert slot is not None and self.visit.ride is not None
        held = self.visit.booked.slot if self.visit.booked else None
        if slot.taken and slot is not held:
            raise ServiceDeskError(
                "The time they picked was just taken. Offer another."
            )
        if not slot.fits(visit_hours(v, self.visit.services)):
            raise ServiceDeskError(
                "The visit no longer fits that time before closing. Offer an earlier one."
            )
        if self.visit.ride == "loaner" and not self.loaner_eligible():
            raise ServiceDeskError(
                "The visit no longer qualifies for a loaner. Offer the shuttle or the lounge."
            )
        if "warranty_repair" in self.visit.services and not v.in_warranty(self.today):
            raise ServiceDeskError(
                f"The {v.name} is out of warranty now. Offer a diagnostic."
            )
        for a in self.customer_appointments():
            same_car = (
                a.status == "booked" and a is not self.visit.booked and a.vehicle is v
            )
            if same_car and (both := set(a.services) & set(self.visit.services)):
                names = ", ".join(service_label(s, v) for s in both).lower()
                raise ServiceDeskError(
                    f"The {names} is already booked for the {v.name} on "
                    f"{spoken(a.slot.starts_at)} ({a.id}). Offer to move that one instead, "
                    "or leave it off this visit."
                )
        recall = v.recall
        return {
            "car": v.label,
            "services": [
                {"service": service_label(s, v), "price": self.price(s)}
                for s in self.visit.services
            ],
            "concern": self.visit.problem.summary,
            "openRecallNotAdded": recall.title
            if recall and "recall" not in self.visit.services
            else None,
            "when": spoken(slot.starts_at),
            "advisor": slot.advisor,
            "ride": self.visit.ride,
            "estimate": self.estimate(),
            "approvalLimit": self.visit.approval_limit,
            "unsafeToDrive": self.visit.problem.unsafe,
            "updatesBooking": self.visit.booked.id if self.visit.booked else None,
        }

    async def book(self) -> Appointment:
        self.read_back()
        v, slot, ride = self.visit.vehicle, self.visit.slot, self.visit.ride
        assert v is not None and slot is not None and ride is not None
        booked = self.visit.booked
        if booked is None:
            booked = Appointment(
                id=self._ro(),
                vehicle=v,
                services=[],
                slot=slot,
                ride=ride,
                booked_on_call=True,
            )
            self.appointments.append(booked)
        else:
            booked.slot.taken = False
        slot.taken = True
        booked.vehicle = v
        booked.services = list(self.visit.services)
        booked.slot = slot
        booked.ride = ride
        booked.concern = self.visit.problem.summary
        booked.approval_limit = self.visit.approval_limit
        booked.unsafe = self.visit.problem.unsafe
        booked.status = "booked"
        self.visit.booked = booked
        self._changed()
        await self._publish_appointments()
        await self._publish_visit()
        return booked

    # ---- appointments already booked ----

    def appointment(self, appointment_id: str) -> Appointment:
        for a in self.customer_appointments():
            if a.id == appointment_id:
                if a.status == "cancelled":
                    raise ServiceDeskError(f"{a.id} is already cancelled.")
                return a
        ids = ", ".join(
            a.id for a in self.customer_appointments() if a.status == "booked"
        )
        raise ServiceDeskError(
            f"No appointment {appointment_id!r} on this file. Booked: {ids or 'none'}."
        )

    def move_read_back(self, appointment_id: str, slot_id: str) -> dict[str, Any]:
        a = self.appointment(appointment_id)
        slot = self._slot(slot_id)
        if slot is a.slot:
            raise ServiceDeskError("It's already booked for that time.")
        if slot.taken:
            raise ServiceDeskError("That time is taken. Offer another.")
        if not slot.fits(a.hours):
            raise ServiceDeskError(
                "That time won't fit it before closing. Offer an earlier one."
            )
        return {
            "appointment": a.to_json(),
            "from": spoken(a.slot.starts_at),
            "to": spoken(slot.starts_at),
            "advisor": slot.advisor,
        }

    async def move(self, appointment_id: str, slot_id: str) -> Appointment:
        self.move_read_back(appointment_id, slot_id)
        a = self.appointment(appointment_id)
        slot = self._slot(slot_id)
        a.slot.taken = False
        slot.taken = True
        a.moved_from = a.slot.starts_at
        a.slot = slot
        if a is self.visit.booked:
            self.visit.slot = slot
        self._changed()
        await self._publish_appointments()
        await self._publish_visit()
        return a

    def cancel_read_back(self, appointment_id: str) -> dict[str, Any]:
        return {"appointment": self.appointment(appointment_id).to_json()}

    async def cancel(self, appointment_id: str) -> Appointment:
        a = self.appointment(appointment_id)
        a.status = "cancelled"
        a.slot.taken = False
        if a is self.visit.booked:
            self.visit.booked = None
        self._changed()
        await self._publish_appointments()
        await self._publish_visit()
        return a

    # ---- the callback: a car in the shop ----

    async def decide(self, repair: RepairId, decision: RepairDecision) -> Finding:
        shop = self._require_shop()
        f = shop.finding(repair)
        if shop.decisions.get(f.id) != decision:
            shop.decisions[f.id] = decision
            self._changed()
        await self._publish_shop()
        return f

    async def arrange_pickup(
        self,
        ride: PickupRide | None = None,
        shuttle_from: str | None = None,
        collected_by: str | None = None,
    ) -> None:
        shop = self._require_shop()
        before = (shop.ride, shop.shuttle_from, shop.collected_by)
        if ride is not None:
            shop.ride = ride
        if shuttle_from is not None:
            shop.shuttle_from = shuttle_from.strip()
        if collected_by is not None:
            shop.collected_by = collected_by.strip()
        if shop.ride == "shuttle" and not shop.shuttle_from:
            shop.ride = before[0]
            raise ServiceDeskError(
                f"Ask where the shuttle should pick them up (within {SHUTTLE_MILES} miles)."
            )
        if (shop.ride, shop.shuttle_from, shop.collected_by) != before:
            self._changed()
        await self._publish_shop()

    async def note_concern(self, concern: str) -> None:
        shop = self._require_shop()
        concern = " ".join(concern.split())
        if not concern:
            raise ServiceDeskError("Pass what they noticed, in a few words.")
        if concern not in shop.concerns:
            shop.concerns.append(concern)
            self._changed()
        await self._publish_shop()

    async def end_early(self, reason: EndedEarly) -> None:
        shop = self._require_shop()
        if shop.decisions:
            raise ServiceDeskError(
                "They've already decided on a repair: finish with how they'll get the car back."
            )
        shop.ended_early = reason
        self._changed()
        await self._publish_shop()

    async def share_findings(self) -> Finding:
        """The repair that needs doing now, once the customer is on the line."""
        shop = self._require_shop()
        if shop.ended_early:
            raise ServiceDeskError("This call already ended without the update.")
        if not shop.findings_shared:
            shop.findings_shared = True
            self._changed()
        await self._publish_shop()
        return next(f for f in shop.findings if f.urgency == "now")

    def undecided_later(self) -> list[Finding]:
        """Repairs that can wait and haven't been brought up and decided yet."""
        shop = self._require_shop()
        return [
            f
            for f in shop.findings
            if f.urgency == "next_visit" and f.id not in shop.decisions
        ]

    def undecided(self) -> list[Finding]:
        shop = self._require_shop()
        return [
            f
            for f in shop.findings
            if f.urgency == "now" and f.id not in shop.decisions
        ]

    def ready_at(self, repairs: list[Finding] | None = None) -> datetime | None:
        """When the car's ready with these repairs (the approved ones, by default); None means
        it's ready now."""
        shop = self._require_shop()
        visit = SEED["shop_visit"]
        if repairs is None:
            repairs = [
                f for f in shop.findings if shop.decisions.get(f.id) == "approved"
            ]
        hours = sum(f.labor_hours for f in repairs)
        if not hours:
            return None
        if hours <= visit["same_day_hours"]:
            return datetime.combine(self.today, CLOSES)
        ready = time.fromisoformat(visit["next_day_ready"])
        return datetime.combine(business_days(self.today, 1)[0], ready)

    def said(self, at: datetime) -> str:
        """A time the way it's said from today: "today at 5:30 PM", "tomorrow at 10:00 AM"."""
        if at.date() == self.today:
            return f"today at {at:%-I:%M %p}"
        if at.date() == self.today + timedelta(days=1):
            return f"tomorrow at {at:%-I:%M %p}"
        return spoken(at)

    def pickup_summary(self) -> dict[str, Any]:
        shop = self._require_shop()
        ready = self.ready_at()
        shuttle_at = None
        if shop.ride == "shuttle":
            lead = timedelta(minutes=SEED["shop_visit"]["shuttle_lead_minutes"])
            shuttle_at = (
                "in about half an hour" if ready is None else self.said(ready - lead)
            )
        return {
            "readyAt": "now" if ready is None else self.said(ready),
            "totalDue": self.total_due(),
            "lines": self.ticket(),
            "ride": shop.ride,
            "shuttleFrom": shop.shuttle_from if shop.ride == "shuttle" else None,
            "shuttleAt": shuttle_at,
            "collectedBy": shop.collected_by or None,
            "stillToDecide": [f.id for f in self.undecided()],
        }

    def ticket(self) -> list[dict[str, Any]]:
        shop = self._require_shop()
        a = shop.appointment
        lines = [
            {"line": service_label(s, a.vehicle), "price": self.price(s)}
            for s in a.services
        ]
        lines += [
            {"line": f.title, "price": f.price}
            for f in shop.findings
            if shop.decisions.get(f.id) == "approved"
        ]
        return lines

    def total_due(self) -> int:
        return sum(line["price"] for line in self.ticket())

    # ---- the page ----

    async def _publish_customer(self) -> None:
        await self.publish("customer", self._customer_json() if self.customer else None)

    async def _publish_appointments(self) -> None:
        await self.publish(
            "appointments", [a.to_json() for a in self.customer_appointments()]
        )

    async def _publish_visit(self) -> None:
        v, p = self.visit.vehicle, self.visit.problem
        cause = likely_cause(p.sound, p.when)
        await self.publish(
            "visit",
            {
                "car": v.to_json(self.today) if v else None,
                "problem": p.to_json(),
                "likelyCause": cause.cause if cause else None,
                "services": [
                    {"service": s, "label": service_label(s, v), "price": self.price(s)}
                    for s in self.visit.services
                ],
                "failedPart": self.visit.failed_part,
                "slot": _slot_json(self.visit.slot) if self.visit.slot else None,
                "ride": self.visit.ride,
                "approvalLimit": self.visit.approval_limit,
                "estimate": self.estimate(),
                "hours": visit_hours(v, self.visit.services),
                "bookedAs": self.visit.booked.id if self.visit.booked else None,
            },
        )

    async def _publish_shop(self) -> None:
        shop = self._require_shop()
        await self.publish(
            "shop",
            {
                "appointment": shop.appointment.to_json(),
                "findings": [
                    {
                        "repairId": f.id,
                        "title": f.title,
                        "found": f.found,
                        "price": f.price,
                        "parts": f.parts,
                        "labor": f.labor,
                        "urgency": f.urgency,
                        "decision": shop.decisions.get(f.id),
                    }
                    for f in shop.findings
                ],
                "concerns": shop.concerns,
                "findingsShared": shop.findings_shared,
                "endedEarly": shop.ended_early,
                **self.pickup_summary(),
            },
        )

    async def publish_all(self) -> None:
        if self.shop is not None:
            await self._publish_customer()
            await self._publish_shop()
        else:
            await self._publish_visit()

    # ---- internals ----

    def _changed(self) -> None:
        self.revision += 1

    def _ro(self) -> str:
        self._next_ro += 1
        return f"RO-{self._next_ro - 1}"

    def _slot(self, slot_id: str) -> Slot:
        for s in self.slots:
            if s.id == slot_id:
                return s
        raise ServiceDeskError(f"No time {slot_id!r}: use a slot_id from find_times.")

    def _require_customer(self) -> Customer:
        if self.customer is None:
            raise ServiceDeskError(
                "Find their file first: their name, then their number."
            )
        return self.customer

    def _require_vehicle(self) -> Vehicle:
        self._require_customer()
        if self.visit.vehicle is None:
            raise ServiceDeskError("Choose which car first.")
        return self.visit.vehicle

    def _require_shop(self) -> ShopVisit:
        if self.shop is None:
            raise ServiceDeskError("There's no car in the shop on this call.")
        return self.shop

    # ---- the seed ----

    def _seed_slots(self) -> list[Slot]:
        advisors = SEED["advisors"]
        booked = {tuple(x) for x in SEED["booked_slots"]}
        return [
            Slot(
                starts_at=datetime.combine(day, time.fromisoformat(t)),
                advisor=advisors[(i + j) % len(advisors)],
                taken=(day.weekday(), j) in booked,
            )
            for i, day in enumerate(self.days)
            for j, t in enumerate(SEED["slot_times"])
        ]

    def _seed_customer(self, c: dict[str, Any]) -> Customer:
        vehicles = [
            Vehicle(
                id=v["id"],
                year=v["year"],
                model=v["model"],
                miles=v["miles"],
                in_service=date.fromisoformat(v["in_service"])
                if "in_service" in v
                else self.today - timedelta(days=v["in_service_days_ago"]),
                trim=v.get("trim"),
            )
            for v in c["vehicles"]
        ]
        return Customer(name=c["name"], phone=c["phone"], vehicles=vehicles)

    def _seed_appointment(self) -> None:
        seeded = SEED["customers"][0]["appointment"]
        customer = self.customers[0]
        at = time.fromisoformat(seeded["time"])
        slot = next(
            s
            for s in self.slots
            if s.starts_at.time() == at and s.starts_at.date() == self.days[-1]
        )
        slot.taken = True
        self.appointments.append(
            Appointment(
                id="RO-48194",
                vehicle=customer.vehicle(seeded["vehicle_id"]),
                services=list(seeded["services"]),
                slot=slot,
                ride=seeded["ride"],
            )
        )

    def _seed_shop_visit(self) -> None:
        seeded = SEED["shop_visit"]
        customer = self.customers[0]
        vehicle = customer.vehicle(seeded["vehicle_id"])
        dropped_off = Slot(
            datetime.combine(self.today, time.fromisoformat(seeded["dropped_off"])),
            SEED["advisors"][0],
            taken=True,
        )
        appointment = Appointment(
            id="RO-48207",
            vehicle=vehicle,
            services=list(seeded["done"]),
            slot=dropped_off,
            ride="shuttle",
            concern=seeded["concern"],
        )
        self.appointments.append(appointment)
        self.customer = customer
        self.shop = ShopVisit(
            appointment=appointment,
            findings=[
                Finding(**f, labor_rate=seeded["labor_rate"])
                for f in seeded["findings"]
            ],
        )


def _in_part(at: datetime, part: PartOfDay) -> bool:
    if part == "morning":
        return at.hour < 12
    if part == "afternoon":
        return at.hour >= 12
    return True


def _slot_json(s: Slot) -> dict[str, Any]:
    return {"slotId": s.id, "when": spoken(s.starts_at), "advisor": s.advisor}
