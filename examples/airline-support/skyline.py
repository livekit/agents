from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

FareClass = Literal["light", "classic", "flex"]
BagStatus = Literal["in_transit", "delivered", "lost"]

CHANGE_FEE_EUR: dict[FareClass, int] = {"light": 70, "classic": 40, "flex": 0}
BAG_DAILY_ALLOWANCE_EUR = 50  # receipted essentials, per day the bag was late
LOST_BAG_CAP_EUR = 1300

POLICY = f"""Skyline Air customer policy.
- EU261: a flight arriving 3 hours or more late, or cancelled with less than 14 days notice,
  pays EUR 250 up to 1500 km, EUR 400 up to 3500 km, EUR 600 beyond, unless the cause was an
  extraordinary circumstance (weather, air traffic control, security). No compensation then.
- Baggage: receipted essentials are reimbursed up to EUR {BAG_DAILY_ALLOWANCE_EUR} per day
  the bag was late. A lost bag pays its declared value, up to EUR {LOST_BAG_CAP_EUR}.
- Refunds: refundable fares get the full price back; other fares get nothing back.
- Changes: the change fee is EUR 70 on light, EUR 40 on classic, nothing on flex, plus any
  fare difference.
- A reply states the amount exactly as computed, promises nothing else, and never mentions
  internal limits or reviews."""


@dataclass
class Flight:
    number: str
    origin: str
    destination: str
    distance_km: int
    scheduled_arrival: datetime
    actual_arrival: datetime | None  # None: cancelled
    cause: str | None = None
    extraordinary: bool = False
    cancellation_notice_days: int | None = None

    @property
    def delay(self) -> timedelta:
        if self.actual_arrival is None:
            return timedelta(0)
        return max(self.actual_arrival - self.scheduled_arrival, timedelta(0))


@dataclass
class Booking:
    ref: str
    passenger: str
    email: str
    flights: list[str]
    fare_class: FareClass
    refundable: bool
    price_eur: int


@dataclass
class BaggageFile:
    pir: str
    booking_ref: str
    status: BagStatus
    days_late: int
    receipts_eur: int
    declared_value_eur: int


@dataclass
class Payment:
    booking_ref: str
    eur: int
    reason: str


@dataclass
class Skyline:
    """The airline's own systems, in memory: bookings, flights, baggage files, payments."""

    bookings: dict[str, Booking]
    flights: dict[str, Flight]
    bags: dict[str, BaggageFile]
    payments: list[Payment] = field(default_factory=list)

    @classmethod
    def seeded(cls) -> Skyline:
        def at(day: int, hour: int, minute: int = 0) -> datetime:
            return datetime(2026, 9, day, hour, minute)

        flights = [
            # 3h20 late on a short hop: EUR 250
            Flight("SK204", "AMS", "BCN", 1240, at(14, 11, 30), at(14, 14, 50), "crew shortage"),
            # 5h late on a long haul: EUR 600, above what is paid without a person
            Flight("SK880", "AMS", "JFK", 5850, at(15, 13), at(15, 18), "technical fault"),
            # 4h late, thunderstorm: extraordinary, nothing due
            Flight("SK312", "CDG", "ATH", 2100, at(16, 16), at(16, 20), "thunderstorm", True),
            # cancelled two days before: EUR 400
            Flight(
                "SK517",
                "AMS",
                "LIS",
                1850,
                at(18, 10),
                None,
                "aircraft swap",
                cancellation_notice_days=2,
            ),
            Flight("SK101", "AMS", "LHR", 370, at(20, 9), at(20, 9, 10)),
        ]
        bookings = [
            Booking(
                "SKY4F2", "Ana Garcia", "ana.garcia@example.com", ["SK204"], "classic", False, 189
            ),
            Booking(
                "SKY7Q9", "Marcus Lee", "marcus.lee@example.com", ["SK880"], "flex", True, 1240
            ),
            Booking(
                "SKY2M8", "Priya Nair", "priya.nair@example.com", ["SK312"], "light", False, 145
            ),
            Booking(
                "SKY5C1", "Tom Okafor", "tom.okafor@example.com", ["SK517"], "classic", False, 210
            ),
            Booking(
                "SKY9B3", "Eleanor Smith", "eleanor.smith@example.com", ["SK101"], "flex", True, 320
            ),
        ]
        bags = [BaggageFile("AMSSK10422", "SKY9B3", "delivered", 3, 180, 600)]
        return cls(
            bookings={b.ref: b for b in bookings},
            flights={f.number: f for f in flights},
            bags={b.pir: b for b in bags},
        )

    def find(self, ref: str, email: str) -> Booking | None:
        """The booking, only when the sender is its passenger."""
        booking = self.bookings.get(ref.strip().upper())
        if booking is None or booking.email.lower() != email.strip().lower():
            return None
        return booking

    def paid_eur(self, booking_ref: str) -> int:
        return sum(p.eur for p in self.payments if p.booking_ref == booking_ref)

    def pay(self, booking_ref: str, eur: int, reason: str) -> Payment:
        payment = Payment(booking_ref, eur, reason)
        self.payments.append(payment)
        return payment
