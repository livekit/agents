from __future__ import annotations

import dataclasses
import datetime
import hashlib
import os
from dataclasses import dataclass, field

# GETAWAY_NOW pins the clock, so the scenarios' dates stay where they are
NOW = (
    datetime.date.fromisoformat(os.environ["GETAWAY_NOW"])
    if os.environ.get("GETAWAY_NOW")
    else None
)

AIRLINES = ("Skyline", "Northwind", "Bluejet")
HOTELS = ("The {} Grand", "Casa {}", "{} Harbor Inn", "Hotel Aurora", "The Lantern House")
CARDS = {"visa": "the Visa ending 4242", "amex": "the Amex ending 1005"}


class PaymentMissing(Exception):
    """No unused charge with that code covers the booking."""


def seed(*parts: object) -> int:
    """The same number for the same trip, so any city and date has fares and rooms."""
    text = "|".join(str(p).strip().lower() for p in parts)
    return int.from_bytes(hashlib.blake2s(text.encode(), digest_size=4).digest(), "big")


def spoken(when: datetime.date | datetime.datetime) -> str:
    if isinstance(when, datetime.datetime):
        return f"{when:%A, %B %d at %H:%M}"
    return f"{when:%A, %B %d}"


@dataclass(frozen=True)
class Flight:
    airline: str
    number: int
    origin: str
    destination: str
    out: datetime.datetime
    back: datetime.datetime
    fare: int
    """Per traveler, round trip, in dollars."""

    @property
    def id(self) -> str:
        return f"{self.airline[:2].upper()}{self.number}"


@dataclass(frozen=True)
class Hotel:
    name: str
    city: str
    stars: int
    nightly: int


@dataclass
class Charge:
    amount: int
    description: str
    card: str
    used: bool = False


@dataclass
class Travel:
    """The systems the three desks share: what was offered, charged and booked."""

    flights: dict[str, Flight] = field(default_factory=dict)
    charges: dict[str, Charge] = field(default_factory=dict)
    bookings: list[str] = field(default_factory=list)

    @staticmethod
    def today() -> datetime.date:
        return NOW or datetime.date.today()

    def search_flights(
        self, origin: str, destination: str, leave: datetime.date, back: datetime.date
    ) -> list[Flight]:
        found = []
        for airline in AIRLINES:
            for slot, (first, spread) in enumerate(((6, 4), (13, 6))):
                n = seed(airline, origin, destination, leave, back, slot)
                flight = Flight(
                    airline=airline,
                    number=100 + n % 900,
                    origin=origin,
                    destination=destination,
                    out=datetime.datetime.combine(leave, datetime.time(first + n % spread)),
                    back=datetime.datetime.combine(back, datetime.time(first + n // 7 % spread)),
                    fare=10 * (18 + n // 11 % 45),
                )
                # an id names one flight: the next free number when another flight holds it
                while self.flights.setdefault(flight.id, flight) != flight:
                    flight = dataclasses.replace(flight, number=flight.number + 1)
                found.append(flight)
        return sorted(found, key=lambda f: f.fare)

    def search_hotels(self, city: str, check_in: datetime.date) -> list[Hotel]:
        found = []
        for name in HOTELS:
            n = seed(name, city, check_in)
            if n % 4 == 0:
                continue  # full that night
            found.append(Hotel(name.format(city.title()), city, 3 + n % 3, 10 * (9 + n // 5 % 26)))
        return sorted(found, key=lambda h: h.nightly)

    def charge(self, amount: int, description: str, card: str) -> str:
        code = f"PAY-{seed(amount, description, card, len(self.charges)) % 10000:04d}"
        self.charges[code] = Charge(amount, description, card)
        return code

    def redeem(self, code: str, amount: int) -> None:
        charge = self.charges.get(code.strip().upper())
        if charge is None or charge.used or charge.amount != amount:
            raise PaymentMissing
        charge.used = True


# one per process: the payments desk writes it, the flight and hotel desks read it
travel = Travel()
